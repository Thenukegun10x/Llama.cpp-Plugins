"""Investigate TQ-3 outlier: binary tree bias from cos/sin LUT alignment."""
import struct, math, os, sys
import numpy as np

# Load the compiled DLL for real encode/decode
dll_path = os.path.join(os.path.dirname(__file__), 'build', 'Release', 'turbo-quant-plugin.dll')
if not os.path.exists(dll_path):
    dll_path = os.path.join(os.path.dirname(__file__), 'build', 'turbo-quant-plugin.dll')

# ---- Pure Python reimplementation for analysis ----
def f32_to_f16(f):
    bits = struct.pack('f', f)
    u = struct.unpack('I', bits)[0]
    s = (u >> 16) & 0x8000
    e = ((u >> 23) & 0xFF) - 112
    if e > 31: e = 31
    if e < 0: e = 0
    m = (u >> 13) & 0x3FF
    return s | (e << 10) | m

def f16_to_f32(h):
    s = (h & 0x8000) << 16
    e = (h & 0x7C00) >> 10
    m = h & 0x03FF
    if e == 0:
        if m == 0: return 0.0
        shift = 0
        while (m & 0x0400) == 0:
            m <<= 1; shift += 1
        m &= 0x03FF
        e = 113 - shift
    else:
        e += 112
    bits = s | (e << 23) | (m << 13)
    return struct.unpack('f', struct.pack('I', bits))[0]

def build_cos_sin_lut(bits):
    n = 1 << bits
    lut = []
    for i in range(n):
        theta = (i + 0.5) / n * 2.0 * math.pi - math.pi
        lut.append((math.cos(theta), math.sin(theta)))
    return lut

def quant_angle(theta, bits):
    u = (theta + math.pi) / (2.0 * math.pi)
    u = max(0.0, min(0.9999999, u))
    return int(u * (1 << bits))

def write_bits(buf, bit_pos, val, bits):
    byte_off = bit_pos >> 3
    bit_off = bit_pos & 7
    mask = (1 << bits) - 1
    cur = (buf[byte_off] | (buf[byte_off + 1] << 8) | (buf[byte_off + 2] << 16))
    cur &= ~(mask << bit_off)
    cur |= ((val & mask) << bit_off)
    buf[byte_off] = cur & 0xFF
    buf[byte_off + 1] = (cur >> 8) & 0xFF
    buf[byte_off + 2] = (cur >> 16) & 0xFF

def read_bits(buf, bit_pos, bits):
    byte_off = bit_pos >> 3
    bit_off = bit_pos & 7
    mask = (1 << bits) - 1
    cur = (buf[byte_off] | (buf[byte_off + 1] << 8) | (buf[byte_off + 2] << 16))
    return (cur >> bit_off) & mask

def polar_quant_encode(src, bits):
    d = len(src)
    buf = bytearray(4 + (d - 1) * bits // 8 + 8)
    work = list(src)
    bit_pos = 0
    n = d
    while n > 1:
        for i in range(n // 2):
            x, y = work[2 * i], work[2 * i + 1]
            work[i] = math.sqrt(x * x + y * y)
            q = quant_angle(math.atan2(y, x), bits)
            write_bits(buf, bit_pos, q, bits)
            bit_pos += bits
        n //= 2
    fr = f32_to_f16(work[0])
    struct.pack_into('<H', buf, 0, fr)
    return buf

def polar_quant_decode(buf, d, bits):
    lut = build_cos_sin_lut(bits)
    fr = struct.unpack('<H', buf[0:2])[0]
    radius = f16_to_f32(fr)
    
    total_angles = d - 1
    angles = []
    bit_pos = 0
    for _ in range(total_angles):
        angles.append(read_bits(buf, bit_pos, bits))
        bit_pos += bits
    
    levels = 0
    tmp = d
    while tmp > 1:
        tmp >>= 1; levels += 1
    
    level_off = []
    off = 0
    for L in range(levels):
        level_off.append(off)
        off += d >> (L + 1)
    
    dst = np.zeros(d, dtype=np.float32)
    for j in range(d):
        r = radius
        node = 0
        for L in range(levels - 1, -1, -1):
            q = angles[level_off[L] + node]
            bit = (j >> L) & 1
            r *= lut[q][0] if bit == 0 else lut[q][1]
            node = (node << 1) | bit
        dst[j] = r
    return dst

def turboquant_encode(src, bits):
    d = len(src)
    sz = 4 + (d + (d - 1) * bits + 7) // 8 + 3
    buf = bytearray(sz)
    
    # PolarQuant: radius at 0, angles at 4
    work = list(src)
    bit_pos = 0
    n = d
    while n > 1:
        for i in range(n // 2):
            x, y = work[2 * i], work[2 * i + 1]
            work[i] = math.sqrt(x * x + y * y)
            q = quant_angle(math.atan2(y, x), bits)
            write_bits(buf, 4 + bit_pos, q, bits)
            bit_pos += bits
        n //= 2
    fr = f32_to_f16(work[0])
    struct.pack_into('<H', buf, 0, fr)
    
    # Decode to get residual
    decoded = polar_quant_decode(buf, d, bits)
    
    # QJL: alpha = mean(|residual|), store sign bits
    err = src - decoded
    alpha = np.mean(np.abs(err))
    af = f32_to_f16(alpha)
    struct.pack_into('<H', buf, 2, af)
    
    qjl_bit_pos = (d - 1) * bits
    for i in range(d):
        write_bits(buf, 4 + qjl_bit_pos, 1 if err[i] >= 0 else 0, 1)
        qjl_bit_pos += 1
    
    return buf

def turboquant_decode(buf, d, bits):
    decoded = polar_quant_decode(buf, d, bits)
    af = struct.unpack('<H', buf[2:4])[0]
    alpha = f16_to_f32(af)
    
    qjl_bit_pos = (d - 1) * bits
    for i in range(d):
        sign_bit = read_bits(buf, 4 + qjl_bit_pos, 1)
        qjl_bit_pos += 1
        decoded[i] += alpha if sign_bit else -alpha
    
    return decoded


# ---- Analysis ----
np.random.seed(42)

def test_signal(d=256):
    """Realistic KV cache-like signal."""
    t = np.arange(d, dtype=np.float32)
    val = np.zeros(d, dtype=np.float32)
    for f in range(4):
        val += np.cos((0.5 + f * 0.5) * t * np.pi / d) / (f + 1)
    return val * 1.2

def gaussian_random(d=256):
    return np.random.randn(d).astype(np.float32)

def uniform_random(d=256):
    return np.random.uniform(-1, 1, d).astype(np.float32)

# Analyze the LUT for each bit width
print("=" * 70)
print("COS/SIN LUT ANALYSIS")
print("=" * 70)
for bits in range(2, 7):
    n = 1 << bits
    lut = build_cos_sin_lut(bits)
    cos_vals = [c for c, s in lut]
    sin_vals = [s for c, s in lut]
    # The "split ratio" at each entry is max(|cos|,|sin|) / min(|cos|,|sin|)
    ratios = [max(abs(c), abs(s)) / max(1e-10, min(abs(c), abs(s))) for c, s in lut]
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    unique_abs_cos = len(set(round(abs(c), 6) for c in cos_vals))
    unique_abs_sin = len(set(round(abs(s), 6) for s in sin_vals))
    print(f"\nTQ-{bits}: {n} angles")
    print(f"  Unique |cos|: {unique_abs_cos}, Unique |sin|: {unique_abs_sin}")
    print(f"  Cos values: {sorted(set(round(c, 4) for c in cos_vals))}")
    print(f"  Split ratios: {[f'{r:.3f}' for r in sorted(set(round(r, 3) for r in ratios))]}")
    print(f"  Max split ratio: {max_ratio:.3f}, Min: {min_ratio:.3f}")

print("\n" + "=" * 70)
print("RECONSTRUCTION ERROR ANALYSIS (test signal, d=256)")
print("=" * 70)
src = test_signal(256)
nb = np.dot(src, src)
src_mag = math.sqrt(nb)
print(f"Source magnitude: {src_mag:.4f}")

print(f"\n{'Bits':<6} {'TQ_mag':<10} {'PQ_mag':<10} {'TQ_CosSim':<12} {'PQ_CosSim':<12} {'TQ_MSE':<12} {'PQ_MSE':<12}")
print("-" * 74)
for bits in range(2, 7):
    buf_tq = turboquant_encode(src, bits)
    tq = turboquant_decode(buf_tq, 256, bits)
    
    buf_pq = polar_quant_encode(src, bits)
    pq = polar_quant_decode(buf_pq, 256, bits)
    
    nt = np.dot(tq, tq)
    np_ = np.dot(pq, pq)
    dt = np.dot(src, tq)
    dp = np.dot(src, pq)
    
    tq_mag = math.sqrt(nt)
    pq_mag = math.sqrt(np_)
    tq_cos = dt / (tq_mag * src_mag + 1e-10)
    pq_cos = dp / (pq_mag * src_mag + 1e-10)
    tq_mse = np.mean((src - tq) ** 2)
    pq_mse = np.mean((src - pq) ** 2)
    
    print(f"TQ-{bits:<3} {tq_mag:<10.4f} {pq_mag:<10.4f} {tq_cos:<12.4f} {pq_cos:<12.4f} {tq_mse:<12.6f} {pq_mse:<12.6f}")

# Check magnitude inflation specifically: inflated = (||TQ|| - ||PQ||) / ||PQ||
print(f"\n{'Bits':<6} {'Mag Inflation':<16} {'QJL alpha':<12} {'QJL cos contrib':<18}")
print("-" * 52)
for bits in range(2, 7):
    buf_tq = turboquant_encode(src, bits)
    decoded = polar_quant_decode(buf_tq, 256, bits)
    af_val = struct.unpack('<H', buf_tq[2:4])[0]
    alpha = f16_to_f32(af_val)
    
    pq_mag = math.sqrt(np.dot(decoded, decoded))
    tq_mag = math.sqrt(np.dot(turboquant_decode(buf_tq, 256, bits), 
                              turboquant_decode(buf_tq, 256, bits)))
    mag_inflation = (tq_mag - pq_mag) / pq_mag * 100 if pq_mag > 0 else 0
    
    # Expected inflation: ||TQ||² = ||PQ||² + d·alpha² (if signs are independent)
    expected_inflation = math.sqrt(pq_mag**2 + 256 * alpha**2)
    exp_inflation_pct = (expected_inflation - pq_mag) / pq_mag * 100
    
    print(f"TQ-{bits:<3} {mag_inflation:<16.2f}% {alpha:<12.6f} {exp_inflation_pct:<18.2f}%")

# ---- Investigate: what makes TQ-3 special? ----
print("\n" + "=" * 70)
print("BINARY TREE BIAS: level-by-level reconstruction analysis")
print("=" * 70)

def analyze_tree_bias(bits):
    """For a single 2D polar decomposition, analyze the angular bias."""
    lut = build_cos_sin_lut(bits)
    n = 1 << bits
    biases = []
    for i in range(n):
        c, s = lut[i]
        # For a uniform input distribution, the expected output is
        # (E[r*cos(θ̂)], E[r*sin(θ̂)]) where θ̂ is the quantized angle
        # The bias is the deviation from ideal
        cos2 = c * c
        sin2 = s * s
        # If cos2 > sin2, more energy goes to the cos branch
        bias = abs(cos2 - sin2) / (cos2 + sin2 + 1e-10)
        biases.append(bias)
    return biases

for bits in range(2, 7):
    biases = analyze_tree_bias(bits)
    print(f"\nTQ-{bits}: per-angle split bias (0=balanced, 1=fully biased)")
    print(f"  Mean bias: {np.mean(biases):.4f}")
    print(f"  Max bias:  {np.max(biases):.4f}")
    print(f"  Unique biases: {sorted(set(round(b, 4) for b in biases))}")

# Show actual cos/sin pairs for TQ-3
print("\n" + "=" * 70)
print("TQ-3 DETAILED LUT")
print("=" * 70)
for i, (c, s) in enumerate(build_cos_sin_lut(3)):
    theta = math.atan2(s, c)
    ratio = max(abs(c), abs(s)) / max(1e-10, min(abs(c), abs(s)))
    print(f"  angle[{i}] = {theta:7.4f} rad ({theta*180/math.pi:6.1f}°)  cos={c:7.4f}  sin={s:7.4f}  split_ratio={ratio:.3f}")

# Check if TQ-3 has a moment where all signs align badly
print("\n" + "=" * 70)
print("QJL SIGN BIAS ANALYSIS")
print("=" * 70)
for bits in range(2, 7):
    buf_tq = turboquant_encode(src, bits)
    qjl_bit_pos = 255 * bits
    signs = []
    for i in range(256):
        signs.append(read_bits(buf_tq, 4 + qjl_bit_pos, 1))
        qjl_bit_pos += 1
    pct_pos = sum(signs) / 256 * 100
    print(f"TQ-{bits}: {pct_pos:.1f}% positive signs (alpha pushes vector outward)")

# ---- Alternative: test non-uniform codebook for 3-bit ----
print("\n" + "=" * 70)
print("ALTERNATIVE 3-BIT CODEBOOKS")
print("=" * 70)

def encode_decode_with_lut(src, d, bits, custom_lut=None):
    """Encode/decode PolarQuant with a custom LUT."""
    n = 1 << bits
    # Use custom LUT for quantization (find nearest angle by cos/sin)
    buf = bytearray(4 + (d - 1) * bits // 8 + 8)
    work = list(src)
    bit_pos = 0
    nd = d
    while nd > 1:
        for i in range(nd // 2):
            x, y = work[2 * i], work[2 * i + 1]
            work[i] = math.sqrt(x * x + y * y)
            theta = math.atan2(y, x)
            if custom_lut is not None:
                # Find nearest (cos, sin) pair in custom LUT
                best = 0
                best_dot = -2
                for j, (cc, ss) in enumerate(custom_lut):
                    dot = cc * math.cos(theta) + ss * math.sin(theta)
                    if dot > best_dot:
                        best_dot = dot
                        best = j
                q = best
            else:
                q = quant_angle(theta, bits)
            write_bits(buf, 4 + bit_pos, q, bits)
            bit_pos += bits
        nd //= 2
    fr = f32_to_f16(work[0])
    struct.pack_into('<H', buf, 0, fr)
    
    # Decode
    lut = custom_lut if custom_lut is not None else build_cos_sin_lut(bits)
    fr = struct.unpack('<H', buf[0:2])[0]
    radius = f16_to_f32(fr)
    total_angles = d - 1
    angles = []
    bit_pos = 0
    for _ in range(total_angles):
        angles.append(read_bits(buf, bit_pos, bits))
        bit_pos += bits
    levels = 0
    tmp = d
    while tmp > 1:
        tmp >>= 1; levels += 1
    level_off = []
    off = 0
    for L in range(levels):
        level_off.append(off)
        off += d >> (L + 1)
    dst = np.zeros(d, dtype=np.float32)
    for j in range(d):
        r = radius
        node = 0
        for L in range(levels - 1, -1, -1):
            q = angles[level_off[L] + node]
            bit = (j >> L) & 1
            r *= lut[q][0] if bit == 0 else lut[q][1]
            node = (node << 1) | bit
        dst[j] = r
    return dst

def optimal_angle_quantize(src, d, bits, custom_lut):
    """Full encode/decode including QJL with custom LUT."""
    n = 1 << bits
    buf = bytearray(4 + (d + (d - 1) * bits + 7) // 8 + 3)
    
    # PolarQuant encode with custom LUT
    work = list(src)
    bit_pos = 0
    nd = d
    while nd > 1:
        for i in range(nd // 2):
            x, y = work[2 * i], work[2 * i + 1]
            work[i] = math.sqrt(x * x + y * y)
            theta = math.atan2(y, x)
            if custom_lut is not None:
                best = 0
                best_dot = -2
                for j, (cc, ss) in enumerate(custom_lut):
                    dot = cc * math.cos(theta) + ss * math.sin(theta)
                    if dot > best_dot:
                        best_dot = dot
                        best = j
                q = best
            else:
                q = quant_angle(theta, bits)
            write_bits(buf, 4 + bit_pos, q, bits)
            bit_pos += bits
        nd //= 2
    fr = f32_to_f16(work[0])
    struct.pack_into('<H', buf, 0, fr)
    
    lut = custom_lut if custom_lut is not None else build_cos_sin_lut(bits)
    fr = struct.unpack('<H', buf[0:2])[0]
    radius = f16_to_f32(fr)
    total_angles = d - 1
    angles = []
    bit_pos = 0
    for _ in range(total_angles):
        angles.append(read_bits(buf, bit_pos, bits))
        bit_pos += bits
    levels = 0
    tmp = d
    while tmp > 1:
        tmp >>= 1; levels += 1
    level_off = []
    off = 0
    for L in range(levels):
        level_off.append(off)
        off += d >> (L + 1)
    decoded = np.zeros(d, dtype=np.float32)
    for j in range(d):
        r = radius
        node = 0
        for L in range(levels - 1, -1, -1):
            q = angles[level_off[L] + node]
            bit = (j >> L) & 1
            r *= lut[q][0] if bit == 0 else lut[q][1]
            node = (node << 1) | bit
        decoded[j] = r
    
    err = src - decoded
    alpha = np.mean(np.abs(err))
    af = f32_to_f16(alpha)
    struct.pack_into('<H', buf, 2, af)
    qjl_bit_pos = (d - 1) * bits
    for i in range(d):
        write_bits(buf, 4 + qjl_bit_pos, 1 if err[i] >= 0 else 0, 1)
        qjl_bit_pos += 1
    
    # Final decode
    for i in range(d):
        sign_bit = read_bits(buf, 4 + (d - 1) * bits + i, 1)
        decoded[i] += alpha if sign_bit else -alpha
    return decoded

# Option 1: Uniform angles shifted by half-bin (integer multiples of π/4)
# These give angles at 0, π/4, π/2, 3π/4, π, -3π/4, -π/2, -π/4
opt_lut_1 = [(math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)) for i in range(8)]
# Option 2: Mix of balanced and extreme splits
opt_lut_2 = []
for i in range(8):
    # Mix angles between π/8 and π/4 offsets to get varied cos/sin ratios
    theta = (i * 2 * math.pi / 8) + math.pi / 6  # 30° offset
    opt_lut_2.append((math.cos(theta), math.sin(theta)))
# Option 3: Optimized for binary tree - minimize worst-case split bias
# Include angles that give more balanced splits
opt_lut_3 = [
    (math.cos(t), math.sin(t)) 
    for t in [0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6, math.pi, -5*math.pi/6]
]  # Only 7, add one more
opt_lut_3.append((math.cos(-math.pi/6), math.sin(-math.pi/6)))

# Try all options
luts = {
    "Uniform (original)": None,
    "Cardinal angles (0,45,90...)": opt_lut_1,
    "30° offset": opt_lut_2,
}

print(f"Testing on signal (d=256):")
for name, lut in luts.items():
    if lut is not None:
        pq = encode_decode_with_lut(src, 256, 3, lut)
    else:
        pq = encode_decode_with_lut(src, 256, 3)
    pq_mag = math.sqrt(np.dot(pq, pq))
    pq_cos = np.dot(src, pq) / (pq_mag * src_mag + 1e-10)
    pq_mse = np.mean((src - pq) ** 2)
    
    # With full TQ
    if lut is not None:
        tq = optimal_angle_quantize(src, 256, 3, lut)
    else:
        buf = turboquant_encode(src, 3)
        tq = turboquant_decode(buf, 256, 3)
    tq_mag = math.sqrt(np.dot(tq, tq))
    tq_cos = np.dot(src, tq) / (tq_mag * src_mag + 1e-10)
    tq_mse = np.mean((src - tq) ** 2)
    mag_inf = (tq_mag - src_mag) / src_mag * 100
    
    print(f"\n{name}:")
    print(f"  PQ CosSim={pq_cos:.4f} MSE={pq_mse:.6f}")
    print(f"  TQ CosSim={tq_cos:.4f} MSE={tq_mse:.6f} MagInflation={mag_inf:.2f}%")

# ---- Key insight: show that TQ-3's issue is that cos and sin are always ±0.3827 or ±0.9239 ----
print("\n" + "=" * 70)
print("ROOT CAUSE: Cos/Sin magnitude pairs at each bit width")
print("=" * 70)
for bits in range(2, 7):
    lut = build_cos_sin_lut(bits)
    pairs = sorted(set((round(abs(c), 6), round(abs(s), 6)) for c, s in lut))
    print(f"\nTQ-{bits}: (|cos|, |sin|) pairs: {pairs}")
    ratios = [f"{max(p[0],p[1])/min(p[0],p[1]):.3f}" for p in pairs]
    print(f"  Split ratios: {ratios}")
    unique_vals = sorted(set(v for p in pairs for v in p))
    print(f"  Unique |cos| or |sin| values: {unique_vals}")
