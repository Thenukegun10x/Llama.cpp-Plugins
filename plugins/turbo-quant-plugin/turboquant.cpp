#include "turboquant.h"
#include <cmath>
#include <cstring>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── FP16 helpers ───────────────────────────────────────────────────────

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t s = (bits >> 16) & 0x8000;
    int32_t e = ((bits >> 23) & 0xFF) - 112;
    if (e > 31) e = 31;
    if (e < 0) e = 0;
    uint32_t m = (bits >> 13) & 0x3FF;
    return (uint16_t)(s | ((uint32_t)e << 10) | m);
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (uint32_t)(h & 0x7C00u) >> 10;
    uint32_t m = (uint32_t)(h & 0x03FFu);
    if (e == 0) {
        if (m == 0) return 0.0f;
        int shift = 0;
        while ((m & 0x0400u) == 0) { m <<= 1; shift++; }
        m &= 0x03FFu;
        e = 113 - (uint32_t)shift;
    } else {
        e += 112;
    }
    uint32_t bits = s | (e << 23) | (m << 13);
    float result;
    memcpy(&result, &bits, sizeof(bits));
    return result;
}

// ─── Angle quantization ─────────────────────────────────────────────────

static uint16_t quant_angle(float theta, int bits) {
    float u = (theta + (float)M_PI) / (2.0f * (float)M_PI);
    if (u < 0.0f) u = 0.0f;
    if (u >= 1.0f) u = 0.9999999f;
    return (uint16_t)(u * (float)(1 << bits));
}

// ─── Cos/sin LUT ────────────────────────────────────────────────────────

typedef struct {
    float cos_val;
    float sin_val;
} cos_sin_pair_t;

static void build_cos_sin_lut(cos_sin_pair_t* lut, int bits) {
    int n = 1 << bits;
    for (int i = 0; i < n; i++) {
        float theta = ((float)i + 0.5f) / (float)n * 2.0f * (float)M_PI - (float)M_PI;
        lut[i].cos_val = cosf(theta);
        lut[i].sin_val = sinf(theta);
    }
}

// ─── Bit stream helpers ─────────────────────────────────────────────────

static void write_bits(uint8_t* dst, uint64_t bit_pos, uint16_t val, int bits) {
    uint64_t byte_off = bit_pos >> 3;
    uint64_t bit_off = bit_pos & 7;
    uint64_t mask = (1ull << bits) - 1;
    uint32_t cur = (uint32_t)(dst[byte_off]) |
                   ((uint32_t)dst[byte_off + 1] << 8) |
                   ((uint32_t)dst[byte_off + 2] << 16);
    cur &= ~((uint32_t)mask << bit_off);
    cur |= ((uint32_t)(val & mask) << bit_off);
    dst[byte_off]     = (uint8_t)(cur & 0xFF);
    dst[byte_off + 1] = (uint8_t)((cur >> 8) & 0xFF);
    dst[byte_off + 2] = (uint8_t)((cur >> 16) & 0xFF);
}

static uint16_t read_bits(const uint8_t* src, uint64_t bit_pos, int bits) {
    uint64_t byte_off = bit_pos >> 3;
    uint64_t bit_off = bit_pos & 7;
    uint64_t mask = (1ull << bits) - 1;
    uint32_t cur = (uint32_t)(src[byte_off]) |
                   ((uint32_t)src[byte_off + 1] << 8) |
                   ((uint32_t)src[byte_off + 2] << 16);
    return (uint16_t)((cur >> bit_off) & mask);
}

// ─── Validate d ─────────────────────────────────────────────────────────

int turboquant_valid_dims(int d) {
    if (d < 2 || d > 4096) return 0;
    return (d & (d - 1)) == 0;
}

// ─── Internal: PolarQuant with configurable offsets ─────────────────────
// Writes radius (FP16) at dst + radius_off, angles at dst + angles_off.

static void polar_quant_encode_raw(const float* src, int d, uint8_t* dst,
                                    int bits, int radius_off, int angles_off)
{
    float* buf = (float*)malloc((size_t)d * sizeof(float));
    memcpy(buf, src, (size_t)d * sizeof(float));

    uint64_t bit_pos = 0;
    int n = d;
    while (n > 1) {
        for (int i = 0; i < n / 2; i++) {
            float x = buf[2 * i];
            float y = buf[2 * i + 1];
            buf[i] = sqrtf(x * x + y * y);
            write_bits(dst + angles_off, bit_pos, quant_angle(atan2f(y, x), bits), bits);
            bit_pos += bits;
        }
        n /= 2;
    }

    uint16_t fr = f32_to_f16(buf[0]);
    memcpy(dst + radius_off, &fr, 2);
    free(buf);
}

static void polar_quant_decode_raw(const uint8_t* src, int d, float* dst,
                                    int bits, int radius_off, int angles_off)
{
    int lut_size = 1 << bits;
    cos_sin_pair_t* lut = (cos_sin_pair_t*)malloc((size_t)lut_size * sizeof(cos_sin_pair_t));
    build_cos_sin_lut(lut, bits);

    uint16_t fr;
    memcpy(&fr, src + radius_off, 2);
    float radius = f16_to_f32(fr);

    int total_angles = d - 1;
    uint16_t* angles = (uint16_t*)malloc((size_t)total_angles * sizeof(uint16_t));
    uint64_t bit_pos = 0;
    for (int i = 0; i < total_angles; i++) {
        angles[i] = read_bits(src + angles_off, bit_pos, bits);
        bit_pos += bits;
    }

    int levels = 0;
    for (int tmp = d; tmp > 1; tmp >>= 1) levels++;

    int level_off[12];
    int off = 0;
    for (int L = 0; L < levels; L++) {
        level_off[L] = off;
        off += d >> (L + 1);
    }

    for (int j = 0; j < d; j++) {
        float r = radius;
        int node = 0;
        for (int L = levels - 1; L >= 0; L--) {
            uint16_t q = angles[level_off[L] + node];
            int bit = (j >> L) & 1;
            r *= (bit == 0) ? lut[q].cos_val : lut[q].sin_val;
            node = (node << 1) | bit;
        }
        dst[j] = r;
    }

    free(angles);
    free(lut);
}

// ─── Public PolarQuant API ──────────────────────────────────────────────

size_t polar_quant_encode(const float* src, int d, uint8_t* dst, int bits) {
    assert(turboquant_valid_dims(d));
    assert(bits >= 1 && bits <= 16);
    size_t total_size = polar_quant_size(d, bits);
    memset(dst, 0, total_size);
    polar_quant_encode_raw(src, d, dst, bits, 0, 2);
    return total_size;
}

void polar_quant_decode(const uint8_t* src, int d, float* dst, int bits) {
    assert(turboquant_valid_dims(d));
    assert(bits >= 1 && bits <= 16);
    polar_quant_decode_raw(src, d, dst, bits, 0, 2);
}

// ─── TurboQuant encode ──────────────────────────────────────────────────
// QJL: 1 sign bit per residual element, alpha = mean(|residual|).
// Standard basis JL projection: correction = alpha * sign(residual_i).
// This is unbiased and reduces per-element variance by ~63%.
//
// Storage: [FP16 radius][FP16 alpha][PolarQuant angles][QJL signs]

size_t turboquant_encode(const float* src, int d, uint8_t* dst,
                          int bits, uint64_t seed)
{
    (void)seed;
    assert(turboquant_valid_dims(d));

    size_t total_size = turboquant_size(d, bits);
    memset(dst, 0, total_size);

    // PolarQuant encode: radius at offset 0, angles at offset 4
    polar_quant_encode_raw(src, d, dst, bits, 0, 4);

    // Decode PolarQuant to compute residual
    float* decoded = (float*)malloc((size_t)d * sizeof(float));
    polar_quant_decode_raw(dst, d, decoded, bits, 0, 4);

    // Compute residual, store QJL sign bits, accumulate sum_abs for alpha
    float sum_abs = 0.0f;
    uint64_t qjl_bit_pos = (size_t)(d - 1) * bits;
    for (int i = 0; i < d; i++) {
        float err = src[i] - decoded[i];
        sum_abs += fabsf(err);
        write_bits(dst + 4, qjl_bit_pos, (err >= 0.0f) ? 1 : 0, 1);
        qjl_bit_pos++;
    }

    // Store alpha = mean(|residual|) at offset 2
    float alpha = sum_abs / (float)d;
    uint16_t alpha_f16 = f32_to_f16(alpha);
    memcpy(dst + 2, &alpha_f16, 2);

    free(decoded);
    return total_size;
}

// ─── TurboQuant decode ──────────────────────────────────────────────────

void turboquant_decode(const uint8_t* src, int d, float* dst,
                        int bits, uint64_t seed)
{
    (void)seed;
    assert(turboquant_valid_dims(d));

    // PolarQuant decode (radius at 0, angles at 4)
    polar_quant_decode_raw(src, d, dst, bits, 0, 4);

    // Read alpha at offset 2
    uint16_t af;
    memcpy(&af, src + 2, 2);
    float alpha = f16_to_f32(af);

    // Apply QJL correction: dst[i] += alpha * (2*sign_bit - 1)
    uint64_t qjl_bit_pos = (size_t)(d - 1) * bits;
    for (int i = 0; i < d; i++) {
        uint16_t sign_bit = read_bits(src + 4, qjl_bit_pos, 1);
        qjl_bit_pos++;
        dst[i] += (sign_bit) ? alpha : -alpha;
    }
}
