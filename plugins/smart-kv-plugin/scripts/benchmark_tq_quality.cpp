/* TQ quality benchmark with Q4_0, Q8_0, F16 comparisons */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../turbo-quant-plugin/turboquant.h"

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-10));
}

// FP16 round-trip
static void roundtrip_f16(const float* src, float* dst, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t bits; memcpy(&bits, &src[i], 4);
        uint32_t s = (bits >> 16) & 0x8000;
        int32_t e = ((bits >> 23) & 0xFF) - 112;
        if (e > 31) e = 31; if (e < 0) e = 0;
        uint32_t m = (bits >> 13) & 0x3FF;
        uint16_t h = (uint16_t)(s | ((uint32_t)e << 10) | m);
        // back to f32
        uint32_t bs = (uint32_t)(h & 0x8000u) << 16;
        uint32_t be = (uint32_t)(h & 0x7C00u) >> 10;
        uint32_t bm = (uint32_t)(h & 0x03FFu);
        if (be == 0) { if (bm == 0) { dst[i] = 0; continue; }
            int shift = 0; while ((bm & 0x0400u) == 0) { bm <<= 1; shift++; }
            bm &= 0x03FFu; be = 113 - (uint32_t)shift;
        } else { be += 112; }
        uint32_t result = bs | (be << 23) | (bm << 13);
        memcpy(&dst[i], &result, 4);
    }
}

// Q8_0: block of 32, scale FP16 + 8-bit values
#define QK8_0 32
static void quant_q8_0(const float* src, uint8_t* dst, int n) {
    for (int i = 0; i < n; i += QK8_0) {
        float amax = 0;
        for (int j = 0; j < QK8_0 && (i + j) < n; j++) {
            float a = fabsf(src[i + j]); if (a > amax) amax = a;
        }
        if (amax == 0) { memset(dst + i / QK8_0 * (sizeof(uint16_t) + QK8_0), 0, sizeof(uint16_t) + QK8_0); continue; }
        float d = amax / 127.0f;
        uint16_t dq;
        { uint32_t bits; memcpy(&bits, &d, 4);
          uint32_t s = (bits >> 16) & 0x8000;
          int32_t e = ((bits >> 23) & 0xFF) - 112;
          if (e > 31) e = 31; if (e < 0) e = 0;
          uint32_t m = (bits >> 13) & 0x3FF;
          dq = (uint16_t)(s | ((uint32_t)e << 10) | m); }
        memcpy(dst + i / QK8_0 * (sizeof(uint16_t) + QK8_0), &dq, 2);
        for (int j = 0; j < QK8_0 && (i + j) < n; j++) {
            int q = (int)(src[i + j] / d);
            if (q > 127) q = 127; if (q < -127) q = -127;
            dst[i / QK8_0 * (sizeof(uint16_t) + QK8_0) + 2 + j] = (uint8_t)(q & 0xFF);
        }
    }
}

static void dequant_q8_0(const uint8_t* src, float* dst, int n) {
    for (int i = 0; i < n; i += QK8_0) {
        uint16_t dq; memcpy(&dq, src + i / QK8_0 * (sizeof(uint16_t) + QK8_0), 2);
        uint32_t bs = (uint32_t)(dq & 0x8000u) << 16;
        uint32_t be = (uint32_t)(dq & 0x7C00u) >> 10;
        uint32_t bm = (uint32_t)(dq & 0x03FFu);
        float d;
        if (be == 0) { if (bm == 0) { d = 0; } else {
            int shift = 0; while ((bm & 0x0400u) == 0) { bm <<= 1; shift++; }
            bm &= 0x03FFu; be = 113 - (uint32_t)shift;
            uint32_t r = bs | (be << 23) | (bm << 13); memcpy(&d, &r, 4); }
        } else { be += 112;
            uint32_t r = bs | (be << 23) | (bm << 13); memcpy(&d, &r, 4); }
        for (int j = 0; j < QK8_0 && (i + j) < n; j++) {
            int8_t q = (int8_t)src[i / QK8_0 * (sizeof(uint16_t) + QK8_0) + 2 + j];
            dst[i + j] = q * d;
        }
    }
}

// Q4_0: block of 32, scale FP16 + 16 bytes of 4-bit values (low nibble first)
#define QK4_0 32
static void quant_q4_0(const float* src, uint8_t* dst, int n) {
    for (int i = 0; i < n; i += QK4_0) {
        float amax = 0;
        for (int j = 0; j < QK4_0 && (i + j) < n; j++) {
            float a = fabsf(src[i + j]); if (a > amax) amax = a;
        }
        if (amax == 0) { memset(dst + i / QK4_0 * (sizeof(uint16_t) + QK4_0 / 2), 0, sizeof(uint16_t) + QK4_0 / 2); continue; }
        float d = amax / 7.0f;
        uint16_t dq;
        { uint32_t bits; memcpy(&bits, &d, 4);
          uint32_t s = (bits >> 16) & 0x8000;
          int32_t e = ((bits >> 23) & 0xFF) - 112;
          if (e > 31) e = 31; if (e < 0) e = 0;
          uint32_t m = (bits >> 13) & 0x3FF;
          dq = (uint16_t)(s | ((uint32_t)e << 10) | m); }
        memcpy(dst + i / QK4_0 * (sizeof(uint16_t) + QK4_0 / 2), &dq, 2);
        for (int j = 0; j < QK4_0 && (i + j) < n; j += 2) {
            int q0 = (int)(src[i + j] / d + 8.0f); if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            int q1 = (int)(src[i + j + 1] / d + 8.0f); if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            dst[i / QK4_0 * (sizeof(uint16_t) + QK4_0 / 2) + 2 + j / 2] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

static void dequant_q4_0(const uint8_t* src, float* dst, int n) {
    for (int i = 0; i < n; i += QK4_0) {
        uint16_t dq; memcpy(&dq, src + i / QK4_0 * (sizeof(uint16_t) + QK4_0 / 2), 2);
        uint32_t bs = (uint32_t)(dq & 0x8000u) << 16;
        uint32_t be = (uint32_t)(dq & 0x7C00u) >> 10;
        uint32_t bm = (uint32_t)(dq & 0x03FFu);
        float d;
        if (be == 0) { if (bm == 0) { d = 0; } else {
            int shift = 0; while ((bm & 0x0400u) == 0) { bm <<= 1; shift++; }
            bm &= 0x03FFu; be = 113 - (uint32_t)shift;
            uint32_t r = bs | (be << 23) | (bm << 13); memcpy(&d, &r, 4); }
        } else { be += 112;
            uint32_t r = bs | (be << 23) | (bm << 13); memcpy(&d, &r, 4); }
        for (int j = 0; j < QK4_0 && (i + j) < n; j += 2) {
            uint8_t q = src[i / QK4_0 * (sizeof(uint16_t) + QK4_0 / 2) + 2 + j / 2];
            dst[i + j]     = ((int)(q & 0x0F) - 8) * d;
            dst[i + j + 1] = ((int)(q >> 4)    - 8) * d;
        }
    }
}

static void generate_kv_vector(float* out, int d, int idx, unsigned int* seed) {
    unsigned int local = *seed + idx;
    int n_freqs = d / 16;
    for (int i = 0; i < d; i++) {
        float val = 0;
        for (int f = 0; f < n_freqs; f++) {
            local = local * 1103515245 + 12345;
            float freq = 0.5f + (local % 1000) / 500.0f;
            local = local * 1103515245 + 12345;
            float phase = (local % 628) / 100.0f;
            local = local * 1103515245 + 12345;
            float amp = (local % 1000) / 1000.0f;
            val += amp * cosf(freq * i * (float)M_PI / d + phase);
        }
        out[i] = val;
    }
    float mag = 0.3f + 1.7f * (float)(idx % 17) / 16.0f;
    for (int i = 0; i < d; i++)
        out[i] *= mag;
}

int main() {
    int d = 256;
    int n = 5000;
    printf("Quality comparison: TQ vs Q4_0, Q8_0, F16\n");
    printf("========================================\n");
    printf("head_dim=%d, n_vecs=%d\n\n", d, n);

    float* vecs = (float*)malloc(n * d * sizeof(float));
    unsigned int seed = 42;
    for (int i = 0; i < n; i++)
        generate_kv_vector(vecs + i * d, d, i, &seed);

    printf("%-8s  %-10s  %-8s  %-8s\n", "Format", "CosSim", "Bytes/h", "vs F16");
    printf("--------  ----------  --------  --------\n");

    // F16
    {
        float* dec = (float*)malloc(d * sizeof(float));
        double cs = 0;
        for (int i = 0; i < n; i++) {
            roundtrip_f16(vecs + i * d, dec, d);
            cs += cosine_sim(vecs + i * d, dec, d);
        }
        printf("%-8s  %.6f  %-8d  %.2fx\n", "F16", cs / n, d * 2, 1.0);
        free(dec);
    }

    // Q8_0
    {
        const size_t row = (sizeof(uint16_t) + QK8_0) * d / QK8_0;
        double cs = 0;
        for (int i = 0; i < n; i++) {
            uint8_t* q = (uint8_t*)malloc(row);
            float* dec = (float*)malloc(d * sizeof(float));
            quant_q8_0(vecs + i * d, q, d);
            dequant_q8_0(q, dec, d);
            cs += cosine_sim(vecs + i * d, dec, d);
            free(q); free(dec);
        }
        printf("%-8s  %.6f  %-8d  %.2fx\n", "Q8_0", cs / n, (int)row, 2.0);
    }

    // Q4_0
    {
        const size_t row = (sizeof(uint16_t) + QK4_0 / 2) * d / QK4_0;
        double cs = 0;
        for (int i = 0; i < n; i++) {
            uint8_t* q = (uint8_t*)malloc(row);
            float* dec = (float*)malloc(d * sizeof(float));
            quant_q4_0(vecs + i * d, q, d);
            dequant_q4_0(q, dec, d);
            cs += cosine_sim(vecs + i * d, dec, d);
            free(q); free(dec);
        }
        printf("%-8s  %.6f  %-8d  %.2fx\n", "Q4_0", cs / n, (int)row, 4.0);
    }

    // TQ 2-6
    for (int bits = 2; bits <= 6; bits++) {
        size_t sz = turboquant_size(d, bits);
        float* dec = (float*)malloc(d * sizeof(float));
        uint8_t* buf = (uint8_t*)malloc(sz);
        double cs = 0;
        for (int i = 0; i < n; i++) {
            turboquant_encode(vecs + i * d, d, buf, bits, 0);
            turboquant_decode(buf, d, dec, bits, 0);
            cs += cosine_sim(vecs + i * d, dec, d);
        }
        free(buf); free(dec);
        float ratio = (float)(d * 2) / (float)sz;
            printf("TQ-%d    %.6f  %-8zu  %.2fx\n", bits, cs / n, sz, ratio);
    }

    free(vecs);
    return 0;
}
