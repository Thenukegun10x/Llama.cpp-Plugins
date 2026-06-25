#pragma once
#include <stdint.h>
#include <stddef.h>

// ─── PolarQuant ──────────────────────────────────────────────────────────
// Recursive polar coordinate transform for KV cache compression.
//
// Algorithm: recursively pair vector elements, convert (x,y) → (r, θ),
// quantize θ to N bits, recurse on radii until one radius remains.
// Storage: [FP16 final_radius] + [(d-1) * bits_per_angle packed bits]

// Returns packed byte size of a PolarQuant-encoded vector.
// Includes 3-byte padding for safe 3-byte reads in write_bits/read_bits.
static inline size_t polar_quant_size(int d, int bits_per_angle) {
    size_t total_bits = (size_t)(d - 1) * bits_per_angle;
    return 2 + (total_bits + 7) / 8 + 3;
}

// Encode vector src[0..d) into dst. d must be power of 2.
// Returns bytes written (always polar_quant_size(d, bits_per_angle))
size_t polar_quant_encode(const float* src, int d, uint8_t* dst, int bits_per_angle);

// Decode PolarQuant vector back to floats
void polar_quant_decode(const uint8_t* src, int d, float* dst, int bits_per_angle);

// ─── TurboQuant (PolarQuant + QJL residual) ─────────────────────────────
// Storage layout:
//   [0..2) FP16 final_radius  (2 bytes)
//   [2..4) FP16 alpha         (2 bytes) — mean absolute residual
//   [4..)  packed PolarQuant angles + QJL sign bits
//
// Decode: element_i = polar_decode_i + alpha * sign(residual_i)
// QJL correction is UNBIASED and reduces per-element variance by ~63%.

static inline size_t turboquant_size(int d, int bits_per_angle) {
    size_t polar_bits = (size_t)(d - 1) * bits_per_angle;
    size_t total_bits = polar_bits + (size_t)d;
    return 4 + (total_bits + 7) / 8 + 3; // radius + alpha + packed angles + signs + padding
}

// Encode with QJL residual correction.
// seed is reserved for future non-basis JL projections (unused with standard basis).
size_t turboquant_encode(const float* src, int d, uint8_t* dst,
                          int bits_per_angle, uint64_t seed);
void turboquant_decode(const uint8_t* src, int d, float* dst,
                        int bits_per_angle, uint64_t seed);

// ─── Utility ────────────────────────────────────────────────────────────
int turboquant_valid_dims(int d);
