#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// TurboQuant 3-bit: pure PolarQuant, no QJL (block size 32).
// Input x[] is already WHT-rotated by the preceding TURBO_WHT graph op.
// Dequant: val = CENTROIDS_3BIT[idx] * norm
static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ x, block_turbo3_0 * __restrict__ y) {
    static constexpr float CENTROIDS[8] = {
        -0.190685f, -0.117832f, -0.065717f, -0.021460f,
         0.021460f,  0.065717f,  0.117832f,  0.190685f
    };

    // L2 norm of the 32-element block
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO3; ++i) {
        norm_sq += x[i] * x[i];
    }
    const float norm     = sqrtf(norm_sq);
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    y->norm = __float2half(norm);

    // Zero packed arrays
    for (int i = 0; i < QK_TURBO3 / 4; ++i) { y->qs[i]    = 0; }
    for (int i = 0; i < QK_TURBO3 / 8; ++i) { y->signs[i] = 0; }

    for (int i = 0; i < QK_TURBO3; ++i) {
        const float v = x[i] * inv_norm;

        // Nearest 3-bit centroid (8 levels, Lloyd-Max for N(0, 1/128))
        int idx;
        if      (v < -0.154259f) { idx = 0; }
        else if (v < -0.091775f) { idx = 1; }
        else if (v < -0.043589f) { idx = 2; }
        else if (v <  0.000000f) { idx = 3; }
        else if (v <  0.043589f) { idx = 4; }
        else if (v <  0.091775f) { idx = 5; }
        else if (v <  0.154259f) { idx = 6; }
        else                     { idx = 7; }

        (void)CENTROIDS; // used only at dequant time
        // Lower 2 bits → qs, upper 1 bit → signs
        y->qs[i / 4]    |= (uint8_t)((idx & 0x3) << ((i % 4) * 2));
        y->signs[i / 8] |= (uint8_t)(((idx >> 2) & 0x1) << (i % 8));
    }
}

// TurboQuant 4-bit: 3-bit PolarQuant + 1-bit QJL sign (block size 128).
// Input x[] is already WHT-rotated. QJL residual correction is omitted in
// this first implementation (rnorm=0, signs=0). The PolarQuant term alone
// gives ~3.5-bit quality which is sufficient to unblock testing.
static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ x, block_turbo4_0 * __restrict__ y) {
    // L2 norm of all 128 elements
    float norm_sq = 0.0f;
    for (int i = 0; i < QK_TURBO4; ++i) {
        norm_sq += x[i] * x[i];
    }
    const float norm     = sqrtf(norm_sq);
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

    y->norm  = __float2half(norm);
    y->rnorm = __float2half(0.0f);  // QJL not yet implemented

    for (int i = 0; i < QK_TURBO4 * 3 / 8; ++i) { y->qs[i]    = 0; }
    for (int i = 0; i < QK_TURBO4 / 8;     ++i) { y->signs[i] = 0; }

    for (int i = 0; i < QK_TURBO4; ++i) {
        const float v = x[i] * inv_norm;

        int idx;
        if      (v < -0.154259f) { idx = 0; }
        else if (v < -0.091775f) { idx = 1; }
        else if (v < -0.043589f) { idx = 2; }
        else if (v <  0.000000f) { idx = 3; }
        else if (v <  0.043589f) { idx = 4; }
        else if (v <  0.091775f) { idx = 5; }
        else if (v <  0.154259f) { idx = 6; }
        else                     { idx = 7; }

        // Pack 3-bit index: 8 indices per 3 bytes
        const int bit_off  = i * 3;
        const int byte_idx = bit_off / 8;
        const int bit_pos  = bit_off % 8;
        y->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            y->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
        }
        // signs: QJL not yet implemented
    }
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
