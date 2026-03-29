// TurboQuant Fast Walsh-Hadamard Transform for CUDA/HIP
// Ports the Metal turbo-wht.h implementation to CUDA/HIP.
// Each GPU thread processes one group of 128 F32 elements sequentially.
// Sign arrays match the CPU/Metal implementations (seed=42, seed=1042).

#include "turbo-wht.cuh"
#include "common.cuh"

#include <cstring>

// WHT sign arrays — must match ggml-cpu/ops.cpp turbo_wht_s1/s2 and Metal turbo-wht.h
static __constant__ float k_turbo_wht_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static __constant__ float k_turbo_wht_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

// One thread per group of 128 elements.
// direction=0: forward (s1 → FWHT → s2)
// direction=1: inverse (s2 → FWHT → s1)
static __global__ void k_turbo_wht_f32(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        int   direction,
        int64_t num_groups) {

    const int64_t g = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (g >= num_groups) {
        return;
    }

    const float * s_first  = (direction == 0) ? k_turbo_wht_s1 : k_turbo_wht_s2;
    const float * s_second = (direction == 0) ? k_turbo_wht_s2 : k_turbo_wht_s1;

    const float * in  = src + g * 128;
    float       * out = dst + g * 128;

    float x[128];

    // Apply first sign array
    for (int i = 0; i < 128; i++) {
        x[i] = in[i] * s_first[i];
    }

    // Fast Walsh-Hadamard Transform butterfly (7 stages)
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // Normalize (1/sqrt(128)) and apply second sign array
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        out[i] = x[i] * inv_sqrt_128 * s_second[i];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_total  = ggml_nelements(src);
    GGML_ASSERT(n_total % 128 == 0);
    const int64_t num_groups = n_total / 128;

    const float * src_d = (const float *) src->data;
    float       * dst_d = (float *)       dst->data;

    const int num_blocks = (num_groups + 255) / 256;
    k_turbo_wht_f32<<<num_blocks, 256, 0, ctx.stream()>>>(src_d, dst_d, direction, num_groups);
}
