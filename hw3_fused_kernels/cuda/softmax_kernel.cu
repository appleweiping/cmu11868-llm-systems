// CMU 11-868 Assignment 3 — fused attention-softmax CUDA kernel.
//
// One warp/block cooperatively softmaxes one attention row (the `to_len` axis),
// applying the additive mask, subtracting the row max, exponentiating and
// normalising in a single kernel launch. Mirrors the reference math in
// hw3_fused_kernels/fused_kernels.py (attn_softmax_fw / attn_softmax_bw), which
// is what kernel_tests/test_softmax_fw.py and test_softmax_bw.py check the GPU
// output against at atol/rtol 1e-3.
//
// Requires an NVIDIA GPU + nvcc; this build machine is CPU-only, so the Python
// reference is the verified artifact and this file is the GPU deliverable.
//
// Build:  nvcc -O3 -c softmax_kernel.cu

#include <cuda_runtime.h>
#include <cub/cub.cuh>

const float REDUCE_FLOAT_INF_NEG = -100000000.f;

// Warp-level reductions ------------------------------------------------------
template <int WARPS>
__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
  return val;
}

template <int WARPS>
__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, offset);
  return val;
}

// Fused forward: one block per (batch, head, query) row ----------------------
// inp:  [batch, nhead, from_len, to_len]
// mask: [batch, 1, 1, to_len]  (additive, already * -1e8 for padding)
__global__ void ker_attn_softmax(float *inp, const float *attn_mask,
                                 int from_len, int to_len, bool mask_future) {
  int block_dim = blockDim.x;
  int batch_id = blockIdx.y;
  int query_id = blockIdx.x;  // flattened (head, from) index
  int head_from = query_id;

  float *row = inp + ((long)batch_id * gridDim.x + head_from) * to_len;
  const float *mask = attn_mask + (long)batch_id * to_len;

  // 1) row max (with mask)
  float lmax = REDUCE_FLOAT_INF_NEG;
  for (int i = threadIdx.x; i < to_len; i += block_dim) {
    float v = row[i] + mask[i];
    lmax = max(lmax, v);
  }
  lmax = warpReduceMax<1>(lmax);

  // 2) exp + local sum
  float lsum = 0.f;
  for (int i = threadIdx.x; i < to_len; i += block_dim) {
    float v = expf(row[i] + mask[i] - lmax);
    row[i] = v;  // store unnormalised exp back
    lsum += v;
  }
  lsum = warpReduceSum<1>(lsum);
  float inv = 1.f / (lsum + 1e-12f);

  // 3) normalise
  for (int i = threadIdx.x; i < to_len; i += block_dim) row[i] *= inv;
}

void launch_attn_softmax(float *inp, const float *attn_mask, int batch_size,
                         int nhead, int from_len, int to_len, cudaStream_t stream) {
  dim3 grid(nhead * from_len, batch_size);
  dim3 block(32);
  ker_attn_softmax<<<grid, block, 0, stream>>>(inp, attn_mask, from_len, to_len, false);
}

// Fused backward: dx = s * (grad - sum(grad * s)) per row --------------------
__global__ void ker_attn_softmax_bw(float *grad, const float *soft_inp,
                                    int to_len) {
  int block_dim = blockDim.x;
  int row_id = blockIdx.x * gridDim.y + blockIdx.y;
  float *g = grad + (long)row_id * to_len;
  const float *s = soft_inp + (long)row_id * to_len;

  float local = 0.f;
  for (int i = threadIdx.x; i < to_len; i += block_dim) local += g[i] * s[i];
  local = warpReduceSum<1>(local);

  for (int i = threadIdx.x; i < to_len; i += block_dim)
    g[i] = s[i] * (g[i] - local);
}

void launch_attn_softmax_bw(float *out_grad, const float *soft_inp, int rows,
                            int to_len, cudaStream_t stream) {
  dim3 grid(rows, 1);
  dim3 block(32);
  ker_attn_softmax_bw<<<grid, block, 0, stream>>>(out_grad, soft_inp, to_len);
}
