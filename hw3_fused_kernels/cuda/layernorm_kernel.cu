// CMU 11-868 Assignment 3 — fused LayerNorm CUDA kernel (forward + backward).
//
// One block normalises one row over the feature axis: it computes mean and
// variance in a single block reduction, then writes the normalised + affine
// output, stashing mean and 1/std for the backward pass. Mirrors the reference
// math in hw3_fused_kernels/fused_kernels.py (layernorm_fw / layernorm_bw),
// which kernel_tests/test_layernorm_fw.py and test_layernorm_bw.py check the GPU
// output against at atol/rtol 1e-3.
//
// Requires an NVIDIA GPU + nvcc; CPU-only build ships this as the GPU deliverable
// with the Python reference as the verified artifact.
//
// Build:  nvcc -O3 -c layernorm_kernel.cu

#include <cuda_runtime.h>

template <int WARPS>
__device__ __forceinline__ float blockReduceSum(float val, float *shared) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, offset);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.f;
  if (wid == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

// Forward: inp [rows, hidden] -> out [rows, hidden]; saves mean, rstd per row.
__global__ void ker_layernorm(const float *inp, const float *gamma,
                              const float *beta, float *out, float *means,
                              float *rstds, int hidden, float eps) {
  __shared__ float shared[32];
  int row = blockIdx.x;
  const float *x = inp + (long)row * hidden;
  float *y = out + (long)row * hidden;

  float lsum = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) lsum += x[i];
  float mean = blockReduceSum<1>(lsum, shared) / hidden;
  __shared__ float s_mean;
  if (threadIdx.x == 0) s_mean = mean;
  __syncthreads();
  mean = s_mean;

  float lvar = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float d = x[i] - mean;
    lvar += d * d;
  }
  float var = blockReduceSum<1>(lvar, shared) / hidden;
  float rstd = rsqrtf(var + eps);

  if (threadIdx.x == 0) {
    means[row] = mean;
    rstds[row] = rstd;
  }
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
    y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
}

void launch_layernorm(const float *inp, const float *gamma, const float *beta,
                      float *out, float *means, float *rstds, int rows,
                      int hidden, float eps, cudaStream_t stream) {
  ker_layernorm<<<rows, 256, 0, stream>>>(inp, gamma, beta, out, means, rstds,
                                          hidden, eps);
}

// Backward: dinp = rstd*(dxhat - mean(dxhat) - xhat*mean(dxhat*xhat)).
// dgamma/dbeta reduced across rows via atomics.
__global__ void ker_layernorm_bw(const float *out_grad, const float *inp,
                                 const float *gamma, const float *means,
                                 const float *rstds, float *dinp, float *dgamma,
                                 float *dbeta, int hidden) {
  __shared__ float shared[32];
  int row = blockIdx.x;
  const float *g = out_grad + (long)row * hidden;
  const float *x = inp + (long)row * hidden;
  float *dx = dinp + (long)row * hidden;
  float mean = means[row], rstd = rstds[row];

  float sum_dxhat = 0.f, sum_dxhat_xhat = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float xhat = (x[i] - mean) * rstd;
    float dxhat = g[i] * gamma[i];
    sum_dxhat += dxhat;
    sum_dxhat_xhat += dxhat * xhat;
    atomicAdd(&dgamma[i], g[i] * xhat);
    atomicAdd(&dbeta[i], g[i]);
  }
  float m1 = blockReduceSum<1>(sum_dxhat, shared) / hidden;
  __shared__ float s_m1, s_m2;
  if (threadIdx.x == 0) s_m1 = m1;
  float m2 = blockReduceSum<1>(sum_dxhat_xhat, shared) / hidden;
  if (threadIdx.x == 0) s_m2 = m2;
  __syncthreads();

  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float xhat = (x[i] - mean) * rstd;
    float dxhat = g[i] * gamma[i];
    dx[i] = rstd * (dxhat - s_m1 - xhat * s_m2);
  }
}

void launch_layernorm_bw(const float *out_grad, const float *inp,
                         const float *gamma, const float *means,
                         const float *rstds, float *dinp, float *dgamma,
                         float *dbeta, int rows, int hidden, cudaStream_t stream) {
  ker_layernorm_bw<<<rows, 256, 0, stream>>>(out_grad, inp, gamma, means, rstds,
                                             dinp, dgamma, dbeta, hidden);
}
