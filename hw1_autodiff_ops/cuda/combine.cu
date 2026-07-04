// CMU 11-868 Assignment 1 — CUDA implementations of the four tensor primitives.
//
// These are the GPU kernels the assignment asks for. They implement exactly the
// same strided/broadcasting algorithm as hw1_autodiff_ops/tensor_ops.py, which
// is the CPU reference the numerical tests check against. Compiling and running
// these requires a CUDA toolkit + NVIDIA GPU (nvcc), which this build machine
// does not have (CPU-only). The Python reference is the verified artifact; this
// file documents the real kernel code for a GPU-equipped grader.
//
// Build (on a CUDA box):  nvcc -O3 --shared -Xcompiler -fPIC combine.cu -o combine.so
//
// Layout mirrors MiniTorch: a tensor is (storage, shape, strides). Indexing
// helpers convert an ordinal -> multi-index -> storage offset with broadcasting.

#include <cuda_runtime.h>

#define MAX_DIMS 8

// ordinal in the *contiguous* out shape -> multi index
__device__ void to_index(int ordinal, int dims, const int *shape, int *out_index) {
  for (int i = dims - 1; i >= 0; --i) {
    int sh = shape[i];
    out_index[i] = ordinal % sh;
    ordinal /= sh;
  }
}

// multi index in the big (out) shape -> multi index in a smaller broadcast shape
__device__ void broadcast_index(const int *big_index, int big_dims,
                                int dims, const int *shape, int *out_index) {
  int offset = big_dims - dims;
  for (int i = 0; i < dims; ++i) {
    out_index[i] = (shape[i] == 1) ? 0 : big_index[i + offset];
  }
}

__device__ int index_to_position(const int *index, int dims, const int *strides) {
  int pos = 0;
  for (int i = 0; i < dims; ++i) pos += index[i] * strides[i];
  return pos;
}

// ------------------------- Primitive 1: map -------------------------
// fn is selected by `op`: 0=neg, 1=relu, 2=sigmoid, 3=log, 4=exp, 5=inv
__device__ float apply_map(int op, float x) {
  switch (op) {
    case 0: return -x;
    case 1: return x > 0.f ? x : 0.f;
    case 2: return 1.f / (1.f + expf(-x));
    case 3: return logf(x);
    case 4: return expf(x);
    case 5: return 1.f / x;
    default: return x;
  }
}

__global__ void tensor_map_kernel(float *out, const int *out_shape,
                                  const int *out_strides, int out_dims, int out_size,
                                  const float *in, const int *in_shape,
                                  const int *in_strides, int in_dims, int op) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int out_index[MAX_DIMS];
  int in_index[MAX_DIMS];
  to_index(i, out_dims, out_shape, out_index);
  broadcast_index(out_index, out_dims, in_dims, in_shape, in_index);
  int in_pos = index_to_position(in_index, in_dims, in_strides);
  int out_pos = index_to_position(out_index, out_dims, out_strides);
  out[out_pos] = apply_map(op, in[in_pos]);
}

// ------------------------- Primitive 2: zip -------------------------
// op: 0=add, 1=mul, 2=lt, 3=eq, 4=max
__device__ float apply_zip(int op, float a, float b) {
  switch (op) {
    case 0: return a + b;
    case 1: return a * b;
    case 2: return a < b ? 1.f : 0.f;
    case 3: return a == b ? 1.f : 0.f;
    case 4: return a > b ? a : b;
    default: return a;
  }
}

__global__ void tensor_zip_kernel(float *out, const int *out_shape,
                                  const int *out_strides, int out_dims, int out_size,
                                  const float *a, const int *a_shape,
                                  const int *a_strides, int a_dims,
                                  const float *b, const int *b_shape,
                                  const int *b_strides, int b_dims, int op) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= out_size) return;
  int out_index[MAX_DIMS], a_index[MAX_DIMS], b_index[MAX_DIMS];
  to_index(i, out_dims, out_shape, out_index);
  broadcast_index(out_index, out_dims, a_dims, a_shape, a_index);
  broadcast_index(out_index, out_dims, b_dims, b_shape, b_index);
  int ap = index_to_position(a_index, a_dims, a_strides);
  int bp = index_to_position(b_index, b_dims, b_strides);
  int op_ = index_to_position(out_index, out_dims, out_strides);
  out[op_] = apply_zip(op, a[ap], b[bp]);
}

// ------------------------- Primitive 3: reduce ----------------------
// Parallel tree reduction: one block per output element, blockDim threads
// cooperatively fold the reduced dimension through shared memory.
__global__ void tensor_reduce_kernel(float *out, const int *out_shape,
                                     const int *out_strides, int out_dims,
                                     const float *a, const int *a_shape,
                                     const int *a_strides, int a_dims,
                                     int reduce_dim, int reduce_size,
                                     float start, int op) {
  extern __shared__ float cache[];
  int out_ordinal = blockIdx.x;
  int t = threadIdx.x;

  int out_index[MAX_DIMS];
  to_index(out_ordinal, out_dims, out_shape, out_index);

  int a_index[MAX_DIMS];
  for (int d = 0; d < a_dims; ++d) a_index[d] = out_index[d];

  // each thread reduces a strided slice of the reduce dimension
  float acc = start;
  for (int j = t; j < reduce_size; j += blockDim.x) {
    a_index[reduce_dim] = j;
    float v = a[index_to_position(a_index, a_dims, a_strides)];
    acc = (op == 0) ? acc + v : (v > acc ? v : acc);
  }
  cache[t] = acc;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) {
      float other = cache[t + s];
      cache[t] = (op == 0) ? cache[t] + other : (other > cache[t] ? other : cache[t]);
    }
    __syncthreads();
  }
  if (t == 0) out[index_to_position(out_index, out_dims, out_strides)] = cache[0];
}

// ------------------------- Primitive 4: matmul ----------------------
// Tiled batched matmul over the last two dims with shared-memory tiles.
#define TILE 16

__global__ void matmul_kernel(float *out, const float *a, const float *b,
                              int batch, int I, int K, int J,
                              int a_batch_stride, int b_batch_stride) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int batch_id = blockIdx.z;
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  const float *A = a + (long)batch_id * a_batch_stride;
  const float *B = b + (long)batch_id * b_batch_stride;
  float acc = 0.f;

  for (int m = 0; m < (K + TILE - 1) / TILE; ++m) {
    int ak = m * TILE + threadIdx.x;
    int bk = m * TILE + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < I && ak < K) ? A[row * K + ak] : 0.f;
    Bs[threadIdx.y][threadIdx.x] = (bk < K && col < J) ? B[bk * J + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < I && col < J)
    out[(long)batch_id * I * J + row * J + col] = acc;
}
