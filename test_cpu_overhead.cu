
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e = call;                                                      \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void empty_kernel() {}
__global__ void busy_kernel(float *A, size_t N) {
  auto local_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_x < N)
    A[local_x] += 1;
}

int main() {
  const int N = 10000;
  using clock = std::chrono::high_resolution_clock;

  // force context creation
  CUDA_CHECK(cudaFree(0));
  void *hptr = nullptr;
  void *dptr = nullptr;
  size_t total_bytes = 4 * 1 << 20; // 1024 * 1024 * 1024 -> 2^30
  // total_bytes = 576 * 2 * 128;
  CUDA_CHECK(cudaMallocHost(&hptr, total_bytes));
  CUDA_CHECK(cudaMalloc(&dptr, total_bytes));

  dim3 threads (256);
  int num_of_float = total_bytes / sizeof(float);
  dim3 blocks ((num_of_float + 256 - 1) / 256);
  // at first warmup a little bit

  // for (int i = 0; i < 100; ++i) {
  //     empty_kernel<<<1,1>>>();
  //     busy_kernel<<<blocks, threads>>>((float*) dptr, total_bytes / sizeof(float));
  //     CUDA_CHECK(cudaMemcpyAsync(dptr, hptr, 0, cudaMemcpyHostToDevice));
  //     CUDA_CHECK(cudaMemcpyAsync(dptr, hptr, total_bytes, cudaMemcpyHostToDevice));
  // }

  // 1) first measure plain enqueue+sync
  auto t0 = clock::now();
  cudaStream_t s0;
  CUDA_CHECK(cudaStreamCreate(&s0));
  for (int i = 0; i < N; i++) {
      // busy_kernel<<<blocks, threads, 0, s0>>>((float*) dptr, total_bytes / sizeof(float));
      cudaMemcpyAsync(dptr, hptr, total_bytes, cudaMemcpyHostToDevice, s0);
  }



  double host_us =
      std::chrono::duration<double>(clock::now() - t0).count() * 1e6;
  std::cout << "Plain enqueue: " << host_us << " us\n";


  CUDA_CHECK(cudaStreamSynchronize(s0));

  // 2) build the CUDA Graph
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < N; i++) {
      // busy_kernel<<<blocks, threads, 0, s>>>((float*) dptr, total_bytes / sizeof(float));
      cudaMemcpyAsync(dptr, hptr, total_bytes, cudaMemcpyHostToDevice, s);
  }
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));

  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // 3) launch the graph once and time it
  t0 = clock::now();
  CUDA_CHECK(cudaGraphLaunch(graphExec, s));


  double graph_us =
      std::chrono::duration<double>(clock::now() - t0).count() * 1e6;
  std::cout << "Graph-launched:    " << graph_us << " us\n";
  std::cout << "Delta T in us: " << host_us - graph_us << " us\n";

  CUDA_CHECK(cudaStreamSynchronize(s));

  // cleanup
  CUDA_CHECK(cudaGraphExecDestroy(graphExec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(s));

  return 0;
}
