#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                << " (" #call ") at " << __FILE__ << ":" << __LINE__    \
                << std::endl;                                           \
      std::exit(1);                                                     \
    }                                                                   \
  } while(0)

using clock = std::chrono::high_resolution_clock;

int main(){
  const size_t total_bytes = 1ULL << 30;  // 1 GiB
  const size_t chunk_bytes = 1ULL << 20;  // 1 MiB
  const int    N = total_bytes / chunk_bytes;

  // Allocate pinned host + device buffers
  void *h, *d;
  CUDA_CHECK(cudaMallocHost(&h, total_bytes));
  CUDA_CHECK(cudaMalloc(&d,      total_bytes));

  // Warm up
  CUDA_CHECK(cudaMemcpy(d, h, total_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Test 1: simple async copies ---
  cudaStream_t s1;
  CUDA_CHECK(cudaStreamCreate(&s1));
  auto t0 = clock::now();
  for(int i = 0; i < N; i++){
    CUDA_CHECK(cudaMemcpyAsync(
      (char*)d + i*chunk_bytes,
      (char*)h + i*chunk_bytes,
      chunk_bytes,
      cudaMemcpyHostToDevice,
      s1));
  }
  CUDA_CHECK(cudaStreamSynchronize(s1));
  auto t1 = clock::now();
  double t_simple = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "Simple async: "
            << (double)total_bytes / t_simple / 1e9
            << " GB/s\n";
  CUDA_CHECK(cudaStreamDestroy(s1));

  // --- Test 2: CUDA Graph of the same copies ---
  cudaStream_t s2;
  CUDA_CHECK(cudaStreamCreate(&s2));
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  CUDA_CHECK(cudaStreamBeginCapture(s2, cudaStreamCaptureModeGlobal));
  for(int i = 0; i < N; i++){
    CUDA_CHECK(cudaMemcpyAsync(
      (char*)d + i*chunk_bytes,
      (char*)h + i*chunk_bytes,
      chunk_bytes,
      cudaMemcpyHostToDevice,
      s2));
  }
  CUDA_CHECK(cudaStreamEndCapture(s2, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  t0 = clock::now();
  CUDA_CHECK(cudaGraphLaunch(graphExec, s2));
  CUDA_CHECK(cudaStreamSynchronize(s2));
  t1 = clock::now();
  double t_graph = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "CUDA Graph: "
            << (double)total_bytes / t_graph / 1e9
            << " GB/s\n";

  // Cleanup
  CUDA_CHECK(cudaGraphExecDestroy(graphExec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaStreamDestroy(s2));
  CUDA_CHECK(cudaFree(d));
  CUDA_CHECK(cudaFreeHost(h));
  return 0;
}
