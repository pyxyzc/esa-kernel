#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>
#include <torch/extension.h>

#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()

/**
 * This kernel performs:
 * repre_cache[repre_repre_index[i]] =
 * mean(key_cache[key_repre_index[i]], 0)
 *
 * @param key_cache: [N, block_size, dim]
 * @param repre_cache: [N, dim]
 * @param key_repre_index: [S]
 * @param repre_repre_index: [S]
 */
__global__ void extract_repre_fp32(const float* key_cache,
                                   float* repre_cache,
                                   const int* block_table,
                                   const int* repre_index,
                                   int block_size,
                                   int dim,
                                   int num_blocks,
                                   int key_rows,
                                   int repre_rows)
{
    int idx = blockIdx.x;
    if (idx >= num_blocks) { return; }
    int index1 = block_table[idx];
    int index2 = repre_index[idx];
    if (index1 < 0 || index1 >= key_rows || index2 < 0 || index2 >= repre_rows) {
        return;
    }
    const float* key_ptr = key_cache + index1 * block_size * dim;
    float* repre_ptr = repre_cache + index2 * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < block_size; ++j) { sum += key_ptr[j * dim + d]; }
        repre_ptr[d] = sum / block_size;
    }
}

__global__ void extract_repre_bf16(const __nv_bfloat16* key_cache,
                                   __nv_bfloat16* repre_cache,
                                   const int* block_table,
                                   const int* repre_index,
                                   int block_size,
                                   int dim,
                                   int num_blocks,
                                   int key_rows,
                                   int repre_rows)
{
    int idx = blockIdx.x;
    if (idx >= num_blocks) { return; }
    int index1 = block_table[idx];
    int index2 = repre_index[idx];
    if (index1 < 0 || index1 >= key_rows || index2 < 0 || index2 >= repre_rows) {
        return;
    }
    const __nv_bfloat16* key_ptr = key_cache + index1 * block_size * dim;
    __nv_bfloat16* repre_ptr = repre_cache + index2 * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            sum += __bfloat162float(key_ptr[j * dim + d]);
        }
        repre_ptr[d] = __float2bfloat16(sum / block_size);
    }
}

__global__ void extract_repre_fp16(const __half* key_cache,
                                   __half* repre_cache,
                                   const int* block_table,
                                   const int* repre_index,
                                   int block_size,
                                   int dim,
                                   int num_blocks,
                                   int key_rows,
                                   int repre_rows)
{
    int idx = blockIdx.x;
    if (idx >= num_blocks) { return; }
    int index1 = block_table[idx];
    int index2 = repre_index[idx];
    if (index1 < 0 || index1 >= key_rows || index2 < 0 || index2 >= repre_rows) {
        return;
    }
    const __half* key_ptr = key_cache + index1 * block_size * dim;
    __half* repre_ptr = repre_cache + index2 * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            sum += __half2float(key_ptr[j * dim + d]);
        }
        repre_ptr[d] = __float2half(sum / block_size);
    }
}

extern "C" void esa_repre(torch::Tensor key_cache,
                          torch::Tensor repre_cache,
                          torch::Tensor block_table,
                          torch::Tensor repre_table)
{
    TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
    TORCH_CHECK(repre_cache.is_cuda(), "repre_cache must be a CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
    TORCH_CHECK(repre_table.is_cuda(), "repre_index must be a CUDA tensor");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(repre_cache.is_contiguous(), "repre_cache must be contiguous");

    // Shape validations based on expected contract:
    // key_cache: [N, block_size, dim], repre_cache: [M, dim]
    TORCH_CHECK(key_cache.dim() == 3, "key_cache must be 3D [N, block_size, dim]");
    TORCH_CHECK(repre_cache.dim() == 2, "repre_cache must be 2D [M, dim]");
    TORCH_CHECK(block_table.dim() == 1 && repre_table.dim() == 1,
                "block_table and repre_index must be 1-D");
    TORCH_CHECK(block_table.size(0) == repre_table.size(0),
                "block_table and repre_index must have the same length");

    // Indices must be int32 on device and contiguous for the kernel
    if (block_table.scalar_type() != at::kInt || !block_table.is_contiguous()) {
        block_table = block_table.to(at::kInt).contiguous();
    }
    if (repre_table.scalar_type() != at::kInt || !repre_table.is_contiguous()) {
        repre_table = repre_table.to(at::kInt).contiguous();
    }

    int block_size = key_cache.size(1);
    int dim = repre_cache.size(-1);
    int num_blocks = block_table.size(0);
    int key_rows = key_cache.size(0);
    int repre_rows = repre_cache.size(0);

    int threads = dim < 1024 ? dim : 1024;
    int blocks = num_blocks;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        key_cache.scalar_type(),
        "esa_repre_cuda",
        ([&] {
            if constexpr (std::is_same_v<scalar_t, float>) {
                extract_repre_fp32<<<blocks, threads>>>(key_cache.data_ptr<float>(),
                                                        repre_cache.data_ptr<float>(),
                                                        block_table.data_ptr<int>(),
                                                        repre_table.data_ptr<int>(),
                                                        block_size,
                                                        dim,
                                                        num_blocks,
                                                        key_rows,
                                                        repre_rows);
            } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
                extract_repre_fp16<<<blocks, threads>>>(
                    reinterpret_cast<__half*>(key_cache.data_ptr()),
                    reinterpret_cast<__half*>(repre_cache.data_ptr()),
                    block_table.data_ptr<int>(),
                    repre_table.data_ptr<int>(),
                    block_size,
                    dim,
                    num_blocks,
                    key_rows,
                    repre_rows);
            } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
                extract_repre_bf16<<<blocks, threads>>>(
                    reinterpret_cast<__nv_bfloat16*>(key_cache.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(repre_cache.data_ptr()),
                    block_table.data_ptr<int>(),
                    repre_table.data_ptr<int>(),
                    block_size,
                    dim,
                    num_blocks,
                    key_rows,
                    repre_rows);
            }
        }));
}
