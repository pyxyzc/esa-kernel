#include "esa_utils.h"
#include "esa_kernels.cu"

void esa_repre(torch::Tensor key_cache, torch::Tensor repre_cache, torch::Tensor block_table, torch::Tensor repre_table){
    int block_size = key_cache.size(1);
    int dim = repre_cache.size(-1);
    int threads = dim;
    int blocks = block_table.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, key_cache.scalar_type(), "esa_repre_cuda", ([&] {
        extract_repre<scalar_t><<<blocks, threads>>>(
            key_cache.data_ptr<scalar_t>(),
            repre_cache.data_ptr<scalar_t>(),
            block_table.data_ptr<int>(),
            repre_table.data_ptr<int>(),
            block_size,
            dim);
        }));
}

void esa_retrieval(const std::vector<torch::Tensor> &query_list, torch::Tensor repre_cache, torch::Tensor q_index, torch::Tensor repre_index, torch::Tensor score, torch::Tensor score_sorted, torch::Tensor index_ranged, torch::Tensor index_sorted, torch::Tensor batch_offset, torch::Tensor workspace){
    int s = q_index.size(0);
    int dim = repre_cache.size(1);
    int batch = query_list.size();
    dim3 numThreads = {(unsigned int)(32)};
    dim3 numBlocks = {(unsigned int)(s)};

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, repre_cache.scalar_type(), "esa_retrieval_cuda", [&]{
        if constexpr (std::is_same_v<scalar_t, float>) {
            float** Q_ptrs = nullptr;
            cudaMallocManaged(&Q_ptrs, batch * sizeof(float*));
            for(int i = 0; i < batch; ++i) {
            Q_ptrs[i] = query_list[i].data_ptr<float>();
            }
            printf("is float32\n");
            size_t bytes = numThreads.x * sizeof(float);
            retrieval_kernel_fp32<<<numBlocks, numThreads, bytes>>>(Q_ptrs, repre_cache.data_ptr<float>(), score.data_ptr<float>(), repre_index.data_ptr<int>(), q_index.data_ptr<int>(), dim, s);
            CUDA_CHECK(cudaFree(Q_ptrs));
            void* temp_workspace = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
            temp_workspace = workspace.data_ptr<int>();
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    score.data_ptr<float>(),  score_sorted.data_ptr<float>(),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
        } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
            __half** Q_ptrs = nullptr;
            cudaMallocManaged(&Q_ptrs, batch * sizeof(__half*));
            for(int i = 0; i < batch; ++i) {
                Q_ptrs[i] = reinterpret_cast<__half*>(query_list[i].data_ptr());
            }
            printf("is float16\n");
            size_t bytes = numThreads.x * sizeof(float);
            retrieval_kernel_fp16<<<numBlocks, numThreads, bytes>>>(Q_ptrs,
                    reinterpret_cast<__half*>(repre_cache.data_ptr()),
                    reinterpret_cast<__half*>(score.data_ptr()),
                    reinterpret_cast<int*>(repre_index.data_ptr()),
                    reinterpret_cast<int*>(q_index.data_ptr()),
                    dim, s);
            CUDA_CHECK(cudaFree(Q_ptrs));
            void* temp_workspace = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__half*>(score.data_ptr()),
                    reinterpret_cast<__half*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
            temp_workspace = workspace.data_ptr<int>();
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__half*>(score.data_ptr()),
                    reinterpret_cast<__half*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
        } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
            __nv_bfloat16** Q_ptrs = nullptr;
            cudaMallocManaged(&Q_ptrs, batch * sizeof(__nv_bfloat16*));
            for(int i = 0; i < batch; ++i) {
                Q_ptrs[i] = reinterpret_cast<__nv_bfloat16*>(query_list[i].data_ptr());
            }
            printf("is bfloat16\n");
            size_t bytes = numThreads.x * sizeof(float);
            retrieval_kernel_bf16<<<numBlocks, numThreads, bytes>>>(Q_ptrs,
                    reinterpret_cast<__nv_bfloat16*>(repre_cache.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                    reinterpret_cast<int*>(repre_index.data_ptr()),
                    reinterpret_cast<int*>(q_index.data_ptr()),
                    dim, s);
            CUDA_CHECK(cudaFree(Q_ptrs));
            void* temp_workspace = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
            temp_workspace = workspace.data_ptr<int>();
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp_workspace, temp_bytes,
                    reinterpret_cast<__nv_bfloat16*>(score.data_ptr()),
                    reinterpret_cast<__nv_bfloat16*>(score_sorted.data_ptr()),
                    index_ranged.data_ptr<int>(), index_sorted.data_ptr<int>(),
                    s, batch, batch_offset.data_ptr<int>(), batch_offset.data_ptr<int>() + 1);
        }
    });
}

void esa_topk(torch::Tensor score, torch::Tensor index, torch::Tensor offsets, torch::Tensor score_out, torch::Tensor index_out, torch::Tensor workspace){
    void* temp_workspace = nullptr;
    size_t temp_bytes = 0;
    size_t B = offsets.size(0) - 1;
    size_t total = score.size(0);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
    // NOTE: Don't use malloc, just reuse the workspace, but the first call of
    // SortPairsDescending is necesssary to determine the workspace size.
    // CUDA_CHECK(cudaMalloc(&temp_workspace, temp_bytes));
    temp_workspace = workspace.data_ptr<int>();
    cub::DeviceSegmentedRadixSort::SortPairsDescending(
            temp_workspace, temp_bytes,
            score.data_ptr<float>(),  score_out.data_ptr<float>(),
            index.data_ptr<int>(), index_out.data_ptr<int>(),
            total, B, offsets.data_ptr<int>(), offsets.data_ptr<int>() + 1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(esa_retrieval)
    TORCH_BINDING_COMMON_EXTENSION(esa_topk)
    TORCH_BINDING_COMMON_EXTENSION(esa_repre)
}
