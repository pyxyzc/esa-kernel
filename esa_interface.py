import cupy as cp
import torch

# compile and load the CUDA kernel
module = cp.RawModule(path='repre_compute.cu', backend='nvcc', options=('--std=c++14',))
extract_repre_kernel = module.get_function('extract_repre')

# compile and load the retrieval kernel
retrieval_module = cp.RawModule(path='retrieval_kernel.cu', backend='nvcc', options=('--std=c++14',))
retrieval_kernel_3 = retrieval_module.get_function('retrieval_kernel_3')

def extract_repre_gpu(key_cache, repre_cache, block_table, block_table_2):
    """
    GPU wrapper for extract_repre kernel.

    key_cache: cupy.ndarray, shape (N, block_size, dim), dtype float32
    block_table, block_table_2: cupy.ndarray, shape (block_number,), dtype int32

    Returns:
      repre_cache: cupy.ndarray, shape (N, dim), dtype float32
    """
    # ensure correct types
    key_cache = cp.asarray(key_cache, dtype=cp.float32)
    repre_cache = cp.asarray(repre_cache, dtype=cp.float32)
    block_table = cp.asarray(block_table, dtype=cp.int32)
    block_table_2 = cp.asarray(block_table_2, dtype=cp.int32)

    N, block_size, dim = key_cache.shape
    block_number = block_table.size

    # launch kernel: grid = block_number, block = dim
    extract_repre_kernel(
        (block_number,), (dim,),
        (key_cache, repre_cache, block_table, block_table_2, block_size, dim)
    )
    return repre_cache

# GPU wrapper for retrieval_kernel_3
def retrieval_score_gpu(Q, K, block_table, batch_index, threads=32):
    """
    GPU wrapper for retrieval_kernel_3.
    Q: cupy.ndarray, shape (B, dim), dtype float32
    K: cupy.ndarray, shape (N, dim), dtype float32
    block_table, batch_index: cupy.ndarray, shape (S,), dtype int32
    threads: int, number of threads per block
    Returns:
      score: cupy.ndarray, shape (S,), dtype float32
    """
    # ensure correct types
    Q = cp.asarray(Q, dtype=cp.float32)
    K = cp.asarray(K, dtype=cp.float32)
    block_table = cp.asarray(block_table, dtype=cp.int32)
    batch_index = cp.asarray(batch_index, dtype=cp.int32)

    B, dim = Q.shape
    S = block_table.size

    # allocate output
    score = cp.empty((S,), dtype=cp.float32)

    # compute shared memory size: threads floats
    shared_mem_bytes = threads * Q.dtype.itemsize

    # launch kernel
    retrieval_kernel_3(
        (S,), (threads,),
        (Q, K, score, block_table, batch_index, dim, B, S),
        shared_mem=shared_mem_bytes
    )

    return score


key_cache = torch.randn(1000, 128, 576, dtype = torch.float32).cuda()

repre_cache = torch.randn(1000, 576, dtype = torch.float32).cuda()

block_table = torch.arange(0, 40, dtype = torch.int32).cuda()

block_table_2 = torch.arange(0, 40, dtype = torch.int32).cuda()

extract_repre_gpu(key_cache, repre_cache, block_table, block_table_2)
