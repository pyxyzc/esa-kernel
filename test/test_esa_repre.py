import numpy as np
import torch
import pytest
import time
import torch.cuda.nvtx as nvtx

from util import print_red, print_green, print_blue, print_yellow


def load_module():
    import esa_kernel as module

    return module


esa_lib = load_module()
esa_repre = esa_lib.esa_repre


@pytest.mark.parametrize("num_repre_blocks", [100])
@pytest.mark.parametrize("dim", [128])
def test_esa_repre(num_repre_blocks, dim):  # extract repre
    print(f"""TEST esa_repre
{' '*4}total number of blocks to extract_repre: {num_repre_blocks}
{' '*4}dim (num_heads * hidden_size): {dim}\n""")
    
    dtype = torch.bfloat16
    N = 2 * num_repre_blocks
    block_size = 128
    key_cache = torch.randn(N, block_size, 8, dim, dtype=dtype).cuda()
    repre_cache = torch.randn(N, 8, dim, dtype=dtype).cuda()
    repre_cache2 = torch.randn(N, 8, dim, dtype=dtype).cuda()

    range_n = np.arange(N)
    rng = np.random.default_rng()
    repre_index = rng.choice(range_n, size=num_repre_blocks, replace=False)
    repre_index = torch.from_numpy(repre_index).to(torch.int32).cuda()

    rng2 = np.random.default_rng()
    block_table = rng2.choice(range_n, size=num_repre_blocks, replace=False)
    block_table = torch.from_numpy(block_table).to(torch.int32).cuda()

    start = time.perf_counter_ns()
    esa_repre(
        key_cache.flatten(-2, -1), repre_cache.flatten(-2, -1), block_table, repre_index
    )
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}[esa_repre] host API time: {duration / 1e6:.3f} ms")

    start = time.perf_counter_ns()
    for r_id, b_id in zip(repre_index, block_table):
        repre_cache2[r_id] = key_cache[b_id].mean(0)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}[naive_repre] host API time: {duration / 1e6:.3f} ms")

    diff = (repre_cache2[repre_index] - repre_cache[repre_index]).abs()
    print_blue(
        f"{' '*4}[esa_repre] repre diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)"
    )
    print("")
    assert diff.mean() < 1e-3
