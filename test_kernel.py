import torch
from torch.utils.cpp_extension import load

my_ext = load(
    name="my_ext",
    sources=["test_kernel.cu"],
    extra_cflags=["-std=c++17"],
)


N = 10000
a = torch.randn(N, device='cuda')
b = torch.randn(N, device='cuda')
c = torch.randn(N, device='cuda')
d = torch.randn(N, device='cuda')
my_ext.launch([a, b, c], d)
diff = (d - (a + b + c)).abs()
print("diff: ", diff)

x = torch.randn(N, device='cuda')
y = torch.randn(N, device='cuda')
z = torch.randn(N, device='cuda')
my_ext.launch([x, y, z], d)
diff = (d - (x + y + z)).abs()
print("diff: ", diff)
