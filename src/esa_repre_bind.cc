#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

extern "C" void esa_repre(torch::Tensor key_cache,
                          torch::Tensor repre_cache,
                          torch::Tensor block_table,
                          torch::Tensor repre_table);

#define STRINGFY(func) #func
#define TORCH_BINDING_COMMON_EXTENSION(func)                                           \
    m.def(STRINGFY(func), &func, STRINGFY(func))

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "ESA cuda kernel for repre computation";
    TORCH_BINDING_COMMON_EXTENSION(esa_repre);
}
