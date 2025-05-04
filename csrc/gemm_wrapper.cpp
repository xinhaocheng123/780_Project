#include <torch/extension.h>
#include "fp8_gemm.h"

void gemm_kernel_wrapper_bf16(torch::Tensor input, torch::Tensor weight, torch::Tensor output) {
    std::vector<void const *> input_tensors = {
        input.data_ptr(), weight.data_ptr()
    };
    std::vector<void *> output_tensors = {
        output.data_ptr()
    };

    gemm_kernel_bf16(input_tensors, output_tensors);
}

void gemm_kernel_wrapper_fp8(torch::Tensor input, torch::Tensor weight, torch::Tensor input_scale, torch::Tensor weight_scale, torch::Tensor output) {
    std::vector<void const *> input_tensors = {
        input.data_ptr(), weight.data_ptr()
    };

    std::vector<void const*> scale_tensors = {
        input_scale.data_ptr(), weight_scale.data_ptr()
    };
    std::vector<void *> output_tensors = {
        output.data_ptr()
    };

    gemm_kernel_fp8(input_tensors, scale_tensors, output_tensors);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_kernel_bf16", &gemm_kernel_wrapper_bf16, "BF16 GEMM kernel");
    m.def("gemm_kernel_fp8", &gemm_kernel_wrapper_fp8, "FP8 GEMM kernel");
}