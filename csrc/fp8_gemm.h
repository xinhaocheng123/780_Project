#pragma once

#include <cutlass/float8.h>
#include <cute/numeric/numeric_types.hpp>

#pragma once
#include <vector>

void gemm_kernel_bf16(std::vector<void const *> input_tensors, std::vector<void*> output_tensors);
void gemm_kernel_fp8(std::vector<void const *> input_tensors,std::vector<void const *> scale_tensors, std::vector<void*> output_tensors);
