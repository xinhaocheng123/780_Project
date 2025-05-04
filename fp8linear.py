import torch
import fp8_gemm
import quantization
from torch import nn
import math


class FP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        input_q, input_scale = quantization.tile_quantize(input, 64)
        weight_q, weight_scale = quantization.block_quantize(weight, 64)
        ctx.save_for_backward(input.to(torch.bfloat16), weight)
        output = torch.empty(input.size(0), weight.size(0), device='cuda', dtype=torch.float16)
        fp8_gemm.gemm_kernel_fp8(input_q, weight_q, input_scale, weight_scale, output)
        return output.to(dtype=torch.bfloat16)
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight 
        grad_weight = grad_output.T @ input
        return grad_input, grad_weight

class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device='cuda'))
        self.bias = nn.Parameter(torch.zeros(out_features, device='cuda'))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        orig_shape = input.shape
        x_2d = input.view(-1, self.in_features).contiguous()
        out = FP8LinearFunction.apply(x_2d, self.weight)
        out = out + self.bias
        return out.view(*orig_shape[:-1], self.out_features)