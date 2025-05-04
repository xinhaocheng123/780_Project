
import torch
import torch.nn as nn
import math
import fp8_gemm
class BF16LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = torch.empty(input.size(0), weight.size(0), device='cuda', dtype=torch.bfloat16)
        fp8_gemm.gemm_kernel_bf16(input, weight, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight  # [B, I]
        grad_weight = grad_output.T @ input  # [O, I]
        return grad_input, grad_weight

class BF16Linear(nn.Module):
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
        out = BF16LinearFunction.apply(x_2d, self.weight)
        out = out + self.bias
        return out.view(*orig_shape[:-1], self.out_features)