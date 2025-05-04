# 
# reuse the quantization logic from https://github.com/interestingLSY/LitePeek.git
#
import torch

import triton
import triton.language as tl

@triton.jit
def _tile_quantize_fwd(
    output_fp8: torch.Tensor,	# (batch_size, hidden_dim), hidden_dim - inner
    output_scale: torch.Tensor,	# (batch_size, hidden_dim/TILE_SIZE), batch_size - inner
    input_activ: torch.Tensor,	# (batch_size, hidden_dim), hidden_dim - inner
    TILE_SIZE: tl.constexpr,
    batch_size: int,
    hidden_dim: tl.constexpr
):
    # grid_shape: (batch_size, )
    batch_size_idx = tl.program_id(0)
    output_fp8 += batch_size_idx*hidden_dim
    input_activ += batch_size_idx*hidden_dim
    output_scale += batch_size_idx
    offs = tl.arange(0, TILE_SIZE)
    for _ in tl.static_range(0, hidden_dim // TILE_SIZE):
        my_input_activ = tl.load(input_activ + offs)
        scale = (tl.max(tl.abs(my_input_activ)) / 256).to(tl.float32)    # DeepSeek chooses 448 here, but here we choose 256 since it can represent numbers near max(abs) more accurately
        scale = tl.maximum(scale, 1e-6)
        my_input_activ = (my_input_activ / scale).to(output_fp8.dtype.element_ty)

        tl.store(output_fp8 + offs, my_input_activ)
        tl.store(output_scale, scale)

        output_fp8 += TILE_SIZE
        input_activ += TILE_SIZE
        output_scale += batch_size

def tile_quantize(input_activ: torch.Tensor, TILE_SIZE: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_activ.is_contiguous()
    assert input_activ.size(-1) % TILE_SIZE == 0
    assert input_activ.stride(-1) == 1
    batch_size = input_activ.numel() // input_activ.size(-1)
    hidden_dim = input_activ.size(-1)
    output_fp8 = torch.empty(batch_size, hidden_dim, dtype=torch.float8_e4m3fn, device=input_activ.device)
    output_scale = torch.empty_strided((batch_size, hidden_dim // TILE_SIZE), (1, batch_size), dtype=torch.float32, device=input_activ.device)
    grid_shape = (batch_size, )
    _tile_quantize_fwd[grid_shape](output_fp8, output_scale, input_activ, TILE_SIZE, batch_size, hidden_dim)
    return output_fp8, output_scale


@triton.jit
def _block_quantize_fwd(
    output_fp8: torch.Tensor,   # (m, n), n - inner
    output_scale: torch.Tensor, # (m/TILE_SIZE, n/TILE_SIZE), n/TILE_SIZE - inner
    input_weight: torch.Tensor, # (m, n), n - inner
    TILE_SIZE: tl.constexpr,
    m: tl.constexpr,    # Since we are always calling this function on weight matrix, we can mark `m` and `n` as tl.constexpr
    n: tl.constexpr
):
    # grid_shape: (m/TILE_SIZE, n/TILE_SIZE)
    m_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    offs_m = m_idx*TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_n = n_idx*TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs = offs_m[:, None]*n + offs_n[None, :]
    my_input_weight = tl.load(input_weight + offs)
    scale = (tl.max(tl.abs(my_input_weight)) / 256).to(tl.float32)
    scale = tl.maximum(scale, 1e-6)
    my_input_weight = (my_input_weight / scale).to(output_fp8.dtype.element_ty)
    tl.store(output_fp8 + offs, my_input_weight)
    tl.store(output_scale + (m_idx*(n//TILE_SIZE)+n_idx), scale)

def block_quantize(input_weight: torch.Tensor, TILE_SIZE: int) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_weight.is_contiguous()
    assert input_weight.dim() == 2
    assert input_weight.stride(-1) == 1
    m = input_weight.size(0)
    n = input_weight.size(1)
    assert m % TILE_SIZE == 0
    assert n % TILE_SIZE == 0
    output_fp8 = torch.empty_like(input_weight, dtype=torch.float8_e4m3fn)
    output_scale = torch.empty_strided((m//TILE_SIZE, n//TILE_SIZE), (n//TILE_SIZE, 1), dtype=torch.float32, device=input_weight.device)
    grid_shape = (m//TILE_SIZE, n//TILE_SIZE)
    _block_quantize_fwd[grid_shape](output_fp8, output_scale, input_weight, TILE_SIZE, m, n)
    return output_fp8, output_scale
