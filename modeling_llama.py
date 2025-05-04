import math
from typing import Optional, Tuple

from fp8linear import FP8Linear
from bf16linear import BF16Linear
import torch
import torch.nn as nn


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        return self.cos[position_ids], self.sin[position_ids]


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation=nn.SiLU()):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

        self.activation = activation

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d)
    return hidden_states.reshape(b, h * n_rep, s, d)


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads=None, precision='bf16'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_groups = num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        Linear = FP8Linear if precision == 'fp8' else BF16Linear
        self.q_proj = Linear(hidden_size, num_heads * self.head_dim)
        self.k_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = Linear(num_heads * self.head_dim, hidden_size)

        # self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim)
        # self.q_proj = FP8Linear(hidden_size, num_heads * self.head_dim);
        # # self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        # # self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim)
        # # self.q_proj = FP8Linear(hidden_size, num_heads * self.head_dim)
        # self.k_proj = FP8Linear(hidden_size, self.num_kv_heads * self.head_dim)
        # self.v_proj = FP8Linear(hidden_size, self.num_kv_heads * self.head_dim)
        # self.out_proj = FP8Linear(num_heads * self.head_dim, hidden_size)
        # self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_size)

    def forward(self, x, cos, sin, mask=None):
        bsz, seqlen, _ = x.size()

        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.kv_groups)
        v = repeat_kv(v, self.kv_groups)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seqlen, -1)
        return self.out_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, kv_heads=None, precision='bf16'):
        super().__init__()
        self.attn = LlamaAttention(hidden_size, num_heads, kv_heads, precision=precision)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.ln1 = LlamaRMSNorm(hidden_size)
        self.ln2 = LlamaRMSNorm(hidden_size)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.mlp(self.ln2(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, intermediate_size, num_heads, kv_heads, num_layers, max_seq_len, precision='bf16'):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(hidden_size, intermediate_size, num_heads, kv_heads, precision=precision)
            for _ in range(num_layers)
        ])
        self.norm = LlamaRMSNorm(hidden_size)
        self.rope = LlamaRotaryEmbedding(hidden_size // num_heads, max_position_embeddings=max_seq_len)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        cos, sin = self.rope(x, pos_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config, precision='bf16'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_layers = config.num_hidden_layers
        self.max_seq_len = config.max_position_embeddings
        self.model = LlamaModel(self.vocab_size, self.hidden_size, self.intermediate_size, self.num_heads, self.num_kv_heads, self.num_layers, self.max_seq_len, precision=precision)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
    def forward(self, input_ids, attention_mask= None,position_ids = None,
        past_key_values= None,
        inputs_embeds = None,labels=None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return logits if loss is None else (logits, loss)
