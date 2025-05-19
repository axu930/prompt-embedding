from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Llama model code: https://github.com/meta-llama/llama/blob/main/llama/model.py
# See also: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

# Models
class LlamaModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.n_layer)]
        )
        self.norm = LlamaRMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        freqs_cis = precompute_freqs_cis(dim=config.head_dim, end=config.block_size, theta=config.rope_theta)
        mask = (torch.arange(config.block_size)[:,None] - torch.arange(config.block_size)[None,:] >= 0) 

        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        self.register_buffer("mask",torch.where(mask,0,float('-inf')), persistent=False)

    def forward(self,
                toks: torch.Tensor,
                ) -> torch.Tensor:
        _, T = toks.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.embed_tokens(toks)

        for layer in self.layers:
            x = layer(x, self.freqs_cis[:T,:], self.mask[None,None,:T,:T])
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
    
    @torch.no_grad()
    def generate(self,
                 toks: torch.Tensor,
                 top_k: int = 50,
                 temperature: float = 1.0,
                 ) -> torch.Tensor:
        assert temperature > 0, "Temperature needs to be positive"
        assert top_k >= 1, "top_k needs to be at least 1"

        logits = self.forward(toks[:,:self.config.block_size])[:,-1,:] / temperature
        k_val, k_ind = torch.topk(logits, k=top_k, dim=-1)
        probs = F.softmax(k_val, dim=-1)
        toks_next = torch.multinomial(probs,num_samples=1)
        toks = torch.cat(toks_next)
        return toks

    
    @classmethod
    def from_huggingface(cls, config):
        from transformers import AutoModelForCausalLM

        model = LlamaModel(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        model_hf = AutoModelForCausalLM.from_pretrained(config.modelpath)
        sd_hf = model_hf.state_dict()

        for k in sd_keys:
            with torch.no_grad():
                # lm head weights are stored in a separate location
                if k == 'lm_head.weight':
                    sd[k].copy_(model_hf.lm_head.weight)
                else:
                    hf_k = 'model.' + k
                    sd[k].copy_(sd_hf[hf_k])
        print("Llama model loaded from state dict")
        return model


class AttentionOnlyLlamaModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            [LlamaSdpaAttention(config) for _ in range(config.n_layer)]
        )
        self.norm = LlamaRMSNorm(config.n_embd)
        
        freqs_cis = precompute_freqs_cis(dim=config.head_dim, end=config.block_size, theta=config.rope_theta)
        mask = (torch.arange(config.block_size)[:,None] - torch.arange(config.block_size)[None,:] >= 0) 

        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        self.register_buffer('mask',torch.where(mask,0,float('-inf')), persistent=False)


    def forward(self,
                toks: torch.Tensor
                ) -> torch.Tensor:
        _, T = toks.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        x = self.embed_tokens(toks)
        for layer in self.layers:
            x = layer(x, self.freqs_cis[:T,:], self.mask[:T,:T])
        x = self.norm(x)
        return x
    
    @classmethod
    def from_huggingface(cls, config):
        from transformers import AutoModelForCausalLM

        model = AttentionOnlyLlamaModel(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        model_hf = AutoModelForCausalLM.from_pretrained(config.modelpath)
        sd_hf = model_hf.state_dict()

        for k in sd_keys:
            if k.startswith("layers."):
                with torch.no_grad():
                    hf_k = k.split(".")
                    hf_k = ['model'] + hf_k[:2] + ['self_attn'] + hf_k[2:]
                    hf_k = ".".join(hf_k)
                    sd[k].copy_(sd_hf[hf_k])
            else:
                with torch.no_grad():
                    hf_k = 'model.' + k
                    sd[k].copy_(sd_hf[hf_k])
        print("Llama model loaded from state dict")
        return model


# ----- Llama components -----

class LlamaDecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp = LlamaMLP(config)
        self.self_attn = LlamaSdpaAttention(config)
        self.input_layernorm = LlamaRMSNorm(config.n_embd)
        self.post_attention_layernorm = LlamaRMSNorm(config.n_embd)

    def forward(
            self,
            x: torch.Tensor,            
            freqs_cis: torch.Tensor,
            attention_mask: torch.Tensor,
            ):
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis, attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x    


class LlamaSdpaAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_heads = config.n_heads 
        self.n_kv_heads = config.n_kv_heads 
        self.head_dim = config.head_dim

    def forward(self,
                x: torch.Tensor,
                freqs_cis: torch.Tensor,
                attention_mask: torch.Tensor,
                ):
        B, T, C = x.size()

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # (bsz, n_head, seq_len, head_dim)
        xq = xq.view(B, T, self.n_heads, self.head_dim).transpose(1,2)  
        xk = xk.view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)
        xv = xv.view(B, T, self.n_kv_heads, self.head_dim).transpose(1,2)   

        # Apply rotary proj here
        
        print(attention_mask.shape)
        print(xq.shape)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        print(xq.shape)

        qk_matrix = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim) 
        print(qk_matrix.shape)

        qk_matrix = qk_matrix + attention_mask[None, None, :, :] 
        qk_matrix = F.softmax(qk_matrix, dim=-1)

        x = (qk_matrix @ xv).transpose(1,2).contiguous().view(B, T, self.n_heads * self.head_dim) 
        return self.o_proj(x)
    

class LlamaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.int_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.int_size, bias=False)
        self.down_proj = nn.Linear(config.int_size, config.n_embd, bias = False)
        self.act_fn = nn.SiLU()

    def forward(
            self,
            x: torch.Tensor, 
            ):
        x = self.down_proj(self.act_fn(self.gate_proj(x) * self.up_proj(x)))
        return x
    

# Huggingface implementation of RMS norm and rotary embedding
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
            self, 
            hidden_states
            ):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)



def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)