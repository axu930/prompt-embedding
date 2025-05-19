from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Model whitepaper https://arxiv.org/abs/2407.21783
# Configs pulled from huggingface https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf 

#meta-llama/Llama-3.2-3B
@dataclass
class Llama3BConfig:
    modelpath:  str = "meta-llama/Llama-3.2-3B-Instruct"
    block_size: int = 32768    # max sequence length is 131072, this reduces memory overhead
    vocab_size: int = 128256   # number of tokens
    n_layer:    int = 28       # number of layers

    n_embd:     int = 3072     # embedding dimension
    int_size:   int = 8192     # intermediate MLP dimension
    n_heads:    int = 24       # number of attn heads 
    n_kv_heads: int = 8        # number kv heads
    head_dim:   int = 128      # channels per head

    rope_theta: float = 500000.0 # theta value 



# meta-llama/Llama-3.2-1B
# 1 498 482 688 with MLP, 430 442 496 without
@dataclass
class Llama1BConfig:
    modelpath:  str = "meta-llama/Llama-3.2-1B-Instruct"

    block_size: int = 32768  # max sequence length is 131072, this reduces memory overhead
    vocab_size: int = 128256 # number of tokens
    n_layer:    int = 16     # number of layers

    n_embd:     int = 2048   # embedding dimension
    int_size:   int = 8192   # intermediate MLP dimension
    n_heads:    int =  32    # number of attn heads 
    n_kv_heads: int = 8      # number kv heads
    head_dim:   int = 64     # channels per head

    rope_theta: float = 10000.0 # theta value 

