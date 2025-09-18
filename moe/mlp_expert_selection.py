import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from moe_fwd import (
                    nki_rms_norm,
                    nki_lang_topk, 
                    nki_lang_softmax, 
                    generate_input_shapes
)

import os

def nki_isa_topk(g, k=4, TILE_P=128):
    """
    Compute top-k values and indices using NKI ISA operations
    
    Parameters:
    -----------
    g : Tensor
        Input tensor of shape [128, 8]
    k : int
        Number of top values to select (default=4)
    TILE_P : int
        Partition size (default=128)
        
    Returns:
    --------
    expert_indices : Tensor [128, k]
    expert_values : Tensor [128, k]
    """
    # Initialize output arrays
    expert_indices = nl.ndarray((TILE_P, k), dtype = np.float32, buffer = nl.sbuf)
                                
    # expert_indices = nisa.memset(shape=[TILE_P, k], value=0, dtype=g.dtype)
    expert_values = nisa.memset(shape=[TILE_P, k], value=0, dtype=g.dtype)
    
    # Find top 8 values (since that's what the ISA supports)
    top_vals = nisa.max8(src=g)  # Gets top 8 values
    
    # Find indices of these values
    top_indices = nisa.nc_find_index8(data=g, vals=top_vals)
    
    # Since we only want k values, we'll copy just the first k entries
    for i in nl.static_range(k):
        expert_values[:, i] = top_vals[:, i]
        expert_indices[ 0:TILE_P, i:i+1] = top_indices[0:TILE_P, i:i+1]
    
    return expert_indices, expert_values

def nki_isa_topk_reshape(g, k):
    """
    Compute top-k values and indices using NKI ISA operations with efficient reshaping
    """
    TILE_P = 128
    
    # Initialize output arrays (using 128 on the 1st dim only for broadcasting rules)
    expert_indices_XL = nl.ndarray((nl.par_dim(TILE_P), TILE_P, k), dtype=np.float32, buffer=nl.sbuf)

    expert_values = nisa.memset(shape=[TILE_P, k], value=0, dtype=g.dtype)
    
    # Find top 8 values and indices
    top_vals = nisa.max8(src=g)
    top_indices = nisa.nc_find_index8(data=g, vals=top_vals)
    
    # Copy values
    for e in nl.static_range(k):
        expert_values[:, e] = top_vals[:, e]

        # works
        expert_indices_XL[0:TILE_P, 0:TILE_P, e:e+1] = top_indices[0:TILE_P, e:e+1]

    return expert_indices_XL, expert_values


@nki.jit(debug_kernel=True)
def sample_selection_kernel(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias):
    '''     
    Input tensors:
        t = (128, 128)
        scale = (1, 128)
        gate_weight = (128, 8) 
        gate_bias = (1, 8)
        mlp1_weight = (8, 64, 128)
        mlp1_bias = (8, 64)
        mlp2_weight = (8, 128, 32)
        mlp2_bias = (8, 128)
    
    Output tensor:
        out_weight = (64, 128)
    '''  

    # set parameters
    k = 4
    batch_size = t.shape[0]                    # 128
    num_experts = mlp1_weight.shape[0]         # 8
    intermediate_size = mlp1_weight.shape[1]   # 64
    hidden_size = mlp1_weight.shape[2]         # 128
    tp_degree = 4

    # Load tiles

    t = nl.load(t)
    scale = nl.load(scale)
    gate_bias = nl.load(gate_bias)
    gate_weight = nl.load(gate_weight)

    # RMSNorm
    t = nki_rms_norm(t, scale)
    
    # Gate projection
    g = nl.matmul(t, gate_weight)
    g = nl.add(g, gate_bias)
    
    expert_indices, expert_values = nki_isa_topk(g, k) # (128, 4)

    selected_weights = nl.ndarray((batch_size, k, nl.par_dim(intermediate_size), hidden_size), dtype=mlp1_weight.dtype, buffer=nl.sbuf)
    
    for b in nl.static_range(batch_size):
        
        for e in nl.static_range(k):

            expert_index_view = nl.ndarray((1, 1), dtype = expert_indices.dtype, buffer = nl.sbuf)

            # need to use DMA copy to extract (1,1) index from the tensor
            nisa.dma_copy(dst =  expert_index_view[0:1, 0:1], src = expert_indices[b:b+1, e:e+1])

            selected_weights[b, e, :, :] = nl.load(mlp1_weight[expert_index_view[0, 0], :, :])  #(64, 128)
            

    one_weight = selected_weights[0, 0, :, :] # (64, 128)

    out_weight = nl.ndarray(shape = one_weight.shape, dtype = one_weight.dtype, buffer = nl.hbm)

    nl.store(out_weight, one_weight)

    return out_weight

if __name__ == "__main__":

    t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = generate_input_shapes()

    out = sample_selection_kernel(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias) # returns shape (64,128)

    breakpoint()
