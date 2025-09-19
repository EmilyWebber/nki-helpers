import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from moe_fwd_transposed import (
                    nki_rms_norm,
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


def load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp1_weight, expert_indices):
    "Set for MLP1"    
    selected_weights = nl.ndarray((batch_size, k, nl.par_dim(intermediate_size), hidden_size), dtype=mlp1_weight.dtype, buffer=nl.sbuf)
    
    for b in nl.static_range(batch_size):
        
        for e in nl.static_range(k):

            expert_index_view = nl.ndarray((1, 1), dtype = expert_indices.dtype, buffer = nl.sbuf)

            # need to use DMA copy to extract (1,1) index from the tensor
            nisa.dma_copy(dst =  expert_index_view[0:1, 0:1], src = expert_indices[b:b+1, e:e+1])

            selected_weights[b, e, :, :] = nl.load(mlp1_weight[expert_index_view[0, 0], :, :])  #(64, 128)

    return selected_weights

def load_mlp_bias(batch_size, k, intermediate_size, hidden_size, mlp1_bias_T, expert_indices):
    "Set for MLP1"
    
    selected_bias_T = nl.ndarray((batch_size, nl.par_dim(intermediate_size), k), dtype = mlp1_bias_T.dtype, buffer = nl.sbuf)

    for b in nl.static_range(batch_size):
    
            for e in nl.static_range(k):

                expert_index_view = nl.ndarray((1, 1), dtype = expert_indices.dtype, buffer = nl.sbuf)

                # need to use DMA copy to extract (1,1) index from the tensor
                nisa.dma_copy(dst =  expert_index_view[0:1, 0:1], src = expert_indices[b, e])

                selected_bias_T[b, :, e] = nl.load(mlp1_bias_T[:, expert_index_view[0,0]])
    return selected_bias_T
    

@nki.jit(debug_kernel=True)
def sample_selection_kernel(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T):
    '''     
    Input tensors:
        t = (128, 128)
        scale = (1, 128)
        gate_weight = (128, 8) 
        gate_bias = (1, 8)
        mlp1_weight = (8, 64, 128)
        mlp1_bias_T = (64, 8)
        mlp2_weight = (8, 128, 32)
        mlp2_bias_T = (128, 8)
    
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
    t = nki_rms_norm(t, scale) # (b, hidden) (128, 128)

    # Gate projection
    g = nl.matmul(t, gate_weight)
    g = nl.add(g, gate_bias)
    
    expert_indices, expert_values = nki_isa_topk(g, k) # (128, 4)

    selected_weights = load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp1_weight, expert_indices) # (128, 4, 64, 128)

    selected_bias_T = load_mlp_bias(batch_size, k, intermediate_size, hidden_size, mlp1_bias_T, expert_indices) #(128, 64, 4)

    rt_token_T = nl.ndarray((batch_size, nl.par_dim(intermediate_size), k), dtype=selected_weights.dtype, buffer=nl.sbuf)
    
    for b in nl.static_range(batch_size):

        # pull a token slice to be used on this batch by all experts
        one_token = nl.ndarray( (1, hidden_size), dtype = t.dtype, buffer = nl.sbuf)
        
        nisa.dma_copy(dst = one_token, src = t[b:b+1, 0:hidden_size] )
            
        for e in nl.static_range(k):

            one_bias_T = selected_bias_T[b, :, e] #(64, 1)

            one_weight = selected_weights[b, e, :, :] # (64, 128)

            multiplied = nl.multiply(one_weight, one_token) + one_bias_T # (64, 128)

            one_vector = nl.sum(multiplied, axis=-1) # (64, 1)
                
            rt_token_T[b, 0:intermediate_size, e:e+1, ] = one_vector 

    one_token = rt_token_T[0, :, :] # (64, 4)
    
    out_token = nl.ndarray( shape = one_token.shape, dtype = one_token.dtype, buffer = nl.hbm)

    nl.store( out_token, one_token)

    return out_token
    

if __name__ == "__main__":

    t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T = generate_input_shapes()

    out = sample_selection_kernel(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T) # returns shape (64,128)

    breakpoint()
