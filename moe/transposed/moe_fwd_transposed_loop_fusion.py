import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

import os

from moe_fwd_transposed import (
    generate_input_shapes,
    nki_rms_norm,
    nki_isa_softmax,
    nki_isa_topk,
    load_mlp_weights,
    load_mlp_bias
)

def fused_mlp1(b, k, intermediate_size, hidden_size, mlp1_weight, mlp1_bias_T, expert_indices, t, out_token):

    # MLP1 - select one token, loop through e to select the weights, bias, do the matmul and sum to get one vector 

    one_token = nl.ndarray( (1, hidden_size), dtype = t.dtype, buffer = nl.sbuf) # (1, 128)

    nisa.dma_copy(dst = one_token, src = t[b:b+1, 0:hidden_size] )

    out_token = nl.ndarray( shape = (intermediate_size, k), dtype = t.dtype, buffer = nl.sbuf)
    
    for e in nl.static_range(k):
    
        expert_index_view = nl.ndarray((1, 1), dtype = expert_indices.dtype, buffer = nl.sbuf)
    
        nisa.dma_copy(dst =  expert_index_view[0:1, 0:1], src = expert_indices[b, e])

        one_weight = nl.ndarray( shape = (intermediate_size, hidden_size), dtype = mlp1_weight.dtype, buffer = nl.sbuf) 
        
        one_weight[...] = nl.load(mlp1_weight[expert_index_view[0, 0], :, :])  # (64, 128)

        one_bias_T = nl.ndarray( shape = (intermediate_size, 1), dtype = mlp1_bias_T.dtype, buffer = nl.sbuf)

        one_bias_T[...] = nl.load(mlp1_bias_T[:, expert_index_view[0,0]]) # (64, 1)

        multiplied = nl.multiply(one_weight, one_token) + one_bias_T # (64, 128)

        one_vector = nl.sum(multiplied, axis=-1) # (64, 1)
            
        out_token[:, e:e+1] = one_vector 

    return out_token

def swiglu(intermediate_size, x, swiglu_t, output_size, alpha=1.702, limit=7.0):
    """
    Input: x shape (64, 4)
    Output: shape (32, 4)
    """
    intermediate_size, k = x.shape
    output_size = intermediate_size // 2  # 32
    
    # Generate tensor indices using arange
    i_k = nl.arange(k)[:, None]
    i_out = nl.arange(output_size)[None, :]
    i_int = nl.arange(intermediate_size)[None, :]

    i_out_p = nl.arange(output_size)[:, None]
    i_k_f = nl.arange(k)[None, :]
    
    # Create split pattern indices based on the output tensor size
    i_f_even = 2 * i_out
    i_f_odd = 2 * i_out + 1

    x_transposed = nl.ndarray((k, intermediate_size), dtype=x.dtype, buffer=nl.sbuf)
    
    x_glu = nl.ndarray((k, output_size), dtype=x.dtype, buffer=nl.sbuf)
    x_linear = nl.ndarray((k, output_size), dtype=x.dtype, buffer=nl.sbuf)
    
    x_int = nl.ndarray((k, output_size), dtype=x.dtype, buffer=nl.sbuf)
    scaled_glu = nl.ndarray((k, output_size), dtype=x.dtype, buffer=nl.sbuf)
    sigmoid_glu = nl.ndarray((k, output_size), dtype=x.dtype, buffer=nl.sbuf)
    glu_term = nl.ndarray((k, output_size), dtype=x.dtype, buffer=nl.sbuf)
    linear_plus_one = nl.ndarray((k, output_size), dtype=x.dtype, buffer=nl.sbuf)
    final = nl.ndarray((output_size, k), dtype=x.dtype, buffer=nl.sbuf)
    
    # First DMA transpose the input
    x_transposed[...] = nisa.dma_transpose(x[:, :], axes=(1, 0)) # (1, 4, 64)
    
    # Perform the split using advanced indexing
    x_glu[ i_k, i_out] = x_transposed[i_k, i_f_even]
    x_linear[i_k, i_out] = x_transposed[i_k, i_f_odd]

    # Clamp x_glu (upper bound only)
    x_glu[:, :] = nisa.tensor_scalar(x_glu[ :, :], nl.minimum, limit)

    # take greater than negative limit         
    x_int[:, :] = nisa.tensor_scalar(x_linear[ :, :], nl.maximum, -limit)
    
    # then take less then positive limit
    x_linear[:, :] = nisa.tensor_scalar(x_int[ :, :], nl.minimum, limit)

    # SwiGLU computation
    scaled_glu[:, :] = nisa.tensor_scalar(x_glu[:,:], nl.multiply, alpha)

    sigmoid_glu[:,:] = nisa.activation(op=nl.sigmoid, data=scaled_glu[:,:])
    
    glu_term[:,:] = nisa.tensor_tensor(x_glu[:,:], sigmoid_glu[:,:], nl.multiply)

    linear_plus_one[:,:] = nisa.tensor_scalar(x_linear[:,:], nl.add, 1.0)
    
    # Transpose both terms before the final multiplication
    glu_term_t = nisa.dma_transpose(glu_term[:,:], axes=(1, 0))
    
    linear_plus_one_t = nisa.dma_transpose(linear_plus_one[:,:], axes=(1, 0))
    
    final[:, :] = nisa.tensor_tensor(glu_term_t, linear_plus_one_t, nl.multiply) # (1, 32, 4)

    # Store result using indices
    swiglu_t[:, :] = final[:, :]
    
    return swiglu_t

@nki.jit(debug_kernel=True)
def moe_mlp_fwd_fused(x, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T):
    '''     
    Input tensors:
        x = (128, 128)
        scale = (1, 128)
        gate_weight = (128, 8) 
        gate_bias = (1, 8)
        mlp1_weight = (8, 64, 128)
        mlp1_bias_T = (64, 8)
        mlp2_weight = (8, 128, 32)
        mlp2_bias_T = (128, 8)
    
    Output tensor:
        t = (128, 128)
    '''  
    # set parameters
    k = 4
    batch_size = x.shape[0]                    # 128
    num_experts = mlp1_weight.shape[0]         # 8
    intermediate_size = mlp1_weight.shape[1]   # 64
    swiglu_output_size = intermediate_size // 2  # 32
    hidden_size = mlp1_weight.shape[2]         # 128
    tp_degree = 4
    hidden_by_tp = hidden_size // tp_degree # 32
    
    # Load tiles
    x_sbuf = nl.load(x)
    t = nl.load(x)
    scale = nl.load(scale)
    gate_bias = nl.load(gate_bias)
    gate_weight = nl.load(gate_weight)

    # RMSNorm
    t = nki_rms_norm(t, scale) # (b, hidden) (128, 128)

    # Gate projection
    g = nl.matmul(t, gate_weight)
    g = nl.add(g, gate_bias)
    
    expert_indices, expert_values = nki_isa_topk(g, k) # (128, 4)
    expert_weights = nki_isa_softmax(expert_values) # (128, 4)
    expert_weights_T = nl.transpose(expert_weights) # (4, 128)

    for b in nl.static_range(batch_size): 
        
        out_token = nl.ndarray( shape = (intermediate_size, k), dtype = t.dtype, buffer = nl.sbuf)

        out_token[...] = fused_mlp1(b, k, intermediate_size, hidden_size, mlp1_weight, mlp1_bias_T, expert_indices, t, out_token)

        swiglu_t = nl.ndarray((swiglu_output_size, k), dtype=x.dtype, buffer=nl.sbuf)
        
        # swiglu 
        swiglu_t[...] = swiglu(intermediate_size, out_token, swiglu_t, swiglu_output_size, alpha=1.702, limit=7.0)

        # mlp2 - loops through e, select one token, select the weights, bias, do the matmul

        # weighted sum - select one token, one expert weight, take the wegighted sum 

    out_swiglu = nl.ndarray(shape = swiglu_t.shape, dtype = swiglu_t.dtype, buffer = nl.hbm)
    nl.store(out_swiglu, swiglu_t)
    return out_swiglu


if __name__ == "__main__":

    t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T = generate_input_shapes()

    out = moe_mlp_fwd_fused(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T) # returns shape (64,128)

    breakpoint()
