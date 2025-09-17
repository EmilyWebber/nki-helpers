import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
from moe_fwd import (
                    nki_rms_norm,
                    nki_lang_topk, 
                    nki_lang_softmax, 
                    generate_input_shapes,
                    load_mlp_weights,
                    load_mlp_bias,
                    nki_isa_topk_reshape
)

import os

def select_bias(batch_size, expert_indices, k, intermediate_size, selected_mlp1_bias_T, mlp1_bias_T ):
    
    for b in nl.static_range(batch_size):

        for e in nl.static_range(k):

            # tile view of the expert ID shape (1,1)
            expert_index_view = expert_indices[0, b:b+1, e:e+1]
        
            # move the selected expert down
            one_expert_bias = nl.load(mlp1_bias_T[:, expert_index_view[0,0]])
        
            selected_mlp1_bias_T[b, 0:intermediate_size, e:e+1] = one_expert_bias
            
    return selected_mlp1_bias_T

@nki.jit
def sample_token_projection(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias):
    '''     
    Input tensors:
        t = (128, 128)
        scale = (1, 128)
        gate_weight = (128, 8) 
        gate_bias = (1, 8)
        mlp1_weight = (8, 64, 128)
        mlp1_bias_T = (64, 8) - transpose on HBM
        mlp2_weight = (8, 128, 32)
        mlp2_bias = (8, 128)
    
    Output tensor:
        out token projection
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

    # on sbuf
    expert_indices, expert_values = nki_isa_topk_reshape(g, k) # (1, 128, 4) and (128, 4)
    
    expert_weights = nki_lang_softmax(expert_values)

    selected_mlp1_weights = load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp1_weight, expert_indices, mlp='1')

    selected_mlp1_bias_T = nl.ndarray((batch_size, nl.par_dim(intermediate_size), k), dtype=mlp1_bias.dtype, buffer=nl.sbuf)
    
    selected_mlp1_bias_T = select_bias(batch_size, expert_indices, k, intermediate_size, selected_mlp1_bias_T , mlp1_bias_T) # on sbuf as (128, 64, 4)

    direction = 'down'
    
    if direction == 'down':
        out_dim = intermediate_size
        
    elif direction == 'up':
        in_dim = t.shape[-1]
        out_dim = hidden_size

    # this shape maybe needs to be changed so we're always moving at 32-dim or more at a time
    rt_token = nl.ndarray((batch_size, nl.par_dim(k), out_dim), dtype=selected_mlp1_weights.dtype, buffer=nl.sbuf)
    
    for b in nl.static_range(batch_size):

        # pull a token slice to be used on this batch by all experts
        if direction == 'down':
            one_token = t[b:b+1, 0:hidden_size]

        for e in nl.static_range(k):

            if direction == 'up':

                one_token = t[ b, e:e+1, 0:in_dim] # (1, 32)

            one_bias = selected_mlp1_bias_T[b, :, e] # (64, 1)

            one_biasT = nl.transpose(one_bias) #(1, 64)

            multiplied = nl.multiply(selected_mlp1_weights[b, e, :, :], one_token) + one_bias # (64, 128)

            one_vector = nl.sum(multiplied, axis=-1) # (64, 1)
                
            rt_token[b, e:e+1, 0:out_dim] = nl.transpose(one_vector) # (1, 64)


    one_token = rt_token[0:1, 0:k, 0:out_dim][0]

    out_array = nl.ndarray((k, out_dim), dtype = rt_token.dtype, buffer = nl.hbm)

    nl.store(out_array[:, :], one_token)
    
    return out_array


if __name__ == "__main__":

    t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = generate_input_shapes()

    out = sample_token_projection(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias.T, mlp2_weight, mlp2_bias) # returns 1, 64, 4
    
    breakpoint()
