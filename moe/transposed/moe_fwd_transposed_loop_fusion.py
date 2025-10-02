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

    # load mlp1 and mlp2 weights and biases 

    selected_weights_1 = load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp1_weight, expert_indices, mlp='1') # (128, 4, 64, 128)

    selected_bias_T_1 = load_mlp_bias(batch_size, k, intermediate_size, hidden_size, mlp1_bias_T, expert_indices, mlp='1') #(128, 64, 4)

    selected_weights_2 = load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp2_weight, expert_indices, mlp='2')
    
    selected_bias_T_2 = load_mlp_bias(batch_size, k, intermediate_size, hidden_size, mlp2_bias_T, expert_indices, mlp='2')

    expert_weights = nki_isa_softmax(expert_values) # (128, 4)

    expert_weights_T = nl.transpose(expert_weights) # (, 128)

    # do the expert weight and transpose here

    # for b in nl.static_range(batch_size): 

        # mlp1

        # swiglu

        # mlp2

        # weighted sum 





        


    
    out_t = nl.ndarray( shape = t.shape, dtype = t.dtype, buffer = nl.hbm)
    nl.store(out_t, t)
    return out_t








if __name__ == "__main__":

    t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T = generate_input_shapes()

    out = moe_mlp_fwd_fused(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias_T, mlp2_weight, mlp2_bias_T) # returns shape (64,128)

    breakpoint()
