'''
Inefficient but operational way to extract the MLP bias given an expert index
'''

import numpy as np
import random

import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

def generate_shapes(intermediate_size, num_experts, batch, k ):

    mlp1_bias = np.random.rand(num_experts, intermediate_size).astype(np.float16)

    expert_indices = []

    for b in range(batch):
        # Generate random integers within the specified range and of the desired length
        selected_experts  = random.sample(range(num_experts), k)
        expert_indices.append(selected_experts)

    return mlp1_bias, np.array(expert_indices, dtype=np.int8)

def python_bias_selection(mlp1_bias, expert_indices, batch):
    
    selected_biases = []
    
    for b in range(batch):
    
        token_experts = []
        
        index_slice = expert_indices[b]
    
        for e in index_slice: 
            expert = mlp1_bias[e, :]
            token_experts.append(expert)
    
        selected_biases.append(token_experts)
    
    selected_biases = np.array(selected_biases, dtype=np.int8)

    return selected_biases

@nki.jit
def nki_bias_selection(mlp1_bias, expert_indices, batch, k, intermediate_size):
    '''
    Takes two matrices, uses indices from one matrix to identify values from a second matrix
    Input:
        mlp1_bias: (8, 64)
        expert_indices: (128, 4)
    Returns:
        selected_mlp1_bias: (128, 4, 64)
    '''
    # Create output tensor with k as partition dimension
    selected_mlp1_bias = nl.ndarray((batch, nl.par_dim(k), intermediate_size), 
                                   dtype=mlp1_bias.dtype, buffer=nl.hbm)

    expert_indices = nl.load(expert_indices)

    for b in nl.static_range(batch):

        for e in nl.static_range(k):

            # tile view of the expert ID shape (1,1)
            expert_id = expert_indices[b, e]
        
            # create the view of the tensor on hbm using the tile for the index
            one_expert_bias = mlp1_bias[expert_id, :]

            one_expert_tile = nl.load(one_expert_bias)
        
            one_expert_tile_T = nl.transpose(one_expert_tile)
        
            nl.store(selected_mlp1_bias[b, e:e+1, 0:intermediate_size], value = one_expert_tile_T)

    return selected_mlp1_bias

if __name__ == "__main__":
        
    intermediate_size = 64
    num_experts = 8
    batch = 128
    k = 4
    
    mlp1_bias, expert_indices = generate_shapes(intermediate_size, num_experts, batch, k)

    # selected_biases = python_bias_selection(mlp1_bias, expert_indices, batch, k, intermediate_size )
    
    selected_mlp1_bias = nki_bias_selection(mlp1_bias, expert_indices, batch, k, intermediate_size)

    
