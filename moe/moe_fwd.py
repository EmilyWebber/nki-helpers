'''
This is a 12-step tutorial that shows how to iteratively develop and improve a NKI kernels for mixture of experts. We focus on the MLP contraction for forward pass, and specifically the context encoding part. We use a TP-degree of 4, assuming a single Trn2 chip which has a default of 4 logical neuron cores. We'll also assume an input context length of 128K, an output sequence length of 4096, and experts per token of 4.

For simplicity, we'll start with a much smaller context length of only 128 tokens and a hidden size of 512. Then we'll work up to the larger shapes throughout the tutorial. We'll also start without the router, adding this later in the tutorial.
'''
import argparse
import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

def generate_input_shapes(tp=4, context_length = 128, hidden_size = 128, num_experts = 32):
    '''
    Generates all shapes used throughout the tutorial, but takes different parameters based on which version you want to test.
    '''

    intermediate_size_per_device = (hidden_size * 2) // tp

    experts_per_device = num_experts // tp 
    
    t = np.random.randn(context_length, hidden_size).astype(np.float16)
    
    scale = np.ones((1, hidden_size)).astype(np.float16)

    gate_weight = np.random.randn(hidden_size, experts_per_device).astype(np.float16)

    gate_bias = np.random.randn(1, experts_per_device).astype(np.float16)

    mlp1_weight = np.random.randn(experts_per_device, intermediate_size_per_device, hidden_size).astype(np.float16)
    
    mlp1_bias = np.random.randn(experts_per_device, intermediate_size_per_device).astype(np.float16)

    mlp2_weight = np.random.randn(experts_per_device, hidden_size,  hidden_size // tp).astype(np.float16)

    mlp2_bias = np.random.randn(experts_per_device, hidden_size).astype(np.float16)

    return t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias

###############################
# v1 - write MOE fwd in Numpy #
###############################

def rms_norm(x, scale, eps=1e-05):  
    
    # Convert to float32 for better numerical stability
    t = x.astype(np.float32)
    
    # Compute mean of squares along last dimension
    mean_squared = np.mean(t**2, axis=-1, keepdims=True)
    
    # Compute rsqrt(mean + eps) directly
    rsqrt = 1.0 / np.sqrt(mean_squared + eps)
    
    # Normalize
    t = t * rsqrt

    # Apply scale - broadcasting happens automatically
    t = t * scale
    
    # Convert back to original dtype
    return t.astype(x.dtype)

def softmax(expert_values):
        
    # Subtract max for numerical stability (along expert dimension)
    max_values = np.max(expert_values, axis=-1, keepdims=True)
    
    shifted_values = expert_values - max_values
    
    # Compute exp of shifted values
    exp_values = np.exp(shifted_values)
    
    # Sum across expert dimension (axis=-1) and keep dims for broadcasting
    sum_exp = np.sum(exp_values, axis=-1, keepdims=True)
    
    # Normalize
    expert_weights = exp_values / sum_exp

    return expert_weights

def sigmoid(x):
    # Numerically stable sigmoid
    x_safe = x.copy()
    mask = x_safe > 0
    
    result = np.zeros_like(x_safe)
    # Where x > 0
    result[mask] = 1 / (1 + np.exp(-x_safe[mask]))
    # Where x <= 0
    exp_x = np.exp(x_safe[~mask])
    result[~mask] = exp_x / (1 + exp_x)
    
    return result

def swiglu(x, alpha=1.702, limit=7.0):
    # Split into two halves along last dimension
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    
    # Clamp the input values
    x_glu = np.clip(x_glu, None, limit)
    x_linear = np.clip(x_linear, -limit, limit)
    
    # Compute SwiGLU with numerically stable sigmoid
    out_glu = x_glu * sigmoid(alpha * x_glu)
    
    # Add 1 to linear part and multiply
    return out_glu * (x_linear + 1)

            
def v1(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias):
    '''
    Does all operations for the MLPBlock forward pass in pure Numpy, assuming tiny shapes.

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
        t_out = (128, 128)
        
    '''

    # experts per token
    k = 4
    
    t = rms_norm(t, scale)
    g = np.matmul(t, gate_weight) + gate_bias

    expert_indices = np.argsort(-g, axis=-1)[:, :k]
    expert_values = np.take_along_axis(g, expert_indices, axis=-1)
    expert_weights = softmax(expert_values)

    # MLP 1
    selected_mlp1_weights = mlp1_weight[expert_indices] # (128, 4, 64, 128)
    selected_mlp1_bias = mlp1_bias[expert_indices] # (128, 4, 64)

    # Equivalent to torch.einsum("beck,bk->bec", mlp1_weight, t)
    t_expanded = t[:, None, None, :].transpose(0, 1, 3, 2)  # (128, 1, 128, 1)
    t = np.matmul(selected_mlp1_weights, t_expanded)  #  (128, 4, 64, 1)
    t = t.squeeze(-1)  # s (128, 4, 64)
    
    t = swiglu(t)    

    # MLP 2
    selected_mlp_weight_2 = mlp2_weight[expert_indices] # (128, 4, 128, 32
    
    selected_mlp_bias_2 = mlp2_bias[expert_indices] # (128, 4, 128)

    # Equivalent to torch.einsum("beck,bek->bec", mlp2_weight, t)
    t_expanded = t[..., None]  # (128, 4, 32, 1)
    t = np.matmul(selected_mlp_weight_2, t_expanded)  #  (128, 4, 128, 1)
    t = t.squeeze(-1)  # (128, 4, 32)
    t = t + selected_mlp_bias_2

    # Equivalent to torch.einsum("bec,be->bc", t, expert_weights)
    t = t * expert_weights[..., None] # (128, 4, 128)
    t = np.sum(t, axis=1)  # (128, 128)

    print ('Done with numpy moe fwd!')
    return t


##################################
# v2 - write MOE fwd in NKI Lang #
##################################

def nki_rms_norm(x, scale, eps=1e-05):
    # Convert to float32 for better numerical stability
    t = x.astype(np.float32)
    
    # Compute mean of squares along last dimension
    mean_squared = nl.mean(nl.square(t), axis=-1, keepdims=True)
    
    # Compute rsqrt(mean + eps) directly
    rsqrt = 1.0 / nl.sqrt(mean_squared + eps)
    
    # Normalize
    t = t * rsqrt
    
    # Apply scale - broadcasting happens automatically
    t = nl.multiply(t, scale)
    
    # Convert back to original dtype
    return t.astype(x.dtype)

    
def nki_lang_topk(g, k=4, TILE_P = 128):
    # g shape: (128, 8)

    num_experts = 8
    
    # Initialize output arrays
    expert_indices = nl.ndarray((TILE_P, k), dtype=g.dtype, buffer=nl.sbuf)
    expert_values = nl.ndarray((TILE_P, k), dtype=g.dtype, buffer=nl.sbuf)

    # Create constants with proper types
    zero = nl.zeros((TILE_P, 1), dtype=g.dtype, buffer=nl.sbuf)
    one = nl.ones((TILE_P, 1), dtype=g.dtype, buffer=nl.sbuf)
    neg_inf = nl.full((TILE_P, 1), fill_value=float('-inf'), dtype=g.dtype, buffer=nl.sbuf)

    zero_e = nl.zeros((TILE_P, num_experts), dtype=g.dtype, buffer=nl.sbuf)
    one_e = nl.ones((TILE_P, num_experts), dtype=g.dtype, buffer=nl.sbuf)
    
    # Create range of expert indices [0,1,2,3,4,5,6,7]
    expert_range = nl.ndarray((1, num_experts), dtype=g.dtype, buffer=nl.sbuf)
    
    # do not unroll this loop, keep it in order
    for i in nl.static_range(num_experts):
        expert_range[0, i] = i
    
    # For each of the k experts we want to select
    for i in nl.static_range(k):
        
        # Find the maximum values across all 128 tokens
        m = nl.max(g, axis=-1) 

        # Assign values to all 128 tokens in kth position from expert values in g
        expert_values[:, i] = m

        max_vals = nl.broadcast_to(m, shape = (g.shape)) # (128, 8)

        # Find positions where values equal the max
        condition = nl.equal(g, max_vals) # (128, 8)

        # Convert boolean condition to integers (0s and 1s)
        condition_int = nl.where(condition, one_e, zero_e) # (128, 8)

        # captures the value of the index as an integer
        indices = nl.multiply(condition_int, expert_range) # (128, 8)

        # reduce along f-dim 
        index = nl.sum(indices, axis=-1) # (128, 1)

        expert_indices[:, i] = index
        
        # Set selected values in g to -inf for next iteration
        g[...] = nl.where(condition, neg_inf, g)
    
    return expert_indices, expert_values 


def nki_lang_softmax(expert_values):
        
    # Subtract max for numerical stability (along expert dimension)
    max_values = nl.max(expert_values, axis=-1, keepdims=True)
    
    shifted_values = expert_values - max_values
    
    # Compute exp of shifted values
    exp_values = nl.exp(shifted_values)
    
    # Sum across expert dimension (axis=-1) and keep dims for broadcasting
    sum_exp = nl.sum(exp_values, axis=-1, keepdims=True)
    
    # Normalize
    expert_weights = exp_values / sum_exp

    return expert_weights


@nki.jit
def nki_swiglu(x, alpha=1.702, limit=7.0):
    """
    Input: x shape (128, 4, 64)
    Output: shape (128, 4, 32)
    """
    batch_size, num_experts, intermediate_size = x.shape
    output_size = intermediate_size // 2  # 32
    
    # Create output tensor with k as partition dimension
    result = nl.ndarray((batch_size, nl.par_dim(num_experts), output_size), 
                       dtype=x.dtype, buffer=nl.sbuf)

    # Generate indices for all dimensions using mgrid
    i_b, i_e, i_f = nl.mgrid[0:1, 0:1, 0:output_size]
    
    # Create indices for even and odd columns
    i_f_even = 2 * i_f
    i_f_odd = 2 * i_f + 1

    ones = nl.ones((1, 1, output_size), dtype=x.dtype, buffer=nl.sbuf)
    
    for b in nl.static_range(batch_size):
        for e in nl.static_range(num_experts):
            
            # Get both glu and linear parts using all advanced indexing
            x_glu = x[i_b + b, i_e + e, i_f_even]
            x_linear = x[i_b + b, i_e + e, i_f_odd]
            
            # Clamp values
            x_glu = nl.minimum(x_glu, limit)  # upper bound only
            x_linear = nl.minimum(nl.maximum(x_linear, -limit), limit)  # both bounds
            
            # Compute sigmoid(alpha * x_glu)
            scaled_glu = nl.multiply(x_glu, alpha)
            sigmoid_glu = nl.sigmoid(scaled_glu)
            
            # Compute final result: (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)
            glu_term = nl.multiply(x_glu, sigmoid_glu)
            linear_plus_one = nl.add(x_linear, ones)
            final = nl.multiply(glu_term, linear_plus_one)
            
            # Store result
            result[i_b + b, i_e + e, i_f] = final
            
    return result


def load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp_weight, expert_indices, mlp='1'):

    tp_degree = 4
    
    if '1' in mlp:
        rt = nl.ndarray((batch_size, k, nl.par_dim(intermediate_size), hidden_size), 
                                     dtype=mlp_weight.dtype, buffer=nl.sbuf)
    elif '2' in mlp:
        rt = nl.ndarray((batch_size, k, nl.par_dim(hidden_size), hidden_size // tp_degree ), 
                             dtype=mlp_weight.dtype, buffer=nl.sbuf)
    
    for b in nl.static_range(batch_size):
        for e in nl.static_range(k):
            rt[b, e, :, :] = nl.load(mlp_weight[expert_indices[b, e]])
    
    return rt

def load_mlp_bias(batch_size, k, intermediate_size, hidden_size, expert_indices, mlp1_bias, selected_mlp_bias, mlp='1'):

    if '1' in mlp:
        fill_val = intermediate_size
    elif '2' in mlp:
        fill_val = hidden_size
    
    for b in nl.static_range(batch_size):
    
            for e in nl.static_range(k):
    
                # tile view of the expert ID shape (1,1)
                expert_id = expert_indices[b, e]
            
                # create the view of the tensor on hbm using the tile on sbuf for the index
                one_expert_bias = mlp1_bias[expert_id, :]
    
                # slow but working
                one_expert_tile = nl.load(one_expert_bias)
            
                one_expert_tile_T = nl.transpose(one_expert_tile)
            
                nl.store(selected_mlp_bias[b, e:e+1, 0:fill_val], value = one_expert_tile_T)
                
    return selected_mlp_bias

def first_token_projection(batch_size, k, intermediate_size, hidden_size, selected_mlp1_weights, selected_mlp1_bias, t):

    rt_token = nl.ndarray((batch_size, nl.par_dim(k), intermediate_size), dtype=selected_mlp1_weights.dtype, buffer=nl.sbuf)
    
    for b in nl.static_range(batch_size):

        # pull a token slice to be used on this batch by all experts
        one_token = t[b:b+1, 0:hidden_size]
        
        for e in nl.static_range(k):
            
            one_bias = nl.load(selected_mlp1_bias[b, e, :]) #(1, 128)

            multiplied = nl.multiply(selected_mlp1_weights[b, e, :, :], one_token) + one_bias # (64, 128)
            
            one_vector = nl.sum(multiplied, axis=-1) # (64, 1)

            rt_token[b, e:e+1, 0:intermediate_size] = nl.transpose(one_vector) # (1, 64)

    return rt_token

def mlp_runner(batch_size, k, intermediate_size, hidden_size, t, mlp_weight, mlp_bias, expert_indices, mlp='1' ):

    # MLP1
    if '1' in mlp:

        selected_mlp1_weights = load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp_weight, expert_indices, mlp='1' )
    
        selected_mlp1_bias = nl.ndarray((batch_size, nl.par_dim(k), intermediate_size), 
                                       dtype=mlp_bias.dtype, buffer=nl.hbm)
    
        selected_mlp1_bias = load_mlp_bias(batch_size, k, intermediate_size, hidden_size, expert_indices, mlp_bias, selected_mlp1_bias, mlp='1')
    
        # bias is loaded and added here too 
        t_out = first_token_projection(batch_size, k, intermediate_size, hidden_size, selected_mlp1_weights, selected_mlp1_bias, t) 

    # MLP 2
    elif '2' in mlp:
        
        selected_mlp2_weights = load_mlp_weights(batch_size, k, intermediate_size, hidden_size, mlp_weight, expert_indices, mlp='2')
    
        selected_mlp2_bias = nl.ndarray((batch_size, nl.par_dim(k), hidden_size), 
                                       dtype=mlp_bias.dtype, buffer=nl.hbm)
    
        selected_mlp2_bias = load_mlp_bias(batch_size, k, intermediate_size, hidden_size, expert_indices, mlp_bias, selected_mlp2_bias, mlp='2')

        # to do add second token projection
    
        # to do add all_reduce here
    
        # to do add bias 

        # filler
        t_out = nl.ones((128, 128), dtype=mlp_bias.dtype, buffer=nl.sbuf)

    return t_out


@nki.jit
def v2(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias):
    '''
    Does all operations for the MLPBlock forward pass in NKI Lang, assuming tiny shapes.
    
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
        t_out = (128, 128)
        
    '''

    # set parameters
    k = 4
    batch_size = t.shape[0]                    # 128
    num_experts = mlp1_weight.shape[0]         # 8
    intermediate_size = mlp1_weight.shape[1]   # 64
    hidden_size = mlp1_weight.shape[2]         # 128

    result = nl.ndarray((t.shape), dtype = t.dtype, buffer = nl.shared_hbm)

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

    # topk
    
    expert_indices, expert_values = nki_lang_topk(g, k) # (128, 4)
    expert_weights = nki_lang_softmax(expert_values)
    
    # MLP1
    t_o =  mlp_runner(batch_size, k, intermediate_size, hidden_size, t, mlp1_weight, mlp1_bias, expert_indices, mlp='1' )
    
    # to do add swiglu here
    t_o = nki_swiglu(t_o)

    # MLP2
    t_o =  mlp_runner(batch_size, k, intermediate_size, hidden_size, t_o, mlp2_weight, mlp2_bias, expert_indices, mlp='2' )

    # to do take weighted sum of experts
    
    nl.store(result[...], value = t)
    
    return result



def main(version):
    
    if 'numpy' in version:

        # call generate shapes for this version 
        t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = generate_input_shapes(context_length = 128, 
                                                                                                                 hidden_size = 128)
        t_out = v1(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias)

    if 'lang' in version:

        t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = generate_input_shapes(context_length = 128, 
                                                                                                                 hidden_size = 128)
        
        t_out = v2(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias)

    assert t.shape == t_out.shape

if __name__ == "__main__":
    
    main(version = 'lang')
