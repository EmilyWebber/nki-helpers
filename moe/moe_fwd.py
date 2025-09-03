'''
This is a 12-step tutorial that shows how to iteratively develop and improve a NKI kernels for mixture of experts. We focus on the MLP contraction for forward pass, and specifically the context encoding part. We use a TP-degree of 4, assuming a single Trn2 chip which has a default of 4 logical neuron cores. We'll also assume an input context length of 128K, an output sequence length of 4096, and experts per token of 4.

For simplicity, we'll start with a much smaller context length of only 128 tokens and a hidden size of 512. Then we'll work up to the larger shapes throughout the tutorial. We'll also start without the router, adding this later in the tutorial.
'''

import numpy as np

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
            
def v1(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias):
    '''
    Input tensors:
        t = (128, 512)
        scale = (1, 512)
        gate_weight = (512, 8)
        gate_bias = (1, 8)
        mlp1_weight = (8, 256, 512)
        mlp1_bias = (8, 256)
        mlp2_weight = (8, 512, 128)
        mlp2_bias = (8, 512)

    Output tensor:
        t_out = (128, 512)
        
    '''

    # experts per token
    k = 4
    
    t = rms_norm(t, scale)

    # a linear transformation for the gate projection
    g = np.matmul(t, gate_weight) + gate_bias

    # find the experts using the gate logits from above, doing full sort like source code
    expert_indices = np.argsort(-g, axis=-1)[:, :k]
    
    # Get the corresponding values
    expert_values = np.take_along_axis(g, expert_indices, axis=-1)

    # softmax on the expert values
    expert_weights = softmax(expert_values)

    selected_mlp1_weights = mlp1_weight[expert_indices]

    selected_mlp1_bias = mlp1_bias[expert_indices]

    # t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
    # t = swiglu(t)    

    # mlp_weight_2 = mlp2_weight[indices]
    # mlp_bias_2 = mlp2_bias[indices]
    # t = torch.einsum("beck,bek->bec", mlp2_weight, t) + mlp_bias_2
    
    # t = torch.einsum("bec,be->bc", t, expert_weights)
    
    return t

def generate_input_shapes(tp=4, context_length = 128000, hidden_size = 2880, num_experts = 32):

    intermediate_size_per_device = (hidden_size * 2) // tp

    experts_per_device = num_experts // tp 
    
    t = np.random.randn(context_length, hidden_size).astype(np.float16)
    
    scale = np.ones(hidden_size).astype(np.float16)

    gate_weight = np.random.randn(hidden_size, experts_per_device).astype(np.float16)

    gate_bias = np.random.randn(1, experts_per_device).astype(np.float16)

    mlp1_weight = np.random.randn(experts_per_device, intermediate_size_per_device, hidden_size).astype(np.float16)
    
    mlp1_bias = np.random.randn(experts_per_device, intermediate_size_per_device).astype(np.float16)

    mlp2_weight = np.random.randn(experts_per_device, hidden_size,  hidden_size // tp).astype(np.float16)

    mlp2_bias = np.random.randn(experts_per_device, hidden_size).astype(np.float16)

    return t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias

def main(version):
    
    if 'numpy' in version:

        # call generate shapes for this version 
        t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = generate_input_shapes(context_length = 128, hidden_size=512)

        t_out = v1(t, scale, gate_weight, gate_bias, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias)

        assert t.shape == t_out.shape

if __name__ == "__main__":
    
    main(version = 'numpy')
