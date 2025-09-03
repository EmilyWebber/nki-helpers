'''
This is a 12-step tutorial that shows how to iteratively develop and improve a NKI kernels for mixture of experts. We focus on the MLP contraction for forward pass, and specifically the context encoding part. We use a TP-degree of 4, assuming a single Trn2 chip which has a default of 4 logical neuron cores. We'll also assume an input context length of 128K, and an output sequence length of 4096.

For simplicity, we'll start with a much smaller context lenghth of only 128 tokens. Then we'll work up to the larger context length throughout the tutorial. We'll also start without the router, adding this later in the tutorial.
'''

import numpy as np

###############################
# v1 - write MOE fwd in Numpy #
###############################
def v1(t, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias):
    '''
    Input tensors:
        t = (128, 2880)
        mlp1_weight = (8, 1440, 2880)
        mlp1_bias = (8, 1440)
        mlp2_weight = (8, 2880, 720)
        mlp2_bias = (8, 2880)

    Output tensor:
        t_out = (128, 2880)
        
    '''

    # just the rmsnorm
    # t = norm(t)

    # a linear transformation
    # g = gate(t)

    # for numpy, use np.argsort probably
    # experts = topk(g)
    
    # expert_weights = softmax(experts)

    # indices = experts.indices

    # mlp_weight_1 = mlp1_weight[indices]
    # mlp_bias_1 = mlp1_bias[indices]
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
    
    mlp1_weight = np.random.randn(experts_per_device, intermediate_size_per_device, hidden_size).astype(np.float16)
    
    mlp1_bias = np.random.randn(experts_per_device, intermediate_size_per_device).astype(np.float16)

    mlp2_weight = np.random.randn(experts_per_device, hidden_size,  hidden_size // tp).astype(np.float16)

    mlp2_bias = np.random.randn(experts_per_device, hidden_size).astype(np.float16)

    return t, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias



def main(version):
    
    if 'numpy' in version:

        # call generate shapes for this version 
        t, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = generate_input_shapes(context_length = 128)
        
        t_out = v1(t, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias)

        assert t.shape == t_out.shape

if __name__ == "__main__":
    main(version = 'numpy')
