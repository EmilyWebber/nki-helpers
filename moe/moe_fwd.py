import numpy as np

###############################
# v1 - write MOE fwd in Numpy #
###############################
def v1():

    # takes a token, mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias
    
    # t = norm(t)

    # g = gate(t)

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
    
    return





def main(version):
    if 'numpy' in version:
        v1()




if __name__ == "__main__":
    main(version = 'numpy')