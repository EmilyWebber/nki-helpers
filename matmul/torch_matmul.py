import torch
from torch_xla.core import xla_model as xm

def cpu_test(lhs, rhs):
        
  output_torch_cpu = torch.matmul(lhs, rhs)

  return output_torch_cpu

def neuron_test(lhs, rhs):

  xla_device = xm.xla_device()
  lhs.to(xla_device)
  rhs.to(xla_device)

  output_torch_neuron = torch.matmul(lhs, rhs)

  return output_torch_neuron


if __name__ == "__main__":

  lhs_shape = (4096, 1024)
  rhs_shape = (1024, 2048)
    
  assert lhs_shape[1] == rhs_shape[0]

  lhs = torch.rand(lhs_shape, dtype=torch.bfloat16)
  rhs = torch.rand(rhs_shape, dtype=torch.bfloat16)

  output_torch_cpu = cpu_test(lhs, rhs)
  
  output_torch_neuron = neuron_test(lhs, rhs)
    
  if torch.allclose(output_torch_cpu, output_torch_neuron, atol=1e-4, rtol=1e-2):
    print("Neuron and CPU match")
  else:
    print("Neuron and CPU differ")
