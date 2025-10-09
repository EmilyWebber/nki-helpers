import torch
from torch_xla.core import xla_model as xm

from matrix_multiplication_nki_kernels import nki_matmul_fully_optimized_

import neuronxcc.nki.typing as nt
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

import os

def cpu_test(lhs, rhs):
        
  output_torch_cpu = torch.matmul(lhs, rhs)

  return output_torch_cpu

def neuron_test(lhs, rhs, use_nki=False):

  xla_device = xm.xla_device()
  lhs.to(xla_device)
  rhs.to(xla_device)

  if not use_nki:

      output_torch_neuron = torch.matmul(lhs, rhs)

  else:
      output_torch_neuron = nki_matmul_fully_optimized_(lhs, rhs)
    
  return output_torch_neuron

def test_medium_shapes():
    
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

def test_large_nki():
    
    device = xm.xla_device()
    
    e = 2048
    s = 49152 

    a = torch.rand((e, s), dtype=torch.bfloat16, device=device)
    
    b = torch.rand((e, s), dtype=torch.bfloat16, device = device)
                   
    ab = nki_matmul_fully_optimized_(a, b) # gives (s, s)
    

if __name__ == "__main__":
     
    lhsT = nt.tensor[[8192, 4096], nl.bfloat16]
    rhs = nt.tensor[[8192, 8192], nl.bfloat16]

    print("Benchmarking nki_matmul_fully_optimized")

    bench_func = nki.benchmark(warmup=5, iters=10, save_neff_name='file.neff', save_trace_name='profile.ntff' )(nki_matmul_fully_optimized_)
    
    bench_func(lhsT, rhs)
    
    latency_res = bench_func.benchmark_result.nc_latency
    
    p99 = latency_res.get_latency_percentile(99)
    
    print("Latency: {:.2f} ms (P99)".format(p99 / 1000.0))

    


    
