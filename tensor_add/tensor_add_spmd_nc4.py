import torch_neuronx
from neuronxcc import nki
import neuronxcc.nki.language as nl
import torch
from torch_xla.core import xla_model as xm 

@nki.jit
def nki_tensor_add_kernel_(a_input, b_input):
  """NKI kernel to compute element-wise addition of two input tensors

  This kernel assumes strict input/output sizes can be uniformly tiled to [128,512]

  Args:
      a_input: a first input tensor
      b_input: a second input tensor

  Returns:
      c_output: an output tensor
  """
  # Create output tensor shared between all SPMD instances as result tensor
  c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

  # Calculate tile offsets based on current 'program'
  offset_i_x = nl.program_id(0) * 128
  offset_i_y = nl.program_id(1) * 512

  # Generate tensor indices to index tensors a and b
  ix = offset_i_x + nl.arange(128)[:, None]
  iy = offset_i_y + nl.arange(512)[None, :]

  # Load input data from device memory (HBM) to on-chip memory (SBUF)
  # We refer to an indexed portion of a tensor as an intermediate tensor
  a_tile = nl.load(a_input[ix, iy])
  b_tile = nl.load(b_input[ix, iy])

  # compute a + b
  c_tile = a_tile + b_tile

  # store the addition results back to device memory (c_output)
  nl.store(c_output[ix, iy], value=c_tile)

  # Transfer the ownership of `c_output` to the caller
  return c_output

def nki_tensor_add_nc(a_input, b_input, num_cores = 2):
  """NKI kernel caller to compute element-wise addition of two input tensors

  This kernel caller lifts tile-size restriction, by applying the kernel on tiles of the inputs/outputs

  Args:
      a_input: a first input tensor, of shape [N*128, M*512]
      b_input: a second input tensor, of shape [N*128, M*512]

  Returns:
      a tensor of shape [N*128, M*512], the result of a_input + b_input
  """

  # The SPMD launch grid denotes the number of kernel instances.
    
  grid_p = a_input.shape[0] // (128 * num_cores)
  grid_f = a_input.shape[1] // 512

  # for NC, just add the param nl.nc(num_cores) as the second argument (that is the number of physical cores to use)
  return nki_tensor_add_kernel_[nl.spmd_dim(grid_p, nl.nc(num_cores)), grid_f](a_input, b_input)

def test_torch_spmd(num_p_tiles, num_f_tiles, num_cores):
    
    device = xm.xla_device()
    
    # this will create an SPMD launch grid of shape (p_dim, f_dim)
    
    a = torch.rand((num_p_tiles * 128, num_f_tiles * 512), dtype=torch.bfloat16).to(device=device)
    b = torch.rand((num_p_tiles * 128, num_f_tiles * 512), dtype=torch.bfloat16).to(device=device)
    
    output_nki = nki_tensor_add_nc(a, b, num_cores = num_cores)
    print(f"output_nki={output_nki}")
    
    output_torch = a + b
    print(f"output_torch={output_torch}")

    t_sum = output_torch.sum().item()

    n_sum = output_nki.sum().item()
    
    if t_sum == n_sum:
        print("NKI and Torch match")
    else:
        print("NKI and Torch differ")

        breakpoint()
if __name__ == "__main__":

    # works
    test_torch_spmd(num_p_tiles = 8, num_f_tiles = 8, num_cores = 1)
    
    # works
    test_torch_spmd(num_p_tiles = 8, num_f_tiles = 8, num_cores = 2)

    # works
    test_torch_spmd(num_p_tiles = 8, num_f_tiles = 8, num_cores = 3)

    # works up to num_core = 4 b/c of LNC=2
    # keeping LNC = 2 enables this kernel to integrate more easily with NxDI
    test_torch_spmd(num_p_tiles = 8, num_f_tiles = 8, num_cores = 4)

    
