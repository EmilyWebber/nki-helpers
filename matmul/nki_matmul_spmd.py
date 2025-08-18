import torch_neuronx
from neuronxcc import nki
import neuronxcc.nki.language as nl
import torch
from torch_xla.core import xla_model as xm 
import neuronxcc.nki.isa as nisa


@nki.jit
def nki_matmul_basic_(lhsT, rhs):
  """NKI kernel to compute a 64x128x512 matrix multiplication operation

  Args:
      lhsT: an input tensor of shape [128,128], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
      rhs: an input tensor of shape [128,128], a right hand side argument of the
        matrix multiplication
  Returns:
      result: the resulting output tensor of shape [128,128]
  """
  result = nl.ndarray((128, 128), dtype=lhsT.dtype, buffer=nl.shared_hbm)

  # Calculate tile offsets based on current 'program,' use same dimensions as tile sizes (will this work if the arrays aren't the same size, eg non-square matmuls?

  offset_i_x = nl.program_id(0) * 128
  offset_i_y = nl.program_id(1) * 128

  # Generate tensor indices to index tensors a and b
  ix = offset_i_x + nl.arange(128)[:, None]
  iy = offset_i_y + nl.arange(128)[None, :]

  # Loading the inputs (HBM->SBUF)
  lhs_tile = nl.load(lhsT[ix, iy])
  rhs_tile = nl.load(rhs[ix, iy])

  # Perform the matrix-multiplication
  result_psum = nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

  # Copy the result from PSUM back to SBUF, and cast to expected output data-type
  result_sbuf = nl.copy(result_psum, dtype=result.dtype)

  nl.store(result[ix, iy], value=result_sbuf)

  return result


def nki_matmul_nc(a_input, b_input, num_cores = 2):
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
  grid_f = a_input.shape[1] // 128

  # for NC, just add the param nl.nc(num_cores) as the second argument (that is the number of physical cores to use)
    # this will create an SPMD launch grid of shape (p_dim, f_dim)
  return nki_matmul_basic_[nl.spmd_dim(grid_p, nl.nc(num_cores)), grid_f](a_input, b_input)

def test_torch_spmd(num_p_tiles, num_f_tiles, num_cores):
    
    device = xm.xla_device()
        
    a = torch.rand((num_p_tiles * 128, num_f_tiles * 128), dtype=torch.bfloat16).to(device=device)
    b = torch.rand((num_p_tiles * 128, num_f_tiles * 128), dtype=torch.bfloat16).to(device=device)
    
    output_nki = nki_matmul_nc(a.T, b, num_cores = num_cores)
    print(f"output_nki={output_nki}")
    
    output_torch = torch.matmul(a, b)
    
    print(f"output_torch={output_torch}")
    
    allclose = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    
    if allclose:
        
        print("NKI and Torch match")
        
    else:
        
        print("NKI and Torch differ")

        breakpoint()
        
if __name__ == "__main__":

    # works
    test_torch_spmd(num_p_tiles = 1, num_f_tiles = 1, num_cores = 1)

    # works
    test_torch_spmd(num_p_tiles = 1, num_f_tiles = 1, num_cores = 2)

    # breaks
    # test_torch_spmd(num_p_tiles = 2, num_f_tiles = 2, num_cores = 2)

    # breaks
    # test_torch_spmd(num_p_tiles = 8, num_f_tiles = 8, num_cores = 4)

    
