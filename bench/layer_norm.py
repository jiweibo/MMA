import torch

import triton
import triton.language as tl
from benchmark import perf_report, Benchmark, do_bench


@triton.jit
def _layer_norm_fwd_fused(
    Out,
    A,
    Weight,
    Bias,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
  # position of elements processed by this program
  row = tl.program_id(0)
  Out += row * stride
  A += row * stride
  # compute mean
  mean = 0
  _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
  for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + cols, mask=cols < N, other=0.,
                eviction_policy="evict_last").to(tl.float32)
    _mean += a
  mean = tl.sum(_mean, axis=0) / N
  # compute variance
  _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
  for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    a = tl.load(A + cols, mask=cols < N, other=0.,
                eviction_policy="evict_last").to(tl.float32)
    a = tl.where(cols < N, a - mean, 0.)
    _var += a * a
  var = tl.sum(_var, axis=0) / N
  rstd = 1 / tl.sqrt(var + eps)
  # write-back mean/std
  tl.store(Mean + row, mean)
  tl.store(Rstd + row, rstd)
  # multiply by weight and add bias
  for off in range(0, N, BLOCK_SIZE):
    cols = off + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    weight = tl.load(Weight + cols, mask=mask)
    bias = tl.load(Bias + cols, mask=mask)
    a = tl.load(A + cols, mask=mask, other=0.,
                eviction_policy="evict_first").to(tl.float32)
    a_hat = (a - mean) * rstd
    out = a_hat * weight + bias
    # write-back
    tl.store(Out + cols, out, mask=mask)


class LayerNorm(torch.autograd.Function):

  @staticmethod
  def forward(ctx, a, normalized_shape, weight, bias, eps):
    # allocate output
    out = torch.empty_like(a)
    # reshape input data into 2D tensor
    a_arg = a.reshape(-1, a.shape[-1])
    M, N = a_arg.shape
    mean = torch.empty((M,), dtype=torch.float32, device="cuda")
    rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // a.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristic for number of wraps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    _layer_norm_fwd_fused[(M,)](
        out,
        a_arg,
        weight,
        bias,
        mean,
        rstd,
        a_arg.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


def layer_norm_triton(a, normalized_shape, weight, bias, eps):
  return LayerNorm.apply(a, normalized_shape, weight, bias, eps)


def test_layer_norm(M, N, dtype, eps=1e-5, device="cuda"):
  torch.manual_seed(0)
  # create data
  x_shape = (M, N)
  w_shape = (x_shape[-1],)
  weight = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=False)
  bias = torch.rand(w_shape, dtype=dtype, device="cuda", requires_grad=False)
  x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")

  y_tri = layer_norm_triton(x, w_shape, weight, bias, eps)
  y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias,
                                         eps).to(dtype)

  triton.testing.assert_almost_equal(y_tri, y_ref)


@perf_report(
    Benchmark(x_names=['N'],
              x_vals=[512 * i for i in range(2, 32)],
              line_arg='provider',
              line_vals=['torch', 'triton'],
              line_names=['Torch', 'Triton'],
              styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
              ylabel='GB/s',
              plot_name='layer_norm',
              args={
                  'M': 4096,
                  'dtype': torch.float16
              }))
def bench_layer_norm(M, N, dtype, provider, eps=1e-5, device='cuda'):
  # create data
  x_shape = (M, N)
  w_shape = (x_shape[-1],)
  weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=False)
  bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=False)
  x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
  if provider == 'torch':
    y_fwd = lambda: torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps
                                                  )
  elif provider == 'triton':
    y_fwd = lambda: layer_norm_triton(x, w_shape, weight, bias, eps)
  gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
  ms, min_ms, max_ms = do_bench(y_fwd, rep=500)
  return gbps(ms), gbps(max_ms), gbps(min_ms)


test_layer_norm(1151, 8192, torch.float32)
bench_layer_norm.run(save_path='.', print_data=True)
