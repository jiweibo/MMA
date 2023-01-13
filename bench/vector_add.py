import numpy as np
import torch
import tvm
from tvm import topi
from tvm import te, auto_scheduler
from benchmark import perf_report, Benchmark, do_bench

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector
    y_ptr,  # *Pointer* to second input vector
    output_ptr,  # *Pointer* to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
    # NOTE: `constexpr` so it can be used as a shape value
):
  # There are multiple 'program's processing different data. We identify which program
  # we are here
  pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
  # This program will process inputs that are offset from the initial data.
  # for instance, if you had a vector of length 256 and block_size of 64, the programs
  # would each access the elements [0:64, 64:128, 128:192, 192:256].
  # Note that offsets is a list of pointers
  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  # Create a mask to guard memory operations against out-of-bounds accesses
  mask = offsets < n_elements
  # Load x and y from DRAM, masking out any extra elements in case the input is not a
  # multiple of the block size
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  # Write x + y back to DRAM
  tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
  # We need to preallocate the output
  output = torch.empty_like(x)
  assert x.is_cuda and y.is_cuda and output.is_cuda
  n_elements = output.numel()
  # The SPMD launch grid denotes the number of kernel instances that run in parallel.
  # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int]
  # In this case, we use a 1D grid where the size is the number of blocks
  grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
  # NOTE:
  #  - each torch.tensor object is implicitly converted into a pointer to its first element.
  #  - `triton.jit`'ed functions can be index with a launch grid to obtain a callable GPU kernel
  #  - don't forget to pass meta-parameters as keywords arguments
  add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
  # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
  # running asynchronously at this point.
  return output


@auto_scheduler.register_workload
def vector_add(size, dtype):
  A = te.placeholder((size,), name="A", dtype=dtype)
  B = te.placeholder((size,), name="B", dtype=dtype)
  C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
  return [A, B, C]


@perf_report(
    Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(10, 27)],
        x_log=True,
        line_arg='provider',
        line_vals=['pytorch', 'triton', 'tvm'],
        line_names=['pytorch', 'triton', 'tvm'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='vector_add',
        args={},
    ))
def benchmark(size, provider):
  x_np = np.random.uniform(size=size).astype(np.float32)
  y_np = np.random.uniform(size=size).astype(np.float32)
  gbps = lambda ms: size * 4 / ms * 1e-6
  if provider == "pytorch":
    x_torch = torch.from_numpy(x_np).cuda()
    y_torch = torch.from_numpy(y_np).cuda()
    y_fwd = lambda: x_torch + y_torch
    ms, min_ms, max_ms = do_bench(y_fwd, rep=500)
  elif provider == "tvm":
    target = tvm.target.Target('nvidia/nvidia-t4')
    task = tvm.auto_scheduler.SearchTask(func=vector_add,
                                         args=(size, "float32"),
                                         target=target)
    log_file = 'vector_add.json'
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2)
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)

    dev = tvm.cuda(0)
    x_tvm = tvm.nd.array(x_np, device=dev)
    y_tvm = tvm.nd.array(y_np, device=dev)
    out_tvm = tvm.nd.array(np.zeros(size, dtype=x_tvm.dtype),
                           dev)  #tvm.nd.empty((size,), device=dev)
    y_fwd = lambda: func(x_tvm, y_tvm, out_tvm)
    ms, min_ms, max_ms = do_bench(y_fwd, rep=500)
  elif provider == "triton":
    x_torch = torch.from_numpy(x_np).cuda()
    y_torch = torch.from_numpy(y_np).cuda()
    ms, min_ms, max_ms = do_bench(lambda: add(x_torch, y_torch), rep=500)

  return gbps(ms), gbps(min_ms), gbps(max_ms)


benchmark.run(print_data=True, save_path='.')
