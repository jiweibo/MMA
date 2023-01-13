import numpy as np
import torch
import tvm
from tvm import topi
from tvm import te, auto_scheduler
from benchmark import perf_report, Benchmark, do_bench


@auto_scheduler.register_workload
def reduce_sum(M, N, dtype):
  A = te.placeholder((M, N), name="A", dtype=dtype)
  C = topi.sum(A, axis=1)
  return [A, C]


@perf_report(
    Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(1, 32)],
        line_arg='provider',
        line_vals=['pytorch', 'tvm'],
        line_names=['pytorch', 'tvm'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='reduce_sum',
        args={"M": 1024},
    ))
def benchmark_(M, N, provider):
  x_shape = (M, N)
  x_np = np.random.uniform(size=(M, N)).astype(np.float32)
  x_torch = torch.from_numpy(x_np)
  dev = tvm.cuda(0)
  x_tvm = tvm.nd.array(x_np, device=dev)

  gbps = lambda ms: M * (N - 1) / ms * 1e-6
  if provider == "pytorch":
    y_fwd = lambda: torch.sum(x_torch, dim=1)
    ms, min_ms, max_ms = do_bench(y_fwd, rep=500)
  elif provider == "tvm":
    target = tvm.target.Target('nvidia/nvidia-t4')
    task = tvm.auto_scheduler.SearchTask(func=reduce_sum,
                                         args=(M, N, "float32"),
                                         target=target)
    log_file = 'reduce_sum.json'
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2)
    task.tune(tune_option)
    sch, args = task.apply_best(log_file)
    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target)

    out_tvm = tvm.nd.empty((M,), device=dev)
    y_fwd = lambda: func(x_tvm, out_tvm)
    ms, min_ms, max_ms = do_bench(y_fwd, rep=500)

  return gbps(ms), gbps(min_ms), gbps(max_ms)


benchmark_.run(print_data=True, save_path='.')
