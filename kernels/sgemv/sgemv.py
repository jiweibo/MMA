from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the cuda kernel as a python module
lib = load(
    name="sgemv_lib",
    sources=["sgemv.cu"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 200,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event)  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    # 理论访存量: 读 A、x，写 y，各访问一次
    bytes_moved = (a.numel() + b.numel() + out.numel()) * a.element_size()
    bandwidth = bytes_moved / (mean_time * 1e-3) / 1e9  # GB/s
    print(
        f"{out_info:>13}: {out_val}, time:{mean_time:.8f}ms, bandwidth:{bandwidth:.2f}GB/s"
    )
    if show_all:
        print(out)
    return out.clone(), mean_time


print("-" * 80)
M, N, K = 16384, 1, 4096
a = torch.randn((M, K)).cuda().float().contiguous()
b = torch.randn((K, N)).cuda().float().contiguous()
c = torch.randn((M, N)).cuda().float().contiguous()
run_benchmark(lib.sgemv_k32_f32, a, b, "k32f32", c)
run_benchmark(lib.sgemv_k128_f32x4, a, b, "k128f32", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")
print("-" * 80)

M, N, K = 4194304, 1, 16
a = torch.randn((M, K)).cuda().float().contiguous()
b = torch.randn((K, N)).cuda().float().contiguous()
c = torch.randn((M, N)).cuda().float().contiguous()
run_benchmark(lib.sgemv_k16_f32, a, b, "k16f32", c)
run_benchmark(lib.sgemv_k16_f32x4, a, b, "k16f32x4", c)
run_benchmark(partial(torch.matmul, out=c), a, b, "f32_th")
print("-" * 80)
