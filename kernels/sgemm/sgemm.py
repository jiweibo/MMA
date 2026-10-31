import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="sgemm_lib",
    sources=[
        "sgemm.cu",
        "sgemm_cublas.cu"
    ],
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

MAX_TFLOPS = -1


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 2,
    iters: int = 20,
    show_all: bool = False,
):
    global MAX_TFLOPS

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)

    if a.size(0) > 1024 or a.size(1) > 1024 or b.size(1) > 1024:
        iters = 10

    if out is not None:
        out.fill_(0)

    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            perf_func(a, b)
    torch.cuda.synchronize()

    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten()[:2].detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}"[:10] for v in out_val]
    TFLOPS = (2 * M * N * K) * 1e-9 / mean_time

    # caculate TFLOPS improved.
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(
            f"{out_info:>35}: {out_val}, time:{mean_time}ms, "
            # f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}(+{improve:.2f}%)"
        )
    else:
        print(
            f"{out_info:>35}: {out_val}, time:{mean_time}ms, "
            # f"swizzle: {swizzle_stride:<4}, TFLOPS: {TFLOPS:<6.2f}"
        )
    if show_all:
        print(out)
    return out, mean_time


Ms = [4096, 8192, 16384]
Ns = [4096, 8192, 16384]
Ks = [2048, 4096, 8192]
MAX_M, MAX_N, MAX_K = 16384, 16384, 8192
# pre allocate for fast profiling.
A = torch.randn((MAX_M, MAX_K), dtype=torch.float).cuda()
B = torch.randn((MAX_K, MAX_N), dtype=torch.float).cuda()
C = torch.randn((MAX_M, MAX_N), dtype=torch.float).cuda()
torch.cuda.synchronize()

MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]

for M, N, K in MNKs:
    MAX_TFLOPS = -1
    print("-" * 130)
    print(" " * 55 + f"M={M}, N={N}, K={K}")
    a = A[:M, :K].contiguous()
    b = B[:K, :N].contiguous()
    c = C[:M, :N].contiguous()
    torch.cuda.synchronize()

    # CUDA Cores FP32
    # run_benchmark(lib.sgemm_naive_f32, a, b, "f32(naive)", c)
    run_benchmark(lib.sgemm_sliced_k_f32, a, b, "f32(slice_k)", c)
    run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4, a, b, "f32x4(t8x8sk)", c)
    run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf, a, b, "f32x4(t8x8bcf)", c)
    run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf, a, b, "f32x4(t8x8dbuf)", c)
    run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf_offset, a, b, "f32x4(t8x8bcf_offset)", c)
    run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_offset, a, b, "f32x4(t8x8bcf_dbuf_offset)", c)
    run_benchmark(lib.sgemm_cublas, a, b, "f32(cublas)", c)
    run_benchmark(partial(torch.matmul, out=c), a, b, "f32(torch)")
