ncu --csv --log-file simple_gemm.csv --print-units=base  --metrics gpu__time_duration.sum  ./tests/simple_cute_gemm_example

python3 ../tools/parse_ncu.py simple_gemm.csv