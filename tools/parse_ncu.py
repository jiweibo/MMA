"""
Parse `ncu --csv --log-file out.csv ....` output csv info.

```
==PROF== Connected to process PID (.../build/tests/simple_cute_gemm_example)
==PROF== Disconnected from process PID
==WARNING== Found outstanding GPU clock reset, trying to revert...Success.
"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","20798","simple_cute_gemm_example","127.0.0.1","ampere_h16816gemm_256x128_ldg8_stages_32x3_tn","2024-Nov-13 18:14:36","1","7","Command line profiler metrics","gpu__time_duration.sum","nsecond","152192"
...
```
"""

import numpy as np
import argparse
from collections import defaultdict
from typing import DefaultDict, List

def _load_csv(path:str) -> DefaultDict[str, DefaultDict[str, list]]:
    """load csv file and return structure info.

    Args:
        path (str): csv file path.
    Returns:
        {
            "ampere_h16816gemm_256x128_ldg8_stages_32x3_tn": {
                "gpu__time_duration.sum": [100, 200]
            }
        }
    """
    with open(path, 'r') as fp:
        lines = fp.readlines()
    
    ret = defaultdict(lambda: defaultdict(list))
    for line in lines:
        if line.startswith('=='): continue
        fields = line.split(",\"")
        kernel_name = fields[4].replace('"', '')
        metric_name = fields[-3].replace('"', '')
        try:
            tsec = float(fields[-1].replace('"', ''))
        except:
            continue
        ret[kernel_name][metric_name].append(tsec)
    
    return ret


def _print_csv_info(kernel_info: DefaultDict[str, DefaultDict[str, List]]):
    """print kernel info.

    Args:
        kernel_info (DefaultDict[str, DefaultDict[str, List]]): result from _load_csv.
    """
    print("kernel_name\tmetrics_name\tmean\tmedian\tstd\tnumber")
    for kernel_name, kernel_struct in kernel_info.items():
        for metric_name, metric_struct  in kernel_struct.items():
            mean_info = np.mean(metric_struct)
            std_info = np.std(metric_struct)
            median_info = np.median(metric_struct)
            num = len(metric_struct)
            print(f"{kernel_name}\t{metric_name}\t{mean_info}\t{median_info}\t{std_info}\t{num}")


def main():
    args = parse_args()
    _print_csv_info(_load_csv(args.csv))


def parse_args():
    parser = argparse.ArgumentParser("Parser for ncu csv out.")
    parser.add_argument("csv", help="ncu output csv file.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
