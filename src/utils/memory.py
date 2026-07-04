import gc
import tracemalloc
import torch


def memory_report(tag=""):
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2

        print(
            f"[GPU][{tag}] "
            f"allocated={allocated:.2f}MB "
            f"reserved={reserved:.2f}MB "
            f"max={max_allocated:.2f}MB"
        )

    current, peak = tracemalloc.get_traced_memory()

    print(
        f"[CPU][{tag}] "
        f"current={current/1024**2:.2f}MB "
        f"peak={peak/1024**2:.2f}MB"
    )
