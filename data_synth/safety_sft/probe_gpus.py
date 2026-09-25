"""Probe which GPU indices actually work in this container."""
import os, time, torch

def _probe(i):
    t0 = time.time()
    try:
        torch.cuda.set_device(i)
        _ = torch.zeros(1, device=f"cuda:{i}")
        return f"OK ({time.time()-t0:.2f}s) name={torch.cuda.get_device_name(i)}"
    except Exception as e:
        return f"FAIL ({time.time()-t0:.2f}s) {type(e).__name__}: {e}"

def main():
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    n = torch.cuda.device_count()
    print("torch.cuda.device_count() =", n)
    print("torch:", torch.__version__, " cuda:", torch.version.cuda)
    for i in range(n):
        print(f"[gpu {i}] {_probe(i)}", flush=True)

if __name__ == "__main__":
    main()
