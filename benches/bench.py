"""Local benchmarking.

One method per run (fresh process). Appends to JSON.

    python benches/bench.py run -m smelt --model microsoft/bitnet-b1.58-2B-4T
    python benches/bench.py run -m hf-gpu --model microsoft/bitnet-b1.58-2B-4T
    python benches/bench.py plot bench_results.json -o benchmark.png
"""

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def hw_info():
    info = {
        "cpu": platform.processor() or platform.machine(),
        "cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / 1e9, 1)

    return info


def sync(device):
    if device != "cpu":
        torch.cuda.synchronize()


def bench_pp(model, tokenizer, pp, device="cpu"):
    ids = torch.randint(0, tokenizer.vocab_size, (1, pp), device=device)
    with torch.no_grad():
        model(ids)

    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        model(ids)

    sync(device)
    return pp / (time.perf_counter() - t0)


def bench_tg(model, tokenizer, tg, device="cpu"):
    prompt = "The meaning of life is"
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(ids, max_new_tokens=4, do_sample=False)

    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tg, do_sample=False)

    sync(device)
    n = out.shape[1] - ids.shape[1]
    return n / (time.perf_counter() - t0)


def results_dict(model, tok, pp, tg, mem, dev):
    return {
        "pp_ts": bench_pp(model, tok, pp, dev),
        "tg_ts": bench_tg(model, tok, tg, dev),
        "mem_mb": mem,
    }


def load_cpu(model_id, quantize=False, threads=None):
    torch.set_num_threads(threads or psutil.cpu_count(logical=False))
    gc.collect()
    rss0 = psutil.Process().memory_info().rss
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    if quantize:
        import smelt

        smelt.quantize(model)

    gc.collect()
    mem = (psutil.Process().memory_info().rss - rss0) / 1e6
    return model, mem, "cpu"


def run_smelt(args):
    model, mem, dev = load_cpu(args.model, quantize=True, threads=args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    return results_dict(model, tok, args.pp, args.tg, mem, dev)


def run_hf_cpu(args):
    model, mem, dev = load_cpu(args.model, quantize=False, threads=args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    return results_dict(model, tok, args.pp, args.tg, mem, dev)


def run_hf_gpu(args):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    mem = torch.cuda.max_memory_allocated() / 1e6
    dev = next(model.parameters()).device
    tok = AutoTokenizer.from_pretrained(args.model)
    return results_dict(model, tok, args.pp, args.tg, mem, dev)


METHODS = {"smelt": run_smelt, "hf-cpu": run_hf_cpu, "hf-gpu": run_hf_gpu}


def do_run(args):
    hw = hw_info()
    print(f"method: {args.method}  model: {args.model}")
    print(f"pp={args.pp} tg={args.tg}  hw: {hw}\n")

    result = METHODS[args.method](args)
    result.update(
        method=args.method,
        model=args.model,
        pp=args.pp,
        tg=args.tg,
        hw=hw,
    )

    print(f"pp{args.pp}: {result['pp_ts']:.1f} tok/s")
    print(f"tg{args.tg}: {result['tg_ts']:.1f} tok/s")
    print(f"mem:   {result['mem_mb']:.0f} MB")

    out = Path(args.output)
    existing = json.loads(out.read_text()) if out.exists() else []
    existing.append(result)
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nappended to {out}")


def do_plot(args):
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.plot).read_text())
    labels = []
    for d in data:
        hw = d["hw"]
        dev = hw.get("gpu") if "gpu" in d["method"] else hw["cpu"]
        labels.append(f"{d['method']}\n({dev})")

    colors = [
        "#e74c3c" if "smelt" in d["method"] else "#2ecc71" if "gpu" in d["method"] else "#3498db"
        for d in data
    ]

    _fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key, title in [
        (axes[0], "pp_ts", f"Prefill (pp{data[0]['pp']}) tok/s"),
        (axes[1], "tg_ts", f"Decode (tg{data[0]['tg']}) tok/s"),
        (axes[2], "mem_mb", "Peak memory (MB)"),
    ]:
        vals = [d[key] for d in data]
        bars = ax.barh(labels, vals, color=colors)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                v + max(vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}",
                va="center",
                fontsize=10,
            )
        ax.set_title(title)
        ax.invert_yaxis()

    name = data[0]["model"].split("/")[-1]
    plt.suptitle(name, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(Path(args.out), dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")

    hdr = f"{'method':<20} {'pp tok/s':>10} {'tg tok/s':>10} {'mem MB':>10}"
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    for d in data:
        print(f"{d['method']:<20} {d['pp_ts']:>10.1f} {d['tg_ts']:>10.1f} {d['mem_mb']:>10.0f}")


def main():
    p = argparse.ArgumentParser(description="smelt-bench")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run")
    run.add_argument("-m", "--method", required=True, choices=METHODS)
    run.add_argument("--model", required=True)
    run.add_argument("-p", "--pp", type=int, default=512)
    run.add_argument("-n", "--tg", type=int, default=128)
    run.add_argument("-t", "--threads", type=int, default=None)
    run.add_argument("-o", "--output", default="bench_results.json")

    plot = sub.add_parser("plot")
    plot.add_argument("plot", help="path to bench_results.json")
    plot.add_argument("-o", "--out", default="benchmark.png")

    args = p.parse_args()
    if args.cmd == "run":
        do_run(args)

    elif args.cmd == "plot":
        do_plot(args)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
