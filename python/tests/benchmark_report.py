"""Write the benchmark results table (tests/benchmark_results.md). Usage: python tests/benchmark_report.py"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import evaluate, CONFIG
from kmz.systems import BENCHMARK

L = ["# Benchmark results (10 systems)", "",
     "Scenario: ψ observed over the whole duration, states over a short window. RMS errors on ψ, independent test trajectories.", "",
     "| System | Selected dictionary | p | Window: model / floor | End: test model / announced / floor / constant | Coverage ±2 bars | Basins | Accuracy |",
     "|---|---|---|---|---|---|---|---|"]
raw = {}
for name in CONFIG:
    r = evaluate(name); K, H = r["K"], r["H_mem"]; b = r["basins"]
    blocks = ", ".join(x.split(" (")[0] if x.startswith("V") else x for x in r["blocks"])
    L.append(f"| {BENCHMARK[name].title} | {blocks} | {r['p']}{'' if r['memory_stable'] else ' (unstable)'} | "
             f"{r['err_test'][H]:.3f} / {r['floor'][H]:.3f} | {r['err_test'][K]:.3f} / {r['err_val'][K]:.3f} / "
             f"{r['floor'][K]:.3f} / {r['std'][K]:.3f} | {r['test_coverage_2s']:.1%} | {b.n_basins} | "
             + (f"{b.accuracy:.1%}" if b.n_basins > 1 else "–") + " |")
    raw[name] = dict(blocks=r["blocks"], p=r["p"], memory_stable=r["memory_stable"], err_test=r["err_test"].tolist(),
                     err_val=r["err_val"].tolist(), floor=r["floor"].tolist(), std=r["std"].tolist(),
                     coverage=r["test_coverage_2s"], n_basins=b.n_basins, accuracy=b.accuracy, time_s=r["total_time"])
    print(L[-1], flush=True)
d = os.path.dirname(__file__)
open(os.path.join(d, "benchmark_results.md"), "w").write("\n".join(L) + "\n")
json.dump(raw, open(os.path.join(d, "benchmark_results.json"), "w"), indent=1)
