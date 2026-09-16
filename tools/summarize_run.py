"""Print the actual results of the latest run under pacman_runs/ and copy evidence into results/."""
import csv, json, shutil, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
runs = sorted((root / "pacman_runs").glob("*/"))
if not runs:
    sys.exit("no runs found")
run = runs[-1]
print("RUN DIR:", run)

config = json.loads((run / "config.json").read_text())
summary = json.loads((run / "training_summary.json").read_text())
comparison = json.loads((run / "comparison.json").read_text()) if (run / "comparison.json").exists() else None
rows = list(csv.DictReader((run / "training.csv").open()))

print("\n== config ==")
for k in ["exploration", "episodes_requested", "learning_rate", "device", "python", "platform"]:
    print(f"{k}: {config[k]}")
print("packages:", config["packages"])

print("\n== training summary ==")
for k, v in summary.items():
    print(f"{k}: {v}")
el = summary["elapsed_seconds_including_periodic_demos"]
print(f"elapsed: {el/60:.1f} min ({el/3600:.2f} h)")
if rows:
    updates_rows = [r for r in rows if r["mean_loss"] not in ("nan", "")]
    print(f"episodes logged: {len(rows)} | first episode with learning updates: {updates_rows[0]['episode'] if updates_rows else 'NONE'}")
    scores = [float(r["score"]) for r in rows]
    print(f"training score first 25 mean: {sum(scores[:25])/len(scores[:25]):.1f} | last 25 mean: {sum(scores[-25:])/len(scores[-25:]):.1f}")
    print(f"training score min/max: {min(scores):.0f}/{max(scores):.0f} | truncated (time-limited) games: {sum(r['truncated']=='True' for r in rows)}")

if comparison:
    b, a = comparison["before"], comparison["after"]
    print("\n== evaluation (5 fixed seeds, 5% exploration) ==")
    print(f"{'Game':>4} {'Seed':>5} {'Before':>8} {'After':>8} {'Change':>8}")
    for i, (s, bs, as_) in enumerate(zip(b["seeds"], b["scores"], a["scores"]), 1):
        print(f"{i:>4} {s:>5} {bs:>8.0f} {as_:>8.0f} {as_-bs:>+8.0f}")
    print(f"{'Mean':>4} {'':>5} {b['mean']:>8.1f} {a['mean']:>8.1f} {a['mean']-b['mean']:>+8.1f}")
    print("time-limited before/after:", sum(b["time_limited"]), "/", sum(a["time_limited"]))
    print("eval steps before:", b["steps"], "| after:", a["steps"])

demo = run / "demo_scores.json"
if demo.exists():
    print("\n== intermediate demonstration scores (seed 101 only) ==")
    for d in json.loads(demo.read_text()):
        print(f"episode {d['episode']}: {d['scores'][0]:.0f}")

# Copy evidence (no .pt checkpoints; those stay in the ZIP).
res = root / "results"
res.mkdir(exist_ok=True)
for name in ["config.json", "training.csv", "training_summary.json", "comparison.json", "baseline.json", "demo_scores.json", "training_dashboard.png"]:
    if (run / name).exists():
        shutil.copy(run / name, res / name)
(res / "demos").mkdir(exist_ok=True)
for gif in sorted((run / "demos").glob("*.gif")):
    shutil.copy(gif, res / "demos" / gif.name)
print("\ncopied evidence to", res)
print("files:", sorted(p.relative_to(res).as_posix() for p in res.rglob("*") if p.is_file()))
print("checkpoints in run dir:", sorted(p.name for p in run.glob("*.pt")))
zips = sorted((root / "pacman_runs").glob("*.zip"))
print("ZIP:", zips[-1] if zips else "not yet created", f"({zips[-1].stat().st_size/2**20:.1f} MiB)" if zips else "")
