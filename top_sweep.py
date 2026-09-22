#!/usr/bin/env python3
"""
PufferLib Sweep Analyzer
Finds and displays the top N hyperparameter configurations from sweep logs.
Usage:
    python3 top_sweep.py [--logs logs/drone] [--top 10] [--metric perf] [--details]
"""

import os
import sys
import glob
import argparse
import configparser

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze PufferLib sweep results and display top hyperparameter configurations.")
    parser.add_argument("--logs", "--log_dir", default="logs/drone", help="Directory containing sweep .ini files (default: logs/drone)")
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of top configurations to show (default: 10)")
    parser.add_argument("--metric", default="perf", choices=["perf", "score", "collisions", "episode_return"],
                        help="Metric to sort by (default: perf)")
    parser.add_argument("--details", action="store_true", default=True, help="Show detailed parameter breakdown for top runs")
    return parser.parse_args()

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def load_series_last(series_str, default=0.0):
    if not series_str:
        return default
    parts = [p.strip() for p in series_str.split(",") if p.strip()]
    if not parts:
        return default
    return safe_float(parts[-1], default)

def find_checkpoint(run_id, base_dir="checkpoints/drone"):
    run_dir = os.path.join(base_dir, run_id)
    if not os.path.isdir(run_dir):
        return None
    bins = sorted(glob.glob(os.path.join(run_dir, "*.bin")))
    return bins[-1] if bins else None

def main():
    args = parse_args()
    log_dir = args.logs
    # Handle possible leading slash
    if not os.path.exists(log_dir) and log_dir.startswith("/"):
        alt = log_dir.lstrip("/")
        if os.path.exists(alt):
            log_dir = alt

    if not os.path.isdir(log_dir):
        print(f"Error: Directory '{log_dir}' not found.")
        sys.exit(1)

    ini_pattern = os.path.join(log_dir, "*.ini")
    files = glob.glob(ini_pattern)
    if not files:
        print(f"No .ini sweep logs found in '{log_dir}'.")
        sys.exit(1)

    runs = []
    for filepath in files:
        filename = os.path.basename(filepath)
        if not filename.startswith("sweep_") and "_" not in filename:
            continue

        cfg = configparser.ConfigParser()
        try:
            cfg.read(filepath)
        except Exception:
            continue

        if "metrics" not in cfg:
            continue

        m = cfg["metrics"]
        perf = load_series_last(m.get("env/avoid/perf", m.get("env/perf", "0")))
        score = load_series_last(m.get("env/avoid/score", m.get("env/score", "0")))
        collisions = load_series_last(m.get("env/avoid/collisions", "1.0"))
        ep_ret = load_series_last(m.get("env/episode_return", "0"))
        ep_len = load_series_last(m.get("env/episode_length", "0"))
        steps = load_series_last(m.get("agent_steps", "0"))
        uptime = load_series_last(m.get("uptime", "0"))

        run_id = cfg.get("base", "run_id", fallback=os.path.splitext(filename)[0])
        checkpoint = find_checkpoint(run_id)

        runs.append({
            "run_id": run_id,
            "filepath": filepath,
            "checkpoint": checkpoint,
            "perf": perf,
            "score": score,
            "collisions": collisions,
            "ep_ret": ep_ret,
            "ep_len": ep_len,
            "steps": steps,
            "uptime": uptime,
            "cfg": cfg,
        })

    if not runs:
        print(f"Found {len(files)} files in '{log_dir}', but none contained valid sweep metrics.")
        sys.exit(1)

    # Sort
    reverse_sort = (args.metric != "collisions")
    runs.sort(key=lambda r: r[args.metric], reverse=reverse_sort)

    top_runs = runs[:args.top]

    print("=" * 105)
    print(f"  PUFFERLIB SWEEP ANALYZER: TOP {len(top_runs)} RUNS (Analyzed {len(runs)} Total Runs in {log_dir})")
    print(f"  Sorted by: {args.metric.upper()} ({'Highest First' if reverse_sort else 'Lowest First'})")
    print("=" * 105)

    print(f"{'Rank':<5} {'Run ID':<26} {'Perf':<8} {'Score':<8} {'Coll%':<8} {'Return':<9} {'Length':<7} {'Steps':<9} {'Uptime':<8}")
    print("-" * 105)

    for i, r in enumerate(top_runs, 1):
        coll_pct = f"{r['collisions']*100.0:.1f}%"
        steps_m = f"{r['steps']/1e6:.1f}M"
        uptime_s = f"{r['uptime']:.1f}s"
        print(f"#{i:<4} {r['run_id']:<26} {r['perf']:<8.4f} {r['score']:<8.1f} {coll_pct:<8} {r['ep_ret']:<9.1f} {r['ep_len']:<7.0f} {steps_m:<9} {uptime_s:<8}")

    print("=" * 105)

    if args.details:
        print()
        print("#" * 105)
        print("  DETAILED HYPERPARAMETER CONFIGURATIONS FOR TOP RUNS")
        print("#" * 105)

        for i, r in enumerate(top_runs, 1):
            c = r["cfg"]
            get_env = lambda k, d="-": c.get("env", k, fallback=d)
            get_trn = lambda k, d="-": c.get("train", k, fallback=d)
            get_pol = lambda k, d="-": c.get("policy", k, fallback=d)
            get_vec = lambda k, d="-": c.get("vec", k, fallback=d)

            print()
            print("-" * 80)
            print(f"  RANK #{i}: {r['run_id']}")
            print(f"  Metrics   : Perf={r['perf']:.4f} | Score={r['score']:.1f} | Collision Rate={r['collisions']*100:.1f}% | Ep Length={r['ep_len']:.0f}")
            if r['checkpoint']:
                print(f"  Checkpoint: {r['checkpoint']}")
            print("-" * 80)

            print("  [OBSTACLE AVOIDANCE REWARDS & GEOMETRY]")
            print(f"    collision_penalty   = {get_env('collision_penalty'):<12} safety_margin      = {get_env('safety_margin')}")
            print(f"    alpha_proximity     = {get_env('alpha_proximity'):<12} alpha_vel          = {get_env('alpha_vel')}")
            print(f"    hover_alpha_dist    = {get_env('hover_alpha_dist'):<12} alpha_hover        = {get_env('alpha_hover')}")
            line_sp = get_env('line_spacing', get_env('center_tower_radius'))
            tow_sp = get_env('tower_spacing', get_env('circle_radius'))
            print(f"    tower_radius        = {get_env('tower_radius'):<12} line_spacing       = {line_sp}")
            print(f"    tower_spacing       = {tow_sp:<12} oob_penalty        = {get_env('oob_penalty')}")

            print("  [PPO TRAINING HYPERPARAMETERS]")
            print(f"    learning_rate       = {get_trn('learning_rate'):<12} gamma              = {get_trn('gamma')}")
            print(f"    gae_lambda          = {get_trn('gae_lambda'):<12} ent_coef            = {get_trn('ent_coef')}")
            print(f"    clip_coef           = {get_trn('clip_coef'):<12} vf_coef             = {get_trn('vf_coef')}")
            print(f"    vf_clip_coef        = {get_trn('vf_clip_coef'):<12} max_grad_norm       = {get_trn('max_grad_norm')}")
            print(f"    momentum            = {get_trn('momentum'):<12} replay_ratio        = {get_trn('replay_ratio')}")
            print(f"    horizon             = {get_trn('horizon'):<12} minibatch_size      = {get_trn('minibatch_size')}")

            print("  [NETWORK ARCHITECTURE & SCALE]")
            print(f"    hidden_size         = {get_pol('hidden_size'):<12} num_layers          = {get_pol('num_layers')}")
            print(f"    total_agents        = {get_vec('total_agents'):<12} total_timesteps     = {get_trn('total_timesteps')}")

        print()
        print("=" * 105)
        best = top_runs[0]
        if best["checkpoint"]:
            print("  HOW TO VISUALIZE AND RUN THE #1 BEST POLICY IN 3D:")
            print(f"    ./drone {best['checkpoint']}")
            print("  OR EVALUATE WITH PUFFER:")
            print(f"    ./puffer eval {best['checkpoint']}")
        print("=" * 105)

if __name__ == "__main__":
    main()
