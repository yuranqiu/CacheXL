#!/usr/bin/env python3
"""
并行运行三个方案: ReAct, BoT, CacheXL
默认并行后台执行: python scripts/run_all.py
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

METHODS = {
    "react": {"script": "src/run_react.py", "name": "ReAct"},
    "bot": {"script": "src/run_bot.py", "name": "BoT"},
    "cachexl": {"script": "src/run_cachexl.py", "name": "CacheXL"},
}

PIDS_FILE = PROJECT_ROOT / "experiments" / ".running_processes.txt"


def save_pids(processes: dict):
    """保存进程 PID"""
    PIDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PIDS_FILE, "w") as f:
        for method, proc in processes.items():
            f.write(f"{method}:{proc.pid}\n")


def load_pids() -> dict:
    """加载之前保存的 PID"""
    processes = {}
    if PIDS_FILE.exists():
        with open(PIDS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    method, pid = line.split(":", 1)
                    try:
                        processes[method] = int(pid)
                    except ValueError:
                        pass
    return processes


def run_method(method: str):
    """运行单个方法（后台运行，日志写入文件）"""
    if method not in METHODS:
        return None

    script_path = PROJECT_ROOT / METHODS[method]["script"]
    cmd = [sys.executable, str(script_path)]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    log_dir = PROJECT_ROOT / "experiments" / method / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{method}_{int(time.time())}.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,
        )
    return proc


def main():
    parser = argparse.ArgumentParser(description="并行运行 ReAct/BoT/CacheXL 实验")
    parser.add_argument(
        "--methods",
        default="react,bot,cachexl",
        help="要运行的方法，逗号分隔 (react,bot,cachexl)",
    )
    parser.add_argument("--status", action="store_true", help="查看运行状态")
    parser.add_argument("--kill", action="store_true", help="停止所有运行中的进程")
    args = parser.parse_args()

    methods_to_run = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.status:
        print("=== 运行状态 ===")
        saved_pids = load_pids()
        if not saved_pids:
            print("没有正在运行的进程")
            return

        import psutil

        for method, pid in saved_pids.items():
            try:
                proc = psutil.Process(pid)
                if proc.is_running():
                    print(f"[{METHODS[method]['name']}] Running (PID: {pid})")
                else:
                    print(f"[{METHODS[method]['name']}] Exited (PID: {pid})")
            except psutil.NoSuchProcess:
                print(f"[{METHODS[method]['name']}] Not found (PID: {pid})")
        return

    if args.kill:
        print("=== 停止所有进程 ===")
        saved_pids = load_pids()
        if not saved_pids:
            print("没有正在运行的进程")
            return

        import psutil

        for method, pid in saved_pids.items():
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                print(f"[{METHODS[method]['name']}] Terminated (PID: {pid})")
            except psutil.NoSuchProcess:
                print(f"[{METHODS[method]['name']}] Not found (PID: {pid})")

        PIDS_FILE.unlink(missing_ok=True)
        print("所有进程已停止")
        return

    print(f"并行启动: {methods_to_run}")
    processes = {}
    for method in methods_to_run:
        proc = run_method(method)
        if proc:
            processes[method] = proc
            print(f"[{METHODS[method]['name']}] Started (PID: {proc.pid})")

    save_pids(processes)
    print(f"\n所有任务已在后台启动，日志文件位于: experiments/*/logs/")


if __name__ == "__main__":
    main()
