#!/usr/bin/env python3
"""
停止所有正在运行的方案进程
用法: python scripts/stop_all.py [--force]
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIDS_FILE = PROJECT_ROOT / "experiments" / ".running_processes.txt"


def is_windows():
    return sys.platform == "win32"


def kill_process_tree(pid: int, force: bool = False):
    """终止进程及其所有子进程"""
    try:
        if is_windows():
            # Windows: 使用 taskkill 终止进程树
            flag = "/F" if force else ""
            cmd = f"taskkill /T {flag} /PID {pid}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            # 组合 stdout 和 stderr 进行检查
            output = result.stdout + result.stderr

            # 检查成功: 返回码为0，或者输出包含"成功"(中文)或"success"(英文)
            success = (
                result.returncode == 0
                or "成功" in output
                or "success" in output.lower()
                or "terminated" in output.lower()
            )

            # 检查进程是否已不存在（也是成功状态）
            if not success and result.returncode != 0:
                if (
                    "not found" in output.lower()
                    or "不存在" in output
                    or "无法找到" in output
                    or "could not find" in output.lower()
                    or "no running" in output.lower()
                ):
                    return True  # 进程已经不存在，也算成功
            return success
        else:
            # Unix/Linux/macOS: 使用 pgrep 查找子进程并终止
            import signal
            import os

            # 首先尝试终止子进程
            try:
                # 使用 pgrep 查找子进程
                pgrep_result = subprocess.run(
                    ["pgrep", "-P", str(pid)], capture_output=True, text=True
                )
                if pgrep_result.returncode == 0:
                    for child_pid_str in pgrep_result.stdout.strip().split("\n"):
                        if child_pid_str:
                            try:
                                child_pid = int(child_pid_str)
                                kill_process_tree(child_pid, force)
                            except ValueError:
                                pass
            except Exception:
                pass

            # 然后终止当前进程
            try:
                # 使用跨平台兼容的信号值
                # SIGTERM = 15 (终止信号), SIGKILL = 9 (强制终止)
                sig = 9 if force else 15
                os.kill(pid, sig)
                return True
            except ProcessLookupError:
                return False
    except Exception as e:
        print(f"  错误: {e}")
        return False


def check_process_status(pid: int) -> str:
    """检查进程状态"""
    try:
        if is_windows():
            result = subprocess.run(
                f'tasklist /FI "PID eq {pid}" /NH',
                shell=True,
                capture_output=True,
                text=True,
            )
            return "running" if str(pid) in result.stdout else "not_found"
        else:
            # Unix: 使用 kill -0 检查进程是否存在
            result = subprocess.run(["kill", "-0", str(pid)], capture_output=True)
            return "running" if result.returncode == 0 else "not_found"
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="停止所有运行中的实验进程")
    parser.add_argument(
        "--force", action="store_true", help="强制终止 (SIGKILL / taskkill /F)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="仅显示将要终止的进程, 不实际执行"
    )
    args = parser.parse_args()

    if not PIDS_FILE.exists():
        print("没有正在运行的进程记录")
        return

    print("=== 停止所有运行中的进程 ===\n")

    with open(PIDS_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("没有正在运行的进程")
        PIDS_FILE.unlink(missing_ok=True)
        return

    stopped = []
    failed = []
    already_exited = []

    for line in lines:
        if ":" not in line:
            continue

        method, pid_str = line.split(":", 1)
        method = method.strip()

        try:
            pid = int(pid_str.strip())
        except ValueError:
            print(f"[{method}] 无效的 PID: {pid_str}")
            failed.append((method, pid_str, "invalid_pid"))
            continue

        # 检查进程状态
        status = check_process_status(pid)

        if status == "not_found":
            print(f"[{method}] 进程已退出 (PID: {pid})")
            already_exited.append((method, pid))
            continue

        if args.dry_run:
            print(f"[{method}] 将要终止 (PID: {pid})")
            continue

        # 终止进程
        print(f"[{method}] 正在终止 (PID: {pid})...", end=" ")

        if kill_process_tree(pid, force=args.force):
            print("成功")
            stopped.append((method, pid))
        else:
            print("失败")
            failed.append((method, pid, "termination_failed"))

    # 清理 PID 文件
    if stopped or already_exited:
        PIDS_FILE.unlink(missing_ok=True)

    # 输出总结
    print("\n=== 总结 ===")
    print(f"成功停止: {len(stopped)} 个")
    print(f"已退出: {len(already_exited)} 个")
    print(f"失败: {len(failed)} 个")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
