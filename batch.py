#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from datetime import datetime

# =========【请按需修改路径/默认参数】=========
PROGRAM    = "src/separation/FastMNMF2.py"   # ← 你的主程序文件路径（就是贴出来的这份 .py）
INPUT_WAV  = "src/separation/00_02_normal.wav"
OUT_ROOT   = "batch_runs"          # 批量结果根目录
FS         = 48000                 # 与主程序一致即可（用不到，这里仅做注释）
NFFT       = 1024
N_BASIS    = 64
N_ITER     = 150
N_MIC      = 4
ALGO       = "IP"
INIT_SCM   = "angle"               # 固定为 angle
G_EPS      = "5e-10"
GPU        = 0                    # CPU：-1；若要用 GPU，改成 0/1...
# ============================================

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    # 角度列表：1,6,11,...,176
    angles = list(range(1, 180, 5))
    n_sources_list = [2, 3]

    # 记录一个总览日志
    master_log = os.path.join(OUT_ROOT, f"master_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(master_log, "w", encoding="utf-8") as mlog:
        mlog.write(f"PROGRAM={PROGRAM}\nINPUT_WAV={INPUT_WAV}\nOUT_ROOT={OUT_ROOT}\n")
        mlog.write(f"ALGO={ALGO}, INIT_SCM={INIT_SCM}, NFFT={NFFT}, N_BASIS={N_BASIS}, N_ITER={N_ITER}\n")
        mlog.write(f"Angles={angles}\nSources={n_sources_list}\n\n")

    total_jobs = len(angles) * len(n_sources_list)
    job_idx = 0

    for n_src in n_sources_list:
        for ang in angles:
            job_idx += 1
            # 目录名明确表明运行条件
            run_dir = os.path.join(
                OUT_ROOT,
                f"S{n_src}_angle{ang:03d}deg_init-{INIT_SCM}_algo-{ALGO}_nfft{NFFT}_K{N_BASIS}_iter{N_ITER}"
            )
            os.makedirs(run_dir, exist_ok=True)

            # 单个任务日志
            run_log = os.path.join(run_dir, "run.log")
            err_log = os.path.join(run_dir, "run.err")

            # 组装命令行
            cmd = [
                sys.executable, PROGRAM,
                "--input_fname", INPUT_WAV,
                "--gpu", str(GPU),
                "--n_fft", str(NFFT),
                "--n_source", str(n_src),
                "--n_basis", str(N_BASIS),
                "--n_iter_init", "50",
                "--init_SCM", INIT_SCM,
                "--n_iter", str(N_ITER),
                "--g_eps", str(G_EPS),
                "--n_mic", str(N_MIC),
                "--n_bit", "64",
                "--algo", ALGO,
                "--init_angle_deg", str(ang),
                "--save_path", run_dir,
            ]

            # 控制台提示
            print(f"[{job_idx}/{total_jobs}] Running: S={n_src}, angle={ang}°  →  {run_dir}")

            # 执行并记录日志
            with open(run_log, "w", encoding="utf-8") as lf, open(err_log, "w", encoding="utf-8") as ef:
                lf.write("CMD: " + " ".join(cmd) + "\n\n")
                try:
                    subprocess.run(cmd, check=True, stdout=lf, stderr=ef)
                except subprocess.CalledProcessError as e:
                    print(f"  →  FAILED (see {err_log})")
                else:
                    print("  →  OK")

if __name__ == "__main__":
    main()
