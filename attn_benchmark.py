import sys
import os
import math
import torch
import torch.nn as nn
from torch import autograd
import pandas as pd
import numpy as np
import argparse
import triton
from components.nn_basic import AttnBlock, FastAttnTriton

class FlashAttnBlock(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v):
        return FastAttnTriton.apply(q, k, v, False)

def calc_flops(batch_size, seq_len, d_model):
    # 标准 Attention 的理论 FLOPs 计算
    # Forward: QK^T (2*B*N^2*D) + softmax(忽略) + Attn*V (2*B*N^2*D) = 4*B*N^2*D
    fwd_flops = 4 * batch_size * (seq_len ** 2) * d_model
    # Backward 通常是 Forward 的 2.5 倍
    bwd_flops = 10 * batch_size * (seq_len ** 2) * d_model
    return fwd_flops, bwd_flops

def benchmark_attention(mode="base", profile_mem=False, target_seq=None):
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    results = []

    print(f"开始测试模式: {mode.upper()}")
    dtype = torch.float16
    for d in d_models:
        for seq in seq_lens:
            # 如果开启了内存分析，仅对指定的 seq_len 进行分析，防止生成的文件过大
            if profile_mem and target_seq is not None and seq != target_seq:
                continue

            try:
                torch.cuda.empty_cache()
                
                # 初始化模型
                if mode == "flash":
                    model = FlashAttnBlock().cuda()
                else:
                    model = AttnBlock().cuda()
                
                if mode == "compile":
                    model = torch.compile(model)

                # 初始化输入张量
                q = torch.randn(batch_size, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                k = torch.randn(batch_size, seq, d, device='cuda', dtype=dtype, requires_grad=True)
                v = torch.randn(batch_size, seq, d, device='cuda', dtype=dtype, requires_grad=True)

                # 1. Warmup
                # 预热让编译器和 CUDA 启动开销完成
                warmup_steps = 10 if mode == "compile" else 5
                for _ in range(warmup_steps):
                    out = model(q, k, v)
                    out.sum().backward()
                    q.grad = k.grad = v.grad = None
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # 2. Memory Profiling
                if profile_mem and seq == target_seq:
                    torch.cuda.memory._record_memory_history(max_entries=100000)
                    out = model(q, k, v)
                    out.sum().backward()
                    torch.cuda.memory._dump_snapshot(f"memory_snapshot_{mode}_{seq}.pickle")
                    torch.cuda.memory._record_memory_history(enabled=False)
                    print(f"已保存内存快照: memory_snapshot_{mode}_{seq}.pickle (拖入 pytorch.org/memory_viz 查看)")

                # 正式测速
                fwd_times, bwd_times = [], []
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                test_iters = 50
                for _ in range(test_iters):
                    # 前向
                    start_event.record()
                    out = model(q, k, v)
                    end_event.record()
                    torch.cuda.synchronize()
                    fwd_times.append(start_event.elapsed_time(end_event))

                    loss = out.sum()
                    if q.grad is not None:
                        q.grad.zero_()
                        k.grad.zero_()
                        v.grad.zero_()     

                    # 后向
                    start_event.record()
                    loss.backward()
                    end_event.record()
                    torch.cuda.synchronize()
                    bwd_times.append(start_event.elapsed_time(end_event))

                # 4. 统计指标计算
                fwd_mean = np.mean(fwd_times)
                bwd_mean = np.mean(bwd_times)
                mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

                fwd_flops, bwd_flops = calc_flops(batch_size, seq, d)
                fwd_tflops = (fwd_flops / (fwd_mean / 1000.0)) / (10**12)
                bwd_tflops = (bwd_flops / (bwd_mean / 1000.0)) / (10**12)

                results.append({
                    "mode": mode,
                    "d_model": d,
                    "seq_len": seq,
                    "fwd_time(ms)": fwd_mean,
                    "bwd_time(ms)": bwd_mean,
                    "fwd_TFLOPS": fwd_tflops,
                    "bwd_TFLOPS": bwd_tflops,
                    "peak_mem(MB)": mem_peak_mb,
                    "status": "Success"
                })

                del q, k, v, out, loss
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results.append({
                        "mode": mode, "d_model": d, "seq_len": seq,
                        "fwd_time(ms)": float('nan'), "bwd_time(ms)": float('nan'),
                        "fwd_TFLOPS": float('nan'), "bwd_TFLOPS": float('nan'),
                        "peak_mem(MB)": float('nan'), "status": "OOM"
                    })
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e

    df = pd.DataFrame(results)
    output_filename = f"attn_benchmark_{mode}.csv"
    df.to_csv(output_filename, index=False)
    print(f"[{mode}] 结果已保存至 {output_filename}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["base", "compile", "flash"], default="base", help="Attention mode")
    parser.add_argument("--profile_mem", action="store_true", help="Enable memory profiling")
    parser.add_argument("--target_seq", type=int, default=4096, help="Target seq_len for memory profiling to avoid huge files")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required.")
        sys.exit(1)

    benchmark_attention(mode=args.mode, profile_mem=args.profile_mem, target_seq=args.target_seq)