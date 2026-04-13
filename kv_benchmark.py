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
from components.nn_basic import AttnBlock, FastAttnTriton, MHA, TransformerLM, RMSNorm


class KVCacheAttnBlock(nn.Module):
    def __init__(self, d_model, num_heads, device, dtype):
        super().__init__()
        self.mha = MHA(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype)
    
    def reset_cache(self):
        self.mha.reset_cache()
    
    def forward(self, x, use_kv_cache):
        batch_size, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.mha(x, token_positions, use_kv_cache=use_kv_cache)

def calc_flops_kvcache(batch_size, seq_len, d_model, use_kv_cache):
    """计算 KVCache 模式下的 FLOPs（生成阶段仅计算 1 个 token）"""
    if not use_kv_cache:
        # 无缓存：全序列计算
        fwd_flops = 4 * batch_size * (seq_len ** 2) * d_model
    else:
        # 有缓存：仅计算新 token（seq_len=1）与历史 seq_len 的交互
        fwd_flops = 4 * batch_size * seq_len * 1 * d_model
    bwd_flops = 10 * batch_size * seq_len * d_model  # 反向仅参考
    return fwd_flops, bwd_flops
def benchmark_kvcache(profile_mem=False, target_seq=4096):
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192]
    results = []
    dtype = torch.float16
    device = "cuda"

    print("开始 KVCache 性能测试...")
    for d in d_models:
        num_heads = d // 16
        for seq in seq_lens:
            try:
                torch.cuda.empty_cache()
                model = KVCacheAttnBlock(d_model=d, num_heads=num_heads, device=device, dtype=dtype).to(device)
                
                for use_kv_cache in [False, True]:
                    # 构造输入
                    x = torch.randn(batch_size, seq, d, device=device, dtype=dtype)
                    x_gen = torch.randn(batch_size, 1, d, device=device, dtype=dtype)

                    # ===================== Warmup =====================
                    if use_kv_cache:
                        model.reset_cache()
                        model(x, use_kv_cache=True)  # 提前建好缓存，不计入时间！

                    for _ in range(5):
                        if use_kv_cache:
                            model(x_gen, use_kv_cache=True)
                        else:
                            model(x, use_kv_cache=False)
                    # ==================================================

                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    # ===================== ✅ 正确测速 =====================
                    fwd_times = []
                    for _ in range(50):
                        torch.cuda.synchronize()
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        
                        start_event.record()

                        if use_kv_cache:
                            # ✅ 只跑生成 1 token！缓存已经建好了！
                            out = model(x_gen, use_kv_cache=True)
                        else:
                            # 无缓存：跑完整序列
                            out = model(x, use_kv_cache=False)

                        end_event.record()
                        torch.cuda.synchronize()
                        fwd_times.append(start_event.elapsed_time(end_event))
                    # ==========================================================

                    fwd_mean = np.mean(fwd_times)
                    mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

                    results.append({
                        "kv_cache": "on" if use_kv_cache else "off",
                        "d_model": d,
                        "seq_len": seq,
                        "fwd_time(ms)": round(fwd_mean, 2),
                        "peak_mem(MB)": round(mem_peak_mb, 2),
                        "status": "Success"
                    })

                del model
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    for use_kv_cache in [False, True]:
                        results.append({
                            "kv_cache": "on" if use_kv_cache else "off",
                            "d_model": d,
                            "seq_len": seq,
                            "fwd_time(ms)": float('nan'),
                            "peak_mem(MB)": float('nan'),
                            "status": "OOM"
                        })
                    break
                else:
                    raise e

    df = pd.DataFrame(results)
    df.to_csv("kvcache_benchmark.csv", index=False)
    print("测试完成！结果已保存至 kvcache_benchmark.csv")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile_mem", action="store_true", help="启用内存分析")
    parser.add_argument("--target_seq", type=int, default=4096, help="内存分析目标序列长度")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA 环境运行测试！")
        sys.exit(1)

    benchmark_kvcache(profile_mem=args.profile_mem, target_seq=args.target_seq)