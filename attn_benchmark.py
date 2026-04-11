# Benchmarking PyTorch Attention, 用来感受我们为什么要使用FA?
import sys
import os
import torch
from torch import Tensor
import timeit
from jaxtyping import Float, Bool
import pandas as pd
import numpy as np
import argparse
from components.nn_basic import AttnBlock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def benchmark_attention(compile=False):
    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]
    results = [] # 用于存储所有实验结果的列表

    for d in d_models:
        for seq in seq_lens:
            try:
                torch.cuda.empty_cache()
                model = AttnBlock().cuda()
                if compile:
                    # 编译模型
                    model = torch.compile(model)
                # 1. 随机产生 Q, K, V
                q = torch.randn(batch_size, seq, d, device='cuda', requires_grad=True)
                k = torch.randn(batch_size, seq, d, device='cuda', requires_grad=True)
                v = torch.randn(batch_size, seq, d, device='cuda', requires_grad=True)
                # 3. Warmup (包含前向和反向，让编译器彻底完成编译)
                for _ in range(5):
                    out = model(q=q, k=k, v=v)
                    out.sum().backward()
                    # 清理梯度，防止显存累加
                    q.grad = None
                    k.grad = None
                    v.grad = None
                
                # 列表
                fwd_times=[]
                bwd_times=[]

                torch.cuda.reset_peak_memory_stats()

                # 测 100次
                for _ in range(100):
                    # 前向传播
                    torch.cuda.synchronize()
                    forward_start=timeit.default_timer()
                    out = model(q=q, k=k, v=v)
                    torch.cuda.synchronize()
                    forward_end=timeit.default_timer()

                    fwd_times.append((forward_end - forward_start) * 1000) # 转换为毫秒
                    # 反向传播
                    loss = out.sum() 
                    if q.grad is not None:
                        q.grad.zero_()
                        k.grad.zero_()
                        v.grad.zero_()     
            
                    torch.cuda.synchronize()
                    backward_start=timeit.default_timer()
                    loss.backward()
                    torch.cuda.synchronize()
                    backward_end=timeit.default_timer()
                    bwd_times.append((backward_end - backward_start) * 1000)
                #测量显存
                mem_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)

                # 5. 计算统计数据并记录
                results.append({
                    "d_model": d,
                    "seq_len": seq,
                    "fwd_mean (ms)": np.mean(fwd_times),
                    "fwd_std (ms)": np.std(fwd_times),
                    "bwd_mean (ms)": np.mean(bwd_times),
                    "bwd_std (ms)": np.std(bwd_times),
                    "memory (MB)": mem_allocated_mb,
                    "status": "Success"
                })
                # 清理显存以备下一次循环
                del q, k, v, out, loss
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # 显存爆炸了
                    results.append({
                        "d_model": d,
                        "seq_len": seq,
                        "fwd_mean (ms)": float('nan'),
                        "fwd_std (ms)": float('nan'),
                        "bwd_mean (ms)": float('nan'),
                        "bwd_std (ms)": float('nan'),
                        "memory (MB)": float('nan'),
                        "status": "OOM"
                    })
                    torch.cuda.empty_cache() # 清理一下
                    # 如果一个 d_model 爆了，更大的 seq_len 肯定也爆，可以直接跳出内层循环
                    break 
                else:
                    # 其他运行时错误
                    raise e

    # 使用 Pandas 整理和输出结果
    df = pd.DataFrame(results)
    df_formatted = df.copy()
    float_cols = df_formatted.select_dtypes(include=['float']).columns
    for col in float_cols:
        df_formatted[col] = df_formatted[col].map('{:,.2f}'.format)
    # 生成 Markdown
    markdown_table = df_formatted.to_markdown(index=False)
    print(markdown_table)
    if compile:
        output_filename = "attn_benchmark_compile.md"
    else:
        output_filename = "attn_benchmark.md"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("# Attention Benchmark Results\n\n")
            f.write(markdown_table)
        print(f"\n结果已成功保存到文件: {output_filename}")
    except IOError as e:
        print(f"\n错误：无法写入文件 {output_filename}。原因: {e}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Whether to use compile.")
    parser.add_argument("--compile",action="store_true",help="Path to config file")
    args = parser.parse_args()
    # 确保有可用的 CUDA 设备
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
    else:
        results_df = benchmark_attention(compile=args.compile)