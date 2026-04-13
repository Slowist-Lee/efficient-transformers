#!/bin/bash

# 遇到错误即停止
set -e

# AI Generated

# 1. 运行常规性能基准测试

echo ">>> 0. Running Flash Attention..."
python attn_benchmark.py --mode flash

echo ">>> 1. Running Base Attention..."
python attn_benchmark.py --mode base

echo ">>> 2. Running Compiled Attention..."
python attn_benchmark.py --mode compile


# 2. 运行内存分析 (针对特定长度，如 seq_len=4096)
echo ">>> 4. Running Memory Profiling (seq=4096)..."
python attn_benchmark.py --mode base --profile_mem --target_seq 4096
python attn_benchmark.py --mode flash --profile_mem --target_seq 4096
echo "提示: 可以将生成的 .pickle 文件拖入 https://pytorch.org/memory_viz 进行可视化分析。"

# 3. 整合 CSV 生成最终 Markdown Report
echo ">>> 5. Generating Unified Report..."
python -c "
import pandas as pd
import glob

files = glob.glob('attn_benchmark_*.csv')
df_list = [pd.read_csv(f) for f in files]
merged_df = pd.concat(df_list, ignore_index=True)

# 根据 d_model 和 seq_len 排序方便对比
merged_df = merged_df.sort_values(by=['d_model', 'seq_len', 'mode'])

# 格式化浮点数
float_cols = merged_df.select_dtypes(include=['float']).columns
for col in float_cols:
    merged_df[col] = merged_df[col].map('{:,.2f}'.format)

with open('final_report_bwd.md', 'w') as f:
    f.write('# Attention Benchmark Report\n\n')
    f.write(merged_df.to_markdown(index=False))
print('最终报告已生成: final_report_bwd.md')
"

# 4. 可选: NSYS Profiling 提示
echo ""
echo "========================================"
echo "      NSYS Profiling (NVIDIA Nsight)    "
echo "========================================"
echo "如果需要查看内核执行细节（例如哪个算子耗时最多），请手动运行以下命令："
echo "nsys profile -t cuda,nvtx -o base_profile --force-overwrite true python attn_benchmark.py --mode base"
echo "然后使用 Nsight Systems UI 打开 base_profile.nsys-rep 文件。"