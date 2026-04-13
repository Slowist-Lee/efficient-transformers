# Efficient Transformers

## Introduction

本项目在 CS336 前两个课程作业的基础上进行融合及修改，手写实现了 Transformers 以及 Efficient LLM 的一些Techniques.

## Quick Start Up

本项目使用`uv`管理环境。

- 训练模型：（超参数及路径均可在`config.yaml`内调整）

```bash
uv run trainer.py
```

其中 `datasets/` 下需要自行下载 TinyStories 数据集，下载方法：

```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

## Reports

- GPU 型号： NVIDIA GeForce RTX 5090  

- [Attention Report](results/final_report.md): 测试Attention Score的计算耗时
    - 由测量结果来看，原先Attention计算存在大量耗时在 `element-wise kernel` 上。改成Flash Attention后前向传播时间提升了12倍左右。
- [Flash Attention](results/final_report_bwd.md): FlashAttention的反向传播使用triton重构后性能提升10倍左右
- [KV Cache](results/kvcache_benchmark.csv): 在提前进行warmup缓存之后，生成速度变快，可以看到序列越长，启用KVCache的效果越好

## To Do

打算实现一些新技术，目前只是实现了基础KV Cache，还可以进一步。例如：

- KVCache Evict
- KVCache Quantization
- AWQ
- MLA
- 一些并行，例如DDP

另外，目前实验对于MEM的测量不够严谨，所以也没做过相关优化，可以再考虑一下

