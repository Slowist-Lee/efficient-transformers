import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.nn_basic import *

from components.data_basic import *
from components.optim import *
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import timeit

def train(config_path, resume: str = "auto",use_wandb: bool = False):
    config=OmegaConf.load(config_path)
    print("Initialize config.")
    device = config.model.device
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    print(f"Initialize tokenizer: {config.data.tokenizer_name}")
    config.model.vocab_size = tokenizer.vocab_size
    print(f"vocal_size: {tokenizer.vocab_size}")

    if os.path.exists(config.train.bin_path):
        print(f"skipping load txt, {config.train.bin_path} already exists.")
    else:
        os.makedirs(os.path.dirname(config.train.bin_path), exist_ok=True)
        read_bytes = int(config.train.get("read_bytes", 1024 * 1024 * 100))
        with open(config.train.txt_path, "r", encoding="utf-8") as f:
            text = f.read(read_bytes)
            ids = tokenizer.encode(text, add_special_tokens=False)
            np.array(ids, dtype=np.uint16).tofile(config.train.bin_path)

    model = TransformerLM(**OmegaConf.to_container(config.model))
    model.to(device)
    model.train()

    # 优化器相关的参数
    optimizer=AdamW(model.parameters(),**OmegaConf.to_container(config.optim))
    output_dir = config.train.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    alpha_max = config.optim.lr
    alpha_min = alpha_max * 0.1
    max_norm=config.train.max_norm

    start_step=0

    # log相关
    log_every=10
    print("Begin Training...")


    # steps相关
    tc=config.train.max_steps
    max_steps=config.train.max_steps
    tw = int(config.train.get("warmup_steps", 10))

    forward_lst=[]
    backward_lst=[]

    # warmup
    warmup_benchmark=0
    for _ in range(warmup_benchmark):
        # 随机批次的随机数据
        x = torch.randint(0, config.model.vocab_size, (config.train.batch_size, config.model.context_length), device=device)
        y = torch.randint(0, config.model.vocab_size, (config.train.batch_size, config.model.context_length), device=device)
        optimizer.zero_grad()
        logits = model(x)
        loss = Cross_Entropy(logits.view(-1, config.model.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    # 正式训练
    for step in range(start_step, max_steps):
        current_lr = get_lr_cosine_schedule(step, alpha_max, alpha_min, tw, tc)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr 
        # 随机批次的随机数据，为了防止IO，测纯算子时间
        x = torch.randint(0, config.model.vocab_size, (config.train.batch_size, config.model.context_length), device=device)
        y = torch.randint(0, config.model.vocab_size, (config.train.batch_size, config.model.context_length), device=device)
        
        torch.cuda.synchronize()
        forward_start=timeit.default_timer()

        logits=model(x)

        torch.cuda.synchronize()
        forward_end=timeit.default_timer()

        forward_lst.append(forward_end-forward_start)
        

        logits_flat = logits.view(-1, config.model.vocab_size)
        y_flat=y.view(-1)
        loss=Cross_Entropy(logits_flat,y_flat)
        optimizer.zero_grad()

        torch.cuda.synchronize()
        backward_start=timeit.default_timer()

        loss.backward()

        torch.cuda.synchronize()
        backward_end=timeit.default_timer()

        backward_lst.append(backward_end-backward_start)

        grad_norm = gradient_clipping(model.parameters(),max_norm)
        optimizer.step()

        if step % log_every == 0 or step == tc - 1:
            print(f"Step: {step}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}, GradNorm: {grad_norm:.4f}")
    print("Forward:",forward_lst)
    print("Backward:",backward_lst)
    forward_arr = np.array(forward_lst, dtype=np.float64)
    backward_arr = np.array(backward_lst, dtype=np.float64)

    # 秒
    print(f"Forward mean: {forward_arr.mean():.6f} s, std: {forward_arr.std(ddof=0):.6f} s")
    print(f"Backward mean: {backward_arr.mean():.6f} s, std: {backward_arr.std(ddof=0):.6f} s")

    # 毫秒（更直观）
    print(f"Forward mean: {forward_arr.mean()*1000:.3f} ms, std: {forward_arr.std(ddof=0)*1000:.3f} ms")
    print(f"Backward mean: {backward_arr.mean()*1000:.3f} ms, std: {backward_arr.std(ddof=0)*1000:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or generate with TransformerLM")
    
    # 通用参数
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    # 训练参数
    parser.add_argument("--train", action="store_true", help="Run training.")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum number of new tokens to generate")

    args = parser.parse_args()
    train(args.config)
