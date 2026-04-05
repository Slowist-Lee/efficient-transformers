from .components.nn_basic import *
from .components.data_basic import *
from .components.optim import *
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import os
import argparse
from pathlib import Path
import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

def _find_latest_checkpoint(output_dir: str) -> str | None:
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return str(checkpoints[-1])


def train(config_path, resume: str = "auto", use_wandb: bool = False):
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
    optimizer=AdamW(model.parameters(),**OmegaConf.to_container(config.optim))

    output_dir = config.train.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    max_norm=config.train.max_norm
    dataset=load_memmap_dataset(config.train.bin_path)
    tc=config.train.max_steps

    alpha_max = config.optim.lr
    alpha_min = alpha_max * 0.1
    tw = int(config.train.get("warmup_steps", 10))
    save_every = int(config.train.get("save_every", 100))
    log_every = int(config.train.get("log_every", 10))

    start_step = 0
    resume_path = None
    if resume == "auto":
        resume_path = _find_latest_checkpoint(output_dir)
    elif resume and resume != "none":
        resume_path = resume
    if resume_path and os.path.exists(resume_path):
        start_step = load_checkpoint(resume_path, model, optimizer)
        print(f"Resumed from checkpoint: {resume_path}, start_step={start_step}")
    else:
        print("No checkpoint found. Cold start training.")

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=config.train.get("wandb_project", "efficient-transformers"), config=OmegaConf.to_container(config))
        except Exception as e:
            print(f"WandB init failed, continue without WandB: {e}")
    
    print("Begin Training...")
    for step in range(start_step, tc):
        current_lr = get_lr_cosine_schedule(step, alpha_max, alpha_min, tw, tc)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr 
        x, y = get_batch(
            dataset=dataset,
            batch_size=config.train.batch_size,
            context_length=config.model.context_length,
            device=device
        )

        logits=model(x)
        logits_flat = logits.view(-1, config.model.vocab_size)
        y_flat=y.view(-1)

        loss=Cross_Entropy(logits_flat,y_flat)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = gradient_clipping(model.parameters(),max_norm)
        optimizer.step()

        if step % log_every == 0 or step == tc - 1:
            print(f"Step: {step}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}, GradNorm: {grad_norm:.4f}")
            if wandb_run is not None:
                wandb_run.log({
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": float(current_lr),
                    "grad_norm": float(grad_norm),
                })

        if (step + 1) % save_every == 0 or step == tc - 1:
            ckpt_path = os.path.join(output_dir, f"checkpoint_step_{step + 1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()

def decode(config_path: str, checkpoint_path: str, prompt: str, max_new_tokens: int = 10000):
    config = OmegaConf.load(config_path)
    device = config.model.device
    context_length = config.model.context_length
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    print(f"Tokenizer '{config.data.tokenizer_name}' loaded.")
    eos_id=tokenizer.eos_token_id
    config.model.vocab_size = tokenizer.vocab_size
    model = TransformerLM(**OmegaConf.to_container(config.model))
    model.to(device)
    load_checkpoint(checkpoint_path, model, optimizer=None)
    model.eval()
    print("model loaded!")
    input_ids=tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 确保不超过模型接受的输入长度
            input_cond = input_tensor if input_tensor.size(1) <= context_length else input_tensor[:, -context_length:]
            logits = model(input_cond) 
            # 保持 batch 维度，便于后续拼接
            next_token_logits = logits[:, -1, :]
            prob=softmax(next_token_logits)
            cdf = torch.cumsum(prob, dim=-1)
            u = torch.rand(prob.shape[:-1] + (1,), device=prob.device) 
            mask = cdf > u
            next_token_id = torch.argmax(mask.to(torch.int), dim=-1, keepdim=True)
            # 检查EOS：
            if eos_id is not None and next_token_id[0, 0].item() == eos_id:
                print("\n[EOS token generated. Stopping.]")
                break   
            input_tensor = torch.cat((input_tensor, next_token_id), dim=1)
    # 仅解码新生成部分，不包含输入 prompt
    generated_token_ids = input_tensor[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(generated_token_ids)
    print(generated_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or generate with TransformerLM")
    
    # 通用参数
    parser.add_argument("--config", default="./cs336_basics/config.yaml", help="Path to config file")

    # 训练参数
    parser.add_argument("--train", action="store_true", help="Run training.")
    parser.add_argument("--resume", default="auto", help="Checkpoint path for resuming training ('auto', 'path/to/ckpt', or 'none')")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging for training")

    # 生成参数
    parser.add_argument("--generate", type=str, metavar="PROMPT", help="Run generation with the given prompt.")
    parser.add_argument("--checkpoint", type=str, default="auto", help="Checkpoint path for generation ('auto' or 'path/to/ckpt')")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum number of new tokens to generate")

    args = parser.parse_args()

    if args.train:
        train(args.config, resume=args.resume, use_wandb=args.wandb)
    elif args.generate:
        output_dir = OmegaConf.load(args.config).train.get("output_dir", "outputs")
        
        checkpoint_path = args.checkpoint
        if checkpoint_path == "auto":
            checkpoint_path = _find_latest_checkpoint(output_dir)
            if checkpoint_path is None:
                print(f"Error: Could not automatically find a checkpoint in '{output_dir}'.")
                print("Please train a model first or specify a checkpoint with --checkpoint.")
                exit(1)
        
        decode(
            config_path=args.config,
            checkpoint_path=checkpoint_path,
            prompt=args.generate,
            max_new_tokens=args.max_tokens
        )
    else:
        print("No action specified. Please use --train to start training or --generate 'Your Prompt' to generate text.")
        parser.print_help()
