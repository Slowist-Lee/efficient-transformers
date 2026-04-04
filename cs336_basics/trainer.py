from .nn_basic import *
from .data_basic import *
from .optim import *
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import os
import torch

torch.autograd.set_detect_anomaly(True)

def train(config_path):
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
        with open(config.train.txt_path, "r", encoding="utf-8") as f:
            text = f.read(1024 * 2) # 直接读100MB文本
            ids = tokenizer.encode(text, add_special_tokens=False)
            # 一次性转为 uint16 并存盘
            np.array(ids, dtype=np.uint16).tofile(config.train.bin_path)
    # ** 解包传入
    model = TransformerLM(**OmegaConf.to_container(config.model))
    print(model.parameters())
    
    model.to(device)
    # 设置为训练模式
    model.train()
    optimizer=AdamW(model.parameters(),**OmegaConf.to_container(config.optim))
    max_norm=config.train.max_norm
    dataset=load_memmap_dataset(config.train.bin_path)
    max_steps=config.train.max_steps
    
    # 在进入 for 循环前，设定好你的参数
    alpha_max = config.optim.lr # 例如 5e-4
    alpha_min = alpha_max * 0.1            # 退火到十分之一，比如 5e-5
    tc = config.train.max_steps            # 总步数
    tw = 10 # 预热步数，建议设为总步数的 2% ~ 5% (根据总步数自己定)
    
    print("Begin Training...")
    for step in range(tc):
        current_lr = get_lr_cosine_schedule(step, alpha_max, alpha_min, tw, tc)
        for param_group in optimizer.param_groups:
            # 你需要确认你的自定义 AdamW 是用 "lr" 还是 "alpha"
            param_group["lr"] = current_lr 
        x, y = get_batch(
            dataset=dataset,
            batch_size=config.train.batch_size,
            context_length=config.model.context_length,
            device=device
        )
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"权重参数 {name} 已损坏 (NaN)")
                break
        logits=model(x)
        print(logits)
        logits_flat = logits.view(-1, config.model.vocab_size)
        print(logits_flat.shape)
        y_flat=y.view(-1)
        print(y_flat.shape)

        loss=Cross_Entropy(logits_flat,y_flat)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(),max_norm)
        # if step % 10 == 0:
        print(loss)
        print(f"Epoch: {step}, Loss: {loss.item():.4f}")
        # 更新参数
        optimizer.step()

if __name__=="__main__":
    train('./cs336_basics/config.yaml')