import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from components.data_basic import get_batch, load_memmap_dataset
from components.nn_basic import Cross_Entropy, TransformerLM
from components.optim import AdamW, get_lr_cosine_schedule, gradient_clipping


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def ensure_train_bin(config) -> None:
	bin_path = config.train.bin_path
	if os.path.exists(bin_path):
		return

	os.makedirs(os.path.dirname(bin_path), exist_ok=True)
	tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
	read_bytes = int(config.train.get("read_bytes", 1024 * 1024 * 100))
	with open(config.train.txt_path, "r", encoding="utf-8") as f:
		text = f.read(read_bytes)
	ids = tokenizer.encode(text, add_special_tokens=False)
	np.array(ids, dtype=np.uint16).tofile(bin_path)


def _is_diverged(loss_value: float) -> bool:
	return (not math.isfinite(loss_value)) or loss_value > 20.0


def run_one_experiment(
	config_path: str,
	lr: float,
	use_norm: bool,
	max_steps: int,
	seed: int,
	output_root: Path,
	run_name: str,
) -> dict:
	set_seed(seed)
	config = OmegaConf.load(config_path)
	config.optim.lr = float(lr)
	config.model.use_norm = bool(use_norm)
	config.train.max_steps = int(max_steps)

	ensure_train_bin(config)

	device = config.model.device
	tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
	config.model.vocab_size = tokenizer.vocab_size

	model = TransformerLM(**OmegaConf.to_container(config.model))
	model.to(device)
	model.train()

	optimizer = AdamW(model.parameters(), **OmegaConf.to_container(config.optim))
	dataset = load_memmap_dataset(config.train.bin_path)

	alpha_max = float(config.optim.lr)
	alpha_min = alpha_max * 0.1
	tw = int(config.train.get("warmup_steps", 10))
	tc = int(config.train.max_steps)
	max_norm = float(config.train.max_norm)

	run_dir = output_root / run_name
	run_dir.mkdir(parents=True, exist_ok=True)
	csv_path = run_dir / "metrics.csv"

	rows = []
	diverged = False
	for step in range(tc):
		current_lr = get_lr_cosine_schedule(step, alpha_max, alpha_min, tw, tc)
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr

		x, y = get_batch(
			dataset=dataset,
			batch_size=int(config.train.batch_size),
			context_length=int(config.model.context_length),
			device=device,
		)

		logits = model(x)
		logits_flat = logits.view(-1, int(config.model.vocab_size))
		y_flat = y.view(-1)

		loss = Cross_Entropy(logits_flat, y_flat)
		optimizer.zero_grad()
		loss.backward()
		grad_norm = gradient_clipping(model.parameters(), max_norm)
		optimizer.step()

		loss_value = float(loss.item())
		row = {
			"step": step,
			"loss": loss_value,
			"lr": float(current_lr),
			"grad_norm": float(grad_norm),
			"use_norm": int(use_norm),
			"base_lr": float(lr),
		}
		rows.append(row)

		print(
			f"[{run_name}] step={step:04d} loss={loss_value:.6f} "
			f"lr={current_lr:.2e} grad_norm={grad_norm:.4f}"
		)

		if _is_diverged(loss_value):
			diverged = True
			print(f"[{run_name}] divergence detected, early stop at step {step}.")
			break

	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)

	final_loss = rows[-1]["loss"]
	min_loss = min(r["loss"] for r in rows)
	return {
		"run_name": run_name,
		"csv_path": str(csv_path),
		"use_norm": use_norm,
		"lr": float(lr),
		"steps": len(rows),
		"diverged": diverged,
		"final_loss": float(final_loss),
		"min_loss": float(min_loss),
	}


def plot_runs(results: list[dict], output_root: Path) -> str | None:
	try:
		import matplotlib.pyplot as plt
	except Exception:
		print("matplotlib not installed; skip plotting. You can still use CSV for curve plotting.")
		return None

	plt.figure(figsize=(10, 6))
	for result in results:
		data = np.genfromtxt(result["csv_path"], delimiter=",", names=True)
		if data.size == 0:
			continue
		if data.shape == ():
			steps = np.array([int(data["step"])])
			losses = np.array([float(data["loss"])])
		else:
			steps = data["step"]
			losses = data["loss"]
		label = (
			f"{result['run_name']} (lr={result['lr']:.1e}, "
			f"diverged={result['diverged']})"
		)
		plt.plot(steps, losses, label=label)

	plt.title("RMSNorm Ablation Learning Curves")
	plt.xlabel("Step")
	plt.ylabel("Train Loss")
	plt.grid(alpha=0.3)
	plt.legend()
	out_png = output_root / "learning_curves.png"
	plt.tight_layout()
	plt.savefig(out_png, dpi=150)
	plt.close()
	return str(out_png)


def write_summary(results: list[dict], output_root: Path) -> str:
	lines = [
		"# RMSNorm Ablation Summary",
		"",
		"## Results",
	]
	for r in results:
		lines.append(
			f"- {r['run_name']}: lr={r['lr']:.2e}, steps={r['steps']}, "
			f"final_loss={r['final_loss']:.4f}, min_loss={r['min_loss']:.4f}, "
			f"diverged={r['diverged']}"
		)

	best_lr_run = next((r for r in results if r["run_name"] == "no_norm_best_lr"), None)
	low_lr_run = next((r for r in results if r["run_name"].startswith("no_norm_low_lr")), None)
	if best_lr_run and low_lr_run:
		lines.extend(
			[
				"",
				"## Comments",
				(
					"- 在最佳学习率下去掉 RMSNorm 后通常更容易不稳定（loss 抖动更大，"
					"甚至出现发散）。"
				),
				(
					"- 降低学习率后训练稳定性通常会改善，但收敛速度会变慢，"
					"最终 loss 未必优于带 RMSNorm 的模型。"
				),
				"- 建议在报告中结合 learning_curves.png 和上述数值进行对比说明。",
			]
		)

	out_md = output_root / "summary.md"
	out_md.write_text("\n".join(lines), encoding="utf-8")
	return str(out_md)


def main() -> None:
	parser = argparse.ArgumentParser(description="RMSNorm ablation experiments")
	parser.add_argument("--config", default="components/config.yaml", help="Path to config file")
	parser.add_argument(
		"--best_lr",
		type=float,
		default=None,
		help="Best LR from your previous baseline. Default: use config.optim.lr",
	)
	parser.add_argument(
		"--low_lrs",
		type=float,
		nargs="+",
		default=None,
		help="Lower LRs to test after removing RMSNorm, e.g. --low_lrs 1e-4 5e-5",
	)
	parser.add_argument("--steps", type=int, default=None, help="Override max training steps")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output_dir", default="ablation_tests/results/layer_norm_ablation")
	args = parser.parse_args()

	config = OmegaConf.load(args.config)
	best_lr = float(config.optim.lr if args.best_lr is None else args.best_lr)
	low_lrs = args.low_lrs if args.low_lrs is not None else [best_lr * 0.5, best_lr * 0.25]
	max_steps = int(config.train.max_steps if args.steps is None else args.steps)

	output_root = Path(args.output_dir)
	output_root.mkdir(parents=True, exist_ok=True)

	print("=" * 80)
	print("LayerNorm Ablation: remove RMSNorm and compare learning rates")
	print(f"Config: {args.config}")
	print(f"Best LR: {best_lr}")
	print(f"Lower LRs: {low_lrs}")
	print(f"Max steps: {max_steps}")
	print("=" * 80)

	results = []
	results.append(
		run_one_experiment(
			config_path=args.config,
			lr=best_lr,
			use_norm=False,
			max_steps=max_steps,
			seed=args.seed,
			output_root=output_root,
			run_name="no_norm_best_lr",
		)
	)

	for i, lr in enumerate(low_lrs, start=1):
		results.append(
			run_one_experiment(
				config_path=args.config,
				lr=float(lr),
				use_norm=False,
				max_steps=max_steps,
				seed=args.seed + i,
				output_root=output_root,
				run_name=f"no_norm_low_lr_{i}",
			)
		)

	curve_path = plot_runs(results, output_root)
	summary_path = write_summary(results, output_root)

	print("\nExperiment finished.")
	for r in results:
		print(
			f"- {r['run_name']}: final_loss={r['final_loss']:.4f}, "
			f"diverged={r['diverged']}, csv={r['csv_path']}"
		)
	if curve_path:
		print(f"- learning curves: {curve_path}")
	print(f"- summary: {summary_path}")


if __name__ == "__main__":
	main()
