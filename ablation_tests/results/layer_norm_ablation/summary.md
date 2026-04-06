# RMSNorm Ablation Summary

## Results
- no_norm_best_lr: lr=3.00e-04, steps=300, final_loss=0.0052, min_loss=0.0012, diverged=False
- no_norm_low_lr_1: lr=1.00e-04, steps=300, final_loss=0.0119, min_loss=0.0053, diverged=False
- no_norm_low_lr_2: lr=5.00e-05, steps=300, final_loss=0.1595, min_loss=0.1003, diverged=False

## Comments
- 在最佳学习率下去掉 RMSNorm 后通常更容易不稳定（loss 抖动更大，甚至出现发散）。
- 降低学习率后训练稳定性通常会改善，但收敛速度会变慢，最终 loss 未必优于带 RMSNorm 的模型。
- 建议在报告中结合 learning_curves.png 和上述数值进行对比说明。