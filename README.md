# tert
ternary bert. actually mostly focused on the baseline model now before making the codebase work for efficient ternary training. have implemented the following:
- masking schedule
- batch size schedule
- one-cycle and trapezoidal LR schedules
- flex-attention "single-batch" training

## TODO
- support for better optimizers (sf adamw, muon, psgd)
- int8/fp8 mixed precision training
- better eval, eval during training
- cut cross-entropy (when apple fixes it...)
