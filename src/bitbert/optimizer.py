import torch.optim
from torchao.prototype.low_bit_optim import AdamW8bit, AdamW4bit
from heavyball.utils import set_torch
import heavyball.utils
heavyball.utils.compile_mode = "reduce-overhead"
set_torch()
import heavyball
from muon import Muon


OPTIMIZERS = {
    "torch_adamw": torch.optim.AdamW,
    "heavyball_adamw": heavyball.AdamW,
    "adamw_8bit": AdamW8bit,
    "adamw_4bit": AdamW4bit,
    "sf_adamw": heavyball.SFAdamW,
    "psgd": heavyball.PSGDKron,
    "muon": Muon,
}

def get_optimizer(model, optimizer_name: str, initial_lr: float):
    if optimizer_name == "muon":
        # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
        muon_param_names = set([
            n for n, p in model.named_parameters()
            if p.ndim >= 2 and not "tok_embeddings" in n
            and not "norm" in n
            and not "output" in n
        ])
        muon_params = [
            p for n, p in model.named_parameters()
            if n in muon_param_names
        ]
        adamw_params = [p for n, p in model.named_parameters()
                        if n not in muon_param_names]
        adamw_params.extend(model.tok_embeddings.parameters())
        # Find everything else -- these will be optimized by AdamW
        optimizer = Muon(
            muon_params,
            lr=0.02,
            momentum=0.95,
            adamw_params=adamw_params,
            adamw_lr=initial_lr,
            adamw_betas=(0.90, 0.95),
            adamw_wd=0.01
        )

        # find keys missing from the optimizer state
        for n, p in model.named_parameters():
            if p not in optimizer.state_dict():
                raise ValueError(f"Missing parameter {n} in optimizer state")

        return optimizer

    return OPTIMIZERS[optimizer_name](model.parameters(), lr=initial_lr)
