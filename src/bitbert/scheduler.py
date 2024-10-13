import torch

def get_wsd_scheduler(
    optimizer,
    total_steps,
    warmup_frac=0.05,
    decay_frac=0.1,
    initial_lr_scale=0.1,
    final_lr_scale=0.0 # decay to 0
):
    warmup_steps = int(total_steps * warmup_frac)
    decay_steps = int(total_steps * decay_frac)
    # WSD = warmup, stable, decay
    def lr_lambda(step):
        # warmup
        if step < warmup_steps:
            # interpolate between initial_lr_scale and 1.0
            alpha = step / warmup_steps # alpha is the weight of the 1.0 term
            return alpha * 1.0 + (1.0 - alpha) * initial_lr_scale
        # stable
        elif step < total_steps - decay_steps:
            return 1.0
        elif decay_steps == 0:
            return 1.0
        else:
            steps_remaining = total_steps - step
            steps_remaining = max(steps_remaining, 0)
            # interpolate between 1.0 and final_lr_scale
            alpha = steps_remaining / decay_steps # alpha is weight of the 1.0 term
            return alpha * 1.0 + (1.0 - alpha) * final_lr_scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_one_cycle_scheduler(
    optimizer, total_steps, warmup_frac=0.05
):
    warmup_steps = total_steps * warmup_frac
    # one cycle = warmup, decay. it's just wsd with no "stable"
    return get_wsd_scheduler(optimizer, total_steps, warmup_frac, 1 - warmup_frac)

# def get_batch_size_scheduler(
#     initial_batch_size, final_batch_size, total_steps, warmup_frac=0.1
# ):
#     """
#     Returns a simple function that gives the batch size for a given step.
#     """"
class BatchSizeSchedule:
    """
    Manages increasing effective batch size, by tracking gradient accumulation steps.
    """
    def __init__(
        self,
        initial_batch_size,
        final_batch_size,
        total_steps,
        warmup_frac=0.1,
        batch_size_increment=32 # amount to increase by each step. assumed same as microbatch size
    ):
        assert  initial_batch_size % batch_size_increment == 0
        assert  final_batch_size % batch_size_increment == 0
        warmup_steps = int(total_steps * warmup_frac) # microbatch steps
        initial_accum_iters = initial_batch_size // batch_size_increment
        final_accum_iters = final_batch_size // batch_size_increment
        # Calculate average accumulation iterations (float)
        avg_accum_iters = (initial_accum_iters + final_accum_iters) / 2.0

        # Compute total optimizer steps during warmup
        warmup_optimizer_steps = warmup_steps / avg_accum_iters

        # Determine the number of accumulation levels (batch sizes)
        num_levels = final_accum_iters - initial_accum_iters + 1

        self.batch_size_increment = batch_size_increment
        self.steps_per_batch_size = max(1, int(round(warmup_optimizer_steps / num_levels)))
        self.current_accum_iters = initial_accum_iters
        self.final_accum_iters = final_accum_iters
        self.optimizer_steps_remaining_at_current_batch_size = self.steps_per_batch_size
        self.microbatch_steps_remaining_before_optimizer_step = initial_accum_iters


    def step(self):
        """
        Manage internal state, and returns True if should step the optimizer.
        """
        step_optimizer = False
        # first, figure out whether to step the optimizer
        self.microbatch_steps_remaining_before_optimizer_step -= 1
        if self.microbatch_steps_remaining_before_optimizer_step == 0:
            step_optimizer = True
            # figure out whether to increase the batch size
            self.optimizer_steps_remaining_at_current_batch_size -= 1
            if self.optimizer_steps_remaining_at_current_batch_size == 0:
                self.current_accum_iters = min(
                    self.current_accum_iters + 1, self.final_accum_iters
                )
                self.optimizer_steps_remaining_at_current_batch_size = (
                    self.steps_per_batch_size
                )
            # reset the microbatch steps remaining
            self.microbatch_steps_remaining_before_optimizer_step = (
                self.current_accum_iters
            )
        return step_optimizer

    def get_current_batch_size(self):
        return self.current_accum_iters * self.batch_size_increment

class MaskingSchedule:
    def __init__(
        self,
        total_steps,
        start_phase=0.1, # time spent at initial ratio before decaying
        end_phase=0.1, # time spent at final ratio before stopping
        initial_mask_ratio=0.30,
        final_mask_ratio=0.15
    ):
        pass
# usage:
# batch_size_scheduler = BatchSizeSchedule(initial_batch_size, final_batch_size, total_steps)
# for i, batch in enumerate(dataloader):
#     if batch_size_scheduler.step():
#         optimizer.step()
#         optimizer.zero_grad()
#     # do training
