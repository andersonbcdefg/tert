import torch


def get_wsd_scheduler(
  optimizer,
  total_steps,
  warmup_frac=0.05,
  decay_frac=0.1,
  final_frac=0.0,
  initial_lr_scale=0.1,
  final_lr_scale=0.0,  # decay to 0
):
  warmup_steps = int(total_steps * warmup_frac)
  decay_steps = int(total_steps * decay_frac)
  final_steps = int(total_steps * final_frac)
  stable_start = warmup_steps
  stable_end = total_steps - decay_steps - final_steps

  def lr_lambda(step):
    # Warmup phase: linear increase from initial_lr_scale to 1.0
    if step < warmup_steps:
      return initial_lr_scale + (1.0 - initial_lr_scale) * (step / warmup_steps)

    # Stable phase: constant learning rate of 1.0
    if step < stable_end:
      return 1.0

    # Skip decay if decay_steps is 0
    if decay_steps == 0:
      return 1.0

    # Decay phase: linear decrease from 1.0 to final_lr_scale
    if step < total_steps - final_steps:
      steps_into_decay = step - stable_end
      decay_progress = steps_into_decay / decay_steps
      return 1.0 + (final_lr_scale - 1.0) * decay_progress

    # Final phase: constant final learning rate
    return final_lr_scale

  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_one_cycle_scheduler(optimizer, total_steps, warmup_frac=0.05):
  warmup_steps = total_steps * warmup_frac
  # one cycle = warmup, decay. it's just wsd with no "stable"
  return get_wsd_scheduler(optimizer, total_steps, warmup_frac, 1 - warmup_frac)


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
    batch_size_increment=32,  # amount to increase by each step. assumed same as microbatch size
  ):
    assert initial_batch_size % batch_size_increment == 0
    assert final_batch_size % batch_size_increment == 0
    warmup_steps = int(total_steps * warmup_frac)  # microbatch steps
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
        self.optimizer_steps_remaining_at_current_batch_size = self.steps_per_batch_size
      # reset the microbatch steps remaining
      self.microbatch_steps_remaining_before_optimizer_step = self.current_accum_iters
    return step_optimizer

  def get_current_batch_size(self):
    return self.current_accum_iters * self.batch_size_increment


class MaskingScheduleCollator:
  def __init__(
    self,
    total_steps,
    tokenizer,
    max_length: int,
    start_phase=0.1,  # time spent at initial ratio before decaying
    end_phase=0.1,  # time spent at final ratio before stopping
    initial_mask_ratio=0.30,
    final_mask_ratio=0.15,
  ):
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.total_steps = total_steps
    self.start_steps = int(start_phase * total_steps)
    self.end_steps = int(end_phase * total_steps)
    self.initial_mask_ratio = initial_mask_ratio
    self.final_mask_ratio = final_mask_ratio
    self.step = 0

  def __call__(self, batch):
    self.step += 1
    # time spent at initial ratio before decaying
    if self.step <= self.start_steps:
      mask_prob = self.initial_mask_ratio
    # time spent at final ratio before stopping
    elif self.step > self.total_steps - self.end_steps:
      mask_prob = self.final_mask_ratio
    # otherwise interpolate between initial and final
    else:
      numerator = self.step - self.start_steps
      denominator = self.total_steps - self.start_steps - self.end_steps
      alpha = numerator / denominator  # weight of the final_mask_ratio
      mask_prob = (
        alpha * self.final_mask_ratio + (1.0 - alpha) * self.initial_mask_ratio
      )

    tokenized = self.tokenizer.encode_batch(
      [x["text"] for x in batch],
      max_length=self.max_length,
      collate_strategy="longest",
    )
    tokenized["input_ids"] = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    tokenized["attention_mask"] = torch.tensor(
      tokenized["attention_mask"], dtype=torch.long
    )

    # Randomly mask 15% of tokens
    masked_indices = torch.bernoulli(
      torch.full(tokenized["input_ids"].shape, mask_prob)
    ).bool()

    # do not allow padding tokens to be masked
    # use the attention_mask to zero those positions
    masked_indices = tokenized["attention_mask"].bool() & masked_indices

    # Create labels tensor
    labels = torch.full(tokenized["input_ids"].shape, -100, dtype=torch.long)
    labels[masked_indices] = tokenized["input_ids"][masked_indices]

    # Apply masking to input_ids
    tokenized["input_ids"][masked_indices] = self.tokenizer.mask_id
    tokenized["labels"] = labels

    return tokenized
