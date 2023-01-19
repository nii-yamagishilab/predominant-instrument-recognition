import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

# restart gamma 0.7 is too large

def cosine_warmup_restart_exponential_decay(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 1,
        gamma: float = 0.9,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        gamma:
            exponential decaying factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        cur_gam = max(gamma ** ((float(num_cycles) * progress) // 1.0), 8e-7)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))) * cur_gam
        # return max(0.001, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))) * cur_gam)
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def exponential_decay_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        step_size: int,
        gamma: float = 0.99,
        last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        cur = (current_step - num_warmup_steps)
        return max(0.0001, gamma ** (cur//step_size + cur % step_size / step_size))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def step_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        cur = (current_step - num_warmup_steps)
        if cur > (3 * step_size):
            return gamma ** 3
        if cur > (2 * step_size):
            return gamma ** 2
        if cur > step_size:
            return gamma
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)