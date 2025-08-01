# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def is_liger_kernel_enabled(job_config: JobConfig) -> bool:
    """Check if Liger-Kernel fused linear cross entropy is enabled"""
    return job_config.liger_kernel.enable_fused_linear_cross_entropy


@torch.compiler.disable
def liger_fused_linear_cross_entropy_loss(
    lin_weight: torch.Tensor, 
    hidden_states: torch.Tensor, 
    labels: torch.Tensor
) -> torch.Tensor:
    """Liger-Kernel fused linear cross entropy loss function"""
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except ImportError:
        raise ImportError(
            "liger-kernel is not installed. Please install it with: pip install liger-kernel"
        )
    
    # Reshape hidden_states to 2D (batch_size * seq_len, hidden_dim) if needed
    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
    
    # Flatten labels to 1D if needed
    labels = labels.flatten()
    
    loss_fn = LigerFusedLinearCrossEntropyLoss(reduction="mean", return_z_loss=False)
    return loss_fn(lin_weight, hidden_states, labels)


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


def build_cross_entropy_loss(job_config: JobConfig):
    loss_fn = cross_entropy_loss
    if job_config.training.compile:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def build_liger_fused_loss(job_config: JobConfig):
    """Build Liger-Kernel fused linear cross entropy loss or fallback to standard"""
    if is_liger_kernel_enabled(job_config):
        logger.info("Using Liger-Kernel fused linear cross entropy loss")
        
        def fused_loss_wrapper(pred_or_loss: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # If pred_or_loss is already a scalar loss (from fused computation), return it
            if pred_or_loss.dim() == 0:
                return pred_or_loss
            # Otherwise, apply standard cross entropy (fallback case)
            return cross_entropy_loss(pred_or_loss, labels)
        
        loss_fn = fused_loss_wrapper
        # Note: Don't compile the fused loss wrapper since Liger-Kernel has torch.compile compatibility issues
        if job_config.training.compile:
            logger.info("Liger-Kernel fused loss is not compiled due to torch.compile compatibility issues")
        return loss_fn
    else:
        return build_cross_entropy_loss(job_config)


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """

    @functools.wraps(unwrapped_loss_fn)
    def accumulated_loss_fn(*args, **kwargs):
        loss = unwrapped_loss_fn(*args, **kwargs)
        return loss / accumulation_steps

    return accumulated_loss_fn
