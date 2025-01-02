"""
---
title: Utility functions for DDPM experiment
summary: >
  Utility functions for DDPM experiment
---

# Utility functions for [DDPM](index.html) experiemnt
"""
import torch.utils.data


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape
      输入：
            consts，一个张量，通常包含一些与时间步长相关的常量参数，例如 alpha 或 beta 系列参数。
            t: 一个张量，包含需要从 consts 中提取的索引。通常是时间步（timestep）的张量
    """
    c = consts.gather(-1, t)  # 根据索引 t，从 consts 的最后一个维度中提取对应值
    return c.reshape(-1, 1, 1, 1)  # 将张量扩展为 4D，自动计算批量大小 batch_size，适应神经网络的特征图feature map[batch_size, channels, height, width]进行广播操作
