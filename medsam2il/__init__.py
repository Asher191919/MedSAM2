"""MedSAM2 交互式肿瘤分割模仿学习 (Interactive Segmentation Imitation Learning)。

状态 = 当前图像、当前分割结果、历史点击
动作 = 下一次点击(正/负)、框选或确认;  目标 = 用尽可能少的交互得到高 Dice。
"""

__version__ = "0.1.0"
