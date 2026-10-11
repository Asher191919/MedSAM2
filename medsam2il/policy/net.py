"""BC 策略网络: 双分支 (原始状态 trunk + SAM2 图像嵌入分支) -> 点击热图 + 确认头。

嵌入分支是文献里交互式分割策略网络的标准做法 (SIMPLE/PCA-DION 等):
SAM2 image encoder 的语义特征远强于从头训练的 CNN, 每图只需算一次。
embed_ch=0 时退化为纯原始状态网络 (轻量替身管线), 行为不变。

结构 (改进2/3):
  - in_ch: 状态输入通道 (灰度 11 / RGB 13, 由数据决定);
  - stem -> 32x32 高分辨率特征: 高分辨率误差头与亚格子点击偏移头都吃它
    (16x16 网格 = 8px 格子, 小病灶定位天然糊; 32x32 = 4px 精度);
  - trunk -> 16x16: 点击热图 / 停手 / Dice 回归 (动作空间仍是网格单元);
  - err_head: 32x32 上预测 FN/FP 误差图 (辅助监督, 迫使特征学会定位误差);
  - offset_head: 32x32 上预测所选单元内的点击偏移 (dy,dx, 格为单位, |·|<=0.5),
    让点击精度从"格子中心"细化到"格子内 4px 级"。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from medsam2il.policy.state import (N_STATE_CH, build_state, cell_center,
                                    n_state_ch)


class InteractionPolicyNet(nn.Module):
    def __init__(self, grid: int = 16, embed_ch: int = 0, in_ch: int = N_STATE_CH):
        super().__init__()
        self.grid = grid
        self.embed_ch = embed_ch
        self.in_ch = in_ch
        self.dual_mode = False   # 双修正协议 (train_bc 按 config 回写)
        self.tau = 0.5        # 分类器停手阈值 (stop_mode=classifier 用)
        self.stop_mode = "classifier"  # dice_regression | classifier
        self.dice_theta = 0.85  # 回归停手阈值: 预测 Dice >= theta 即停 (校准后回写)
        self.apply_theta = 0.5  # 双修正: 次通道峰值 >= theta 才附加第二击 (校准后回写)
        self.fine_offset = False  # 亚格子点击偏移 (train_bc 按 config 回写)
        self._last_offset = (0.0, 0.0)   # act() 附带产出, 供执行器细化像素

        def block(cin, cout, stride=1):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.stem = nn.Sequential(
            block(in_ch, 32, 2),   # 128 -> 64
            block(32, 64, 2),      # -> 32 (高分辨率特征)
        )
        self.trunk = nn.Sequential(
            block(64, 96, 2),      # 32 -> 16
            block(96, 96),
            block(96, 64),
        )
        if embed_ch > 0:
            self.embed_branch = nn.Sequential(
                nn.Conv2d(embed_ch, 32, 1), nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True), nn.Conv2d(32, 32, 1), nn.ReLU(inplace=True))
            self.fuse = nn.Sequential(nn.Conv2d(64 + 32, 64, 1),
                                      nn.ReLU(inplace=True))
        else:
            self.embed_branch = None
            self.fuse = None
        self.click_head = nn.Conv2d(64, 2, 1)          # [B,2,g,g] 每格 pos/neg
        # 高分辨率误差头 (32x32): 训练时预测 GT 的 FN/FP 误差图 (推理时不用 GT),
        # 迫使共享特征学会"从图像-分割不一致中找误差"—— 修正点击的核心能力。
        # 32x32 分辨率让误差定位精确到 4px (旧的 16x16 头只有 8px)。
        self.err_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.Conv2d(32, 2, 1))
        # 亚格子偏移头 (32x32): 在所选网格单元内预测点击偏移 (dy,dx), 格为单位。
        # tanh*0.5 限幅 -> 偏移不超过半个格子; 让点击从"格子中心"细化到格内。
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True), nn.Conv2d(32, 2, 1))
        self.stop_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(64, 32), nn.ReLU(),
                                       nn.Linear(32, 1))
        # Dice 回归头: 直接预测"当前分割相对 GT 的 Dice"(训练时 GT 现成),
        # 停手条件 = 预测 Dice >= 目标 (实验结论: 劣于分类器, 保留开关)
        self.dice_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                       nn.Linear(64, 32), nn.ReLU(),
                                       nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, embed: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor]:
        """返回 (heat[2,g,g], stop[B], err[2,32,32], dice_logit[B], off[2,32,32])。"""
        f = self.stem(x)                                # [B,64,32,32] 高分辨率
        h = self.trunk(f)                               # [B,64,16,16]
        if h.shape[-1] != self.grid:
            h = F.interpolate(h, size=(self.grid, self.grid), mode="bilinear",
                              align_corners=False)
        if self.embed_branch is not None:
            if embed is None:
                embed = x.new_zeros(x.shape[0], self.embed_ch,
                                    self.grid, self.grid)
            h = self.fuse(torch.cat([h, self.embed_branch(embed)], dim=1))
        heat = self.click_head(h)                       # [B,2,g,g]
        err = self.err_head(f)                          # [B,2,32,32] 高分辨率误差图
        off = self.offset_head(f)                       # [B,2,32,32] 亚格子偏移
        # detach: stop 分支不得干扰 trunk 的定位特征 (否则点击热图退化)
        hd = h.detach()
        stop = self.stop_head(hd).squeeze(-1)           # [B]
        dice_logit = self.dice_head(hd).squeeze(-1)    # [B] 原始 logit
        return heat, stop, err, dice_logit, off

    @torch.no_grad()
    def act(self, image: np.ndarray, prob: np.ndarray, pos_map: np.ndarray,
            neg_map: np.ndarray, step: int, max_clicks: int, conf: float = 0.0,
            embed: np.ndarray | None = None, hyp_dis: np.ndarray | None = None,
            score_std: float = 0.0
            ) -> tuple[str, int, int, bool] | str:
        """推理接口 (与医生策略同签名, 无 GT); 返回网格单元坐标。

        停止判定: p_stop >= tau (tau 在验证集上校准) 且至少交互过一次。
        fine_offset 开启时, 附带把亚格子偏移写入 self._last_offset (格为单位),
        执行器 (_BCAdapter) 用它把"格子中心"细化到格内 4px 级。
        注意: build_state 已返回 0-1 浮点, 不要再做 unpack_state (那会再除 255)。
        """
        s = build_state(image, prob, pos_map, neg_map, step, max_clicks, conf,
                        hyp_dis, score_std)
        e = None if embed is None else torch.from_numpy(
            np.ascontiguousarray(embed.astype(np.float32))[None])
        heat, stop, _, dice_pred, off = self.forward(torch.from_numpy(s[None]), e)
        p = torch.sigmoid(heat[0])                      # [2,g,g]
        ch, cy, cx = np.unravel_index(int(p.argmax()), p.shape)
        if step >= 1 and self.stop_mode == "dice_regression" and \
                float(dice_pred[0]) >= self.dice_theta:
            self._last_offset = (0.0, 0.0)
            return "stop"
        if step >= 1 and self.stop_mode == "classifier" and \
                float(torch.sigmoid(stop[0])) >= self.tau:
            self._last_offset = (0.0, 0.0)
            return "stop"
        self._last_offset = self._offset_in_cell(off, int(cy), int(cx))
        return "click", int(cy), int(cx), bool(ch == 0)

    def _offset_in_cell(self, off: torch.Tensor, cy: int, cx: int
                        ) -> tuple[float, float]:
        """所选网格单元 (cy,cx) 内的点击偏移 (格为单位, |·| <= 0.5)。

        32x32 特征上该单元对应 r x r 块 (r = 特征分辨率 / grid), 取块均值平滑。
        fine_offset 关闭时返回 (0,0) —— 行为与旧版完全一致 (格子中心)。
        """
        if not self.fine_offset:
            return (0.0, 0.0)
        r = off.shape[-1] // self.grid
        if r < 1:
            return (0.0, 0.0)
        blk = off[0, :, cy * r:(cy + 1) * r, cx * r:(cx + 1) * r].mean(dim=(1, 2))
        return (float(torch.tanh(blk[0]) * 0.5), float(torch.tanh(blk[1]) * 0.5))

    def click_pixel(self, cy: int, cx: int, size: int) -> tuple[int, int]:
        return cell_center(cy, cx, self.grid, size)

    @torch.no_grad()
    def top_candidates(self, image: np.ndarray, prob: np.ndarray,
                       pos_map: np.ndarray, neg_map: np.ndarray, step: int,
                       max_clicks: int, conf: float = 0.0,
                       embed: np.ndarray | None = None,
                       hyp_dis: np.ndarray | None = None,
                       score_std: float = 0.0, k: int = 3, sep: int = 2
                       ) -> list[tuple[int, int, bool, float]]:
        """热图峰值前 k 个候选点击 [(cy, cx, positive, score)], 空间非极大抑制。

        测试时候选搜索用: 候选交给分割器逐个试错, 用 predicted-IoU 选优。
        sep: 候选间的最小网格距离 (切比雪夫), 保证候选落在不同误差区域。
        """
        s = build_state(image, prob, pos_map, neg_map, step, max_clicks, conf,
                        hyp_dis, score_std)
        e = None if embed is None else torch.from_numpy(
            np.ascontiguousarray(embed.astype(np.float32))[None])
        heat = self.forward(torch.from_numpy(s[None]), e)[0]
        p = torch.sigmoid(heat[0]).reshape(-1).cpu().numpy()
        order = np.argsort(-p)
        picked: list[tuple[int, int, bool, float]] = []
        for idx in order:
            ch, v = int(idx) // (self.grid * self.grid), int(idx) % (self.grid * self.grid)
            cy, cx = v // self.grid, v % self.grid
            if any(max(abs(cy - q[0]), abs(cx - q[1])) <= sep for q in picked):
                continue
            picked.append((cy, cx, bool(ch == 0), float(p[idx])))
            if len(picked) >= k:
                break
        return picked


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_net(cfg, embed_ch: int = 0, image: np.ndarray | None = None
             ) -> InteractionPolicyNet:
    """按配置构造策略网络 (输入通道随图像灰度/彩色自适应)。"""
    return InteractionPolicyNet(
        grid=int(cfg.raw["data"]["grid"]), embed_ch=embed_ch,
        in_ch=n_state_ch(image))
