import math
import warnings

import torch


class ZClip:
    """
    ZClip: Adaptive Spike Mitigation for LLM Pre-Training

    Dynamically adjusts gradient clipping thresholds using an EMA of the gradient norm's z-score.
    Returns both the final gradient norm tensor and a dictionary of logging statistics.
    """

    def __init__(self, alpha=0.97, z_thres=2.5, warmup_steps=25, eps=1e-6):
        self.alpha = alpha
        self.z_thres = z_thres
        self.warmup_steps = warmup_steps
        self.eps = eps

        self.step_count = 0
        self.warmup_norms = []

        # EMA statistics
        self.mu = 0.0
        self.v = 0.0
        self.is_warmed_up = False

    def __call__(self, accelerator, parameters):
        """
        Applies ZClip and returns (final_norm, logs).
        """
        if isinstance(parameters, torch.Tensor):
            params_list = [parameters]
        else:
            params_list = list(parameters)

        if len(params_list) == 0:
            return torch.tensor(0.0), {}

        # Compute the current gradient norm WITHOUT clipping
        g_t_tensor = accelerator.clip_grad_norm_(params_list, max_norm=float("inf"))

        if g_t_tensor is None:
            warnings.warn("ZClip received None for the gradient norm. Bypassing ZClip.")
            return None, {}

        g_t = g_t_tensor.item()

        # Initialize default logs for this step
        logs = {
            "zclip/raw_norm": g_t,
            "zclip/clipped_norm": g_t,
            "zclip/mu": self.mu,
            "zclip/sigma": math.sqrt(self.v) if self.v > 0 else 0.0,
            "zclip/z_score": 0.0,
            "zclip/is_spike": False,
            "zclip/warmed_up": self.is_warmed_up,
        }

        # Gracefully handle invalid gradients (NaN/Inf)
        if math.isnan(g_t) or math.isinf(g_t):
            return g_t_tensor, logs

        self.step_count += 1

        # Warm-up Phase
        if not self.is_warmed_up:
            self.warmup_norms.append(g_t)
            if self.step_count == self.warmup_steps:
                self.mu = sum(self.warmup_norms) / self.warmup_steps
                self.v = sum((x - self.mu) ** 2 for x in self.warmup_norms) / self.warmup_steps
                self.is_warmed_up = True
                del self.warmup_norms  # Free memory

                # Update logs for the exact step warmup finishes
                logs["zclip/mu"] = self.mu
                logs["zclip/sigma"] = math.sqrt(self.v)
                logs["zclip/warmed_up"] = True

            return g_t_tensor, logs

        # Z-score based Spike Detection
        std_dev = math.sqrt(self.v)
        z_t = (g_t - self.mu) / (std_dev + self.eps)

        logs["zclip/z_score"] = z_t

        g_t_star = g_t
        is_spike = z_t > self.z_thres
        logs["zclip/is_spike"] = is_spike

        if is_spike:
            # Gradient Adjustment (Reciprocal Clipping)
            g_t_star = self.mu + ((self.z_thres**2) / z_t) * std_dev
            logs["zclip/clipped_norm"] = g_t_star

        # Apply the clipping scaling to gradients if a spike was detected
        if g_t_star < g_t:
            clip_coef = g_t_star / (g_t + 1e-6)
            clip_coef_tensor = torch.tensor(clip_coef, device=g_t_tensor.device)

            for p in params_list:
                if p.grad is not None:
                    p.grad.detach().mul_(clip_coef_tensor)

        # Update EMA Statistics
        self.mu = self.alpha * self.mu + (1 - self.alpha) * g_t_star
        self.v = self.alpha * self.v + (1 - self.alpha) * ((g_t_star - self.mu) ** 2)

        final_norm_tensor = torch.tensor(g_t_star, device=g_t_tensor.device) if g_t_star < g_t else g_t_tensor

        return final_norm_tensor, logs
