# compute_tracker.py
import time
from dataclasses import dataclass

@dataclass
class ModelProfile:
    # params in units of parameters (not bytes)
    dense_params: float                    # e.g., 70e9
    moe_expert_params: float = 0.0         # sum over all experts
    experts_total: int = 0                 # total experts
    experts_per_token: int = 0             # active experts per token (top-k)
    extra_fw_passes: float = 0.0           # recompute overhead (e.g., 1.0 means one extra FW)
    optimizer_overhead: float = 0.10       # 5–15% is a common add-on for opt/emb/softmax

@dataclass
class HardwareProfile:
    n_gpus: int
    peak_flops_per_gpu: float              # FLOPs/sec for your precision (e.g., 1.51e15 for H800 FP8)
    utilization_target: float = 0.30       # for planning; not used in accounting

class ComputeTracker:
    """
    Tracks tokens, FLOPs, GPU-hours, and MFU for dense or MoE models.
    Call .update(tokens_in_batch, phase='pretrain'/'sft'/'rl_sample'/'rl_update')
    once per optimizer step (or per microstep if you prefer).
    """
    def __init__(self, model: ModelProfile, hw: HardwareProfile):
        self.m = model
        self.hw = hw
        self.start = time.time()
        self.tokens = 0
        self.flops = 0.0
        self.phase_flops = {}
        self.steps = 0

    def _active_params(self):
        if self.m.experts_total > 0 and self.m.experts_per_token > 0:
            f_active = self.m.experts_per_token / self.m.experts_total
            return self.m.dense_params + f_active * self.m.moe_expert_params
        return self.m.dense_params

    def _fw_bw_multiplier(self, phase):
        # Dense math per token: ~2N (FW) + ~4N (BW) = ~6N
        # RL rollout (sampling): ~2N only (no backprop)
        if phase == 'rl_sample':
            return 2.0 + 2.0 * self.m.extra_fw_passes  # recompute affects FW too
        # training phases with backprop:
        base = 6.0
        # gradient checkpointing / recompute -> extra forward passes:
        base += 2.0 * self.m.extra_fw_passes
        return base

    def estimate_flops(self, tokens, phase):
        N_active = self._active_params()
        mult = self._fw_bw_multiplier(phase)
        fl = mult * N_active * tokens
        fl *= (1.0 + self.m.optimizer_overhead)  # small correction for opt/softmax/emb
        return fl

    def update(self, tokens_in_batch: int, phase: str = 'pretrain'):
        est = self.estimate_flops(tokens_in_batch, phase)
        self.tokens += tokens_in_batch
        self.flops += est
        self.phase_flops[phase] = self.phase_flops.get(phase, 0.0) + est
        self.steps += 1
        return est

    def metrics(self):
        elapsed_s = max(1e-9, time.time() - self.start)
        gpu_seconds = elapsed_s * self.hw.n_gpus
        gpu_hours = gpu_seconds / 3600.0
        peak_total = self.hw.n_gpus * self.hw.peak_flops_per_gpu
        mfu = min(1.0, (self.flops / elapsed_s) / peak_total)  # implied MFU from our accounting
        return {
            "tokens_total": int(self.tokens),
            "flops_total": self.flops,
            "gpu_hours_elapsed": gpu_hours,
            "implied_mfu": mfu,
            "phase_breakdown_flops": {k: v for k, v in self.phase_flops.items()},
            "steps": self.steps,
        }
