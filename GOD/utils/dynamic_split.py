"""Dynamic & Adaptive Layer Splitting for GOD  (completed version).

Implements the improvement proposed in Paper/1 (Improvement4GOD) and
Paper/3 (GOD_Short_4_AI): the static split k=9 (9 shallow Shared-LoRA blocks,
3 deep Task-specific blocks) fails under large domain shift (SAR, medical
X-ray, hyperspectral).

This file provides the two levels of the mechanism:

  Level 1 (Macro): a cheap warm-up profiler that measures a *normalized*,
      scale-free domain-shift signal per Transformer block using the REAL
      GOD loss (ETF + ArcFace), then maps it to a per-task split point k_t
      with an explicit efficiency prior (bias toward large k).

  Level 2 (Micro): per-layer soft gate mixing Shared vs Task-specific LoRA,
      Output_l = alpha_l * Shared_LoRA_l(x) + beta_l * TaskSpecific_LoRA_l,t(x),
      with alpha_l + beta_l = 1.  (Stored PER TASK by the backbone to avoid
      interfering with the computation of previously-learned tasks.)

Design note (why this differs from the naive proposal):
  * GOD's efficient 2-mode inference caches a single shared feature SL_x at a
    GLOBAL depth.  A naive per-task k breaks that cache.  We therefore keep a
    global coarse depth K_coarse (= the paper's k=9) for the shared trunk and
    only let k_t move DOWN to a floor K_floor for domain-shifted tasks; the
    backbone caches the feature at BOTH depths so shifted tasks can be
    re-encoded correctly while natural tasks keep the exact fast path.
  * The old helpers `compute_layer_grad_variance` / `decide_k_from_variance`
    are kept as thin backward-compatible aliases.

Reference: Paper/1. Improvement4GOD.pdf, Paper/3. GOD_Short_4_AI.pdf
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Level 2 — Micro gating
# ---------------------------------------------------------------------------
class LayerGate(nn.Module):
    """Per-layer soft router between Shared LoRA and Task-specific LoRA.

    logits: [2] unnormalized scores -> softmax -> (alpha, beta), alpha+beta=1.
    alpha = weight of Shared LoRA, beta = weight of Task-specific LoRA.

    `beta_min` provides a floor on the task-specific weight so the task adapter
    always receives gradient (prevents the beta->0 "dead branch" failure mode).
    """

    def __init__(self, init_bias_shared: float = 0.5, beta_min: float = 0.0):
        super().__init__()
        p = max(min(init_bias_shared, 0.99), 0.01)
        self.logits = nn.Parameter(
            torch.tensor([math.log(p), math.log(1.0 - p)], dtype=torch.float32)
        )
        self.beta_min = float(beta_min)

    def weights(self):
        w = F.softmax(self.logits, dim=0)
        alpha, beta = w[0], w[1]
        if self.beta_min > 0.0:
            # reserve beta_min for the task branch, renormalize the rest
            beta = self.beta_min + (1.0 - self.beta_min) * beta
            alpha = 1.0 - beta
        return alpha, beta

    def forward(self, shared_out: torch.Tensor, task_out: torch.Tensor):
        alpha, beta = self.weights()
        return alpha * shared_out + beta * task_out


# ---------------------------------------------------------------------------
# Level 1 — Macro: normalized domain-shift profiling + split decision
# ---------------------------------------------------------------------------
def profile_layer_shift(
    network,
    warm_loader,
    device,
    cur_task,
    known_classes,
    warmup_batches=10,
    use_arcface=True,
):
    """Measure a normalized per-block domain-shift signal on the new task.

    Unlike the earlier proxy (mean squared feature norm), this uses the REAL
    GOD objective (ETF cross-entropy, optionally + ArcFace) so that the
    gradients reflect how much each block must move to fit the new task.

    For each Transformer block l we accumulate an approximate Fisher score
        F_l = E[ mean( g^2 ) ]   over q/k/v, A/B of the block's NEWEST adapter,
    where g are gradients of the real loss.  We then return the per-block
    scores; the caller normalizes & maps them to k_t via `decide_split_k`.

    Returns:
        fisher_per_block: list[float] length n_blocks (ViT-B/16 -> 12).
    """
    backbone = network.backbone
    was_training = backbone.training
    backbone.train()

    n_blocks = len(backbone.blocks)
    acc = [0.0] * n_blocks
    counts = [0] * n_blocks

    for batch_idx, (_, inputs, targets) in enumerate(warm_loader):
        if batch_idx >= warmup_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        # task-local labels (heads are per-task of width init_cls)
        fake_targets = targets - known_classes if cur_task > 0 else targets

        network.zero_grad(set_to_none=True)
        # forward through the full network on the current task head
        outputs = network(inputs, cur_task, Train=True)
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, fake_targets)
        if use_arcface and "feature" in outputs:
            try:
                class_logits = network.simple_fc(outputs["feature"], fake_targets)["logits"]
                loss = loss + F.cross_entropy(class_logits, fake_targets)
            except Exception:
                pass
        loss.backward()

        for li, blk in enumerate(backbone.blocks):
            gsum, gcnt = 0.0, 0
            for proj in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj):
                if len(proj.lora_As) == 0:
                    continue
                for mat in (proj.lora_As[-1], proj.lora_Bs[-1]):
                    if mat.grad is not None:
                        gsum += mat.grad.detach().pow(2).mean().item()  # Fisher ~ E[g^2]
                        gcnt += 1
            if gcnt > 0:
                acc[li] += gsum / gcnt
                counts[li] += 1

    network.zero_grad(set_to_none=True)
    if not was_training:
        backbone.eval()
    return [a / max(c, 1) for a, c in zip(acc, counts)]


def decide_split_k(
    fisher_per_block,
    K_coarse=9,
    K_floor=6,
    k_min=6,
    tau=1.15,
    gamma=6.0,
    efficiency_lambda=0.0,
):
    """Map a per-block Fisher profile to a per-task split point k_t.

    Steps:
      1. Normalize to a scale-free profile:  shift_l = F_l / mean_l(F_l).
      2. Shallow-shift score s = mean over shallow blocks (l < K_coarse) of shift_l.
         (A large s means the frozen shallow filters are under heavy pressure
         to change -> domain shift -> we should open more shallow blocks, i.e.
         lower k.)
      3. Smooth, efficiency-biased mapping:
             drop = sigmoid( gamma * (s - tau) )          in [0,1]
             k_t  = round( K_coarse - (K_coarse - k_min) * drop ) - efficiency bias
         with tau set high so k stays at K_coarse unless shift is clear.
      4. Clamp to [max(k_min, K_floor), K_coarse].

    Returns:
        k_t (int), info (dict) for logging.
    """
    n = len(fisher_per_block)
    if n == 0:
        return K_coarse, {"shallow_shift": 0.0, "k": K_coarse}
    mean_f = sum(fisher_per_block) / n + 1e-12
    shift = [f / mean_f for f in fisher_per_block]
    upper_shallow = min(K_coarse, n)
    shallow = sum(shift[:upper_shallow]) / max(upper_shallow, 1)

    drop = 1.0 / (1.0 + math.exp(-gamma * (shallow - tau)))
    k_float = K_coarse - (K_coarse - k_min) * drop
    # optional explicit efficiency prior: push k back up
    k_float = k_float + efficiency_lambda * (K_coarse - k_float)
    lo = max(int(k_min), int(K_floor))
    k = int(round(k_float))
    k = max(lo, min(int(K_coarse), k))

    info = {
        "shallow_shift": shallow,
        "drop": drop,
        "k": k,
        "k_float": k_float,
    }
    return k, info


# ---------------------------------------------------------------------------
# Backward-compatible aliases (old API)
# ---------------------------------------------------------------------------
def compute_layer_grad_variance(backbone, train_loader, device, warmup_batches=20):
    """Deprecated proxy probe (label-free mean|grad| of newest adapter).

    Kept only for backward compatibility; prefer `profile_layer_shift`, which
    uses the real GOD loss and returns a Fisher (E[g^2]) signal.
    """
    backbone.train()
    n_blocks = len(backbone.blocks)
    acc = [0.0] * n_blocks
    counts = [0] * n_blocks
    for batch_idx, (_, inputs, targets) in enumerate(train_loader):
        if batch_idx >= warmup_batches:
            break
        inputs = inputs.to(device)
        backbone.zero_grad(set_to_none=True)
        out, _ = backbone(inputs, [backbone.cur_loraid])
        if isinstance(out, tuple):
            out = out[0]
        loss = (out.float() ** 2).mean()
        loss.backward()
        for li, blk in enumerate(backbone.blocks):
            gsum, gcnt = 0.0, 0
            for proj in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj):
                if len(proj.lora_As) == 0:
                    continue
                for mat in (proj.lora_As[-1], proj.lora_Bs[-1]):
                    if mat.grad is not None:
                        gsum += mat.grad.detach().abs().mean().item()
                        gcnt += 1
            if gcnt > 0:
                acc[li] += gsum / gcnt
                counts[li] += 1
    backbone.zero_grad(set_to_none=True)
    return [a / max(c, 1) for a, c in zip(acc, counts)]


def decide_k_from_variance(
    var_per_block,
    default_k=9,
    k_min=3,
    k_max=10,
    shallow_ratio_threshold=2.0,
):
    """Deprecated magic-threshold mapping; prefer `decide_split_k`."""
    n = len(var_per_block)
    half = n // 2
    shallow = sum(var_per_block[:half]) / max(half, 1)
    deep = sum(var_per_block[half:]) / max(n - half, 1)
    ratio = shallow / (deep + 1e-12)
    if ratio >= shallow_ratio_threshold * 1.5:
        k = max(k_min, default_k - 4)
    elif ratio >= shallow_ratio_threshold:
        k = max(k_min, default_k - 2)
    elif ratio <= 1.0 / shallow_ratio_threshold:
        k = min(k_max, default_k + 1)
    else:
        k = default_k
    info = {"shallow_var": shallow, "deep_var": deep, "ratio": ratio, "k": k}
    return k, info
