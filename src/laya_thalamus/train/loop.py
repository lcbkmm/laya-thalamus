"""Single-device RLCD fine-tune loop (adapted from upstream Laya notebook)."""

from __future__ import annotations

import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward

from laya_thalamus.train.data import load_trace_rows, traces_to_items
from laya_thalamus.train.resolve import resolve_checkpoint_dir


def collate_train_batch(items: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def _pick_device(device: str | None) -> torch.device:
    if device and device != "auto":
        d = torch.device(device)
        if d.type == "cuda" and not torch.cuda.is_available():
            print("[finetune] CUDA requested but unavailable; using CPU")
            return torch.device("cpu")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_checkpoint(
    model: Any,
    tok: Any,
    cfg: dict[str, Any],
    out_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out_dir / "encoder"))
    tok.save_pretrained(str(out_dir / "tokenizer"))
    (out_dir / "rl_agent_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if meta:
        (out_dir / "finetune_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def run_finetune(
    *,
    dataset: str | Path,
    base_model: str,
    output_dir: str | Path,
    subfolder: str | None = None,
    device: str | None = "auto",
    epochs: int = 3,
    micro_batch: int = 4,
    grad_accum: int = 4,
    group_size: int = 4,
    lr_encoder: float = 2.5e-5,
    lr_head: float = 1.0e-4,
    sigma_start: float = 0.4,
    sigma_end: float = 0.1,
    ce_weight: float = 1.0,
    max_items: int = 0,
    seed: int = 42,
    token: str | None = None,
) -> dict[str, Any]:
    """Fine-tune and write a Laya-native checkpoint under ``output_dir``."""
    torch.manual_seed(seed)
    random.seed(seed)

    out = Path(output_dir)
    device_t = _pick_device(device)
    use_cuda = device_t.type == "cuda"
    print(f"[finetune] device={device_t} base={base_model!r}")

    model_dir = resolve_checkpoint_dir(base_model, subfolder=subfolder, token=token)
    print(f"[finetune] checkpoint={model_dir}")

    with open(model_dir / "rl_agent_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = dict(cfg)
    cfg["gradient_checkpointing"] = bool(use_cuda)
    max_len = int(cfg.get("max_len") or 1024)
    head_max_len = int(cfg.get("head_max_len") or 256)

    tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    model = build_model(cfg, encoder_dir=str(model_dir / "encoder"), pretrained=True)
    weights = load_file(str(model_dir / "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    if use_cuda:
        try:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.head_checkpointing = True
        except Exception as e:
            print(f"[finetune] gradient checkpointing skipped: {e}")
    model.to(device_t)
    model.train()

    rows = load_trace_rows(dataset)
    items = traces_to_items(
        rows, tok, max_len=max_len, head_max_len=head_max_len, limit=max_items
    )
    if not items:
        raise RuntimeError(
            f"No training items built from {dataset}. "
            "Need traces with state + questions + answers."
        )
    print(f"[finetune] dataset={dataset} rows={len(rows)} items={len(items)}")

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr_encoder},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=0.01,
    )
    steps_per_epoch = max(1, len(items) // max(1, micro_batch * grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps_per_epoch * epochs), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history: list[dict[str, Any]] = []
    t_start = time.time()
    for epoch in range(epochs):
        random.seed(seed + epoch)
        random.shuffle(items)
        epoch_loss = 0.0
        epoch_reward = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        accum_step = 0
        progress = epoch / max(1, epochs - 1)
        sigma = sigma_start + (sigma_end - sigma_start) * progress
        t0 = time.time()

        for b_idx in range(0, len(items), micro_batch):
            chunk = items[b_idx : b_idx + micro_batch]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.float16)
                if use_cuda
                else torch.autocast("cpu", enabled=False)
            )
            with autocast_ctx:
                logits, act = model(
                    batch["input_ids"].to(device_t),
                    batch["attention_mask"].to(device_t),
                    batch["marker_pos"].to(device_t),
                    batch["marker_mask"].to(device_t),
                    batch["qtype"].to(device_t),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device_t)
            k = mask.sum(-1, keepdim=True).float().clamp(min=1.0)
            target = batch["target"].to(device_t)

            g = group_size if use_cuda else max(1, min(group_size, 2))
            eps = torch.randn((g,) + logits.shape, device=device_t) * sigma
            eps = eps * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(
                    q,
                    target.unsqueeze(0),
                    batch["qtype"].to(device_t),
                    mask,
                    w_sph=0.75,
                    w_rps=1.0,
                )
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (
                2 * max(sigma, 1e-4) ** 2
            )
            loss_rl = -(adv * logp).mean()
            loss_ce = -(
                target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (loss_rl + ce_weight * loss_ce) / grad_accum + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * grad_accum
            epoch_reward += float(r.mean().item())
            n_batches += 1
            if n_batches % 20 == 0:
                print(
                    f"  epoch {epoch+1}/{epochs} step {n_batches} "
                    f"loss={loss.item()*grad_accum:.4f} reward={r.mean().item():.3f}"
                )

        avg_loss = epoch_loss / max(1, n_batches)
        avg_r = epoch_reward / max(1, n_batches)
        print(
            f"[finetune] epoch {epoch+1}/{epochs} done in {time.time()-t0:.1f}s "
            f"avg_loss={avg_loss:.4f} avg_reward={avg_r:.3f}"
        )
        history.append(
            {"epoch": epoch + 1, "avg_loss": avg_loss, "avg_reward": avg_r}
        )
        _save_checkpoint(
            model,
            tok,
            cfg,
            out / "checkpoint_latest",
            meta={"epoch": epoch + 1, "avg_loss": avg_loss, "avg_reward": avg_r},
        )

    meta = {
        "base_model": base_model,
        "base_checkpoint": str(model_dir),
        "dataset": str(dataset),
        "n_rows": len(rows),
        "n_items": len(items),
        "epochs": epochs,
        "device": str(device_t),
        "elapsed_s": round(time.time() - t_start, 2),
        "history": history,
    }
    _save_checkpoint(model, tok, cfg, out, meta=meta)
    # Keep a copy of original tokenizer special files if missing
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        src = model_dir / "tokenizer" / name
        dst = out / "tokenizer" / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    print(f"[finetune] wrote {out}")
    return meta
