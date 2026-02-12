"""
Verify that grid_ae weights remain unchanged after training.
Compares pretrained AE weights against the trained diffusion checkpoint.

Usage:
    python scripts/verify_frozen_weights.py \
        --pretrained logs/checkpoints/gridae/128latent/last-v1.ckpt \
        --trained logs/wandb_logs/LG3DContact/p7yt5x77/checkpoints/last.ckpt
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Verify grid_ae weights are frozen during training")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained AE checkpoint", default="logs/checkpoints/gridae/128latent/best-epoch=16-val/total_loss=0.0948.ckpt")
    parser.add_argument("--trained", type=str, help="Path to trained diffusion checkpoint", default="logs/wandb_logs/LG3DContact/p7yt5x77/checkpoints/last.ckpt")
    args = parser.parse_args()

    pretrained_ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    trained_ckpt = torch.load(args.trained, map_location="cpu", weights_only=False)

    pretrained_sd = pretrained_ckpt.get("state_dict", pretrained_ckpt)
    trained_sd = trained_ckpt.get("state_dict", trained_ckpt)

    # Pretrained AE keys: "model.xxx" -> remap to "grid_ae.xxx"
    ae_keys = {k[6:]: k for k in pretrained_sd if k.startswith("model.")}

    matched, changed, missing = 0, 0, 0
    changed_keys = []

    for short_key, orig_key in sorted(ae_keys.items()):
        trained_key = f"grid_ae.{short_key}"
        if trained_key not in trained_sd:
            missing += 1
            print(f"  MISSING  {trained_key}")
            continue

        pre_w = pretrained_sd[orig_key]
        tra_w = trained_sd[trained_key]

        if pre_w.shape != tra_w.shape:
            print(f"  SHAPE MISMATCH  {trained_key}: {pre_w.shape} vs {tra_w.shape}")
            changed += 1
            continue

        diff = (pre_w - tra_w).abs().max().item()
        if diff > 0:
            changed += 1
            changed_keys.append((trained_key, diff))
        else:
            matched += 1

    print(f"\n{'='*60}")
    print(f"Results: {matched} identical, {changed} changed, {missing} missing")
    print(f"{'='*60}")

    if changed_keys:
        print(f"\nWARNING: {changed} grid_ae parameters changed during training!")
        for key, diff in changed_keys:
            print(f"  {key}: max diff = {diff:.6e}")
    else:
        print("\nAll grid_ae weights are identical. Freezing worked correctly.")


if __name__ == "__main__":
    main()
