"""
Utility for comparing DDPM and GaussianDiffusion implementations.

This module provides functions to identify and quantify differences between
two diffusion sampling processes that use the same model and input noise.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

# Import types for detection
from common.model.diff.dm.ddpm import DDPM
from common.model.diff.mdm.gaussian_diffusion import GaussianDiffusion


@dataclass
class StepRecord:
    """Record of intermediate values at a single denoising step."""
    timestep: int

    # Input state differences
    x_t_diff: float = 0.0

    # Model prediction differences
    pred_noise_diff: float = 0.0
    pred_x0_diff: float = 0.0

    # Posterior mean and variance differences
    mean_diff: float = 0.0
    log_variance_diff: float = 0.0

    # Output state difference
    x_t_minus_1_diff: float = 0.0

    # Raw values for debugging (optional)
    raw_values: Dict[str, Any] = field(default_factory=dict)


def _compare_schedules(ddpm, gaussian) -> Dict[str, Dict[str, float]]:
    """
    Compare schedule parameters between DDPM and GaussianDiffusion.

    Args:
        ddpm: DDPM instance
        gaussian: GaussianDiffusion instance

    Returns:
        Dict mapping parameter names to difference statistics
    """
    params_to_compare = [
        'betas',
        'alphas_cumprod',
        'posterior_variance',
        'posterior_log_variance_clipped',
        'posterior_mean_coef1',
        'posterior_mean_coef2',
        'sqrt_alphas_cumprod',
        'sqrt_one_minus_alphas_cumprod',
        'sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod'
    ]

    diff = {}
    for param in params_to_compare:
        try:
            # Get DDPM value (PyTorch tensor, registered as buffer)
            ddpm_val = getattr(ddpm, param)
            if isinstance(ddpm_val, torch.Tensor):
                ddpm_val = ddpm_val.cpu().float()
            else:
                ddpm_val = torch.tensor(ddpm_val).float()

            # Get GaussianDiffusion value (NumPy array)
            gauss_val = getattr(gaussian, param)
            if isinstance(gauss_val, np.ndarray):
                gauss_val = torch.from_numpy(gauss_val).float()
            elif isinstance(gauss_val, torch.Tensor):
                gauss_val = gauss_val.cpu().float()
            else:
                gauss_val = torch.tensor(gauss_val).float()

            # Compute differences
            abs_diff = (ddpm_val - gauss_val).abs()
            diff[param] = {
                'max_abs_diff': abs_diff.max().item(),
                'mean_abs_diff': abs_diff.mean().item(),
                'diff_at_t0': abs_diff[0].item() if abs_diff.numel() > 0 else 0.0,
                'diff_at_t_last': abs_diff[-1].item() if abs_diff.numel() > 0 else 0.0,
            }
        except AttributeError as e:
            diff[param] = {'error': str(e)}

    return diff


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    (Copied from GaussianDiffusion for standalone use)
    """
    if isinstance(arr, np.ndarray):
        arr = torch.from_numpy(arr)
    res = arr.to(device=timesteps.device, dtype=torch.float32)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _compare_single_step(
    ddpm,
    gaussian,
    model: nn.Module,
    x_t_ddpm: torch.Tensor,
    x_t_gaussian: torch.Tensor,
    t: int,
    condition_ddpm: torch.Tensor,
    condition_gaussian: torch.Tensor,
    shared_noise: torch.Tensor,
    store_raw: bool = False
) -> StepRecord:
    """
    Compare a single denoising step between DDPM and GaussianDiffusion.

    Args:
        ddpm: DDPM instance
        gaussian: GaussianDiffusion instance
        model: The shared noise prediction model
        x_t_ddpm: Current state for DDPM
        x_t_gaussian: Current state for GaussianDiffusion
        t: Current timestep
        condition_ddpm: Condition tensor for DDPM
        condition_gaussian: Condition tensor for GaussianDiffusion
        shared_noise: Noise to use for both (for fair comparison)
        store_raw: Whether to store raw intermediate values

    Returns:
        StepRecord with comparison metrics
    """
    B = x_t_ddpm.shape[0]
    device = x_t_ddpm.device

    # Create batch timestep tensors
    batch_t = torch.full((B,), t, device=device, dtype=torch.long)

    record = StepRecord(timestep=t)
    record.x_t_diff = torch.norm(x_t_ddpm - x_t_gaussian).item()

    with torch.no_grad():
        # === DDPM predictions ===
        pred_noise_ddpm, pred_x0_ddpm = ddpm.model_predict(model, x_t_ddpm, batch_t, condition_ddpm)
        mean_ddpm, var_ddpm, log_var_ddpm = ddpm.p_mean_variance(model, x_t_ddpm, batch_t, condition_ddpm)

        # === GaussianDiffusion predictions ===
        # Call model directly to get predicted noise
        pred_noise_gaussian = model(x_t_gaussian, gaussian._scale_timesteps(batch_t), cond=condition_gaussian)

        # Compute pred_x0 from noise (without clipping for fair comparison)
        pred_x0_gaussian = (
            _extract_into_tensor(gaussian.sqrt_recip_alphas_cumprod, batch_t, x_t_gaussian.shape) * x_t_gaussian
            - _extract_into_tensor(gaussian.sqrt_recipm1_alphas_cumprod, batch_t, x_t_gaussian.shape) * pred_noise_gaussian
        )

        # Get variance (FIXED_SMALL)
        log_var_gaussian = _extract_into_tensor(
            gaussian.posterior_log_variance_clipped, batch_t, x_t_gaussian.shape
        )

        # Compute mean using posterior coefficients
        mean_gaussian = (
            _extract_into_tensor(gaussian.posterior_mean_coef1, batch_t, x_t_gaussian.shape) * pred_x0_gaussian
            + _extract_into_tensor(gaussian.posterior_mean_coef2, batch_t, x_t_gaussian.shape) * x_t_gaussian
        )

        # Record differences
        record.pred_noise_diff = torch.norm(pred_noise_ddpm - pred_noise_gaussian).item()
        record.pred_x0_diff = torch.norm(pred_x0_ddpm - pred_x0_gaussian).item()
        record.mean_diff = torch.norm(mean_ddpm - mean_gaussian).item()
        record.log_variance_diff = torch.norm(log_var_ddpm - log_var_gaussian).item()

        # Compute next state for both
        if t > 0:
            x_t_minus_1_ddpm = mean_ddpm + (0.5 * log_var_ddpm).exp() * shared_noise
            x_t_minus_1_gaussian = mean_gaussian + (0.5 * log_var_gaussian).exp() * shared_noise
        else:
            x_t_minus_1_ddpm = mean_ddpm
            x_t_minus_1_gaussian = mean_gaussian

        record.x_t_minus_1_diff = torch.norm(x_t_minus_1_ddpm - x_t_minus_1_gaussian).item()

        if store_raw:
            record.raw_values = {
                'pred_noise_ddpm': pred_noise_ddpm.clone(),
                'pred_noise_gaussian': pred_noise_gaussian.clone(),
                'pred_x0_ddpm': pred_x0_ddpm.clone(),
                'pred_x0_gaussian': pred_x0_gaussian.clone(),
                'mean_ddpm': mean_ddpm.clone(),
                'mean_gaussian': mean_gaussian.clone(),
                'log_var_ddpm': log_var_ddpm.clone(),
                'log_var_gaussian': log_var_gaussian.clone(),
                'x_t_minus_1_ddpm': x_t_minus_1_ddpm.clone(),
                'x_t_minus_1_gaussian': x_t_minus_1_gaussian.clone(),
            }

    return record, x_t_minus_1_ddpm, x_t_minus_1_gaussian


def _compute_divergence_summary(step_records: List[StepRecord]) -> Dict[str, float]:
    """
    Compute summary statistics from step records.
    """
    if not step_records:
        return {}

    return {
        'max_x_t_diff': max(r.x_t_diff for r in step_records),
        'max_pred_noise_diff': max(r.pred_noise_diff for r in step_records),
        'max_pred_x0_diff': max(r.pred_x0_diff for r in step_records),
        'max_mean_diff': max(r.mean_diff for r in step_records),
        'max_log_variance_diff': max(r.log_variance_diff for r in step_records),
        'final_output_diff': step_records[-1].x_t_minus_1_diff,
        'cumulative_mean_diff': sum(r.mean_diff for r in step_records),
        'mean_mean_diff': sum(r.mean_diff for r in step_records) / len(step_records),
    }


def compare_diffusion_sampling(
    diffusion0,
    diffusion1,
    model: nn.Module,
    input_data: Dict[str, torch.Tensor],
    n_samples: int = 1,
    seed: int = 42,
    timesteps_to_record: Optional[List[int]] = None,
    compare_schedules: bool = True,
    store_raw: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compare two diffusion implementations step-by-step.

    Args:
        diffusion0: First diffusion instance (DDPM or GaussianDiffusion)
        diffusion1: Second diffusion instance (DDPM or GaussianDiffusion)
        model: The noise prediction model (shared between both)
        input_data: Dict with 'x' (starting noise/data) and 'obj_pc' (conditioning)
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        timesteps_to_record: Specific timesteps to record (None = all)
        compare_schedules: Whether to compare schedule parameters first
        store_raw: Whether to store raw intermediate tensors (memory intensive)
        verbose: Print detailed comparison at each step

    Returns:
        Dict containing:
        - 'schedule_diff': Differences in schedule parameters
        - 'step_records': Per-timestep comparison records
        - 'divergence_summary': Summary statistics of divergence
        - 'first_significant_divergence_timestep': Where significant divergence first occurs
    """
    # Auto-detect and assign DDPM vs GaussianDiffusion
    if isinstance(diffusion0, DDPM) and isinstance(diffusion1, GaussianDiffusion):
        ddpm, gaussian = diffusion0, diffusion1
    elif isinstance(diffusion0, GaussianDiffusion) and isinstance(diffusion1, DDPM):
        ddpm, gaussian = diffusion1, diffusion0
        if verbose:
            print("Note: Swapped diffusion0 and diffusion1 to match expected types (DDPM, GaussianDiffusion)")
    else:
        raise TypeError(
            f"Expected one DDPM and one GaussianDiffusion instance, "
            f"got {type(diffusion0).__name__} and {type(diffusion1).__name__}"
        )

    results = {
        'schedule_diff': {},
        'step_records': [],
        'divergence_summary': {},
        'first_significant_divergence_timestep': None,
        'ddpm_type': type(ddpm).__name__,
        'gaussian_type': type(gaussian).__name__,
    }

    # 1. Compare schedule parameters
    if compare_schedules:
        results['schedule_diff'] = _compare_schedules(ddpm, gaussian)
        if verbose:
            print("=== Schedule Parameter Differences ===")
            for param, diff in results['schedule_diff'].items():
                if 'error' not in diff and diff.get('max_abs_diff', 0) > 1e-6:
                    print(f"  {param}: max_diff={diff['max_abs_diff']:.6e}, t0_diff={diff['diff_at_t0']:.6e}")

    # 2. Setup initial state
    device = next(model.parameters()).device
    x_t_ddpm = input_data['x'].clone().to(device)
    x_t_gaussian = input_data['x'].clone().to(device)

    # Compute conditions
    with torch.no_grad():
        condition_ddpm = model.condition({'obj_pc': input_data['obj_pc'].to(device)})
        condition_gaussian = condition_ddpm.clone()  # Same condition for fair comparison

    # 3. Run step-by-step comparison
    timesteps = ddpm.timesteps
    divergence_threshold = 1e-4

    if verbose:
        print(f"\n=== Step-by-Step Comparison (T={timesteps}) ===")

    for t in reversed(range(timesteps)):
        # Skip if not in requested timesteps
        if timesteps_to_record is not None and t not in timesteps_to_record:
            # Still need to advance the state
            torch.manual_seed(seed + t)
            shared_noise = torch.randn_like(x_t_ddpm) if t > 0 else torch.zeros_like(x_t_ddpm)

            with torch.no_grad():
                B = x_t_ddpm.shape[0]
                batch_t = torch.full((B,), t, device=device, dtype=torch.long)

                # Advance DDPM
                mean_ddpm, _, log_var_ddpm = ddpm.p_mean_variance(model, x_t_ddpm, batch_t, condition_ddpm)
                x_t_ddpm = mean_ddpm + (0.5 * log_var_ddpm).exp() * shared_noise if t > 0 else mean_ddpm

                # Advance GaussianDiffusion
                out = gaussian.p_mean_variance(model, x_t_gaussian, batch_t, condition=condition_gaussian, clip_denoised=False)
                x_t_gaussian = out['mean'] + (0.5 * out['log_variance']).exp() * shared_noise if t > 0 else out['mean']
            continue

        # Generate synchronized noise
        torch.manual_seed(seed + t)
        shared_noise = torch.randn_like(x_t_ddpm) if t > 0 else torch.zeros_like(x_t_ddpm)

        # Compare this step
        record, x_t_ddpm, x_t_gaussian = _compare_single_step(
            ddpm, gaussian, model,
            x_t_ddpm, x_t_gaussian, t,
            condition_ddpm, condition_gaussian,
            shared_noise, store_raw
        )
        results['step_records'].append(record)

        # Track first significant divergence
        if results['first_significant_divergence_timestep'] is None:
            if record.x_t_minus_1_diff > divergence_threshold:
                results['first_significant_divergence_timestep'] = t

        if verbose and (t % 100 == 0 or t < 10):
            print(f"  t={t:4d}: x_diff={record.x_t_diff:.6e}, mean_diff={record.mean_diff:.6e}, "
                  f"logvar_diff={record.log_variance_diff:.6e}, out_diff={record.x_t_minus_1_diff:.6e}")

    # 4. Compute summary statistics
    results['divergence_summary'] = _compute_divergence_summary(results['step_records'])

    if verbose:
        print("\n=== Divergence Summary ===")
        for key, val in results['divergence_summary'].items():
            print(f"  {key}: {val:.6e}")
        if results['first_significant_divergence_timestep'] is not None:
            print(f"  First significant divergence at t={results['first_significant_divergence_timestep']}")

    return results


def print_comparison_report(results: Dict[str, Any]) -> None:
    """
    Print a formatted report of comparison results.
    """
    print("=" * 60)
    print("DIFFUSION COMPARISON REPORT")
    print("=" * 60)

    # Schedule differences
    print("\n[1] Schedule Parameter Differences")
    print("-" * 40)
    schedule_diff = results.get('schedule_diff', {})
    significant_diffs = []
    for param, diff in schedule_diff.items():
        if 'error' in diff:
            print(f"  {param}: ERROR - {diff['error']}")
        elif diff.get('max_abs_diff', 0) > 1e-10:
            significant_diffs.append((param, diff))

    if significant_diffs:
        for param, diff in significant_diffs:
            print(f"  {param}:")
            print(f"    max_abs_diff: {diff['max_abs_diff']:.6e}")
            print(f"    diff_at_t0:   {diff['diff_at_t0']:.6e}")
            print(f"    diff_at_t_last: {diff['diff_at_t_last']:.6e}")
    else:
        print("  No significant differences in schedule parameters.")

    # Divergence summary
    print("\n[2] Divergence Summary")
    print("-" * 40)
    summary = results.get('divergence_summary', {})
    for key, val in summary.items():
        print(f"  {key}: {val:.6e}")

    # First divergence
    first_div = results.get('first_significant_divergence_timestep')
    if first_div is not None:
        print(f"\n[3] First Significant Divergence")
        print("-" * 40)
        print(f"  Timestep: {first_div}")

    print("\n" + "=" * 60)
