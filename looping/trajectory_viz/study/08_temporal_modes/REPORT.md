# Temporal modes

## Hypothesis

Sotaku's recurrent state may contain a reusable rotating or oscillating mode, rather than only smooth drift. A genuine mode should predict the next state on unseen puzzles, survive a second random projection, lose strength when iteration order or cross-feature phase is destroyed, and transfer across checkpoints.

## Method

The analysis used 60 balanced test puzzles: four puzzles from each of five rating buckets in each of the discovery, validation, and final splits. It evaluated the stable plain, collapsed plain, late-state cross-entropy, and combined-margin checkpoints. No final puzzle was used to fit a projection or DMD operator.

Two independent random 16-dimensional projections of the 10,368-dimensional board state were evaluated. The predefined variants were normalized states and normalized updates over iterations 0-128 and 768-1024. Validation selected early normalized updates by mean DMD R2 improvement over a model fitted after shuffling iteration order. Final controls used 32 independent time shuffles and 32 phase-randomized surrogates. Phase randomization preserved every projected channel's Fourier power exactly while removing its phase relationship to other channels.

## Final held-out result

Time ordering is real, but the results do not support a substantial rotating or oscillating latent mode.

On the selected early normalized updates, ordered DMD achieved final R2 of 0.80-0.85 for stable plain, 0.85-0.86 for collapsed plain, 0.90-0.91 for late-state cross-entropy, and 0.90-0.92 for combined margin across the two random projections. Shuffling time reduced R2 by 0.30-0.40 for the two plain checkpoints and by 0.49-0.95 for the late-trained checkpoints.

Phase randomization reduced R2 by only 0.008-0.021. Most predictability therefore comes from each channel's autocorrelation and power spectrum, not a precise phase relationship among channels.

The DMD eigenvalues cluster near the positive real axis. The largest angle among active eigenvalues was 0.09 radians, and no checkpoint showed a large, stable complex pair characteristic of sustained rotation. Simple persistence also predicted the next update better than DMD in seven of eight checkpoint/projection comparisons. Late-state cross-entropy was the only exception, and only slightly: DMD improved over persistence by 0.009 and 0.018.

Raw cross-checkpoint DMD R2 ranged from 0.57 to 0.91, but persistence beat almost every transferred operator. The transfer result supports shared smoothness, not a shared oscillatory law.

## Curvature and Fourier checks

The early normalized state moves smoothly: consecutive projected velocity cosine is 0.90-0.94 on final puzzles. Normalized updates are less straight and differ by checkpoint, but no curvature pattern separates healthy from collapsed models consistently.

Late random projections show period-two power for stable plain, late-state cross-entropy, and combined margin, while collapsed plain is dominated by a much slower component. This is exploratory rather than evidence of a full-state period-two mode. Each projection observes only 16 of 10,368 dimensions, late updates are small, DMD finds no matching active oscillatory eigenvalue, and the late-update variant did not survive validation selection. The likely explanation is an alternating low-energy residual exposed after projection and normalization.

## Controls

- Final holdout: 20 puzzles balanced across all five rating buckets.
- Unseen-puzzle transfer: every projection and DMD operator was fitted on the 20 discovery puzzles.
- Validation selection: 20 separate puzzles chose among four predefined variants.
- Projection control: two independent matched-rank random subspaces gave the same verdict.
- Shuffled-time control: 32 independent shuffles per checkpoint and projection.
- Phase control: 32 Fourier-magnitude-preserving phase randomizations per checkpoint and projection.
- Checkpoint transfer: all four discovery-fit operators were evaluated on all four final checkpoint trajectories.
- Normalization: both full-state-normalized projections and full-update-normalized projections were tested.

## Limitations

Random projections can expose low-energy temporal components but cannot establish that those components dominate the full hidden state. DMD tests an affine one-step approximation; a nonlinear or state-dependent rotation could escape this analysis. The study also tests iteration windows, not trajectories aligned to puzzle-solving events.

## Verdict

Sotaku has strongly ordered and predictable temporal motion, especially during the first 128 iterations. The motion is better described as smooth, mostly real-valued relaxation than as a helix, rotation, or sustained oscillation. No temporal mode found here explains long-horizon health or collapse.

## Artifacts

- `metrics.json`: full configuration, split membership, summaries, controls, DMD spectra, and transfer matrix.
- `dmd_heldout_comparison.png`: ordered versus shuffled-fit DMD for every predefined variant.
- `selected_controls.png`: final phase-randomized and shuffled-time controls in both random projections.
- `dmd_eigenvalues.png`: discovery-fit eigenvalues for the selected variant.
- `checkpoint_transfer.png`: source-to-target checkpoint DMD transfer.
- `curvature_fourier_summary.png`: held-out curvature and Fourier statistics.
