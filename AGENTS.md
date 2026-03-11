# AGENTS.md

See `README.md` for the current SOTA path, key files, and headline results.

## Environment

- Run `source venv/bin/activate` before any `python3` command.

## Modal

- Use `modal run --detach ...` for training so runs survive client disconnection.
- `--detach` is also the safer default for longer eval and analysis runs.
- Modal can preempt GPU workers at any time; design training to resume from checkpoints.
- Set a long enough function timeout for training. In practice this repo uses the 24h maximum.
- Volumes persist indefinitely; treat the output volume as the source of truth for checkpoints and logs.
- A local output stream can die while the Modal worker keeps running. Check `modal container list` and `modal volume ls` for true status.
- `modal app logs` only works while the app is still active. For detached or finished runs, poll the log file from the volume with `modal volume get`.
- Keep experiment code Modal-agnostic: accept an `output_dir` argument and let the Modal wrapper point it at the mounted volume path.
- Do not pipe `modal run --detach` into `tail`, `head`, or similar tools; those commands hang waiting for EOF.

## Checkpoints

- Save model state, optimizer state, training step, and config in resumable checkpoints.
- Load checkpoints before `torch.compile()` so parameter names still match the saved state.
- Save config in checkpoints and verify it on load to avoid resuming the wrong experiment.

## `torch.compile`

- `torch.compile(model)` renames parameters from `foo.weight` to `_orig_mod.foo.weight`.
- Any code that depends on `model.named_parameters()` must be initialized after `torch.compile()`, not before.
- When saving checkpoints or final model weights, strip the `_orig_mod.` prefix:
  `k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()`
