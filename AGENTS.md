# AGENTS.md

See `README.md` for the current SOTA path, key files, and headline results.

## Environment

- Run `source venv/bin/activate` before running Python commands.

## Modal

The general, project-independent version of these rules lives in the public template repo [tips-for-running-modal](https://github.com/chenglou/tips-for-running-modal), which was extracted from this project. When you learn a new general Modal lesson, add it there; keep this section to the rules an agent needs in-context while operating this repo.

- Use `modal run --detach ...` for training so runs survive client disconnection.
- `--detach` is also the safer default for longer eval and analysis runs.
- Modal can preempt GPU workers at any time, and GPU jobs cannot opt out. A preempted job restarts with the same input only when the function sets `retries=` (modal_run.py does), so design training to resume from checkpoints.
- Launch training with the spawn-based entrypoint in modal_run.py plus `--detach`. A `.remote()`-style client holds a connection for the whole run and cancels the input if that connection breaks (laptop sleep, network blip) — this killed three runs on 2026-07-02 and looks deceptively like preemption in the logs.
- Modal's default function timeout is 5 minutes, so set a longer timeout for training. In practice this repo uses the 24h maximum.
- Volumes persist indefinitely without automatic eviction; treat the output volume as the source of truth for checkpoints and logs.
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
- Any code that depends on `model.named_parameters()`—such as EMA, custom optimizers, or weight manipulation—must be initialized after `torch.compile()`, not before.
- When saving checkpoints or final model weights, strip the `_orig_mod.` prefix:
  `k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()`

## Conversation Tone

You're free to think internally in whatever terms you're comfortable with, but when communicating responses, comments and docs, here are the guidelines:

- Do NOT try to be concise or over-compress words. Be generally brief and clear.
- Avoid using common words in uncommon situations, as pseudo-jargons. Established technical terms in their field (e.g. compiler terms) are fine.
  - Bad: `earn`, `win`, `teach`, `road signs`, `seam`, `source-backed`, `source-shaped`, `browser-owned`
  - Good/fine: `call site`, `control flow`, `type narrowing`, `invariant`
  - Bad: `input facts`, `source calls`, `loop facts`
  - Good: `input contracts`, `function calls`, `loops`
  - Bad doc: "An inclusive infinity endpoint overrides the finite default."
  - Good doc: "An explicitly written range, e.g. 0..Infinity, replaces the default."
  - Bad: "Finite-default parameters publish an implicit finite precondition."
  - Good: "A function's number param is assumed to be finite."
  - Bad: "The table lists the supported checks and their effects on each branch. This supports validation at the boundary"
  - Good: "The table shows what each check proves in its true and false branches. E.g. after `const parsed = Number.parseFloat(text)`, the true branch of `if (Number.isFinite(parsed))` knows that `parsed` is finite."
- Don’t state the obvious just to sound thorough
- Preserve the author's tone. Remove generic filler, but don't shorten an explanation so much that it loses context.
- Use descriptive variable names that make the example understandable on its own.
  - Bad: `ptAt`
  - Good: `pointerDownTime`
- A great trick we use is to document a general point along with an example
  - Bad: "An explicitly written range replaces the default."
  - Good: "An explicitly written range, e.g. 0..Infinity, replaces the default."
- When giving examples, you have a bad habit of using variable names that only make sense within the current conversation, not in a general doc.
  - Bad: “Model nullability explicitly. E.g. `hullSpace: HullID | null`”. The name `hullSpace` might make sense in the conversation where I asked you to modify some docs, but it makes no sense on its own
  - Good: replace with `userID: ID | null`. Everyone's familiar with `userID` and its frequent appearances in app dev
- A general point accompanied by long examples is fine/desirable when the general point's too abstract for most people:
  - Bad: "Later arithmetic does not restore finiteness."
  - Good: "Later arithmetic doesn't necessarily make the value finite again. For example, if `value * 2` may be `Infinity`, dividing it by 2 may still return `Infinity` (`Infinity / 2 === Infinity`). You can make it finite again with e.g. `Math.max(-100, -Infinity)`". Notice the pattern "e.g."/"for example"/"like" + a trailing example; those are nice turns of sentence that point to the general idea of what users can write.
- Avoid unclear pronouns like `this` `it` and others when they can refer to multiple things
  - Bad: "If an operation may produce `NaN`, we report its result as unknown unless an earlier check rules that out"
  - Good: "If an operation may produce `NaN`, we report its result as unknown unless an earlier check proves that the operation cannot produce `NaN`". See how we swapped out "that" with "the operation" and rephrased accordingly. Alternatively, attach a noun: write "these checks" instead of bare "these"
- For markdown files, don't do hard line breaks
