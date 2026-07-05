# TODO

- Investigate epiplexity as a way to understand the training and loss dynamics behind the current iterative SOTA regime.
- Investigate influence functions to see whether we can identify the most important training examples and reduce data requirements.
- Publish the July 2026 model artifacts as a GitHub release (like the February checkpoint): the 96.5% ES-polished model, the 95.2% and 94.6% harvest rescues (currently on the sudoku-outputs Modal volume and R2), then prune the ~60 study checkpoints from the volume.

## Open explorations (2026-07-05) — threads to keep warm, not verdicts
- Convergence menu: settledness-as-ES-fitness at 2048; randomized-horizon supervised training (no special iteration count); Jacobian damping at the terminal state (the DEQ-stabilizer trick, mechanistically different from the four failed fixed-point losses); DEQ/monotone-operator prototype (guarantee by construction — open research question whether contraction costs capability).
- ES from scratch, phase two: the CE run reached the uniform predictor — the same phase every backprop run passes through before breaking out. Untried knobs to break through: recalibrate sigma at the floor (3e-3 was chosen at random init), bigger population near the floor (signal is tiny there), fitness at horizon 1 first (learn the single-pass skill, then extend — horizon curriculum from the bottom), easy-puzzle fitness curriculum (more givens = denser signal per puzzle).
- Intermediate-horizon ES, take two: fitness at 256 where the 6.7% seed is actually weak (the 64-iteration take failed by design — mastered horizon, no pressure; that was the horizon choice failing, not the idea).
- muP width test: "d=192 collapses at every LR" is the textbook signature of hyperparameters that don't transfer across width; a muP-parameterized testbed run answers whether wider was ever really broken. Needs careful implementation — wrong muP gives a confident wrong answer.
- ES fitness pool: enlarge from 20K puzzles (we cycle it every ~52 generations; 3M unused rows sit right there — repetition decays in value, fresh data is free).
- Fit our own allocation curve: final quality as a function of (bootstrap steps, ES generations) from the ladder data — our-scale scaling law, heeding the fitting pitfalls (small-range extrapolation, parameter-counting choices).
