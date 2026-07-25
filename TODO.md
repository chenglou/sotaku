# TODO

- Investigate epiplexity as a way to understand the training and loss dynamics behind the current iterative SOTA regime.
- Investigate influence functions to see whether we can identify the most important training examples and reduce data requirements.
- Publish the July model artifacts: the 96.8% settledness model, the best randomized-late-state model, and the late stay-consistency model. Then prune superseded study checkpoints from the Modal volume.

## Next Experiments

- Replicate the step-39K late stay-consistency switch on independent lineages. Use full 25K evaluation before promoting it over the damped late-state recipe.
- Compare recheck-only against recheck plus KL consistency at the late switch. Existing screens show that the second supervised window carries most of the benefit.
- Test a deliberately contractive or monotone recurrent update. The existing checkpoint is not a DEQ and cannot be converted into one after training.
- Run a muP width study before concluding that wider Sotaku models are intrinsically unstable.
- Fit a budget-matched curve over backprop steps and ES generations. Current early-checkpoint results use unfinished learning-rate schedules and are lower bounds.
- Enlarge the ES fitness pool beyond 20K puzzles for runs longer than about 50 generations.
- Treat pure-ES bootstrapping as low priority. A clean next test would retune the CDRGE step size at the horizon-4 transition or test Eggroll-style low-rank directions on a simpler arithmetic task before returning to full Sudoku.

Pixel Sudoku, generated mazes, vector-valued fitness, Hebbian updates, function-preserving growth, and the first CDRGE runs are completed and documented in `es/EXPERIMENTS_ES.md`.
