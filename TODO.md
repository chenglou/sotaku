# TODO

- Investigate epiplexity as a way to understand the training and loss dynamics of the current recommended model.
- Investigate influence functions to see whether we can identify the most important training examples and reduce data requirements.
- Publish the remaining July research artifacts: the 96.8% ES model rewarded for correct answers matching at iterations 1920 and 2048, and the model with a second supervised window and consistency loss added late in training. The recommended model trained on later iterations is already published as v2.0.0. Preserve reproduction records before pruning superseded study checkpoints from the Modal volume.

## Next Experiments

- Complete the [controlled Hyperloop study](looping/hyperloop/README.md): three paired 20K seeds each for the current baseline, one gated state, and four gated states, with full FP32 evaluation through 4096 iterations.
- Replicate adding the second supervised window and consistency loss at step 39K on independent training runs. Compare against the released later-iteration recipe using ordinary FP32 inference and full 25K evaluation.
- Compare adding a second supervised window alone against adding both that window and KL consistency late in training. Existing preliminary experiments show that the second supervised window carries most of the benefit.
- Test a deliberately contractive or monotone recurrent update. The existing checkpoint is not a DEQ and cannot be converted into one after training.
- Run a muP width study before concluding that wider Sotaku models are intrinsically unstable.
- Fit a budget-matched curve over backprop steps and ES generations. Current early-checkpoint results use unfinished learning-rate schedules and do not establish the best result of a dedicated shorter run.
- Enlarge the ES fitness pool beyond 20K puzzles for runs longer than about 50 generations.
- Treat ES training from random initialization as low priority. A next test could retune the CDRGE step size when increasing from one to four model iterations, or test Eggroll-style low-rank directions on a simpler arithmetic task before returning to full Sudoku.

Pixel Sudoku, generated mazes, vector-valued fitness, Hebbian updates, function-preserving growth, and the first CDRGE runs are completed and documented in `es/EXPERIMENTS_ES.md`.
