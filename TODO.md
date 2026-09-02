# TODO

- Investigate epiplexity as a way to understand the training and loss dynamics of the current recommended model.
- Investigate influence functions to see whether we can identify the most important training examples and reduce data requirements.
- Publish the July model artifacts: the 96.8% ES model rewarded for correct answers matching at iterations 1920 and 2048, the best model trained on later iterations, and the model with a second supervised window and consistency loss added late in training. Then prune superseded study checkpoints from the Modal volume.

## Next Experiments

- Replicate adding the second supervised window and consistency loss at step 39K on independent training runs. Use full 25K evaluation before recommending the addition over training on later iterations with inference damping.
- Compare adding a second supervised window alone against adding both that window and KL consistency late in training. Existing preliminary experiments show that the second supervised window carries most of the benefit.
- Test a deliberately contractive or monotone recurrent update. The existing checkpoint is not a DEQ and cannot be converted into one after training.
- Run a muP width study before concluding that wider Sotaku models are intrinsically unstable.
- Fit a budget-matched curve over backprop steps and ES generations. Current early-checkpoint results use unfinished learning-rate schedules and do not establish the best result of a dedicated shorter run.
- Enlarge the ES fitness pool beyond 20K puzzles for runs longer than about 50 generations.
- Treat ES training from random initialization as low priority. A next test could retune the CDRGE step size when increasing from one to four model iterations, or test Eggroll-style low-rank directions on a simpler arithmetic task before returning to full Sudoku.

Pixel Sudoku, generated mazes, vector-valued fitness, Hebbian updates, function-preserving growth, and the first CDRGE runs are completed and documented in `es/EXPERIMENTS_ES.md`.
