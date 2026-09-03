# Sotaku 2 Announcement

Draft only; not posted.

Sotaku 2 is out: a tiny neural net that learns to solve Sudoku without being given Sudoku's rules.

- Solves 99.12% of our fixed 25,000-puzzle sudoku-extreme benchmark at 1,024 iterations.
- Same looped transformer, trained from scratch. About 800K parameters.
- No search, solution checking, or inference-time damping. Still solves 98.63% at 4,096 iterations.
- Evolutionary fine-tuning helped earlier models, but the released model doesn't need it.
- 50K training steps on one H200. The released run took about 5.5 hours including setup, compilation, and validation.

The key training change was simple: on 20% of batches, let the model run for 32-512 iterations without recording gradients, then train it on the next 16 iterations. That gives it practice solving puzzles from the states it actually reaches after running for a while. The other 80% of batches train the first 16 iterations as usual.

Code, recipes, experiment notes, and the checkpoint: https://github.com/chenglou/sotaku

## Optional Follow-Up

Is tying weights through a loop just premature optimization? My hunch is no: it changes the learning problem. An untied network can represent the same computation, but shared weights force training to find an update that works at many iterations. That's a constraint on optimization, not just a way to save parameters.

We tested that hunch with 18 fresh 20K runs: separate compute and parameter comparisons, three seeds, and newly generated puzzles. Shared weights solved more of the hard sudoku-extreme puzzles under our recipe, while independent stages sometimes retained correct answers better. So the evidence supports "different learning behavior," not "sharing always wins." Full results: `looping/weight_tying/RESULTS.md` on the research branch.
