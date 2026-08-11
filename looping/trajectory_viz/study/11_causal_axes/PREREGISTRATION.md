# Causal-axis analysis plan

This plan was fixed before the final split was evaluated.

- Sample 60 canonical `sudoku-extreme` test puzzles with seed 20260811: four puzzles from each of five rating buckets in each of discovery, validation, and final splits.
- Analyze the four checkpoints named by `study/PROTOCOL.md`.
- Test two predefined semantic axes. `answer_evidence` is the normalized output-head direction for the true digit versus the mean other digit. `solvedness_progress` is a ridge direction fitted on discovery board states to the fraction of blank cells currently correct, after removing iteration means.
- Choose the ridge penalty from 0.01, 0.1, 1, 10, and 100 by mean validation partial correlation across all four checkpoints. Do not refit on validation or final puzzles.
- Evaluate the selected axes once on the 20-puzzle final split. Compare the supervised axis with 64 within-iteration label shuffles and the answer-evidence axis with 64 random digit-conditioned axes.
- Apply one pulse after iterations 16 and 512. Continue recurrence to iterations 128 and 1024, respectively. Use signed doses ±0.125, ±0.25, and ±0.5 times each puzzle's natural one-step update norm.
- Every pulse has eight orthogonal random-direction controls with exactly matched norm. Measure immediate, one-step, and endpoint confidence, correct-answer margin, cell accuracy, solved fraction, recovery, and collapse.
- Accept an axis as causal only if the held-out semantic association exceeds its null and the signed dose response exceeds random controls in the predicted direction on at least two checkpoints. Otherwise reject the axis or label the result checkpoint-specific.
