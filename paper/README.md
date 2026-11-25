# Paper draft

`sotaku_short.tex` is the paper, and `sotaku_short.pdf` is the built draft. The main text is 3 pages and is written in `sotaku_short.tex` itself. The appendix is the full account: `body.tex` (how we got to the model, the analysis, the measurement mistakes, related work) followed by `appendix.tex` (tables, hyperparameters and the picture gallery). Tables come from `generated/*.tex`.

## Build

```sh
source venv/bin/activate
python -m paper.scripts.make_tables     # writes paper/generated/table_*.tex, headline_1024.tex and full_test_1024.tex
python -m paper.scripts.make_figures    # writes paper/figures/*.pdf
cd paper && pdflatex sotaku_short && bibtex sotaku_short && pdflatex sotaku_short && pdflatex sotaku_short
```

For arXiv, build the paper and then run `python -m paper.scripts.make_arxiv_bundle` from the repository root. The script writes `paper/arxiv_upload.tar.gz`, which holds the `.tex` files, the built `sotaku_short.bbl` (arXiv does not run bibtex), the generated tables and only the figures the paper uses.

On this machine `pdflatex` needs `PATH=/opt/homebrew/Cellar/texlive/20260301/bin:$PATH`; the `/opt/homebrew/bin` symlinks fail to find their support files.

Both scripts read the new results from `runs/paper/`, which is not in Git. `make_figures.py` also reads the perturbation-map arrays under `looping/basin_diagnostics/slow_puzzles/results/`, which are not in Git either. A figure whose inputs are missing is skipped.

`appendix.tex` is kept short on purpose. It holds the hyperparameters, the tables that the text's claims depend on, and a picture gallery. Results that a reader can look up in the repository's own notes (the design history, single-run sensitivity tables, evolution strategies, the 20K-step runs) are summarized in `body.tex` and listed in the last appendix section, "Where to find the rest".

`references.bib` holds only the sources the paper cites. `references_unused.bib` holds entries that were checked against their papers and then dropped from the text; move an entry back to restore a citation.

## Where each number comes from

- Tables 2 and 3 and Figure 1 (released checkpoint and version 1, benchmark and full test split): `runs/paper/evaluations/{v2_fp32,v1_fp32,*_full_test}/` and `runs/paper/analysis/`.
- Table 1 and the right panel of Figure 4 (the six runs trained for the paper): `runs/paper/evaluations/model_paper_*` and `runs/paper/curves/`, which holds the FP32 score of every saved checkpoint. `runs/paper/cohort/result_paper_*.json` holds the BF16 scores recorded during training.
- The new runs' numbers are also written by hand in the prose and must be edited if the runs are redone: the abstract, introduction and results in `sotaku_short.tex`, the last row of Table 4 and the "Results" paragraph of the section "Training on later iterations" in `body.tex`, and Table 8 with the text around it in `appendix.tex`.
- Table 9 (training on 1,000 puzzles): `runs/paper/small/`, which holds each run's training log, `result_*.json`, `small_data_*.json` (the settings and the 1,000 chosen training rows) and the full-test scores in `status/eval_*_full_test.json`. The numbers are written by hand in `appendix.tex`, and the summary "about 50%" appears in the related-work section of `body.tex`. `small_data.py` builds the training set and runs the repository's trainer on it without editing the trainer; `test_small_data.py` tests it (`python -m unittest paper.test_small_data`).
- Table 10 and Figure 8 (inference precision): `release/PRECISION_RESULTS.md` (top half of the table) and `runs/paper/precision/` (bottom half and the figure).
- Figures 3, 6, 7, 9 and 10: `runs/paper/analysis/v2/trajectory.npz` and the three other checkpoints beside it.
- Figure 11 (perturbation maps): `looping/basin_diagnostics/slow_puzzles/results/20k_20260907/slow/level_*.npz`.
- Figures 12 and 13 are images copied from the repository: the top row of `looping/trajectory_viz/global_pca/late_state_ce_global_pca.png`, and `looping/trajectory_viz/study/12_adversarial_audit/adversarial_gallery.png`.
- Everything else is copied from the result documents in the repository; `scripts/make_tables.py` names the source file beside each block of constants, and the prose numbers come from `looping/EXPERIMENTS_LOOPING.md`, `iters/EXPERIMENTS_ITERS.md`, `stabilize/EXPERIMENTS_STABILIZE.md`, `es/EXPERIMENTS_ES.md`, `STALE_EXPERIMENTS_DOC.md` and the study folders under `looping/`.

## Modal jobs

All jobs for the paper ran in the `coreauto` Modal workspace under the app name `cheng-sotaku-paper`, with volumes `cheng-sotaku-outputs` and `cheng-sotaku-hf-cache`. `modal_jobs.py` has one local entrypoint per kind of job; launch each job as its own detached invocation from the repository root:

```sh
modal run --detach paper/modal_jobs.py::prepare
modal run --detach paper/modal_jobs.py::train --arm late_state_ce --seed 20261119
modal run --detach paper/modal_jobs.py::train --arm vanilla --seed 20261119
modal run --detach paper/modal_jobs.py::evaluate --model paper_cohort/model_paper_late_state_ce_seed20261119.pt
modal run --detach paper/modal_jobs.py::train_small --arm late_state_ce --mixed --total-steps 10000 --weight-decay 0.3
modal run --detach paper/modal_jobs.py::evaluate --model paper_small/model_small1000_late_state_ce_wd0p3_mixed_steps10000_seed20261120.pt --name small1000_late_state_ce_wd0p3_mixed_steps10000_seed20261120_full_test --iters 16,128,1024 --full-test
modal run --detach paper/modal_jobs.py::evaluate --model release/model_late_state_ce.pt --name v2_fp32_full_test --iters 16,128,1024 --full-test
modal run --detach paper/modal_jobs.py::curve --run paper_late_state_ce_seed20261119
modal run --detach paper/modal_jobs.py::analyze --model release/model_late_state_ce.pt --name v2
modal run --detach paper/modal_jobs.py::precision --model release/model_late_state_ce.pt --name v2
```

Every worker writes `status/<job>.json` on the output volume when it starts, finishes or fails, e.g. `modal volume get cheng-sotaku-outputs status/train_paper_vanilla_seed20261119.json -`. Reading these files is how to check progress without listing the workspace's apps or containers.

The released checkpoints were copied to `release/` on the volume with `modal volume put`. Download results with e.g.:

```sh
modal volume get cheng-sotaku-outputs evaluations/v2_fp32 runs/paper/evaluations/
modal volume get cheng-sotaku-outputs analysis/v2 runs/paper/analysis/
```

`jobs.json` records the call IDs of the training runs.

## Scripts

- `scripts/trajectory_stats.py`: per-iteration state size, update size, weakest answer margin, per-puzzle solved flags, example boards, and mean attention maps for one checkpoint.
- `scripts/precision_state.py`: FP32 against BF16 against BF16 arithmetic with the carried state stored in FP32.
- `scripts/checkpoint_curve.py`: re-scores every saved training checkpoint of a run in FP32.
- `scripts/make_tables.py`, `scripts/make_figures.py`: build the LaTeX fragments and figures.
- `scripts/make_arxiv_bundle.py`: writes the source bundle for arXiv.
