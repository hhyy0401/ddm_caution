# Decision caution under changing error penalties

Manuscript and analysis code for a study of how decision thresholds change across error penalties, relative to model-based score-maximizing thresholds.

## Manuscript

- [Compiled paper](paper/main.pdf): six pages of main text, followed by the AI-use declaration and references.
- [LaTeX source](paper/main.tex), [references](paper/refs.bib), and [figures](paper/figs).
- Figure 1 illustrates the task structure and DDM parameters. Its sample trajectory is schematic, not observed data.

The focal contribution is the quantitative comparison between observed and optimal threshold adjustment and the change from above-optimal to below-optimal caution. Penalty-sensitive caution, average overcaution, and small reward losses despite threshold deviations have precedents. The reported reward efficiency is model-predicted, not an observed-score ratio.

## Contents

- `code/`: preprocessing, diffusion-model fitting, comparisons, recovery checks, and figure generation.
- `paper/revision_audit.json`: aggregate checks of the saved estimates.
- `paper/verify_saved_results.py`: audit script requiring the original fitted tables.
- `lean/`: formalization source. A clean Lean build has not been verified for this release.

## Reproduction

Install Python dependencies with `python -m pip install -r requirements.txt`.
Participant-level data and fitted tables are not distributed. Therefore the full empirical analysis cannot be reproduced from this public repository alone.

With authorized data, place `Behavioral_Scores_trials.csv` in `data/` and run `python code/prep.py`. Prepared trial and subject tables are written to `results/`. Fitting and follow-up scripts read this directory; inspect their command-line arguments and input tables before running. Cognitive analyses additionally require composite tables (`results/composites.csv`, and for `hier_abs.py`, `data/all_Scores_filtered_composites.csv`). This release does not provide an end-to-end pipeline for preparing those composites.

The task schematic can be regenerated without participant data using `python paper/make_task_figure.py`. To compile the paper, open the contents of `paper/` in Overleaf or use a LaTeX installation with compatible biblatex and Biber versions. The checked local build used biblatex 3.17 with Biber 2.17; do not mix incompatible versions.

## Interpretation and remaining checks

The manuscript documents unresolved task-protocol details and a task-name discrepancy in the attention-control composite. These require confirmation against the original protocol. The saved cross-validation analysis uses full-data initialization and some full-data quantities; it is not a strictly isolated training-only pipeline. Stronger predictive claims require that reanalysis. Aggregate audits verify saved outputs and do not replace refitting or parameter-uncertainty propagation.

## Commit conventions

Use one or two words per commit message. Do not add AI collaborators or AI co-author trailers. The manuscript retains the authors' declaration of AI assistance.
