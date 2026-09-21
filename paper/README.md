# paper

Two forms of the same write-up:

- `paper.md` — the working draft in Markdown.
- `ieee/` — the IEEE conference-format paper (IEEEtran, two-column), with TikZ/pgfplots
  figures and the compiled `ieee/typedecide.pdf`.

Every number in either comes from `bench/` (`bench/results/*.json`, or a measurement
recorded in the top-level `README.md` / `packages/typedecide/docs/SPEED.md`), or is
arithmetic shown in the text. Nothing is typed in by hand without a source, which is the
only way a results table stays true after a refactor. Numbers attributed to openjev or
sarvam-jev are theirs.

## Build the IEEE paper

No TeX distribution is needed, only [tectonic](https://tectonic-typesetting.github.io),
which downloads IEEEtran, pgfplots and the fonts on first use and reruns LaTeX/BibTeX
until references settle.

```bash
brew install tectonic          # macOS
# Linux: curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh

cd paper/ieee
make            # -> typedecide.pdf
make check      # fails on undefined refs/citations or boxes overfull by more than 5pt
make clean
```

Layout of `ieee/`:

```
typedecide.tex       the paper
references.bib       bibliography (IEEEtran.bst)
figures/*.tex        TikZ/pgfplots sources; data is inline, and a comment at the top of
                     each file names the bench/results file the data was copied from
Makefile
typedecide.pdf       compiled output, committed for convenience
```

CI (`.github/workflows/paper.yml`) rebuilds the PDF on any change under `paper/` and
uploads it as a workflow artifact.

### Things to do before submitting

- Replace the **affiliation placeholder** in the author block of `typedecide.tex`.
- Check the repository URL in the `\repourl` macro near the top of `typedecide.tex`.
- The McNemar result lives in the `\mcnemarp`, `\mcnemardiscordant`,
  `\mcnemaronlycyclic` and `\mcnemaronlyfull` macros near the top of `typedecide.tex`.
  If `bench/compare.mjs` is re-run on regenerated prediction files and prints anything
  different, update the macros **and** re-read Sec. VII-C, whose prose assumes p > 0.05.
- If a number in `bench/results/` changes, update the table, the figure source that
  names that file, and any prose that quotes it. `grep` for the old value.
- Keep the AI-assistance disclosure in the Acknowledgment; IEEE policy requires it.

## Regenerate the evidence

```bash
../bench/fetch_fixtures.sh
cd ../bench && npm install
./run_grid.sh                 # local rows
node summarize.mjs            # the table to paste into §5
./rerun_contested.sh          # cyclic vs full permutation, with the McNemar test
```

Hosted-model rows need a key in the environment (this comparison has **not** been run;
the paper says so):

```bash
export ANTHROPIC_API_KEY=...          # or OPENAI_API_KEY / GEMINI_API_KEY
node api_baseline.mjs --provider anthropic --model <id> \
    --in-price <usd/Mtok> --out-price <usd/Mtok>
```

Responses are cached under `bench/.cache/`, so a rerun costs nothing and an
interrupted run resumes for free. Prices are passed in rather than hardcoded because
they change; take them from the provider's own page on the day you run it and record
the date in the paper.

## Honest framing

This is a short empirical paper, not a modelling contribution. Its strongest parts are
the negative result (no batched branching in ONNX Runtime) and the position-prior
measurement. Keep the claims sized to that: it belongs on arXiv or at a workshop, and
it should not present itself as a new method.
