# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Full build (recommended — handles bibliography and cross-references)
latexmk -pdf main.tex

# Single-pass compile
pdflatex main.tex

# Build bibliography only
bibtex main

# Clean auxiliary files
latexmk -c

# Clean everything including PDF
latexmk -CA
```

The output PDF is `main.pdf`. Auxiliary files (`.aux`, `.log`, `.toc`, `.bbl`, etc.) are generated automatically and can be ignored.

## Document Structure

- `main.tex` — root document: includes preamble, title, all chapters, bibliography
- `include/preambule.tex` — all LaTeX package imports and global settings
- `include/title.tex` — title page (author: Селин Тимур Александрович, supervisor: Завьялова Наталья Александровна, MIPT)
- `references.bib` — bibliography in GOST 7.1 style via `gost71u.bst`
- `parts/` — thesis chapters:
  - `Annotation.tex` — abstract
  - `Chapter0.tex` — Introduction (Введение): motivation, goals, research questions
  - `Chapter1.tex` — Physics of SAR and optical imaging (theoretical background)
  - `Chapter2.tex` — Materials and methods: datasets, problem formulation, metrics (PSNR, SSIM, FID), loss functions, GAN components
  - `Chapter3.tex` — Review of existing approaches: Pix2Pix, CycleGAN, CFRWD-GAN
  - `Chapter4.tex` — Proposed models, architecture adaptations, optimizations
  - `Chapter5.tex` — Results and analysis (in progress)
  - `Chapter6.tex` — Conclusion (in progress)
  - `Appendix.tex` — Appendix
- `images/` — figures organized by model/dataset: `pix2pix/`, `cyclegan/`, `cfrwd/`, `sen12/`, `qxs_sar2opt/`, `layers/`

## Thesis Context

The thesis is a master's qualification work (ВКР магистра) at MIPT's Phystech School of Aerospace Technologies, Department of Computational Physics. The topic is SAR→optical image translation using generative neural networks (GANs, diffusion models).

Key terms used throughout:
- **SAR / РЛИ** — Synthetic Aperture Radar imagery
- **ДЗЗ** — Remote sensing (дистанционное зондирование Земли)
- **GAN / ГАН** — Generative Adversarial Network
- **спекл-шум** — speckle noise (characteristic of SAR images)
- Datasets: SEN12 (Sentinel-1/2 pairs), QXS-SAR2OPT

Bibliography style is GOST 7.1u — cite keys use lowercase author names, e.g. `\cite{isola2017image}`, `\cite{chu2017cyclegan}`, `\cite{wei2023cfrwd}`.
