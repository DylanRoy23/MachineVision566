# Homework 2: Feature Detection and Matching

The programs use the supplied frame sequences in `images/`; no command-line
arguments are required. Run:

```sh
make
make run
```

The three comparisons use frames 100 and 150 for scale and rotation, and
frames 130 and 160 for translation. Edit `src/dataset.cpp` to choose other
frames from the sequences.

Shared interfaces are under `include/hw2/`, their implementations are under
`src/`, and the executable entry points are under `apps/`. The shared sources
are compiled once into `build/libhw2.a` and linked into each program.

Results are written under `output_images/` as annotated images and CSV files.
The four required experiments are:

| Program | Assignment item | Output |
| --- | --- | --- |
| `detector_comparison` | DoG, Harris, Shi-Tomasi, and FAST | count, repeatability, and runtime |
| `feature_selection` | ANMS feature selection | strongest-N versus ANMS coverage |
| `descriptor_matching` | SIFT, BRIEF, and ORB | matches, timing, and confusion matrices |
| `roc_curve` | ratio-test ROC | ROC points, plots, AUC, and best threshold |
