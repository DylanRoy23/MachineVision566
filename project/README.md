# Residential 3D reconstruction: image density and measurement accuracy

Research question: **How does reducing image density affect the accuracy and
measurability of a residential reconstruction?**

## Fixed scope

Reconstruct one exterior corner of the house: two adjoining walls visible from
ground level. Choose a section with identifiable features and accessible points
for physical measurements. Photograph that same section for every experiment.

The required result is a sparse reconstruction with estimated cameras and a dense
point cloud for each usable dataset. A failed reconstruction is an experimental
result and must be reported, including its failure stage. A mesh or a whole-house
model is optional future work. Interior reconstruction, roof capture, SLAM,
NeRFs, Gaussian splatting, and a custom viewer are outside this project's scope.

Use COLMAP as the reconstruction backend. The student's contribution is the
capture design, reproducible dataset preparation, physical validation, controlled
experiment, and interpretation of the geometry and failures. This scope assumes
the instructor permits an existing reconstruction backend; it is a proposal, not
confirmation of instructor approval.

## One controlled experiment

Capture one ordered sequence of approximately 120–160 sharp photos. Use one
camera, lens, resolution, and fixed focus/zoom. Walk around the corner so views
include translation, changing angles, and useful overlap. Keep the scene and
lighting stable. Preserve original files and EXIF metadata. Name them in capture
order with zero padding, such as `0001.jpg`. Alternatively, record about
one minute of video and extract a frame every half second (about 120 images)
using `src/extract_frames.py`. Keep sharp frames with overlapping views.

Create three nested datasets from that one sequence:

| Dataset | Selection | If 160 images were captured |
| --- | --- | --- |
| full | Every image, starting with the first | 160 |
| half | Every second image, starting with the first | 80 |
| quarter | Every fourth image, starting with the first | 40 |

The deliberate change is sampling density along the same capture path; overlap
and baseline between retained views change with it. This does not isolate image
count independently of viewpoint spacing. Select the capture sequence before
examining reconstruction accuracy; use the same measurement targets for all runs.

Process each dataset independently. Fix the COLMAP version, reconstruction
settings, image processing, hardware, and timing boundaries across all runs.
Measure end-to-end reconstruction time, excluding capture, input preparation,
and manual endpoint selection. Record failed runs and disconnected components.
If settings change after a pilot, rerun all three datasets with those settings.

## Physical validation

Before reconstruction, identify and label six physical spans visible in the
photos. Put their descriptions and measured lengths in `data/measurements.csv`.
Use accessible, repeatable endpoints such as identifiable wall or frame corners;
choose checks on both walls and across multiple lengths. Document which endpoints
were measured using annotated photos. Record the measuring tool and its precision.

- Exactly one span has role `scale`. It establishes meters per model unit.
- Five different spans have role `evaluate`. Keep their measured lengths out of
  scale fitting, reconstruction tuning, and endpoint selection decisions.

After each reconstruction, select the corresponding six pairs of 3D endpoints in
a point-cloud viewer. Use the same physical endpoint definitions across runs and
record coordinates in `output/<dataset>/endpoints.csv`. Use one reconstructed
component and its coordinate system per run. Do not combine component coordinates
or independently fit the five check lengths.

The evaluation stage computes a separate scale for each model:

```text
scale = physical reference length / model reference length
predicted length = scale * model check length
absolute error = |predicted length - measured check length|
percentage error = 100 * absolute error / measured check length
```

If an endpoint cannot be located reliably, leave all six coordinate cells for
that span empty. Report it as unavailable. Never silently drop unavailable
measurements or compare only different subsets without identifying them.

## Results to report

For each dataset, report image count, registered-image fraction in the selected
component, disconnected components, runtime, the five individual distance errors,
mean absolute distance error, and the number of available checks out of five.
Compare errors on the same available checks across runs as well as listing all
individual outcomes. Include point-cloud views from comparable viewpoints and
examples of missing or incorrect geometry.

Endpoint availability is a limited measure of usefulness, not surface
completeness. Point count alone does not establish accuracy or coverage.
Without a reference scan, do not claim to measure full surface accuracy.
Manual endpoint selection and tape measurements also contribute uncertainty.

Completion means performing all three runs or documenting their failures,
reporting every planned check, explaining the observed tradeoffs, and delivering
the code, configurations, reconstruction artifacts, and LaTeX report. No accuracy
threshold is assumed before data collection. Dense outputs and measurements still
need real images.

## Repository and current status

```text
project/
  README.md                 Project definition, workflow, and citations
  config.json               Relative paths and dataset strides
  requirements.txt          Python dependency notes
  data/
    images/                 Original ordered photos (ignored by Git)
    measurements.csv        One scale span and five independent checks
  output/                   Generated manifests, models, and results (ignored)
  src/
    project.py              Short command-line entry point
    extract_frames.py       Extract video frames every half second
    utils/
      datasets.py           Implemented: nested manifests and endpoint templates
      reconstruction.py     Coordinate datasets and save run results
      pipeline.py           Sparse and dense stages for one dataset
      colmap.py             Commands, options, logs, and version checks
      model_io.py           Read model and point-cloud metadata
      environment.py        Record hardware details
      evaluation.py         Implemented: scale and distance-error calculations
```

Paths in `config.json` resolve relative to that file. From the repository root:

```sh
python project/src/project.py --help
# Optional video input; requires FFmpeg on PATH:
python project/src/extract_frames.py /path/to/house.mp4
python project/src/project.py prepare
# Requires a CUDA build of COLMAP on PATH (or set reconstruction.colmap):
python project/src/project.py reconstruct
# Fill physical measurements and model endpoints before evaluation:
python project/src/project.py evaluate
```

`extract_frames.py` writes `0001.jpg`, `0002.jpg`, etc. to `data/images/`.
Use `--output /path/to/empty/folder` for a different destination. It refuses
folders containing images to prevent mixing captures or overwriting photos.
FFmpeg is a separate dependency; it is not installed by `requirements.txt`.

`prepare` requires images and writes full/half/quarter manifests containing exact
input paths. It does not copy photos or run reconstruction. It creates blank
endpoint templates and preserves existing selections. A changed capture needs a
new output directory in `config.json` to keep experiments separate.

`reconstruct` runs COLMAP feature extraction, exhaustive matching, mapping, and
multi-view stereo for each manifest in its own `output/<dataset>/colmap`
workspace. COLMAP options live in `config.json` under `reconstruction.options`.
It densifies the component with the most registered images and writes
`output/<dataset>/cloud.ply`. It also writes `reconstruction.json` with the
COLMAP version, commands, hardware, timings, components, and any failure stage.
A recorded run is never overwritten. Rerunning skips it, and changed options
require a new output directory. COLMAP is not installed by this project.

`evaluate` writes `output/<dataset>/evaluation.json`. It rejects unfilled physical
lengths and requires the scale endpoints. The initial CSV contains no invented
measurements. No final report or experimental results are supplied.

## Sources and contributions

- Reconstruction stages and image capture:
  [COLMAP tutorial](https://colmap.github.io/tutorial.html).
- Video frame sampling:
  [FFmpeg fps filter](https://ffmpeg.org/ffmpeg-filters.html#fps-1).
- Planned backend commands:
  [COLMAP command-line interface](https://colmap.github.io/cli.html).
- Johannes L. Schoenberger and Jan-Michael Frahm. *Structure-from-Motion
  Revisited*. CVPR, 2016.
- Johannes L. Schoenberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys.
  *Pixelwise View Selection for Unstructured Multi-View Stereo*. ECCV, 2016.
