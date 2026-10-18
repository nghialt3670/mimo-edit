# MMEdit

A multi-turn image-editing agent built on `smolagents`, with Pillow images and masks as its entity representation. Its installable packages live under `src/`: the editing framework is `framework`, annotations are `annotation`, benchmark evaluation is `evaluation`, and plotting/browser helpers are `visualization`. Notebook work lives in `src/notebooks`; maintenance commands live in `src/scripts`.

- `framework.core` owns agent orchestration and conversation state.
- `framework.tools` owns tool registration, input validation, and routing to editing workflows.
- `framework.inference` owns the shared model pool and task-specific inference helpers: FLUX.2 Klein edits live in `flux2_klein`, SAM 3 segmentation in `sam3`, Grounding DINO region grounding in `grounding_dino`, and ObjectClear removal in `object_clear`. Geometric object operations are implemented by their corresponding registered tools. `ModelPool` owns these fixed backbones.
- `framework.utils` owns shared image and project utilities.

## Install

Install the framework and its optional model stack:

```bash
uv sync --extra models
```

Creating `ModelPool` downloads and eagerly loads Klein, SAM 3, and ObjectClear. Grounding DINO is loaded lazily when `segment_region` is first called. Ensure sufficient memory is available before constructing the pool and using the region tool; the models remain resident for the lifetime of the tools. Gated Hugging Face models require `HF_TOKEN`.

Set the required `RANDOM_SEED` integer in `.env` to seed Python, NumPy, Torch, model pipelines, benchmark prompt selection, and supported LLM APIs. It is the sole seed source; the project fails fast when it is missing or outside the range `0` through `4294967295`.

Install the development formatters and format Python source with:

```bash
uv sync --group dev
uv run --group dev python -m scripts.format
```

Pass file or directory paths to `scripts.format` to format a subset.

## Use the Tools In-Process

Tools invoke the model pipelines directly; there is no inference service or separate server process.

```python
from framework import Agent, image_editing_tools

tools = image_editing_tools()
agent = Agent(model=model, tools=tools)

result = agent.send(
    "Remove the object selected by <box> and return the edited image.",
    objects={"image_0": image, "box_0": [120, 80, 340, 310]},
)
edited = result.message.objects[0]
```

Pass Pillow `Image` instances and grayscale Pillow masks as objects. White mask pixels select the entity. Object-targeting tools accept one mask/object per call; issue another tool call for each additional object. `segment_object` accepts exactly one text prompt, point, box, or scribble and returns one mask; text prompts return the highest-scoring SAM 3 match. `segment_region` first grounds a text description to a box with Grounding DINO, then refines it to a mask with SAM 3. `add_object` sends the selected crop and prompt to the shared FLUX.2 Klein editing pipeline, then composites the result only inside the mask. The tool set shares its `ModelPool`.

For every registered image-editing tool's inputs, behavior, and default backbone, see the detailed [tool reference](./docs/TOOLS.md).

To find completed RefEdit-Bench mixed-mode samples where a spatial placement
used a point or scribble despite a full reference box, run:

```bash
uv run python -m evaluation.spatial_selector_issues \
  .out/evaluation/refedit_bench/mixed --indices-only
```

Set `RERUN_SPATIAL_SELECTOR_ISSUES = True` in the RefEdit-Bench mixed inference
notebook to detect and rerun only those samples in the latest run.

ObjectClear is provided under its upstream S-Lab license, which restricts use to non-commercial purposes. Review the license before enabling that model.

## Configure Inference

The three inference backbones are fixed: FLUX.2 Klein base 4B for editing and
generation, SAM 3 for segmentation, and ObjectClear for removal. Configure
their inference behavior, not their model IDs:

```python
from framework import InferenceConfig, KleinInferenceConfig, image_editing_tools

tools = image_editing_tools(
    inference_config=InferenceConfig(
        klein=KleinInferenceConfig(
            num_inference_steps=4,
            guidance_scale=1.0,
            mask_context_padding=64,
        ),
    )
)
```

`KleinInferenceConfig` controls the shared edit/generation step count,
guidance scale, and context padding around masked generation.
The inference config defaults to 4 steps and guidance scale `1.0`, matching
Black Forest Labs' FLUX.2 Klein 4B example.
Masked generation crops around the selected region with 64 pixels of context by
default, then overlays the generated crop only inside the selected mask.
`Sam3InferenceConfig` controls segmentation mask thresholding.
`ObjectClearInferenceConfig` controls
removal steps, guidance, and minimum working image side. The project-wide
`RANDOM_SEED` controls stochastic operations. `add_object` uses Klein to
generate a contextual crop and composites it only inside the supplied mask or
the rectangular mask built from its box.
Its box uses `[x1, y1, x2, y2]` coordinates with the lower-right edge exclusive.
The Klein base 4B model card reports approximately 13 GB VRAM for GPU use; Apple
Silicon uses MPS with float16, but speed and memory usage depend on the Mac.
Review each checkpoint's license before use.

`apply_filter` works on the full image by default. Pass a mask to filter only one selected object; the result is blended softly at the mask edge while pixels outside the mask remain unchanged.

`edit_object` requires one of three modes:

- `local`: make appearance-only edits such as color, material, or pattern; preserve the object's silhouette. The model sees the object with some context.
- `regional`: permit additions and shape/size changes within an expanded neighborhood. The model sees a crop with padding around the object. `padding` defaults to `0.25`, meaning 25% of the mask bounding-box width and height on each side; use this for edits like adding a hat.
- For `local` and `regional`, the pasted region is the original mask plus any significant change the model made that touches it (found by a noise-aware before/after color difference) inside that neighborhood, feathered at the edge; the rest of the image stays original.
- `global`: send the whole image to the model and return its full result, without mask compositing. Use this when surrounding scene changes are intended; unrelated areas may also change.

Example: `tool_dict["edit_object"](image=image, mask=mask, prompt="add a hat", mode="regional", padding=0.25)`.

See the [FLUX.2 Klein base 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and [Diffusers FLUX.2 guide](https://huggingface.co/docs/diffusers/api/pipelines/flux2) for pipeline and hardware details.

## Benchmark the Agent

`MMEDIT_COMPBENCH_OUTPUT_DIR`, `MMEDIT_REFEDIT_BENCH_OUTPUT_DIR`, and `MMEDIT_MIRA_BENCH_OUTPUT_DIR`, when set, override the benchmark output root; the selected `text` or `mixed` subdirectory is appended automatically.

Benchmark helpers live in [`src/evaluation/`](./src/evaluation/). Run either the [CompBench text-prompt notebook](./src/notebooks/inference/compbench_inference_text.ipynb) or [CompBench mixed-prompt notebook](./src/notebooks/inference/compbench_inference_mixed.ipynb), and likewise the [RefEdit-Bench text-prompt notebook](./src/notebooks/inference/refedit_bench_inference_text.ipynb) or [mixed-prompt notebook](./src/notebooks/inference/refedit_bench_inference_mixed.ipynb), or the [MIRA-Bench text-prompt notebook](./src/notebooks/inference/mira_bench_inference_text.ipynb) or [mixed-prompt notebook](./src/notebooks/inference/mira_bench_inference_mixed.ipynb). Text mode uses each annotation's `original_instruction`; mixed mode uses `mixed_instruction` with visual-reference tokens. CompBench and RefEdit-Bench use `hf://` references and original dataset row indices. MIRA-Bench is file-based; its notebooks download the dataset image snapshot and resolve the image filenames recorded in annotations. By default, inference selects the latest annotation run containing the expected JSONL under `.out/annotation/<benchmark>/run_*/`; MIRA-Bench also falls back to the latest legacy run under `.out/annotation/mirage/` when no canonical MIRA-Bench run exists. Set `MMEDIT_COMPBENCH_ANNOTATION_RUN`, `MMEDIT_REFEDIT_BENCH_ANNOTATION_RUN`, or `MMEDIT_MIRA_BENCH_ANNOTATION_RUN` to a specific `run_<timestamp>`, or the corresponding `MMEDIT_*_ANNOTATIONS` variable to an explicit JSONL path. `ANNOTATE_OUT` selects a non-default annotation root. Each inference execution creates a separate `.out/evaluation/<benchmark>/<text|mixed>/run_<timestamp>/` directory containing `logs/` and `results/`; metrics are created separately under the same run only when the metrics workflow runs. All notebooks run all annotations by default and display a concise tqdm progress bar. Visual prompts are selected reproducibly from the required `RANDOM_SEED` environment variable, combined with each sample ID so a sample keeps its prompt choice across individual, subset, and full runs. Copy `.env.example` to `.env`, set `RANDOM_SEED`, then set `MMEDIT_API_KEY` or `GOOGLE_API_KEY`.

For standalone CompBench inference, use `uv run python -m scripts.compbench_inference_mixed` for mixed prompts or `uv run python -m scripts.compbench_inference_text` for text-only prompts. The mixed script accepts `--tasks add replace` to filter by `metadata.task`; without it, all tasks are selected. The [text-prompt notebook](./src/notebooks/inference/compbench_inference_text.ipynb) uses `TASKS = None` for all tasks or a set such as `{"add", "replace"}`. In both cases the start index refers to the original annotation order, and the limit is applied after task filtering. Pass `--annotations /path/to/annotations.jsonl` or `--annotation-run run_<timestamp>` to choose the input; `--limit 3` is useful for a short smoke test, and `--start-index N` selects its starting annotation index. Each normal invocation creates a fresh run in the matching `mixed` or `text` output directory. Pass `--resume` to continue the latest run, or `--resume --run run_<timestamp>` to resume a specific one; these skip completed samples by default. To overwrite results from index N onward in an existing run while preserving earlier indices, use `--resume --run run_<timestamp> --start-index N --rerun-completed`. This reruns every sample in the selected range after deleting that sample's old log directory and generated source, annotated, output, and target images; it records a `running` result before executing so an interrupted retry is not mistaken for the old completed result. Use `--require-cuda` on NVIDIA servers to fail before loading models if CUDA is unavailable. The scripts can also be invoked from the self-extracting bundle, for example `./evaluation.run -m scripts.compbench_inference_text --limit 3 --require-cuda`.

To create a retry list for a CompBench run, use `uv run python -m scripts.compbench_incomplete_samples --prompt-mode mixed --run run_<timestamp> --output retry_samples.txt` for mixed mode or `--prompt-mode text` for text mode (the default is mixed). It writes `metadata.row_index` values for missing results, non-completed statuses, or completed results whose output image is missing. These dataset row indices can be sparse and are matched back to annotation records when the retry list is run. Then run only those annotations with the matching inference script, for example `uv run python -m scripts.compbench_inference_mixed --samples-file retry_samples.txt --resume --run run_<timestamp> --rerun-completed` or `uv run python -m scripts.compbench_inference_text --samples-file retry_samples.txt --resume --run run_<timestamp> --rerun-completed`. The sample file accepts one row index per line and `#` comments; it is mutually exclusive with task, start-index, and limit filters. `--rerun-completed` ensures listed entries are rerun even if the results manifest says completed.

For standalone RefEdit-Bench inference, use `uv run python -m scripts.refedit_bench_inference_mixed` for mixed prompts (or `uv run python -m scripts.refedit_bench_inference`) and `uv run python -m scripts.refedit_bench_inference_text` for text-only prompts. Pass `--annotations /path/to/annotations.jsonl` or `--annotation-run run_<timestamp>` to choose the input; `--limit 3` is useful for a short smoke test, and `--start-index N` selects its starting annotation index. Each normal invocation creates a fresh run under `.out/evaluation/refedit_bench/<mixed|text>`. Pass `--resume` to continue the latest run, `--resume --run run_<timestamp>` to resume a specific one, and `--resume --run run_<timestamp> --start-index N --rerun-completed` to overwrite results from index N onward. In mixed mode, pass `--resume --rerun-spatial-selector-issues` to rescan the run and rerun only completed samples that used a point or scribble selector for spatial placement.

For MIRA-Bench Replicate inference, run `uv run python -m scripts.mira_bench_inference_replicate`. By default it processes the full benchmark with the four configured Replicate models and creates a timestamped run directory under `.out/evaluation/mira_bench/run_<timestamp>/`. Select samples by zero-based line positions in `benchmark/annotations.jsonl` with `--sample-indices 10 30 4 57`, or use stable annotation IDs with `--sample-ids mirage:<image-path> ...`. Add `--models bagel --dry-run` to prepare the selected samples without creating paid predictions. Set `REPLICATE_API_TOKEN` before a live run. Resume with `--resume --run run_<timestamp>` (or `--resume` to select the latest run); already succeeded results are skipped. An optional `--dataset-dir` accepts either a local snapshot containing `benchmark/` or the `benchmark/` folder itself. Records, source images, selection manifest, inference manifest, and summary are kept in the run directory.

The [RefEdit-Bench text metrics notebook](./src/notebooks/metrics/refedit_bench_metrics_text.ipynb) and [mixed metrics notebook](./src/notebooks/metrics/refedit_bench_metrics_mixed.ipynb) score saved runs without rerunning the agent. Each selects the latest run in its own prompt family by default; set `MMEDIT_REFEDIT_BENCH_TEXT_RUN` or `MMEDIT_REFEDIT_BENCH_MIXED_RUN` to select a specific run. Per-sample scores are written to `<run>/metrics/modified_viescore.jsonl`, separate from `<run>/results/`. The matching command-line entrypoints are `uv run python -m evaluation.refedit_bench_metrics_text` and `uv run python -m evaluation.refedit_bench_metrics_mixed`; pass `--run run_<timestamp>` to select a run explicitly. Their shared implementation is in [`src/evaluation/refedit_bench_viescore.py`](./src/evaluation/refedit_bench_viescore.py), with mask, crop, and score-summary helpers in [`src/evaluation/metrics.py`](./src/evaluation/metrics.py). Scoring uses the original `metadata.edit_instruction_single`, the GT polygon mask, and a tight shared ROI crop before scoring with VIEScore `task="tie"` and GPT-4o. Configure `VIESCORE_ROOT` and `VIESCORE_KEY_PATH` to enable scoring; keep the key file local and uncommitted.

The [CompBench mixed metrics notebook](./src/notebooks/metrics/compbench_metrics_mixed.ipynb) and [text metrics notebook](./src/notebooks/metrics/compbench_metrics_text.ipynb) score existing runs using the five automated metrics from the CompBench paper and released evaluator: LC-T, LC-I, PSNR, SSIM, and LPIPS. They select the latest run by default; set `MMEDIT_COMPBENCH_MIXED_RUN` or `MMEDIT_COMPBENCH_TEXT_RUN` to choose one. Scores are saved under that inference run's `metrics/` directory. The evaluator implementation is [`src/evaluation/compbench_metrics.py`](./src/evaluation/compbench_metrics.py); install its optional dependency with `uv sync --extra metrics` if needed. Both notebooks also provide modified VIEScore **only** for records with task `multi_object_add` or `multi_object_remove` (excluding the single-object `add`/`remove` and `replace` tasks), using the original instruction and ground-truth edit-mask crop. The [CompBench Replicate VIEScore notebook](./src/notebooks/metrics/compbench_metrics_replicate.ipynb) scores successful per-model records under `.out/evaluation/compbench/replicate`, reading its local selection manifest and source/mask files; scores are saved under its `metrics/` directory and summarized by model and task. These VIEScore workflows use the GEdit-Bench `tie` task with GPT-4o, may incur API charges, and require `VIESCORE_ROOT` and `VIESCORE_KEY_PATH`; keep the API key file local and uncommitted.

Both the mixed and Replicate notebooks also run [EditScore](https://github.com/VectorSpaceLab/EditScore) after VIEScore, scoring only `multi_object_add` and `multi_object_remove` with the full source/output images and original instruction. Install the official package into the notebook environment with `pip install -U editscore`; the default local model is `Qwen/Qwen3-VL-4B-Instruct` with `EditScore/EditScore-Qwen3-VL-4B-Instruct`, downloaded from Hugging Face on first use. The scorer reports normalized 0–10 prompt-following, consistency, perceptual-quality, and overall scores. On Apple Silicon, MPS is selected automatically when CUDA is unavailable and the Qwen model is loaded in float16; set `EDITSCORE_DEVICE=mps` to request MPS explicitly. The model requires substantial unified memory. Configure `EDITSCORE_BACKBONE`, `EDITSCORE_MODEL`, `EDITSCORE_LORA`, `EDITSCORE_NUM_PASS`, and `EDITSCORE_SCORE_RANGE` to change the model or self-ensemble settings. Results and cached per-sample scores are stored separately in `metrics/editscore.jsonl` and `metrics/editscore_summary.json`; Replicate summaries are grouped by model and task. The default uses local model inference rather than a paid scoring API. The separate paper-reported CompBench VLM/human ratings for action, location, and viewpoint are not inferred by these automated metrics.

Use the [CompBench metrics debug notebook](./src/notebooks/metrics/compbench_metrics_debug.ipynb) to list the lowest-performing samples independently for SSIM, PSNR, LPIPS, LC-T, and LC-I. Each sample includes input and annotated-input previews, the target/reference and generated output, annotation details, an expandable cleaned trajectory, and the raw message list supplied to the LLM for each model call. Inference runs save the uncleaned per-step agent trace, including model input messages and model outputs, to `logs/<sample>/agent_raw_trajectory.json`; `raw.json` also retains the raw steps alongside other debug metadata. Its replay cell accepts a sample ID and reruns the saved tool-call code without asking the language model to plan again; image-editing models may need to be loaded. Set `TOP_N`, `PROMPT_MODE`, and optionally `TASK_FILTER` in its setup cell.

Inference result records also include `elapsed_seconds` for total per-sample processing, `agent_elapsed_seconds` for the agent run, `agent_timing_totals` for summed prompt assembly/LLM request/tool durations, and `agent_timings` with each prompt assembly, LLM generation request, and individual tool call. LLM request duration includes client request preparation, provider/network latency, retries, and response parsing; it is not a measurement of provider-only inference time.

Use the [RefEdit-Bench text vs. mixed comparison notebook](./src/notebooks/metrics/refedit_bench_compare_text_mixed.ipynb) to pair saved ViEScore rows by sample, report per-metric wins, and visualize text-winning cases with mixed visual-prompt annotations. Expand each case's text or mixed agent trace to inspect its observations, thoughts, actions, and execution results. It selects the latest run in each prompt family by default; the same run environment variables above can select specific runs.

## Annotation and Visualization

MIRA-Bench is distributed through the file-based Hugging Face dataset repository `ziqiangoodgood/MIRAGE` rather than a table split. Its notebook downloads the image files and official JSONL directly, then uses the shared annotation schema and resumable runner.

For MIRA-Bench, each edit mask receives a mask-conditioned LLM parse and one targeted retry if the result is empty, ambiguous, or cannot be aligned to the instruction.

The annotation pipeline is available as `annotation`; its benchmark notebooks share settings, schema checks, resumable conversion, and audit handling through `annotation.workflows`. Visualization/browser helpers are in `visualization`. Use `uv sync --extra visualize` for notebook plotting dependencies. Run annotations with the [CompBench](./src/notebooks/compbench_annotation.ipynb), [MagicBrush](./src/notebooks/magicbrush_annotation.ipynb), [RefEdit-Bench](./src/notebooks/annotation/refedit_bench_annotation.ipynb), or [MIRA-Bench](./src/notebooks/annotation/mira_bench_annotation.ipynb) annotation notebook. Each annotation notebook creates a separate `.out/annotation/<benchmark>/run_<timestamp>/` folder by default; set `RESUME_LAST_RUN = True` in its setup/run cell to continue the latest run, skipping completed samples and retrying audited failures. MIRA-Bench's current release contains the composite instruction and five ordered polygon masks, not `refer_object` or `bbox` fields; its LLM parser runs against the full instruction once per highlighted edit mask. CompBench multi-object add/remove partitions connected masks using projection, distance-watershed, and spatial clustering, and parses each partition separately. Set `RETRY_SAMPLE_IDS` in that notebook to replace specific accepted-but-incorrect annotations. Set `ANNOTATE_OUT` to use a different output root. For RefEdit, `MASK_FROM_START` in the setup cell overrides `REFEDIT_MASK_FROM_START` from `.env`; when false, parsing starts text-only, but conditional mask-guided retry still applies when the text parse is ambiguous or yields no usable reference. Copy `.env.example` to `.env`, set `RANDOM_SEED`, and set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); optional settings control limits, streaming, grounding thresholds, split, and output location. Explore annotations with the matching [CompBench](./src/notebooks/compbench_visualization.ipynb), [MagicBrush](./src/notebooks/magicbrush_visualization.ipynb), or [RefEdit-Bench](./src/notebooks/refedit_bench_visualization.ipynb) visualization notebook. [Annotation statistics](./src/notebooks/annotations_statistics.ipynb) summarizes the generated annotations; [tools_demo.ipynb](./src/notebooks/tools_demo.ipynb) demonstrates the editing tools. Generated JSONL/assets are kept locally in `.out/annotation/` (ignored by Git). Sync audit entries against successful records and sort by original dataset row with `uv run python -m scripts.sync_annotations compbench` (also accepts `magicbrush`, `refedit_bench`, and `mira_bench`); it previews by default, and `--apply` writes both changes. Backfill visual prompt geometry for a benchmark with `uv run python -m scripts.backfill_visual_prompts compbench` (also accepts `magicbrush`, `refedit_bench`, and `mira_bench`).

For mask-guided RefEdit annotations, the mask may identify a relational anchor rather than the object being edited. In that case the parser marks the masked person/entity as an `anchor` and leaves the separately named edit target in the instruction; directly masked edit targets remain `target` references. Replacement instructions use the existing source object as the reference, not the requested replacement. If a masked retry returns no source-present entity, an unambiguous text-first reference is retained; otherwise the audit records each parse mode, entity count, span, role, and source-presence flag.

## Register a Tool

Register a Python handler with its model-facing schema:

```python
from PIL import ImageOps
from framework.tools import register_tool

@register_tool(
    "invert_image",
    "Invert the image colors.",
    {"image": {"type": "image", "description": "Input image."}},
    "image",
)
def invert_image(models, image):
    return ImageOps.invert(image.convert("RGB"))
```

Registered handlers receive a `ToolContext` containing the shared `model_pool`, followed by their declared inputs. This keeps local image operations independent of inference models while model-backed tools reuse loaded pipelines.

## Security

The default local executor is restricted but is not a security sandbox. Use an isolated `smolagents` executor for untrusted generated code.
