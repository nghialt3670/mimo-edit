# MMEdit Framework

A multi-turn image-editing agent built on `smolagents`, with Pillow images and masks as its entity representation. Its installable packages live under `src/`: the editing framework is `framework`, annotations are `annotation`, benchmark evaluation is `evaluation`, and plotting/browser helpers are `visualization`. Notebook work lives in `src/notebooks`; maintenance commands live in `src/scripts`.

- `framework.core` owns agent orchestration and conversation state.
- `framework.tools` owns registered editing operations and their Pillow-facing APIs.

## Install

Install the framework and its optional model stack:

```bash
uv sync --extra models
```

Models load on first use. Run the [model warmup notebook](./src/notebooks/model_warmup.ipynb) before inference to download and load the default tool backbones in sequence; weights remain in the Hugging Face cache, while inference still loads model objects into its own kernel. Gated Hugging Face models require `HF_TOKEN`.

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

Pass Pillow `Image` instances and grayscale Pillow masks as objects. White mask pixels select the entity. Object-targeting tools accept one mask/object per call; issue another tool call for each additional object. `segment_object` accepts exactly one text prompt, point, box, or scribble and returns one mask; text prompts return the highest-scoring match. `generate_object` generates or transforms content in a masked or boxed region with FLUX.2 Klein, optionally guided by a phrase. Models load lazily and are shared through the tool set's model cache.

For every registered image-editing tool's inputs, behavior, and default backbone, see the detailed [tool reference](./docs/TOOLS.md).

ObjectClear is provided under its upstream S-Lab license, which restricts use to non-commercial purposes. Review the license before enabling that model.

## Configure Models

Model selection is independent of tool registration. `edit_object` defaults to FLUX.2 Klein 4B:

```python
from framework import ModelConfig, image_editing_tools

tools = image_editing_tools()
```

`edit_object` and `generate_object` use FLUX.2 Klein 4B (`black-forest-labs/FLUX.2-klein-4B`) by default. Both condition on image content and an edit prompt and run four denoising steps; `generate_object` edits a crop around the selected region and composites the result back into the image. The model card reports approximately 13 GB VRAM for GPU use; Apple Silicon uses MPS with float16, but speed and memory usage depend on the Mac. Both tools share the `edit_object` model/backend configuration. Other tools use ObjectClear for removal/transforms and SAM3 for segmentation. For custom/shared caches, pass `ModelCache(ModelConfig(...))` as the first argument to `image_editing_tools`. Review each checkpoint's license before use.

`apply_filter` works on the full image by default. Pass a mask to filter only one selected object; the result is blended softly at the mask edge while pixels outside the mask remain unchanged.

`edit_object` requires one of three modes:

- `local`: make appearance-only edits such as color, material, or pattern; preserve the object's silhouette and composite within its mask at full edit strength, with a soft transition outside the mask.
- `regional`: permit additions and shape/size changes within an expanded neighborhood. The model sees a crop with padding around the object, and the dilated mask neighborhood is composited at full edit strength with a softly feathered transition beyond it. `padding` defaults to `0.25`, meaning 25% of the mask bounding-box width and height on each side; use this for edits like adding a hat.
- `global`: send the whole image to the model and return its full result, without mask compositing. Use this when surrounding scene changes are intended; unrelated areas may also change.

Example: `tool_dict["edit_object"](image=image, mask=mask, prompt="add a hat", mode="regional", padding=0.25)`.

See the [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and [Diffusers FLUX.2 guide](https://huggingface.co/docs/diffusers/api/pipelines/flux2) for pipeline and hardware details.

## Benchmark the Agent

Benchmark helpers live in [`src/evaluation/`](./src/evaluation/). The [CompBench inference](./src/notebooks/compbench_inference.ipynb) and [RefEdit-Bench inference](./src/notebooks/refedit_bench_inference.ipynb) notebooks use shared path/configuration and OpenAI-compatible model setup from `evaluation.notebook`, then load `hf://` images using their original dataset row indices. They read annotation outputs generated by the project: `.out/annotation/compbench/compbench_local_multiobject_mixed.jsonl` and `.out/annotation/refedit_bench/refedit_bench_mixed.jsonl`; run the corresponding annotation notebook first. Optional `MMEDIT_COMPBENCH_ANNOTATIONS` and `MMEDIT_REFEDIT_BENCH_ANNOTATIONS` variables can override those files. Each inference execution creates a separate `.out/evaluation/<benchmark>/run_<timestamp>/` directory containing `logs/` and `results/`; metrics are created separately under the same run only when the metrics workflow runs. Both run all annotations by default and display a concise tqdm progress bar; use [CompBench debug](./src/notebooks/compbench_debug.ipynb) to inspect one representative sample from each task type. Visual prompts are selected reproducibly with `RANDOM_SEED = 42`; selection combines the seed with each sample ID so a sample keeps its prompt choice across individual, subset, and full runs. Install with `uv sync --extra models --extra benchmark`, copy `.env.example` to `.env`, then set `MMEDIT_API_KEY` or `GOOGLE_API_KEY`.

The standalone [RefEdit-Bench metrics notebook](./src/notebooks/refedit_bench_metrics.ipynb) scores a selected saved run without rerunning the agent; it defaults to the latest run, or can select one with `MMEDIT_REFEDIT_BENCH_RUN`. Per-sample scores are written to `<run>/metrics/modified_viescore.jsonl`, separate from `<run>/results/`. Its implementation is in [`src/evaluation/refedit_bench_viescore.py`](./src/evaluation/refedit_bench_viescore.py). It uses the original `metadata.edit_instruction_single`, the GT polygon mask, and a tight shared ROI crop before scoring with VIEScore `task="tie"` and GPT-4o. Configure `VIESCORE_ROOT` and `VIESCORE_KEY_PATH` to enable scoring; keep the key file local and uncommitted. Run the evaluator from the repository root with `uv run python -m evaluation.refedit_bench_viescore`.

## Annotation and Visualization

The annotation pipeline is available as `annotation`; its benchmark notebooks share settings, Hugging Face loading, schema checks, resumable conversion, and audit handling through `annotation.workflows`. Visualization/browser helpers are in `visualization`. Use `uv sync --extra visualize` for notebook plotting dependencies. Run annotations with the [CompBench](./src/notebooks/compbench_annotation.ipynb), [MagicBrush](./src/notebooks/magicbrush_annotation.ipynb), or [RefEdit-Bench](./src/notebooks/refedit_bench_annotation.ipynb) annotation notebook. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`); optional settings control limits, streaming, grounding thresholds, split, and output location. Explore annotations with the matching [CompBench](./src/notebooks/compbench_visualization.ipynb), [MagicBrush](./src/notebooks/magicbrush_visualization.ipynb), or [RefEdit-Bench](./src/notebooks/refedit_bench_visualization.ipynb) visualization notebook. [Annotation statistics](./src/notebooks/annotations_statistics.ipynb) summarizes the generated annotations; [tools_demo.ipynb](./src/notebooks/tools_demo.ipynb) demonstrates the editing tools. Generated JSONL/assets are kept locally in `.out/annotation/` (ignored by Git). Sync audit entries against successful records and sort by original dataset row with `uv run python -m scripts.sync_annotations compbench` (also accepts `magicbrush` and `refedit_bench`); it previews by default, and `--apply` writes both changes. Backfill visual prompt geometry for a benchmark with `uv run python -m scripts.backfill_visual_prompts compbench` (also accepts `magicbrush` and `refedit_bench`).

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

Handlers receive the shared model cache first, followed by their declared inputs. This keeps local Pillow tools simple while model-backed tools reuse loaded pipelines.

## Security

The default local executor is restricted but is not a security sandbox. Use an isolated `smolagents` executor for untrusted generated code.
