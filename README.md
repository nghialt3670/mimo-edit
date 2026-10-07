# MMEdit Framework

A multi-turn image-editing agent built on `smolagents`, with Pillow images and masks as its entity representation. The project has three packages:

- `core` owns agent orchestration and conversation state.
- `tools` owns registered editing operations and their Pillow-facing APIs.

## Install

Install the framework and its optional model stack:

```bash
uv sync --extra models
```

Models load on first use. Gated Hugging Face models require `HF_TOKEN`.

## Use the Tools In-Process

Tools invoke the model pipelines directly; there is no inference service or separate server process.

```python
from core import Agent
from tools import image_editing_tools

tools = image_editing_tools()
agent = Agent(model=model, tools=tools)

result = agent.send(
    "Remove the object selected by <box> and return the edited image.",
    objects={"image_0": image, "box_0": [120, 80, 340, 310]},
)
edited = result.message.objects[0]
```

Pass Pillow `Image` instances and grayscale Pillow masks as objects. White mask pixels select the entity. Object-targeting tools accept one mask/object per call; issue another tool call for each additional object. `segment_object` accepts exactly one text prompt, point, box, or scribble and returns one mask; text prompts return the highest-scoring match. `generate_object` fills or generates content in one masked or boxed region, and can also place one phrase in a box with GLIGEN. Models load lazily and are shared through the tool set's model cache.

For every registered image-editing tool's inputs, behavior, and default backbone, see the detailed [tool reference](./docs/TOOLS.md).

ObjectClear is provided under its upstream S-Lab license, which restricts use to non-commercial purposes. Review the license before enabling that model.

## Configure Models

Model selection is independent of tool registration. `edit_object` defaults to FLUX.2 Klein 4B:

```python
from tools import ModelConfig, image_editing_tools

tools = image_editing_tools()
```

`edit_object` uses only the FLUX.2 Klein 4B Diffusers pipeline, `black-forest-labs/FLUX.2-klein-4B`. It conditions on the source crop and edit prompt and runs four denoising steps. The model card reports approximately 13 GB VRAM for GPU use; Apple Silicon uses MPS with float16, but speed and memory usage depend on the Mac. `generate_object` uses GLIGEN for phrase-plus-box placement with scheduled-sampling beta 0.3, and Stable Diffusion inpainting for mask/box fills. `model_ids` can override checkpoint IDs by tool name (the GLIGEN route uses `generate_object_gligen`). Other tools use ObjectClear for removal/transforms and SAM3 for segmentation. For custom/shared caches, pass `ModelCache(ModelConfig(...))` as the first argument to `image_editing_tools`. Review each checkpoint's license before use.

`apply_filter` works on the full image by default. Pass a mask to filter only one selected object; the result is blended softly at the mask edge while pixels outside the mask remain unchanged.

`edit_object` requires one of three modes:

- `local`: make appearance-only edits such as color, material, or pattern; preserve the object's silhouette and composite within its mask at full edit strength, with a soft transition outside the mask.
- `regional`: permit additions and shape/size changes within an expanded neighborhood. The model sees a crop with padding around the object, and the dilated mask neighborhood is composited at full edit strength with a softly feathered transition beyond it. `padding` defaults to `0.25`, meaning 25% of the mask bounding-box width and height on each side; use this for edits like adding a hat.
- `global`: send the whole image to the model and return its full result, without mask compositing. Use this when surrounding scene changes are intended; unrelated areas may also change.

Example: `tool_dict["edit_object"](image=image, mask=mask, prompt="add a hat", mode="regional", padding=0.25)`.

See the [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and [Diffusers FLUX.2 guide](https://huggingface.co/docs/diffusers/api/pipelines/flux2) for pipeline and hardware details.

## Benchmark the Agent

Benchmark helpers live in [`evaluation/common.py`](./evaluation/common.py). Use the dedicated [CompBench notebook](./evaluation/compbench.ipynb) or [RefEdit-Bench notebook](./evaluation/refedit_bench.ipynb) to load the corresponding JSONL annotations and fetch `hf://` images using their original split row indices. The agent receives the benchmark instruction and one box, point, or scribble prompt per reference, selected reproducibly from the available prompt types. Both notebooks set `RANDOM_SEED = 42`; the selection uses that seed together with the sample ID, so a sample keeps the same prompt selection when run alone, in a subset, or in a full run. Change `RANDOM_SEED` to explore a different selection. The seed is recorded in each result row and `raw.json`. The agent then uses `segment_object` to produce its own mask. Annotation masks and crops are not used. The RefEdit-Bench notebook runs the full annotation set by default; change `SAMPLE_INDICES` in its run cell to select a subset. The initial user observation contains only the benchmark instruction; reference descriptions, selector mappings, and editing guidance are passed separately as agent instructions. Each sample log directory includes trajectory/raw JSON and a `variables.json` snapshot of the agent namespace; images, masks, arrays, tensors, and byte values are saved as files under `debug_assets/` and referenced from that manifest. Edited/source/annotated/target images go to `results/images/`, with the run summary in `results/results.jsonl`. Install with `uv sync --extra models --extra benchmark`; by default, the notebooks read `~/Datasets/MMEdit/compbench/annotations.jsonl` and `~/Datasets/MMEdit/refedit-bench/annotations.jsonl`, respectively. Override with `MMEDIT_COMPBENCH_ANNOTATIONS` or `MMEDIT_REFEDIT_ANNOTATIONS`, and configure `MMEDIT_API_KEY` (or `GOOGLE_API_KEY`) before running. The notebooks report execution status and timing; compare saved outputs with benchmark targets separately for image-quality scoring.

Use the standalone [Modified VIEScore notebook](./evaluation/refedit_viescore.ipynb) to score existing RefEdit outputs without rerunning the editing agent. Its implementation lives in [`evaluation/refedit_viescore.py`](./evaluation/refedit_viescore.py). By default, it loads RefEdit outputs from `outputs/evaluation/refedit_bench`; set `MMEDIT_REFEDIT_OUTPUT_DIR` to override the output directory. The RefEdit benchmark notebook uses the same default output location. Scoring uses each annotation's original `metadata.edit_instruction_single`, decodes the RefEdit GT polygon mask, and applies the same zero-padding tight mask-bounding-box crop to source and generated images before calling VIEScore `task="tie"` with the GPT-4o API model. The current upstream `gpt4o` backbone defaults its API model name to GPT-4.1, so the evaluator explicitly sets the API model identifier to `gpt-4o`. To enable the optional scorer, check out [Step1X-Edit](https://github.com/stepfun-ai/Step1X-Edit), install its GEdit-Bench VIEScore dependencies, set `VIESCORE_ROOT` to that checkout's `GEdit-Bench` directory, and set `VIESCORE_KEY_PATH` to a local text file containing only the OpenAI API key on its first line. Do not commit the key file. The evaluator scores completed outputs from `results/results.jsonl` and also scans `results/images/*_output.png` so it can recover outputs omitted when that JSONL was overwritten by a later subset run. It saves per-sample scores to `results/metrics/modified_viescore.jsonl` and reports SC/PQ/O means separately for Easy and Hard; unchanged output files reuse cached scores. To run the same evaluator from the repository root without opening a notebook, run `uv run python -m evaluation.refedit_viescore`.

## Register a Tool

Register a Python handler with its model-facing schema:

```python
from PIL import ImageOps
from tools import register_tool

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
