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

ObjectClear is provided under its upstream S-Lab license, which restricts use to non-commercial purposes. Review the license before enabling that model.

## Configure Models

Model selection is independent of tool registration. `edit_object` defaults to FLUX.2 Klein 4B:

```python
from tools import ModelConfig, image_editing_tools

tools = image_editing_tools()
```

`edit_object` uses only the FLUX.2 Klein 4B Diffusers pipeline, `black-forest-labs/FLUX.2-klein-4B`. It conditions on the source crop and edit prompt and runs four denoising steps. The model card reports approximately 13 GB VRAM for GPU use; Apple Silicon uses MPS with float16, but speed and memory usage depend on the Mac. `model_ids` can override checkpoint IDs by tool name (the GLIGEN generation route uses `generate_object_gligen`). Other tools use Stable Diffusion inpainting for `generate_object`, ObjectClear for removal/transforms, and SAM3 for segmentation. For custom/shared caches, pass `ModelCache(ModelConfig(...))` as the first argument to `image_editing_tools`. Review each checkpoint's license before use.

`apply_filter` works on the full image by default. Pass a mask to filter only one selected object; the result is blended softly at the mask edge while pixels outside the mask remain unchanged.

`edit_object` requires one of three modes:

- `local`: make appearance-only edits such as color, material, or pattern; preserve the object's silhouette and composite only within its mask.
- `regional`: permit additions and shape/size changes within an expanded neighborhood. The model sees a crop with padding around the object, and only the dilated mask neighborhood is composited back. `padding` defaults to `0.25`, meaning 25% of the mask bounding-box width and height on each side; use this for edits like adding a hat.
- `global`: send the whole image to the model and return its full result, without mask compositing. Use this when surrounding scene changes are intended; unrelated areas may also change.

Example: `tool_dict["edit_object"](image=image, mask=mask, prompt="add a hat", mode="regional", padding=0.25)`.

See the [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) and [Diffusers FLUX.2 guide](https://huggingface.co/docs/diffusers/api/pipelines/flux2) for pipeline and hardware details.

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
