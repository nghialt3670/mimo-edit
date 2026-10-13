# Image-editing tools

This document describes the tools registered by
[`src/framework/tools/image_editing/`](../src/framework/tools/image_editing/) and built by
[`image_editing_tools`](../src/framework/tools/__init__.py). The registry currently exposes
12 tools:

| Tool | Purpose | Backbone |
| --- | --- | --- |
| [`segment_object`](../src/framework/tools/image_editing/segment_object.py) | Find one object and return its mask | SAM 3 |
| [`segment_region`](../src/framework/tools/image_editing/segment_region.py) | Ground a text description and return its mask | Grounding DINO + SAM 3 |
| [`edit_object`](../src/framework/tools/image_editing/edit_object.py) | Semantically edit one masked object | FLUX.2 Klein base 4B |
| [`remove_object`](../src/framework/tools/image_editing/remove_object.py) | Remove one masked object | ObjectClear |
| [`replace_object`](../src/framework/tools/image_editing/replace_object.py) | Remove an object, then paste a replacement image | ObjectClear + Pillow |
| [`add_object`](../src/framework/tools/image_editing/add_object.py) | Generate an object inside a selected region | FLUX.2 Klein base 4B |
| [`move_object`](../src/framework/tools/image_editing/move_object.py) | Move one object to a point or box | ObjectClear + Pillow |
| [`shift_object`](../src/framework/tools/image_editing/shift_object.py) | Translate one object by an offset | ObjectClear + Pillow |
| [`rotate_object`](../src/framework/tools/image_editing/rotate_object.py) | Rotate one object | ObjectClear + Pillow |
| [`scale_object`](../src/framework/tools/image_editing/scale_object.py) | Resize one object around its center | ObjectClear + Pillow |
| [`flip_object`](../src/framework/tools/image_editing/flip_object.py) | Flip one object horizontally or vertically | ObjectClear + Pillow |
| [`apply_filter`](../src/framework/tools/image_editing/apply_filter.py) | Apply a visual filter globally or to one object | Pillow; no learned model |

Each backbone's checkpoint ID and loading logic live in its inference module;
inference settings are documented in
[Inference configuration and runtime](#inference-configuration-and-runtime).

## Shared interface conventions

- `image` and image-valued results are Pillow `Image` instances. The tools return
  a new image; they do not mutate the input image in place.
- A mask is a Pillow image. Its nonzero/white area selects the object or region.
  Masks are resized to the source image size by the relevant image utilities.
- Boxes are pixel coordinates `[x1, y1, x2, y2]` in the source image. Points and
  offsets use pixel coordinates too; positive x moves right and positive y moves
  down. Box coordinates must be finite, ordered, and inside the image.
- Tools operate on one selected object per call. For multiple objects, segment
  and edit/remove each object in separate calls, using the latest returned image
  as the next call's input.
- `segment_object` creates masks. Other object-targeting tools consume a mask;
  they do not automatically identify the object from a text description.
- `segment_region` grounds a natural-language description to a box with
  Grounding DINO, then uses SAM 3 to return a pixel mask for that region.
- Model-backed tools use one eagerly initialized `ModelPool`. Constructing it
  loads Klein, SAM 3, and ObjectClear together, so confirm that the host has
  enough memory before creating the tool set. Grounding DINO loads only when
  `segment_region` is first used. Reuse the pool across calls.

## Tool interfaces and behavior

### `segment_object`

**Backbone:** SAM 3.

**Inputs**

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | Image | Yes | Source image. |
| `prompt` | String or null | No | Text description of the object to find. |
| `point` | Array or null | No | One pixel point `[x, y]` inside/on the selected object. |
| `box` | Array or null | No | Pixel box `[x1, y1, x2, y2]` around the selection. |
| `scribble` | Array or null | No | Pixel path `[[x, y], ...]` over the object. |

Provide **exactly one** selector: `prompt`, `point`, `box`, or `scribble`.
Text prompting runs SAM 3 instance segmentation, sorts candidate masks by
confidence, and returns the highest-scoring one. Point, box, and scribble
selection use SAM 3's promptable tracker; scribbles are reduced to at most ten
positive points sampled along the path. The output is a single-channel mask
where the selected region is white.

An empty/whitespace-only text prompt, missing selector, multiple selectors,
malformed point, empty scribble, or a text prompt with no matching object raises
an error. If text segmentation finds no match, retry with a concise visible-object
description using `segment_region`, or select the object with a point, box, or
scribble; do not repeat near-identical text prompts.

**Backbone:** `facebook/sam3` by default. Both the text and visual-prompt
pathways share the lazily loaded SAM 3 model bundle.

### `segment_region`

**Backbone:** Grounding DINO (`IDEA-Research/grounding-dino-tiny`) for text
grounding followed by SAM 3 for mask refinement.

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | Image | Yes | Source image. |
| `prompt` | String | Yes | Natural-language description of the object or region. |

Grounding DINO selects the highest-scoring matching box using the configured
box and text thresholds. SAM 3 then refines that box into a single-channel
white-on-black mask. If no region matches, the tool raises an error. The
Grounding DINO model bundle loads lazily on first use and is reused afterward.

### `edit_object`

**Inputs**

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | Image | Yes | Source image. |
| `mask` | Image | Yes | White pixels select the object. Must match image dimensions. |
| `prompt` | String | Yes | Complete description of the desired transformation. |
| `mode` | String | Yes | `local`, `regional`, or `global`. |
| `padding` | Number or null | No | Regional padding from 0 to 1; defaults to `0.25`. |

Write `prompt` as an actionable edit, not a category label: say what the
selected object should look like afterward, include requested attributes, and
preserve unmentioned details. For example, use “Change the selected car to
bright red, keeping its shape and the surrounding scene unchanged” rather than
“a red car.”

The image-edit backend receives a crop and the composed editing instruction:

- **`local`** is intended for appearance-only edits (color, material, pattern).
  The source crop is masked around the selected object. The edit is composited
  at full strength within the mask, with a soft transition just outside its
  boundary; the object's geometry should remain unchanged.
- **`regional`** allows shape changes or nearby additions such as accessories.
  The selected mask is expanded by `padding` times its bounding-box width and
  height on each side, and the corresponding padded crop is edited. The
  expanded region is composited at full strength with a soft transition beyond
  it. Increasing padding gives the model more room but can change more nearby
  content.
- **`global`** sends the entire image with the selected object's approximate
  bounds described in the prompt. It returns the full edited image without
  mask-based compositing, so unrelated areas can change.

**Backbone:** `black-forest-labs/FLUX.2-klein-base-4B`, loaded through
Diffusers' `Flux2KleinPipeline`. Editing uses 50 inference steps and guidance
scale `4.0`. Stochastic operations use the required project `RANDOM_SEED` from
the environment. The base checkpoint is slower than the
distilled model, but is better suited to prompt-guided inpainting. Both editing
and generation use this same backend; there is no alternate model backend.

### `remove_object`

**Inputs:** `image` (Image), `mask` (Image); both required.

Runs ObjectClear with the fixed instruction “remove the selected object,” using
the white mask as the removal region. The mask is resized with nearest-neighbor
sampling; the image and mask are resized for the model and the result is
resized back to the original image dimensions.

**Backbone:** `jixin0101/ObjectClear` by default. It uses eight inference steps,
guidance scale `2.0` and a minimum working image side of 384 pixels
by default. ObjectClear is distributed under its upstream S-Lab license, which
restricts use to non-commercial purposes; review the included license before use.

### `replace_object`

**Inputs**

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | Image | Yes | Source image. |
| `mask` | Image | Yes | Selects the object to remove. |
| `replacement` | Image | Yes | Image to paste at the selected object's bounds. |

First removes the masked object using ObjectClear. It then calculates the
mask's rectangular bounds, resizes the replacement to exactly that width and
height, and pastes it at the bounds' top-left corner. The replacement is fitted
to the rectangle, not clipped to the mask's irregular shape. An empty mask
raises an error.

**Backbone:** ObjectClear (`jixin0101/ObjectClear`) for removal; resizing and
pasting use Pillow.

### `add_object`

**Inputs**

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | Image | Yes | Source image. |
| `prompt` | String | Yes | Description of the requested content or transformation. |
| `mask` | Image or null | No | White pixels select the region to edit. |
| `box` | Array or null | No | In-image pixel region `[x1, y1, x2, y2]` to edit; the lower-right edge is exclusive. |
| `phrase` | String or null | No | Object description for phrase-guided placement within `box`. |

There are two mutually exclusive routes:

1. **Region inpainting:** supply exactly one of `mask` or `box`, and omit
   `phrase`. A box must lie entirely inside the image and is converted to an
   exact rectangular mask. The prompt describes the object to insert.
2. **Phrase-guided placement:** supply `box` and `phrase`, and omit `mask`.
   The phrase is appended to the prompt and the box is converted to a mask.

The general `prompt` must be non-empty. A `phrase` without a box, a mask paired
with a phrase, both mask and box without a phrase, or neither mask nor box
raises an error.

**Backbone**

`add_object` uses the shared FLUX.2 Klein 4B editing pipeline. If a box is
supplied, the tool converts it to a rectangular mask with a softly feathered
inner edge (up to 6 pixels) to reduce hard seams while preserving pixels
outside the box. With a mask, the pipeline generates a contextual crop around
the selected region (64 pixels of context by default), then overlays the
generated crop only inside the mask. `KleinInferenceConfig` controls the step
count, guidance scale, and mask context padding.

### `move_object`

**Inputs:** `image` (Image), `mask` (Image), `target` (Array); all required.

The target is either a destination center point `[x, y]` or a destination box
`[x1, y1, x2, y2]`. The object is extracted as a masked RGBA layer. For a box,
the layer is scaled to fit inside the destination while preserving aspect ratio,
then centered. ObjectClear fills the original mask area before the layer is
pasted at its new position.

**Backbone:** ObjectClear (`jixin0101/ObjectClear`) for filling the source
location; extraction, optional resizing, and pasting use Pillow.

### `shift_object`

**Inputs:** `image` (Image), `mask` (Image), `offset` (Array); all required.

`offset` must be `[dx, dy]` in pixels. The object is extracted, ObjectClear
fills its original location, and the layer is pasted at the original location
plus the requested offset. A positive `dx` shifts right; positive `dy` shifts
down.

**Backbone:** ObjectClear (`jixin0101/ObjectClear`) for filling the source
location; translation and pasting use Pillow.

### `rotate_object`

**Inputs:** `image` (Image), `mask` (Image), `angle` (Number); all required.

Extracts the masked object, rotates it counterclockwise around its center using
Pillow's bicubic resampling, removes the source object with ObjectClear, then
pastes the rotated layer centered on the original center.

**Backbone:** ObjectClear (`jixin0101/ObjectClear`) for filling the source
location; rotation and pasting use Pillow.

### `scale_object`

**Inputs:** `image` (Image), `mask` (Image), `factor` (Number); all required.

`factor` must be greater than zero. The extracted object is resized by this
factor with Lanczos resampling, preserving aspect ratio and keeping its center
fixed. ObjectClear fills the original mask before the scaled layer is pasted.
A factor above `1` enlarges; a factor between `0` and `1` shrinks.

**Backbone:** ObjectClear (`jixin0101/ObjectClear`) for filling the source
location; scaling and pasting use Pillow.

### `flip_object`

**Inputs:** `image` (Image), `mask` (Image), `axis` (String); all required.

`axis="x"` flips horizontally (left-right); `axis="y"` flips vertically
(top-bottom). The flipped object preserves its original position. ObjectClear
fills the source mask before the flipped layer is pasted. Any other axis raises
an error.

**Backbone:** ObjectClear (`jixin0101/ObjectClear`) for filling the source
location; flipping and pasting use Pillow.

### `apply_filter`

**Inputs**

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | Image | Yes | Source image. |
| `filter_name` | String | Yes | One supported filter name. |
| `value` | Number | Yes | Strength from `-1` to `1`; `0` gives each filter's default. |
| `mask` | Image or null | No | Optional white selection for one object. |

Supported filters:

| `filter_name` | Effect |
| --- | --- |
| `blackWhite` | Convert to grayscale RGB. |
| `invert` | Invert RGB colors. |
| `blur` | Gaussian blur; radius is `max(0, (value + 1) * 5)`. |
| `brightness` | Pillow brightness enhancer; factor is `1 + value`. |
| `contrast` | Pillow contrast enhancer; factor is `1 + value`. |
| `saturation` | Pillow color enhancer; factor is `1 + value`. |

Without a mask, the filter applies to the whole image. With a mask, the filtered
image is blended into the original at the selected region with a soft boundary.
Unsupported filter names and values outside `[-1, 1]` raise an error.

**Backbone:** None. Processing uses Pillow filters and compositing only.

## Inference configuration and runtime

The framework uses fixed backbones: FLUX.2 Klein base 4B for editing and generation,
SAM 3 for segmentation, Grounding DINO for text-region grounding, and ObjectClear
for removal. `InferenceConfig` contains their inference settings rather than model IDs:

```python
from framework import (
    InferenceConfig,
    GroundingDinoInferenceConfig,
    KleinInferenceConfig,
    ObjectClearInferenceConfig,
    image_editing_tools,
)

tools = image_editing_tools(
    inference_config=InferenceConfig(
        klein=KleinInferenceConfig(num_inference_steps=6),
        grounding_dino=GroundingDinoInferenceConfig(
            box_threshold=0.35,
            text_threshold=0.25,
        ),
        object_clear=ObjectClearInferenceConfig(guidance_scale=2.5),
    )
)
```

`KleinInferenceConfig` controls `num_inference_steps`, `guidance_scale`, and
`mask_context_padding`.
`GroundingDinoInferenceConfig` controls the box and text confidence thresholds.
`Sam3InferenceConfig` controls `mask_threshold`. `ObjectClearInferenceConfig`
controls `num_inference_steps`,
`guidance_scale`, and `working_min_side`. The required project-wide
`RANDOM_SEED` controls stochastic operations. Editing and generation share
Klein; object transforms share ObjectClear; visual and text segmentation share
SAM 3. Defaults preserve the existing inference settings.

At runtime, the model device is selected in this order: Apple MPS when
available, otherwise CUDA, otherwise CPU. `ModelPool` eagerly loads Klein,
SAM 3, and ObjectClear during construction; it loads Grounding DINO on the first
`segment_region` call. Tools constructed together by
`image_editing_tools()` share the same pool. For explicit pool reuse, pass it
as the first argument:

```python
from framework import InferenceConfig, ModelPool, image_editing_tools

pool = ModelPool(InferenceConfig())
tools = image_editing_tools(pool)
# Reuse this pool and these tools for related calls.
```

`pool` and `inference_config` are mutually exclusive arguments to
`image_editing_tools()`.
