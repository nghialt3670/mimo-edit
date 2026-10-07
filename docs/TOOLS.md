# Image-editing tools

This document describes the tools registered by
[`src/tools/image_editing/`](../src/tools/image_editing/) and built by
[`image_editing_tools`](../src/tools/__init__.py). The registry currently exposes
11 tools:

| Tool | Purpose | Default backbone |
| --- | --- | --- |
| [`segment_object`](../src/tools/image_editing/segment_object.py) | Find one object and return its mask | SAM 3 |
| [`edit_object`](../src/tools/image_editing/edit_object.py) | Semantically edit one masked object | FLUX.2 Klein 4B |
| [`remove_object`](../src/tools/image_editing/remove_object.py) | Remove one masked object | ObjectClear |
| [`replace_object`](../src/tools/image_editing/replace_object.py) | Remove an object, then paste a replacement image | ObjectClear + Pillow |
| [`generate_object`](../src/tools/image_editing/generate_object.py) | Inpaint a region or place a phrase in a box | Stable Diffusion Inpainting or GLIGEN |
| [`move_object`](../src/tools/image_editing/move_object.py) | Move one object to a point or box | ObjectClear + Pillow |
| [`shift_object`](../src/tools/image_editing/shift_object.py) | Translate one object by an offset | ObjectClear + Pillow |
| [`rotate_object`](../src/tools/image_editing/rotate_object.py) | Rotate one object | ObjectClear + Pillow |
| [`scale_object`](../src/tools/image_editing/scale_object.py) | Resize one object around its center | ObjectClear + Pillow |
| [`flip_object`](../src/tools/image_editing/flip_object.py) | Flip one object horizontally or vertically | ObjectClear + Pillow |
| [`apply_filter`](../src/tools/image_editing/apply_filter.py) | Apply a visual filter globally or to one object | Pillow; no learned model |

The entries in the table are the default backbones. Model IDs can be overridden
as described in [Model configuration](#model-configuration).

## Shared interface conventions

- `image` and image-valued results are Pillow `Image` instances. The tools return
  a new image; they do not mutate the input image in place.
- A mask is a Pillow image. Its nonzero/white area selects the object or region.
  Masks are resized to the source image size by the relevant image utilities.
- Boxes are pixel coordinates `[x1, y1, x2, y2]` in the source image. Points and
  offsets use pixel coordinates too; positive x moves right and positive y moves
  down.
- Tools operate on one selected object per call. For multiple objects, segment
  and edit/remove each object in separate calls, using the latest returned image
  as the next call's input.
- `segment_object` creates masks. Other object-targeting tools consume a mask;
  they do not automatically identify the object from a text description.
- Model-backed tools load pipelines lazily. The cache is shared by the tool set
  created in one `image_editing_tools()` call, avoiding repeated pipeline loads
  across calls.

## Tool interfaces and behavior

### `segment_object`

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
an error.

**Backbone:** `facebook/sam3` by default. Both the text and visual-prompt
pathways share the lazily loaded SAM 3 model bundle.

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

**Backbone:** `black-forest-labs/FLUX.2-klein-4B` by default, loaded through
Diffusers' `Flux2KleinPipeline`. The optional
`black-forest-labs/FLUX.2-klein-9b-kv` backend uses `Flux2KleinKVPipeline`.
Both adapters use four inference steps and seed `42` (guidance scale `1.0` for
the 4B model; the 9B-KV pipeline uses its pipeline default). The 9B-KV model
needs about 29 GB VRAM and has a non-commercial license. Neither backend falls
back to Stable Diffusion.

### `remove_object`

**Inputs:** `image` (Image), `mask` (Image); both required.

Runs ObjectClear with the fixed instruction “remove the selected object,” using
the white mask as the removal region. The mask is resized with nearest-neighbor
sampling; the image and mask are resized for the model and the result is
resized back to the original image dimensions.

**Backbone:** `jixin0101/ObjectClear` by default. It uses eight inference steps,
guidance scale `2.0`, and seed `42`. ObjectClear is distributed under its
upstream S-Lab license, which restricts use to non-commercial purposes; review
the included license before use.

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

### `generate_object`

**Inputs**

| Name | Type | Required | Meaning |
| --- | --- | --- | --- |
| `image` | Image | Yes | Source image. |
| `prompt` | String | Yes | Description of content to generate or region fill. |
| `mask` | Image or null | No | White region to inpaint. |
| `box` | Array or null | No | Pixel region `[x1, y1, x2, y2]`. |
| `phrase` | String or null | No | Object description for GLIGEN placement. |

There are two mutually exclusive routes:

1. **Inpainting:** supply exactly one of `mask` or `box`, and omit `phrase`.
   A box is converted to a rectangular mask. The prompt describes what should
   fill the selected region. The mask and box routes independently default to
   Stable Diffusion Inpainting; either route can use FLUX.1 Fill [dev].
2. **Phrase-guided placement:** supply `box` and `phrase`, and omit `mask`.
   GLIGEN places the described object in the specified box, conditioned on the
   general `prompt` and phrase. Uses the box-and-phrase GLIGEN pipeline.

The general `prompt` must be non-empty. A `phrase` without a box, a mask paired
with a phrase, both mask and box without a phrase, or neither mask nor box
raises an error.

**Backbones**

- Mask inpainting: `runwayml/stable-diffusion-inpainting` by default; 50 steps,
  guidance scale `7.5`, seed `42`.
- Box inpainting: `runwayml/stable-diffusion-inpainting` by default; 50 steps,
  guidance scale `7.5`, seed `42`.
- FLUX.1 Fill [dev] (`black-forest-labs/FLUX.1-Fill-dev`) is selectable
  independently for the mask and box inpainting routes; it uses 50 steps,
  guidance scale `30`, seed `42`, and resizes inference dimensions to multiples
  of 16. Its checkpoint is under the FLUX.1 [dev] non-commercial license.
- PowerPaint v2.1 is not currently supported; its upstream implementation
  requires custom BrushNet modules and older pinned dependencies.
- Phrase and box: `masterful/gligen-1-4-inpainting-text-box` by default;
  normalized box coordinates, 50 steps, scheduled-sampling beta `0.3`, seed
  `42`.

Both routes resize the returned image to the input image dimensions if needed.

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

## Model configuration and runtime

Pass a `ModelConfig` when building the tools to override default model IDs:

```python
from tools import ModelConfig, image_editing_tools

tools = image_editing_tools(
    model_config=ModelConfig(
        model_ids={
            "edit_object": "black-forest-labs/FLUX.2-klein-9b-kv",
            "segment_object": "facebook/sam3",
            "generate_object_mask": "black-forest-labs/FLUX.1-Fill-dev",
            "generate_object_box": "black-forest-labs/FLUX.1-Fill-dev",
            "generate_object_gligen": "masterful/gligen-1-4-inpainting-text-box",
            "remove_object": "jixin0101/ObjectClear",
        },
        backends={
            "edit_object": "flux2-klein-9b-kv",
            "generate_object_mask": "flux-fill",
            "generate_object_box": "flux-fill",
        },
    )
)
```

Other supported model-ID override keys are `flip_object`, `move_object`,
`replace_object`, `rotate_object`, `scale_object`, and `shift_object`; each
defaults to the ObjectClear checkpoint. `apply_filter` has no model override
because it does not load a model. `generate_object` and `generate_object_gligen`
are the legacy shared inpainting ID and GLIGEN ID. Prefer the route-specific
`generate_object_mask` and `generate_object_box` ID keys when those routes use
different models; route-specific IDs take precedence over the legacy shared ID.
The supported `backends` keys are `edit_object` (`flux2-klein-4b` or
`flux2-klein-9b-kv`), `generate_object_mask`, and `generate_object_box` (each
`stable-diffusion` or `flux-fill`). The GLIGEN phrase+box route remains separate
and unchanged.

At runtime, the model device is selected in this order: Apple MPS when
available, otherwise CUDA, otherwise CPU. Pipelines are lazy-loaded and cached
by pipeline and model ID in a `ModelCache`; tools constructed together by
`image_editing_tools()` share the same cache. For explicit cache lifecycle or
reuse, pass a `ModelCache` as the first argument:

```python
from tools import ModelCache, ModelConfig, image_editing_tools

cache = ModelCache(ModelConfig(model_ids={"segment_object": "facebook/sam3"}))
tools = image_editing_tools(model_cache=cache)
# Reuse these tools/cache for related calls, then release cached pipelines:
cache.close()
```

`model_cache` and `model_config` are mutually exclusive arguments to
`image_editing_tools()`.
