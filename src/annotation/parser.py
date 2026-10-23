from __future__ import annotations

import os
import time
from typing import Protocol

from PIL import Image

from framework.utils.randomness import seed_random_generators

from .schema import MaskClauseAssignments, ParsedInstruction, RefinementDecision

SYSTEM_PROMPT = """You are a strict semantic parser for image-editing benchmark annotation.
Return only the structured schema requested by the API.

Extract only noun phrases that refer to entities already present in the SOURCE image and should be replaced by visual-reference tokens.
Rules:
- span must be copied verbatim from the instruction, with exact casing and punctuation excluded unless it is part of the noun phrase.
- occurrence is the 0-based occurrence if the exact same span appears multiple times.
- target: entity being edited, removed, moved, recolored, replaced, textured, etc.
- anchor: independently mentioned visible entity used as a spatial/relational anchor.
- source: independently mentioned visible exemplar/source entity whose appearance/texture/content is reused.
- context: other independently necessary visible referent.
- Do not emit newly requested objects that do not exist in the source image; if useful for semantic understanding, include them only with present_in_source=false.
- Set mask_guidance_needed=true when text alone does not confidently identify one visible reference, when the target may be a part of a larger object, or when multiple visible entities could match; otherwise set it false.
- Prefer one complete target referring expression when an anchor phrase is embedded inside it. Example: in 'make the cup to the left of the kettle blue', emit 'the cup to the left of the kettle' as one target span, not an overlapping kettle span.
- Emit separate entities when they are independent arguments. Example: 'move the cup next to the kettle' -> 'the cup' target and 'the kettle' anchor.
- grounding_query should be a concise visual description of the entity itself, preserving useful visible attributes.
- Never rewrite the instruction.
"""

MASKED_REFERENCE_PROMPT = """The input image has the dataset-provided mask overlaid in translucent magenta. Use the highlighted pixels to identify the visible source entity that should become the visual reference. If the instruction contains multiple edits and this mask identifies only one edit region, identify only the source entity for that region; do not annotate the other edits. For multi-object removal, each parse highlights one removed instance; identify that instance, not the whole group of removed instances.
Return exactly one entity: a verbatim phrase from the instruction that identifies the masked entity. Set present_in_source=true and choose its role based on the instruction:
- Use role="target" only when the masked entity itself is the entity being edited.
- Use role="anchor" when the instruction edits a distinct object, part, or accessory associated with the masked entity. In that case, select only the phrase identifying the masked entity; do not include the edited object/part/accessory in the span. Keeping that target phrase outside the span preserves the requested edit in the resulting mixed instruction.
- For replacement instructions (for example, "change X to Y", "replace X with Y", or "swap X with Y"), X is the existing source entity and Y is the requested replacement. Never select Y as the visual reference; it may not exist in the source image.
Examples:
- "Erase the pink frisbee the man is holding." with the MAN masked -> span "the man", role="anchor". The resulting instruction is "Erase the pink frisbee <v1> is holding."
- "Remove the eyeglasses of the white-haired man." with the MAN masked -> span "the white-haired man", role="anchor". Keep "the eyeglasses" outside the span.
- "Change the young boy to a young girl." with the BOY masked -> span "the young boy", role="target"; the young girl is a requested replacement, not a source reference.
- "Change the bird standing sideways to a birdhouse." with the BIRD masked -> span "the bird standing sideways", role="target"; the birdhouse is a requested replacement.
- "Change the color of the shirt of the man in a grey suit." with the MAN masked -> span "the man in a grey suit", role="anchor". Keep "the shirt" and the requested color outside the span.
- "Turn off the light of the microwave that has another microwave on top of it." with the LOWER MICROWAVE masked -> span "the microwave that has another microwave on top of it", role="anchor". Keep "the light" outside the span.
- "Change the surfboard to red." with the SURFBOARD masked -> span "the surfboard", role="target".
When the instruction names multiple instances collectively (for example, "remove the two fish") and does not distinguish them in text, return the same exact collective span for the individually highlighted instance. The annotation pipeline will create one reference for each mask.
Do not select a larger phrase that combines the masked entity with a distinct edited object or accessory. Do not emit other entities, including newly requested objects or replacement objects.
The span must be copied verbatim from the instruction. Never rewrite the instruction.
"""

MASKED_SPATIAL_REFERENCE_PROMPT = """For this CompBench add instruction, replace the usual entity-extraction rules with these mask-guided rules.
The input image has the dataset-provided edit mask overlaid in translucent magenta. The mask marks the destination edit region; it does not imply that the new object is already in the source. For multi_object_add, the mask shown for this parse is one separated partition of the row's edit mask and marks one requested placement. When the dataset mask joins two placements together, the annotation pipeline splits the union spatially and sends each partition separately.
Return exactly one entity whose span is the verbatim spatial or relational phrase in the instruction that identifies this highlighted destination. Prefer a phrase that answers "where should the new object be added?", such as "the leftmost", "the upper left", or "below the red fish". Do not include a second destination, or select the newly requested object's noun phrase or its appearance/action attributes.
When multiple requested objects share one location phrase, return that same phrase for each corresponding mask partition. Never extend a location span across "and" or "or" to include another requested object (for example, in "add a dog left of the black dog and a woman", do not include "and a woman" in the location span).
Keep the span as a complete location expression. If the span includes its leading spatial preposition (for example, "in the middle of the birds" or "behind the tree"), the annotation pipeline will replace that complete expression with "at <v1>" so the destination reference remains connected to the sentence. Use role="target", present_in_source=true, and a concise grounding_query such as "masked addition destination"; the mask, not text-based detection, grounds this reference.
The span must be copied verbatim from the instruction. Never rewrite the instruction.
"""

MIRA_BENCH_MASKED_REFERENCE_PROMPT = """This MIRA-Bench sample contains five edits. The highlighted mask marks the region for only one of those edits. Return exactly one visible source entity from the instruction clause associated with this mask; ignore the other edits.
The mask may cover only a small edited part or accessory, such as eyes, clothing, a shell, dirt, or an item held by a person. In that case, identify the complete visible object that the edit refers to, not just the masked part. Use role="target" when that object itself is edited, and role="anchor" when the instruction edits one of its parts or adds something to it.
Do not return an empty entity list when the highlighted region corresponds to a visible source object described by an instruction clause. Do not select a newly requested object that is not in the source image. The span must be copied verbatim from the instruction.
"""

MIRA_BENCH_ASSIGNMENT_PROMPT = """You are a strict semantic parser for MIRA-Bench image-editing annotation.
The instruction is split into numbered clauses; each clause is one edit. The image shows exactly as many numbered, colored regions as there are clauses. Each region is the location of the object edited by exactly one clause, and each clause owns exactly one region (a one-to-one matching).
For every region, find the clause whose edit refers to the object inside that region. Use the object's category, position (left, middle, right, "second from the left", corners), color and size to decide, and make sure no clause or region is used twice. Several clauses may name similar objects, so compare their positions with the regions carefully; clause order does not necessarily match region numbering.
For each region return:
- mask_index: the region number shown in the image.
- clause_index: the matching clause number.
- span: a phrase copied verbatim from THAT clause naming the existing source object that the clause edits or refers to. For "add X onto Y" use the existing object Y, never the new object X. For "replace Y with X" use Y.
- role: "target" when the object itself is edited, removed or replaced; "anchor" when something is added onto it or one of its parts is edited.
- grounding_query: a short description of the visible object.
Return one assignment per region.
"""

REFINEMENT_PROMPT = """You are reviewing a RefEdit-Bench mixed annotation.

The magenta overlay is the dataset-provided mask that became visual reference <v1>.
The mixed instruction replaced a span from the original instruction with that token.

Decide whether the annotation should be optimized by re-masking a SMALLER, more specific visible entity that is the actual object of the edit (a part, accessory, garment, or held object), instead of a larger host (person, animal, vehicle) that the dataset mask covers.

Optimize when:
- The instruction edits or removes a part/accessory (sunglasses, hat, shirt, frisbee, headlights, ...) but the magenta mask covers the host (the man, the car, ...).
- The mixed instruction is redundant: it still names the edited object in words and uses <v1> only as a host, e.g. "Remove the sunglasses from <v1>."

Do not optimize when:
- The mask already covers the edited entity.
- The mixed instruction already uses <v1> as the edit target, e.g. "Remove <v1>." or "Change <v1> to red."
- The edit is not a segmentable object (global lighting, the whole scene).
- You cannot name a unique visible object that SAM 3 could segment from a short text phrase.

If optimize=true:
- span: exact verbatim phrase from the ORIGINAL instruction for the edited object AND any host/identifier used only to find it (for example "from the man ..."). Do not invent words. The mixed instruction should not keep that identifier, because the visual prompt already points at the object: "Remove the sunglasses from the man standing on the road wearing glasses." -> span "the sunglasses from the man standing on the road wearing glasses" -> mixed "Remove <v1>."
- Keep requested-change wording outside the span: "Change the color of the shirt of the man to red." -> span "the shirt of the man" -> mixed "Change the color of <v1> to red."
- sam3_text: a short SAM 3 text prompt for the visible edited object only (e.g. "sunglasses"), not the host.
- role: "target" when that entity is being edited or removed; "anchor" only if a still-smaller part is edited.
- grounding_query: short visual description of that entity.
If optimize=false, leave span, sam3_text, and grounding_query empty.
"""


_MASK_PALETTE = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
)


def _numbered_masks_overlay(
    image: Image.Image, masks: list[Image.Image]
) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    canvas = image.convert("RGBA")
    for index, mask in enumerate(masks):
        mask = mask.convert("L")
        if mask.size != canvas.size:
            mask = mask.resize(canvas.size, Image.Resampling.NEAREST)
        color = _MASK_PALETTE[index % len(_MASK_PALETTE)]
        layer = Image.new("RGBA", canvas.size, (*color, 0))
        layer.putalpha(mask.point(lambda value: round(value * 0.5)))
        canvas = Image.alpha_composite(canvas, layer)

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(24, min(canvas.size) // 18))
    for index, mask in enumerate(masks):
        bbox = mask.convert("L").getbbox()
        if bbox is None:
            continue
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        color = _MASK_PALETTE[index % len(_MASK_PALETTE)]
        draw.text(
            center,
            str(index + 1),
            font=font,
            fill=color,
            anchor="mm",
            stroke_width=3,
            stroke_fill=(255, 255, 255),
        )
    return canvas.convert("RGB")


class Parser(Protocol):
    def parse(
        self,
        instruction: str,
        *,
        image: Image.Image | None = None,
        mask: Image.Image | None = None,
        masked_reference_only: bool = False,
        mask_guided_spatial_reference: bool = False,
        mira_bench_masked_reference: bool = False,
    ) -> ParsedInstruction: ...


def _mask_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    source = image.convert("RGB")
    mask = mask.convert("L")
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", source.size, (235, 35, 125, 0))
    overlay.putalpha(mask.point(lambda value: round(value * 0.48)))
    return Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")


class GoogleParser:
    """Structured parser using the official google-genai SDK; no Instructor dependency."""

    def __init__(
        self, model: str | None = None, api_key: str | None = None, max_retries: int = 2
    ):
        from google import genai

        self.seed = seed_random_generators()
        raw = model or os.getenv("ANNOTATE_LLM", "google/gemini-2.5-flash-lite")
        self.model = raw.split("/", 1)[1] if raw.startswith("google/") else raw
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key) if key else genai.Client()
        self.max_retries = max_retries

    def parse(
        self,
        instruction: str,
        *,
        image: Image.Image | None = None,
        mask: Image.Image | None = None,
        masked_reference_only: bool = False,
        mask_guided_spatial_reference: bool = False,
        mira_bench_masked_reference: bool = False,
    ) -> ParsedInstruction:
        from google.genai import types

        system_instruction = SYSTEM_PROMPT
        contents: str | list[object] = f"Instruction:\n{instruction}"
        if masked_reference_only and mask_guided_spatial_reference:
            raise ValueError("Choose only one mask-guided parsing mode")
        if masked_reference_only or mask_guided_spatial_reference:
            if image is None or mask is None:
                raise ValueError(
                    "Mask-guided parsing requires both the source image and its mask"
                )
            prompt = (
                MASKED_REFERENCE_PROMPT
                if masked_reference_only
                else MASKED_SPATIAL_REFERENCE_PROMPT
            )
            system_instruction += "\n" + prompt
            if mira_bench_masked_reference:
                if not masked_reference_only:
                    raise ValueError(
                        "MIRA-Bench masked-reference guidance requires masked_reference_only"
                    )
                system_instruction += "\n" + MIRA_BENCH_MASKED_REFERENCE_PROMPT
            from io import BytesIO

            image_buffer = BytesIO()
            _mask_overlay(image, mask).save(image_buffer, format="PNG")
            contents = [
                f"Instruction:\n{instruction}",
                types.Part.from_bytes(
                    data=image_buffer.getvalue(), mime_type="image/png"
                ),
            ]

        def call() -> ParsedInstruction:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0,
                    seed=self.seed,
                    response_mime_type="application/json",
                    response_schema=ParsedInstruction,
                ),
            )
            if response.parsed is not None:
                return ParsedInstruction.model_validate(response.parsed)
            return ParsedInstruction.model_validate_json(response.text)

        return self._with_retries(call)

    def propose_refinement(
        self,
        *,
        original_instruction: str,
        mixed_instruction: str,
        span: str,
        role: str,
        grounding_query: str,
        image: Image.Image,
        mask: Image.Image,
    ) -> RefinementDecision:
        """Ask whether the dataset mask should be replaced by a tighter SAM 3 mask."""
        from io import BytesIO

        from google.genai import types

        image_buffer = BytesIO()
        _mask_overlay(image, mask).save(image_buffer, format="PNG")
        contents = [
            (
                f"Original instruction:\n{original_instruction}\n\n"
                f"Mixed instruction:\n{mixed_instruction}\n\n"
                f"Current visual reference: span={span!r}, role={role}, "
                f"grounding_query={grounding_query!r}"
            ),
            types.Part.from_bytes(data=image_buffer.getvalue(), mime_type="image/png"),
        ]

        def call() -> RefinementDecision:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT + "\n" + REFINEMENT_PROMPT,
                    temperature=0,
                    seed=self.seed,
                    response_mime_type="application/json",
                    response_schema=RefinementDecision,
                ),
            )
            if response.parsed is not None:
                return RefinementDecision.model_validate(response.parsed)
            return RefinementDecision.model_validate_json(response.text)

        return self._with_retries(call)

    def assign_mira_bench_masks(
        self,
        instruction: str,
        clauses: list[str],
        *,
        image: Image.Image,
        masks: list[Image.Image],
    ) -> MaskClauseAssignments:
        """Match every numbered region to its clause in a single LLM call."""
        from io import BytesIO

        from google.genai import types

        buffer = BytesIO()
        _numbered_masks_overlay(image, masks).save(buffer, format="PNG")
        numbered = "\n".join(
            f"{number}. {clause}" for number, clause in enumerate(clauses, 1)
        )
        contents = [
            f"Instruction:\n{instruction}\n\nClauses:\n{numbered}",
            types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
        ]

        def call() -> MaskClauseAssignments:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=MIRA_BENCH_ASSIGNMENT_PROMPT,
                    temperature=0,
                    seed=self.seed,
                    response_mime_type="application/json",
                    response_schema=MaskClauseAssignments,
                ),
            )
            if response.parsed is not None:
                return MaskClauseAssignments.model_validate(response.parsed)
            return MaskClauseAssignments.model_validate_json(response.text)

        return self._with_retries(call)

    def _with_retries(self, call):
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error
