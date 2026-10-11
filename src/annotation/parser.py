from __future__ import annotations

import os
import time
from typing import Protocol

from PIL import Image

from framework.utils.randomness import seed_random_generators

from .schema import ParsedInstruction

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

MASKED_REFERENCE_PROMPT = """The input image has the dataset-provided mask overlaid in translucent magenta. Use the highlighted pixels to identify the visible source entity that should become the visual reference.
Return exactly one entity: a verbatim, non-overlapping phrase from the instruction that identifies the masked entity. Set present_in_source=true and choose its role based on the instruction:
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
Do not select a larger phrase that combines the masked entity with a distinct edited object or accessory. Do not emit other entities, including newly requested objects or replacement objects.
The span must be copied verbatim from the instruction. Never rewrite the instruction.
"""

MASKED_SPATIAL_REFERENCE_PROMPT = """For this CompBench add instruction, replace the usual entity-extraction rules with these mask-guided rules.
The input image has the dataset-provided edit mask overlaid in translucent magenta. The mask marks the destination edit region; it does not imply that the new object is already in the source. For multi_object_add, the mask shown for this parse is one connected component of the row's mask and marks one of the requested placements.
Return exactly one entity whose span is the verbatim spatial or relational phrase in the instruction that identifies this highlighted destination. Prefer a phrase that answers "where should the new object be added?", such as "the leftmost", "the upper left", or "below the red fish". Do not include a second destination, or select the newly requested object's noun phrase or its appearance/action attributes.
Keep the span as a complete location expression. If the span includes its leading spatial preposition (for example, "in the middle of the birds" or "behind the tree"), the annotation pipeline will replace that complete expression with "at <v1>" so the destination reference remains connected to the sentence. Use role="target", present_in_source=true, and a concise grounding_query such as "masked addition destination"; the mask, not text-based detection, grounds this reference.
The span must be copied verbatim from the instruction. Never rewrite the instruction.
"""


class Parser(Protocol):
    def parse(
        self,
        instruction: str,
        *,
        image: Image.Image | None = None,
        mask: Image.Image | None = None,
        masked_reference_only: bool = False,
        mask_guided_spatial_reference: bool = False,
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
