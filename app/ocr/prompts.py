"""
Prompt templates for vision-model image extraction.

Kept separate from app.ocr.image_ocr (traditional OCR, no LLM
involved) and app.generation.llm_generator (generic Ollama request/
response plumbing) so the actual WORDING of these instructions is easy
to find, read, and tune in one place, independent of the code that
sends them.

CODE_EXTRACTION_PROMPT is deliberately NOT a generic "what's in this
image?" question - open-ended prompts invite a vision model to
summarize, explain, or "helpfully" reconstruct code it isn't fully
sure about, which is exactly wrong for extracting exact source code
from a screenshot. A strict, transcription-only instruction (no
explaining, no inferring, no correcting, mark unclear characters
explicitly) keeps the model's output close to a literal transcription,
which is what should be combined with OCR's own transcription (see
app.ocr.image_ocr.combine_ocr_and_vision).
"""

CODE_EXTRACTION_PROMPT = """You are a code extraction system.

Extract ONLY the code and text visible in the image.

Rules:
1. Do not explain the code.
2. Do not infer missing code.
3. Do not correct the code.
4. Preserve all symbols exactly.
5. Preserve indentation and line breaks.
6. If any character is unclear, mark it as [UNCLEAR].
7. Return only the extracted content."""
