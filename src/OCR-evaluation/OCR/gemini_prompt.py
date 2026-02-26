from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Define the Structured Output Schema
class OcrLine(BaseModel):
    raw_transcription: str = Field(
        description="The exact text as it appears (including bskungs yig shorthands)."
    )
    normalized_text: str = Field(
        description="The text with all bskungs yig shorthands expanded into standard Tibetan."
    )
    confidence_score: float = Field(ge=0, le=1)
    orientation: str = Field(description="'horizontal' or 'vertical_90_deg'")
    bounding_box: List[int] = Field(description="[ymin, xmin, ymax, xmax] (0-1000)")

class TibetanManuscript(BaseModel):
    lines: List[OcrLine]

# 2. Setup Client
# client = genai.Client(api_key="YOUR_API_KEY")

SYSTEM_INSTRUCTION = (
    "You are an expert Tibetan paleographer. Perform line-by-line OCR. "
    "For 'raw_transcription', preserve all bskungs yig (shorthands) exactly as written. "
    "For 'normalized_text', expand all shorthands into full, standard Tibetan orthography. "
    "Identify line orientation and provide precise bounding boxes."
)

OCR_USER_PROMPT = "Perform OCR on this Tibetan manuscript page."


def get_ocr_config(model_name: str = "") -> types.GenerateContentConfig:
    """Build a GenerateContentConfig using the structured schema.

    Parameters
    ----------
    model_name : str
        Gemini model identifier.  ``thinking_level`` is only set for
        Gemini 3 models (``"gemini-3"`` in the name).
    """
    kwargs = dict(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=TibetanManuscript,
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    )
    if "gemini-3" in model_name:
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.HIGH,
        )
    return types.GenerateContentConfig(**kwargs)


# 3. Create the Job Configuration
# Using thinking_level='high' for better accuracy on complex scripts
# and media_resolution='high' to catch fine details in Tibetan glyphs.
# config = types.GenerateContentConfig(
#     system_instruction=(
#         "You are an expert Tibetan paleographer. Perform line-by-line OCR. "
#         "For 'raw_transcription', preserve all bskungs yig (shorthands) exactly as written. "
#         "For 'normalized_text', expand all shorthands into full, standard Tibetan orthography. "
#         "Identify line orientation and provide precise bounding boxes."
#     ),
#     response_mime_type="application/json",
#     response_schema=TibetanManuscript,
#     thinking_level="high", # Available in Gemini 3 for deeper reasoning
#     media_resolution="high" # Essential for reading small Tibetan vowel signs
# )

# 4. Prepare Batch Requests (JSONL format)
# Note: In a real batch job, you'd upload a .jsonl file to Google Cloud Storage
# requests = [
#     {
#         "request": {
#             "contents": [{
#                 "parts": [
#                     {"text": "Perform OCR on this Tibetan manuscript page."},
#                     {"file_data": {"mime_type": "image/jpeg", "file_uri": "gs://your-bucket/page_001.jpg"}}
#                 ]
#             }],
#             "generation_config": config
#         }
#     }
# ]

# 5. Submit the Batch Job
# This is asynchronous. It returns a job ID you can poll.
# batch_job = client.batches.create(
#     model="gemini-3-flash",
#     src=requests, # Or use a GCS URI for large batches
#     config={"display_name": "Tibetan_OCR_Batch_01"}
# )
#
# print(f"Batch job created: {batch_job.name}")
