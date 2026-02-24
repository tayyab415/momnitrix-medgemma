"""Quick diagnostic to understand what the API actually returns."""
from google import genai
from google.genai import types as t
from PIL import Image as PILImage
from io import BytesIO
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

img_path = sorted(Path("annotated_data/surgwound/images").glob("*.jpg"))[0]
img = PILImage.open(img_path).convert("RGB")
buf = BytesIO()
img.save(buf, format="JPEG", quality=85)
img_bytes = buf.getvalue()
print(f"Image: {img_path.name}  ({len(img_bytes)} bytes)")

schema = {
    "type": "OBJECT",
    "properties": {
        "value": {
            "type": "STRING",
            "enum": ["Existent", "Non-existent", "UNCERTAIN"],
            "description": "Edema assessment",
        },
        "confidence": {
            "type": "STRING",
            "enum": ["HIGH", "MEDIUM", "LOW"],
        },
        "reasoning": {
            "type": "STRING",
            "description": "One short sentence (max 100 chars) about visible image findings.",
            "max_length": 200,
        },
    },
    "required": ["value", "confidence", "reasoning"],
}

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=t.Content(
        role="user",
        parts=[
            t.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            t.Part.from_text(
                text=(
                    "Is edema (swelling/puffiness) present in the wound or surrounding tissue? "
                    "Be conservative — only mark Existent when visual evidence is clear."
                )
            ),
        ],
    ),
    config=t.GenerateContentConfig(
        system_instruction="You are an expert wound care clinician. Assess the wound image and respond with the JSON schema provided.",
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=schema,
    ),
)

print("response.text repr:", repr(response.text))
print("response.parsed:", getattr(response, "parsed", "N/A"))
print("finish_reason:", response.candidates[0].finish_reason if response.candidates else "none")
print("usage:", response.usage_metadata)
