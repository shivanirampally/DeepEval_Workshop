import os
import logging

from dotenv import load_dotenv
from google import genai
from deepeval.models import GeminiModel


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")


if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not configured.")

if not GEMINI_MODEL:
    raise ValueError("GEMINI_MODEL is not configured.")

logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)


client = genai.Client(api_key=GOOGLE_API_KEY)

gemini_model = GeminiModel(
    model=GEMINI_MODEL,
    api_key=GOOGLE_API_KEY
)


class GeminiLLM:

    def invoke(self, prompt):
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text


llm = GeminiLLM()