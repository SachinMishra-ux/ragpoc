import os
import time
import random
from google import genai
from PIL import Image

class GeminiEmbedder:
    def __init__(self, model_name="gemini-embedding-2", api_key: str | None = None):
        """
        Initializes the Google GenAI client and sets the model.
        The user specified "gemini-embedding-2".
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def embed_image(self, image: Image.Image, max_retries: int = 6, initial_backoff: float = 4.0) -> list[float]:
        """
        Embeds a single PIL Image with automatic exponential backoff retry for rate limits (429).
        Ensures vector output is formatted as a list of Python floats (compatible with float32 in S3 Vectors).
        """
        print(f"Embedding image using model: {self.model_name}...")
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=image
                )
                return [float(v) for v in result.embeddings[0].values]
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(1.0, 3.0)
                    print(f"⏳ Rate limit (429) hit. Backing off for {sleep_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    print(f"Error embedding image: {e}")
                    raise e

        # Final attempt
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=image
        )
        return [float(v) for v in result.embeddings[0].values]

    def embed_text(self, text: str, max_retries: int = 5, initial_backoff: float = 2.0) -> list[float]:
        """
        Embeds a text query with automatic retry on rate limits (429).
        Ensures vector output is formatted as a list of Python floats (compatible with float32 in S3 Vectors).
        """
        print(f"Embedding text using model: {self.model_name}...")
        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                return [float(v) for v in result.embeddings[0].values]
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(0.5, 1.5)
                    print(f"⏳ Rate limit (429) hit on text embedding. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"Error embedding text: {e}")
                    raise e

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return [float(v) for v in result.embeddings[0].values]

    def embed_multimodal(self, text: str, image: Image.Image, max_retries: int = 5, initial_backoff: float = 2.0) -> list[float]:
        """
        Embeds a multimodal query containing both an image (screenshot/table/chart) and text
        using gemini-embedding-2. Includes exponential backoff retry on rate limits.
        """
        print(f"Embedding multimodal (image + text) using {self.model_name}...")
        contents = [image]
        if text and text.strip():
            contents.append(text.strip())

        for attempt in range(max_retries):
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents
                )
                return [float(v) for v in result.embeddings[0].values]
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(0.5, 1.5)
                    print(f"⏳ Rate limit (429) hit on multimodal embedding. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"Error embedding multimodal query: {e}")
                    raise e

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=contents
        )
        return [float(v) for v in result.embeddings[0].values]

