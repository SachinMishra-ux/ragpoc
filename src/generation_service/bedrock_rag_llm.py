import os
from typing import List, Optional
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

DEFAULT_NOVA_MODEL = os.getenv("BEDROCK_NOVA_MODEL", "amazon.nova-2-lite-v1:0")
DEFAULT_BEDROCK_REGION = os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION", "eu-north-1")


class BedrockNovaRAG:
    """
    Multimodal Academic & Engineering RAG inference using Amazon Nova models via AWS Bedrock Converse API.
    Supports receiving base64 rendered document pages and diagrams as multimodal visual context.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_NOVA_MODEL,
        region_name: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.region_name = region_name or DEFAULT_BEDROCK_REGION
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.session_token = os.getenv("AWS_SESSION_TOKEN")

        kwargs = {
            "model": self.model_name,
            "region_name": self.region_name,
            "temperature": temperature,
        }
        bedrock_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        if bedrock_key:
            kwargs["bedrock_api_key"] = bedrock_key
        elif self.access_key and self.secret_key:
            kwargs["aws_access_key_id"] = self.access_key
            kwargs["aws_secret_access_key"] = self.secret_key
        if self.session_token:
            kwargs["aws_session_token"] = self.session_token

        try:
            self.llm = ChatBedrockConverse(**kwargs)
        except Exception as e:
            if "nova-2-lite" in self.model_name:
                fallback_model = "amazon.nova-lite-v1:0"
                print(f"[BedrockNovaRAG] Initializing fallback model {fallback_model} (primary failed: {e})")
                kwargs["model"] = fallback_model
                self.model_name = fallback_model
                self.llm = ChatBedrockConverse(**kwargs)
            else:
                raise e

        self.system_prompt = (
            "You are an expert academic and engineering tutor. "
            "Your answer must be firmly grounded in and focused on the provided document context and page images. "
            "Base your explanation, technical definitions, formulas, and facts directly on what is shown in the provided materials, "
            "avoiding speculative assumptions or unrelated external information. "
            "Format math equations using LaTeX ($...$ or $$...$$) and code inside proper Markdown code blocks. "
            "If the answer cannot be determined from the provided context, state that clearly."
        )

    def answer_question(self, question: str, base64_images: List[str]) -> str:
        """
        Answers a query using the provided list of base64 images as visual context.
        """
        print(f"Sending request to Amazon Bedrock Nova LLM ({self.model_name}, region: {self.region_name}) ...")

        # Build message content with text and multimodal images
        content = [{"type": "text", "text": question}]

        for b64_img in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}"
                }
            })

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=content)
        ]

        response = self.llm.invoke(messages)
        if isinstance(response.content, list):
            parts = []
            for block in response.content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
        return str(response.content)
