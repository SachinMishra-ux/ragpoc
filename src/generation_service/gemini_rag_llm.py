import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

class GeminiRAG:
    def __init__(self, model_name="gemini-2.5-pro", temperature=0.2):
        """
        Initializes the Gemini LLM model.
        Uses a vision-enabled Gemini model by default to handle image context (multimodal).
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
            
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key)
        self.system_prompt = (
            "You are an expert academic and engineering tutor. "
            "Your answer must be firmly grounded in and focused on the provided document context and page images. "
            "Base your explanation, technical definitions, formulas, and facts directly on what is shown in the provided materials, "
            "avoiding speculative assumptions or unrelated external information. "
            "Format math equations using LaTeX ($...$ or $$...$$) and code inside proper Markdown code blocks. "
            "If the answer cannot be determined from the provided context, state that clearly."
        )

    def answer_question(self, question: str, base64_images: list[str]) -> str:
        """
        Answers a question using the provided list of base64 images as context.
        """
        print(f"Sending request to Gemini LLM ({self.llm.model}) ...")
        
        # Build the message content.
        content = [{"type": "text", "text": question}]
        
        # Append the base64 images
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
        return response.content
