import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

class GeminiRAG:
    def __init__(self, model_name="gemini-2.5-pro", temperature=0.2):
        """
        Initializes the Gemini LLM model because Groq decommissioned their vision models.
        Uses a vision-enabled Gemini model by default.
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the provided image context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise."
        )

    def answer_question(self, question: str, base64_images: list[str]) -> str:
        """
        Answers a question using the provided list of base64 images as context.
        """
        print(f"Sending request to Gemini LLM ...")
        
        # Build the message content.
        content = [{"type": "text", "text": question}]
        
        # Then we append the base64 images
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
