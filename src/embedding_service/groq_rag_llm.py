import os
from groq import Groq

class GroqRAG:
    def __init__(self, model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.2):
        """
        Initializes the Groq client. Uses the new LLaMA 4 Scout vision model.
        """
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
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
        print(f"Sending request to Groq LLM ({self.model_name})...")
        
        # Build the message content for the user role.
        # Start with the user's text question.
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
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return response.choices[0].message.content
    
    
