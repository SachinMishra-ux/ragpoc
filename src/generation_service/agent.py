import os
import sys
import uuid
import sqlite3
import base64
import io
from PIL import Image
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "src", ".env"))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from src.embedding_service.s3_vector_manager import S3VectorManager
from src.embedding_service.embedder import GeminiEmbedder
from src.embedding_service.document_processor import render_pdf_page_to_base64

DB_PATH = os.path.join(PROJECT_ROOT, "financial_checkpoints.db")


# ---------------------------------------------------------------------------
# 2. Execution Context Tracker (captures tool calls & multimodal inputs per query)
# ---------------------------------------------------------------------------
class TurnContext:
    def __init__(self):
        self.tool_called: bool = False
        self.sources: List[Dict[str, Any]] = []
        self.images: List[str] = []
        self.query_image_base64: Optional[str] = None
        self.query_image_pil: Optional[Any] = None

    def reset(self):
        self.tool_called = False
        self.sources = []
        self.images = []
        self.query_image_base64 = None
        self.query_image_pil = None


_current_turn_context = TurnContext()


# ---------------------------------------------------------------------------
# 3. RAG Tool Definition
# ---------------------------------------------------------------------------
def create_rag_tool(vector_manager: S3VectorManager, embedder: GeminiEmbedder):
    @tool
    def query_financial_knowledge_base(
        query: str,
        document_name: Optional[str] = None,
        limit: int = 3,
    ) -> str:
        """Searches the Amazon S3 Vectors index for relevant financial documents, annual reports, balance sheets, income statements, or tables matching the query.
        Args:
            query: The search question or semantic query describing the required financial facts or tables.
            document_name: Optional specific document filename to filter on (e.g. 'EY_Financial_report_2025.pdf', 'JPM_Annual_2023.pdf').
            limit: Number of context pages to retrieve (default: 3).
        Returns:
            Structured text excerpt with page numbers, document names, and content snippets.
        """
        global _current_turn_context
        _current_turn_context.tool_called = True
        print(f"\n[Tool: query_financial_knowledge_base] Executing vector search for: '{query}' (doc_filter: {document_name}, limit: {limit})")

        try:
            # 1. Embed query using Gemini Embedding 2 (multimodal if user attached screenshot)
            if _current_turn_context.query_image_pil is not None:
                print(f"[Tool: query_financial_knowledge_base] Performing multimodal (screenshot + text) S3 vector search")
                query_emb = embedder.embed_multimodal(text=query, image=_current_turn_context.query_image_pil)
            else:
                query_emb = embedder.embed_text(query)

            # 2. Query Amazon S3 Vectors
            filter_expr = {"document_name": document_name} if document_name else None
            results = vector_manager.search(query_emb, limit=limit, filter_expr=filter_expr)

            if not results:
                return f"No matching financial documents or records found in Amazon S3 Vectors for query: '{query}'."

            output_lines = [f"Found {len(results)} matching page(s) in Amazon S3 Vectors:\n"]

            for idx, hit in enumerate(results, start=1):
                meta = hit.get("metadata", {})
                doc = meta.get("document_name", "Unknown Document")
                page_num = meta.get("page_number", "?")
                total_pages = meta.get("total_pages", "?")
                distance = hit.get("distance", "N/A")
                snippet = meta.get("text_snippet", "")

                source_record = {
                    "key": hit.get("key"),
                    "distance": distance,
                    "document_name": doc,
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "source_uri": meta.get("source_uri"),
                    "text_snippet": snippet,
                }
                _current_turn_context.sources.append(source_record)

                # Try resolving page image
                local_path = os.path.join(PROJECT_ROOT, "data", doc)
                if os.path.exists(local_path) and isinstance(page_num, int):
                    b64_img = render_pdf_page_to_base64(local_path, page_num)
                    if b64_img:
                        _current_turn_context.images.append(b64_img)

                output_lines.append(f"--- [Match #{idx}] Document: {doc} | Page: {page_num}/{total_pages} | Distance: {distance} ---")
                if snippet:
                    output_lines.append(f"Content:\n{snippet}\n")

            return "\n".join(output_lines)

        except Exception as e:
            print(f"Error querying S3 Vectors inside tool: {e}")
            return f"Error occurred while searching Amazon S3 Vectors: {str(e)}"

    return query_financial_knowledge_base


def extract_message_text(content: Any) -> str:
    """Extracts complete text from string, dict, or mixed list message content from Gemini."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if "text" in part and part["text"]:
                    text_parts.append(part["text"])
                elif "content" in part and isinstance(part["content"], str):
                    text_parts.append(part["content"])
            else:
                text_parts.append(str(part))
        result = ""
        for p in text_parts:
            if not result:
                result = p
            elif result.endswith("\n") or p.startswith("\n"):
                result += p
            else:
                result += "\n\n" + p
        return result
    elif isinstance(content, dict):
        return content.get("text") or content.get("content") or str(content)
    return str(content) if content is not None else ""


# ---------------------------------------------------------------------------
# 4. RAG Agent Builder & Manager
# ---------------------------------------------------------------------------
class FinancialRAGAgent:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        self.memory.setup()

        # Initialize S3 Vectors & Embedder
        self.vector_manager = S3VectorManager()
        self.embedder = GeminiEmbedder(model_name="gemini-embedding-2")

        # Initialize LLM
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1,
            google_api_key=api_key,
        )

        # Define RAG Tool
        self.rag_tool = create_rag_tool(self.vector_manager, self.embedder)

        # Define System Prompt for Selective Tool Calling
        self.system_prompt = (
            "You are an expert financial analyst and conversational AI assistant for corporate financial reports, "
            "annual statements, and SEC filings. You have access to a tool named `query_financial_knowledge_base` "
            "that searches and retrieves relevant document pages from an Amazon S3 Vectors database.\n\n"
            "CRITICAL TOOL USAGE POLICY:\n"
            "1. ONLY invoke the `query_financial_knowledge_base` tool when the user's question asks for specific, "
            "factual, quantitative, or contextual details about companies, financial reports, annual statements, "
            "audits, balance sheets, revenue numbers, margins, or proprietary filings (e.g. EY Financial Report 2025, "
            "JPMorgan Chase Annual Report 2023, etc.).\n"
            "2. DO NOT call the tool for general conversation, greetings (e.g., 'hello', 'how are you', 'what can you do?'), "
            "general accounting principles, high-level definitions (e.g., 'what is EBITDA?'), math formulas, or code generation "
            "unrelated to specific document data. Answer these directly from your knowledge base.\n"
            "3. When you invoke the tool, strictly ground your final answer on the retrieved excerpts. Always cite the "
            "document name and page number (e.g., '[Source: EY_Financial_report_2025.pdf, Page 14]').\n"
            "4. TABULAR FORMATTING: Whenever asked to present financial figures, summaries, comparisons, or data in a "
            "tabular format, ALWAYS construct clean, well-formatted Markdown tables with clear headers and aligned columns.\n"
            "5. CODE FORMATTING: Whenever asked to provide code (Python, SQL, financial calculations), ALWAYS format it "
            "inside proper Markdown code blocks specifying the language identifier (e.g. ```python ... ```).\n"
            "6. Be direct, professional, and accurate. If information is not found in the retrieved documents, state that clearly."
        )

        # Compile Agent with Checkpointer
        self.agent = create_agent(
            model=self.llm,
            tools=[self.rag_tool],
            system_prompt=self.system_prompt,
            checkpointer=self.memory,
        )
        print("✅ Financial RAG Agent successfully compiled with SQLite checkpointer!")

    def run(
        self,
        question: str,
        thread_id: Optional[str] = None,
        document_name: Optional[str] = None,
        limit: int = 3,
        image_base64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the agent for a user question under the given thread_id.
        Supports multimodal queries with screenshots/images and maintains conversational history in SQLite.
        """
        global _current_turn_context
        _current_turn_context.reset()

        active_thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": active_thread_id, "checkpoint_ns": ""}}

        # Process user attached image / screenshot if provided
        clean_b64 = None
        pil_img = None
        if image_base64 and image_base64.strip():
            raw_b64 = image_base64.strip()
            if "," in raw_b64:
                clean_b64 = raw_b64.split(",", 1)[1]
            else:
                clean_b64 = raw_b64
            try:
                img_bytes = base64.b64decode(clean_b64)
                pil_img = Image.open(io.BytesIO(img_bytes))
                if pil_img.mode in ("RGBA", "P"):
                    pil_img = pil_img.convert("RGB")
                print(f"[Agent] Successfully decoded attached user screenshot/image: {pil_img.size} ({pil_img.mode})")
            except Exception as e:
                print(f"Warning: Could not decode attached user image: {e}")
                pil_img = None
                clean_b64 = None

        _current_turn_context.query_image_base64 = clean_b64
        _current_turn_context.query_image_pil = pil_img

        # If a specific document filter was requested by the user, prepend guidance
        query_text = question
        if document_name:
            query_text = f"[Filter: restrict document to '{document_name}'] {question}"

        print(f"\n🤖 Running Financial RAG Agent [Thread: {active_thread_id}]...")
        print(f"User Query: {question} (Image attached: {bool(clean_b64)})")

        try:
            if clean_b64:
                human_content = [
                    {"type": "text", "text": query_text},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{clean_b64}"}
                ]
            else:
                human_content = query_text

            response = self.agent.invoke(
                {"messages": [HumanMessage(content=human_content)]},
                config=config,
            )

            # Extract the final answer text from the last AI message
            messages = response.get("messages", [])
            final_content = ""
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                    final_content = extract_message_text(msg.content)
                    if final_content.strip():
                        break

            if not final_content and messages:
                final_content = extract_message_text(messages[-1].content)

            return {
                "question": question,
                "answer": final_content,
                "thread_id": active_thread_id,
                "tool_called": _current_turn_context.tool_called,
                "user_image": f"data:image/jpeg;base64,{clean_b64}" if clean_b64 else None,
                "sources": _current_turn_context.sources,
                "images": _current_turn_context.images,
                "pages_retrieved": len(_current_turn_context.sources),
            }

        except Exception as e:
            print(f"Error executing agent: {e}")
            raise e

    def get_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Retrieves conversational message history for a given thread_id."""
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        try:
            state = self.agent.get_state(config)
            if not state or not state.values:
                return []

            messages = state.values.get("messages", [])
            history = []
            for msg in messages:
                msg_type = getattr(msg, "type", "")
                if msg_type == "tool" or type(msg).__name__ == "ToolMessage":
                    continue

                content = extract_message_text(msg.content)

                # Check if message contains an attached user image
                user_img = None
                if isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            img_val = part.get("image_url")
                            if isinstance(img_val, dict):
                                user_img = img_val.get("url")
                            elif isinstance(img_val, str):
                                user_img = img_val

                if not content.strip() and not user_img:
                    continue

                role = "user" if isinstance(msg, HumanMessage) or msg_type == "human" else "assistant"
                history.append({"role": role, "content": content, "user_image": user_img})
            return history
        except Exception as e:
            print(f"Error retrieving history for thread {thread_id}: {e}")
            return []

    def list_threads(self) -> List[str]:
        """Lists distinct thread IDs stored in the SQLite database."""
        try:
            with self.conn:
                cur = self.conn.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC")
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error listing threads: {e}")
            return []


# Global singleton agent
_global_agent: Optional[FinancialRAGAgent] = None


def get_agent() -> FinancialRAGAgent:
    global _global_agent
    if _global_agent is None:
        _global_agent = FinancialRAGAgent()
    return _global_agent
