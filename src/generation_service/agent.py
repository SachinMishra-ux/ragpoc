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
from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from src.embedding_service.s3_vector_manager import S3VectorManager
from src.embedding_service.embedder import GeminiEmbedder
from src.embedding_service.document_processor import render_pdf_page_to_base64

DB_PATH = os.path.join(PROJECT_ROOT, "academic_checkpoints.db")
DEFAULT_BEDROCK_REGION = os.getenv("AWS_REGION_2") or os.getenv("BEDROCK_REGION") or "us-east-1"
DEFAULT_NOVA_MODEL = os.getenv("BEDROCK_NOVA_MODEL") or (
    "us.amazon.nova-2-lite-v1:0" if "us-" in DEFAULT_BEDROCK_REGION else "amazon.nova-2-lite-v1:0"
)
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"


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
# 3. Academic RAG Tool Definition
# ---------------------------------------------------------------------------
def create_rag_tool(vector_manager: S3VectorManager, embedder: GeminiEmbedder):
    @tool
    def query_academic_knowledge_base(
        query: str,
        document_name: Optional[str] = None,
        limit: int = 3,
    ) -> Any:
        """Searches the Amazon S3 Vectors database for relevant academic textbooks, engineering lecture notes, scientific papers, circuit diagrams, code examples, formulas, derivations, and problem solutions matching the query across all subjects.
        Args:
            query: The search question or semantic query describing the academic topic, concept, formula, algorithm, theorem, or definition.
            document_name: Optional specific document or textbook filename to restrict the search to (if requested by the user).
            limit: Number of context pages to retrieve (default: 3).
        Returns:
            Structured text excerpt with page numbers, document names, and content snippets along with multimodal page images.
        """
        global _current_turn_context
        _current_turn_context.tool_called = True
        print(f"\n[Tool: query_academic_knowledge_base] Executing vector search for: '{query}' (doc_filter: {document_name}, limit: {limit})")

        try:
            # 1. Embed query using Gemini Embedding 2 (multimodal if user attached screenshot)
            if _current_turn_context.query_image_pil is not None:
                print(f"[Tool: query_academic_knowledge_base] Performing multimodal (screenshot + text) S3 vector search")
                query_emb = embedder.embed_multimodal(text=query, image=_current_turn_context.query_image_pil)
            else:
                query_emb = embedder.embed_text(query)

            # 2. Query Amazon S3 Vectors
            filter_expr = {"document_name": document_name} if document_name else None
            results = vector_manager.search(query_emb, limit=limit, filter_expr=filter_expr)

            if not results:
                return (
                    f"No matching documents found in Amazon S3 Vectors for query: '{query}'. "
                    f"Please proceed to answer the student's question accurately using your general academic foundational knowledge, "
                    f"and explicitly mention that specific textbook excerpts were not found in the indexed library documents."
                )

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
                if not os.path.exists(local_path):
                    try:
                        import boto3
                        s3_bucket = os.getenv("S3_BUCKET_NAME", "academic-rag-documents-493116771407")
                        reg = os.getenv("AWS_REGION", "eu-north-1")
                        ak = os.getenv("AWS_ACCESS_KEY_ID")
                        sk = os.getenv("AWS_SECRET_ACCESS_KEY")
                        s3_kw = {"region_name": reg}
                        if ak and sk:
                            s3_kw["aws_access_key_id"] = ak
                            s3_kw["aws_secret_access_key"] = sk
                        s3 = boto3.client("s3", **s3_kw)
                        os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
                        s3.download_file(s3_bucket, doc, local_path)
                    except Exception:
                        pass

                if os.path.exists(local_path) and isinstance(page_num, int):
                    b64_img = render_pdf_page_to_base64(local_path, page_num)
                    if b64_img:
                        _current_turn_context.images.append(b64_img)

                output_lines.append(f"--- [Match #{idx}] Document: {doc} | Page: {page_num}/{total_pages} | Distance: {distance} ---")
                if snippet:
                    output_lines.append(f"Content:\n{snippet}\n")

            text_output = "\n".join(output_lines)
            if _current_turn_context.images:
                tool_result = [{"type": "text", "text": text_output}]
                for b64 in _current_turn_context.images:
                    tool_result.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
                return tool_result
            return text_output

        except Exception as e:
            print(f"Error querying S3 Vectors inside tool: {e}")
            return f"Error occurred while searching Amazon S3 Vectors: {str(e)}"

    return query_academic_knowledge_base


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
class AcademicRAGAgent:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        self.memory.setup()

        # Initialize S3 Vectors & Embedder
        self.vector_manager = S3VectorManager()
        self.embedder = GeminiEmbedder(model_name="gemini-embedding-2")

        # Define Academic RAG Tool
        self.rag_tool = create_rag_tool(self.vector_manager, self.embedder)

        # Model configuration
        self.nova_model_id = DEFAULT_NOVA_MODEL
        self.gemini_model_id = DEFAULT_GEMINI_MODEL
        self.bedrock_region = DEFAULT_BEDROCK_REGION

        # Define Dynamic System Prompt for All Academic Disciplines
        self.system_prompt = (
            "You are an expert Professor and Academic AI Research Assistant with access to a comprehensive university-level "
            "academic library stored in an Amazon S3 Vectors database. You have access to a tool named `query_academic_knowledge_base` "
            "that searches and retrieves relevant textbook pages, lecture notes, diagrams, and technical documents from this database.\n\n"
            "CRITICAL WORKFLOW & GROUNDING POLICY:\n"
            "1. MANDATORY VECTOR SEARCH FOR ALL ACADEMIC QUESTIONS:\n"
            "   Whenever the user asks ANY academic, educational, scientific, engineering, technical, theoretical, or conceptual "
            "   question across ANY subject (such as programming, computer science, electronics, circuits, fluid mechanics, "
            "   thermodynamics, physics, chemistry, mathematics, mechanics, or any other academic discipline), YOU MUST ALWAYS "
            "   INVOKE the `query_academic_knowledge_base` tool FIRST to search the vector database for relevant literature "
            "   and textbook pages. Never answer academic questions directly from parametric memory on the initial turn without searching.\n"
            "2. STRICT GROUNDING & FOCUS ON RETRIEVED DOCUMENTS:\n"
            "   - When the tool retrieves relevant textbook pages and excerpts, your answer MUST BE FIRMLY GROUNDED IN AND FOCUSED ON "
            "     the retrieved document content.\n"
            "   - Rely on the facts, technical definitions, formulas, derivations, circuit properties, and explanations present in the "
            "     retrieved materials rather than generating unrelated external information or speculative assumptions.\n"
            "   - Connect and synthesize the information from the retrieved pages clearly, directly answering the user's question while "
            "     faithfully preserving the technical accuracy of the source documents.\n"
            "3. MANDATORY CITATIONS:\n"
            "   ALWAYS cite the specific document name and page number for every piece of information retrieved from the library "
            "   (e.g., '[Source: <document_name>, Page <page_number>]').\n"
            "4. PARAMETRIC FALLBACK POLICY (WHEN NO DOCUMENTS ARE FOUND):\n"
            "   If the `query_academic_knowledge_base` tool returns no matching documents, or if the retrieved excerpts do not contain "
            "   sufficient details to fully answer the question, ONLY THEN should you answer the question using your general academic "
            "   foundational knowledge. When this occurs, politely inform the student that specific textbook excerpts were not found "
            "   in the indexed library, and provide your own rigorous explanation.\n"
            "5. EXEMPTION FOR CASUAL / CONVERSATIONAL QUERIES ONLY:\n"
            "   The ONLY questions where you must NOT invoke the tool are pure non-academic conversational pleasantries and greetings "
            "   (e.g., 'hello', 'hi', 'how are you', 'who are you', 'what can you do', 'thank you', 'goodbye'). For these, reply warmly "
            "   and encourage the student to ask an academic question.\n"
            "6. MATHEMATICAL & FORMULA NOTATION:\n"
            "   ALWAYS format all mathematical formulas, equations, integrals, and derivations using standard LaTeX notation "
            "   ($...$ for inline math, and $$...$$ for centered block equations).\n"
            "7. CODE & ALGORITHMIC FORMATTING:\n"
            "   Whenever providing or explaining code (C, C++, Python, Java, SQL, etc.), ALWAYS format it inside proper Markdown code blocks "
            "   specifying the language identifier (e.g., ```c ... ```), include clear comments, and state time/space complexities where relevant.\n"
            "8. TABULAR FORMATTING:\n"
            "   Whenever presenting comparisons, truth tables, properties, or structured parameters, ALWAYS construct clean Markdown tables "
            "   with clear headers and aligned columns.\n"
            "9. CIRCUIT & DIAGRAM ANALYSIS:\n"
            "   When the student attaches a screenshot or asks about a circuit, diagram, or graph, provide a clear step-by-step physical "
            "   and analytical explanation of the visual components based on the retrieved context."
        )

        self.nova_llm = None
        self.nova_agent = None
        self.gemini_llm = None
        self.gemini_agent = None
        self._init_agents()

    def _init_agents(self):
        # 1. Initialize Amazon Nova Lite Agent (Bedrock)
        try:
            # Map amazon.nova-2-lite-v1:0 to cross-region profile if in US region
            nova_model = self.nova_model_id
            if nova_model == "amazon.nova-2-lite-v1:0" and "us-" in self.bedrock_region:
                nova_model = "us.amazon.nova-2-lite-v1:0"
                self.nova_model_id = nova_model

            nova_kwargs = {
                "model": nova_model,
                "region_name": self.bedrock_region,
                "temperature": 0.1,
            }

            # Dedicated credentials for Bedrock Nova LLM
            ak2 = os.getenv("AWS_ACCESS_KEY_ID2") or os.getenv("AWS_ACCESS_KEY_ID")
            sk2 = os.getenv("AWS_SECRET_ACCESS_KEY2") or os.getenv("AWS_SECRET_ACCESS_KEY")
            st2 = os.getenv("AWS_SESSION_TOKEN")
            if ak2 and sk2:
                nova_kwargs["aws_access_key_id"] = ak2
                nova_kwargs["aws_secret_access_key"] = sk2
            if st2:
                nova_kwargs["aws_session_token"] = st2

            try:
                self.nova_llm = ChatBedrockConverse(**nova_kwargs)
            except Exception as ex:
                if "nova-2-lite" in self.nova_model_id:
                    fallback_id = "us.amazon.nova-lite-v1:0" if "us-" in self.bedrock_region else "amazon.nova-lite-v1:0"
                    print(f"[AcademicRAGAgent] Nova 2 fallback to {fallback_id}: {ex}")
                    nova_kwargs["model"] = fallback_id
                    self.nova_model_id = fallback_id
                    self.nova_llm = ChatBedrockConverse(**nova_kwargs)
                else:
                    raise ex

            self.nova_agent = create_agent(
                model=self.nova_llm,
                tools=[self.rag_tool],
                system_prompt=self.system_prompt,
                checkpointer=self.memory,
            )
            print(f"✅ Amazon Nova 2 Lite Agent ({self.nova_model_id}, region: {self.bedrock_region}) successfully compiled!")
        except Exception as e:
            print(f"Notice: Could not compile Amazon Nova Agent: {e}")
            self.nova_agent = None

        # 2. Initialize Google Gemini Agent
        try:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=self.gemini_model_id,
                    temperature=0.1,
                    google_api_key=api_key,
                )
                self.gemini_agent = create_agent(
                    model=self.gemini_llm,
                    tools=[self.rag_tool],
                    system_prompt=self.system_prompt,
                    checkpointer=self.memory,
                )
                print(f"✅ Google Gemini Agent ({self.gemini_model_id}) successfully compiled!")
        except Exception as e:
            print(f"Notice: Could not compile Google Gemini Agent: {e}")
            self.gemini_agent = None

    def run(
        self,
        question: str,
        thread_id: Optional[str] = None,
        document_name: Optional[str] = None,
        limit: int = 3,
        image_base64: Optional[str] = None,
        model_provider: str = "nova",
        llm_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the agent for a user question under the given thread_id.
        Supports selecting between Amazon Nova 2 Lite (Bedrock) and Google Gemini.
        Supports multimodal queries with screenshots/images and maintains conversational history in SQLite.
        """
        global _current_turn_context
        _current_turn_context.reset()

        active_thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": active_thread_id, "checkpoint_ns": ""}}

        # Resolve model provider & active agent
        provider = (model_provider or "nova").lower()
        if provider in ("nova", "amazon", "bedrock"):
            resolved_provider = "nova"
            used_model_id = (llm_model if llm_model and "nova" in llm_model.lower() else None) or self.nova_model_id
            if not self.nova_agent:
                self._init_agents()
            if not self.nova_agent:
                raise ValueError("Amazon Nova Agent is not initialized. Please verify AWS credentials in .env.")
            active_agent = self.nova_agent
        else:
            resolved_provider = "gemini"
            used_model_id = (llm_model if llm_model and "gemini" in llm_model.lower() else None) or self.gemini_model_id
            if not self.gemini_agent:
                self._init_agents()
            if not self.gemini_agent:
                raise ValueError("Google Gemini Agent is not initialized. Please verify GEMINI_API_KEY in .env.")
            active_agent = self.gemini_agent

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

        print(f"\n🤖 Running Academic RAG Agent [{resolved_provider.upper()}: {used_model_id}] [Thread: {active_thread_id}]...")
        print(f"User Query: {question} (Image attached: {bool(clean_b64)})")

        try:
            if clean_b64:
                human_content = [
                    {"type": "text", "text": query_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{clean_b64}"}}
                ]
            else:
                human_content = query_text

            response = active_agent.invoke(
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
                "model_provider": resolved_provider,
                "model_used": used_model_id,
                "user_image": f"data:image/jpeg;base64,{clean_b64}" if clean_b64 else None,
                "sources": _current_turn_context.sources,
                "images": _current_turn_context.images,
                "pages_retrieved": len(_current_turn_context.sources),
            }

        except Exception as e:
            err_str = str(e)
            print(f"Error executing agent ({resolved_provider}): {err_str}")
            if "AccessDeniedException" in err_str and "bedrock:InvokeModel" in err_str:
                raise PermissionError(
                    f"AWS Bedrock Access Denied: The IAM user in .env lacks 'bedrock:InvokeModel' permission on {used_model_id}. "
                    f"Please attach the 'AmazonBedrockFullAccess' policy to your IAM user in the AWS Console, or toggle to Google Gemini in the UI."
                ) from e
            if "Operation not allowed" in err_str:
                raise PermissionError(
                    f"AWS Bedrock Model Access Required: Model access for '{used_model_id}' has not been enabled yet in your AWS account in region {self.bedrock_region}. "
                    f"To enable it: Go to AWS Console -> Amazon Bedrock (ensure region is '{self.bedrock_region}') -> Click 'Model access' in the left sidebar -> Click 'Modify model access' -> Check 'Amazon: Nova Lite' (or 'Nova 2 Lite') -> Click 'Next' / 'Save changes'. "
                    f"Once granted, Amazon Nova inference will work instantly. In the meantime, you can toggle to Google Gemini from the model dropdown."
                ) from e
            raise e

    def get_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Retrieves conversational message history for a given thread_id."""
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        active_agent = self.nova_agent or self.gemini_agent
        if not active_agent:
            return []
        try:
            state = active_agent.get_state(config)
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


# Backward compatibility alias
FinancialRAGAgent = AcademicRAGAgent

# Global singleton agent
_global_agent: Optional[AcademicRAGAgent] = None


def get_academic_agent() -> AcademicRAGAgent:
    global _global_agent
    if _global_agent is None:
        _global_agent = AcademicRAGAgent()
    return _global_agent


# Backward compatibility alias
get_agent = get_academic_agent

