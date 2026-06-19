import os
import json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch.nn.functional as F
from lab11_search import vector_search

import torch
import numpy as np
import warnings
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress some verbose huggingface warnings
warnings.filterwarnings("ignore")


# ==========================================
# Configuration
# ==========================================
class RAGConfig:
    MODEL_CACHE_DIR = "./model_cache"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Models
    EMB_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
    GEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
    RERANK_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

    # Qwen Reranker Specific Prompts
    RERANK_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )
    RERANK_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


# ==========================================
# Prompt Template
# ==========================================
def build_rag_prompt(query: str, contexts: List[str]) -> str:
    """Construct a RAG prompt with retrieved context chunks."""
    context_text = "\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
    return f"""You are a helpful assistant. Use only the provided context to answer the question.
If the context does not contain the answer, say "I don't know based on the provided context."

Context:
{context_text}

Question: {query}

Answer (cite sources using [1], [2], etc. based on context chunks above):"""


# ==========================================
# IO & Document Processing (Code Provided)
# ==========================================


def _make_converter():
    """Create a DocumentConverter that caches models in MODEL_CACHE_DIR."""
    artifacts_path = os.path.join(RAGConfig.MODEL_CACHE_DIR, "docling")
    pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_pages_from_pdf(pdf_path: str, converter=None) -> List[Dict]:
    """Extract text per page from a PDF using docling."""
    pages_data = {}
    try:
        if converter is None:
            converter = _make_converter()
        doc = converter.convert(pdf_path).document
        pdf_name = os.path.basename(pdf_path)

        for item in doc.texts:
            if item.prov:
                page_no = item.prov[0].page_no
                if page_no not in pages_data:
                    pages_data[page_no] = ""
                pages_data[page_no] += item.text + "\n"

        return [
            {"pdf_name": pdf_name, "page_no": k, "text": v}
            for k, v in pages_data.items()
        ]
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {e}")
        return []


def load_course_materials(
    handout_dir: str, cache_path: str = "extracted_pages.json"
) -> List[Dict]:
    """Loads all PDFs from the specified directory. Reuses cache if available."""
    if os.path.exists(cache_path):
        print(f"Loading cached pages from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    all_pages = []
    if os.path.exists(handout_dir):
        print(f"Loading PDFs from {handout_dir} (this may take a while)...")
        converter = _make_converter()
        for file in os.listdir(handout_dir):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(handout_dir, file)
                print(f"Processing {file}...")
                all_pages.extend(extract_pages_from_pdf(pdf_path, converter=converter))
        
        # Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_pages, f, ensure_ascii=False, indent=2)
    else:
        raise FileNotFoundError(f"Directory '{handout_dir}' not found.")
    return all_pages


def chunk_pages(
    pages: List[Dict], chunk_size: int = 512, overlap: int = 64
) -> Tuple[List[str], List[Dict]]:
    """Split pages into overlapping text chunks, preserving source metadata.

    Returns:
        (chunks, chunk_metadata) where each chunk_metadata[i] records
        the pdf_name and page_no that chunk i originated from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = []
    metadata = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        chunks.extend(page_chunks)
        metadata.extend(
            [{"pdf_name": page["pdf_name"], "page_no": page["page_no"]}]
            * len(page_chunks)
        )
    return chunks, metadata


# ==========================================
# Embedding Helpers (Task 2)
# ==========================================
def last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Pools the hidden states by taking the last token of each sequence.
    Required for Qwen3-Embedding.
    """
    ### TODO (Task 2): Implement last_token_pool

    # 1. Check if left padding is used (all elements in the last column of attention_mask are 1).
    # 2. If left padding, simply return the last hidden state: last_hidden_states[:, -1]
    # 3. Otherwise, calculate sequence_lengths = attention_mask.sum(dim=1) - 1
    # 4. Extract the token at sequence_lengths for each sequence in the batch.
    # Note: See slide for the reference implementation.
    last_col = attention_mask[:, -1] # get the last column of attention_mask
    if torch.all(last_col == 1):
        return last_hidden_states[:, -1, :] # return the last hidden state if left padding is used
    
    seq_lens = attention_mask.sum(dim=1).long() - 1
    # calculate sequence lengths by summing attention_mask and subtracting 1 for zero-based indexing

    batch_size, seq_len, hidden = last_hidden_states.size()
    # get batch size, sequence length, and hidden dimension from last_hidden_states

    seq_lens = seq_lens.clamp(min=0, max=seq_len - 1) # clamp sequence lengths to valid range

    batch_indices = torch.arange(batch_size, device=last_hidden_states.device) # create batch indices
    pooled = last_hidden_states[batch_indices, seq_lens, :] # extract the last token hidden states based on sequence lengths

    return pooled # return the pooled embeddings


# ==========================================
# Core RAG System
# ==========================================
class CourseQASystem:
    def __init__(self):
        self.device = RAGConfig.DEVICE

        # Models (Lazy loaded)
        self.emb_tokenizer = None
        self.emb_model = None
        self.gen_tokenizer = None
        self.gen_model = None
        self.rerank_tokenizer = None
        self.rerank_model = None

        # State
        self.chunks = []
        self.chunk_metadata = []
        self.chunk_embeddings = None

    # ---- Model Loading (Lazy) ----

    def _load_embedding_model(self):
        if self.emb_model is None:
            print(f"Loading Embedding Model ({RAGConfig.EMB_MODEL_ID})...")
            self.emb_tokenizer = AutoTokenizer.from_pretrained(
                RAGConfig.EMB_MODEL_ID,
                padding_side="left",
                cache_dir=RAGConfig.MODEL_CACHE_DIR,
            )
            self.emb_model = AutoModel.from_pretrained(
                RAGConfig.EMB_MODEL_ID,
                cache_dir=RAGConfig.MODEL_CACHE_DIR,
                device_map="auto",
            )
            self.emb_model.eval()

    def _load_generation_model(self):
        if self.gen_model is None:
            print(f"Loading Generation Model ({RAGConfig.GEN_MODEL_ID})...")
            self.gen_tokenizer = AutoTokenizer.from_pretrained(
                RAGConfig.GEN_MODEL_ID, cache_dir=RAGConfig.MODEL_CACHE_DIR
            )
            self.gen_model = AutoModelForCausalLM.from_pretrained(
                RAGConfig.GEN_MODEL_ID,
                cache_dir=RAGConfig.MODEL_CACHE_DIR,
                device_map="auto",
            )
            self.gen_model.eval()

    def _load_reranking_model(self):
        if self.rerank_model is None:
            print(f"Loading Reranking Model ({RAGConfig.RERANK_MODEL_ID})...")
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                RAGConfig.RERANK_MODEL_ID, cache_dir=RAGConfig.MODEL_CACHE_DIR
            )
            self.rerank_model = AutoModelForCausalLM.from_pretrained(
                RAGConfig.RERANK_MODEL_ID,
                cache_dir=RAGConfig.MODEL_CACHE_DIR,
                device_map="auto",
            )
            self.rerank_model.eval()

    # ---- Phase 1: Indexing ----

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Embed a list of strings using the embedding model."""
        self._load_embedding_model()

        ### TODO (Task 2): Implement embed_texts
        # 1. Create an empty list `all_embeddings`.
        # 2. Loop over `texts` in batches of `batch_size`.
        # 3. Tokenize `batch` using self.emb_tokenizer (padding=True, truncation=True, max_length=512, return_tensors="pt").
        # 4. Move inputs to `self.device`.
        # 5. Get `outputs` from `self.emb_model(**batch_dict)`.
        # 6. Call `last_token_pool` to get raw embeddings.
        # 7. Normalize them using `F.normalize(..., p=2, dim=1)`.
        # 8. Move to cpu(), convert to numpy(), and append to `all_embeddings`.
        # 9. Return `np.concatenate(all_embeddings, axis=0)`.
        # Note: See slide for the reference implementation.

        all_embeddings = []
        device = self.device

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize the batch of texts with appropriate padding and truncation
            batch_enc = self.emb_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Move tokenized inputs to the appropriate device
            input_ids = batch_enc["input_ids"].to(device)
            attention_mask = batch_enc["attention_mask"].to(device)

            # Get model outputs and pool to get embeddings
            outputs = self.emb_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state # shape: (batch_size, seq_len, hidden_dim)

            # Pool the last hidden states to get a single embedding per input text
            pooled = last_token_pool(last_hidden, attention_mask) # shape: (batch_size, hidden_dim)

            # Normalize the pooled embeddings
            pooled_norm = F.normalize(pooled, p=2, dim=1) # L2 normalization

            # Move embeddings to CPU, convert to numpy, and store
            all_embeddings.append(pooled_norm.detach().float().cpu().numpy())

        if len(all_embeddings) == 0:
            return np.zeros((0, 0), dtype=np.float32) # return empty array if no texts

        return np.concatenate(all_embeddings, axis=0) # concatenate all batch embeddings and return

    def index_pages(self, pages: List[Dict], chunk_size: int = 256, overlap: int = 64):
        """Chunk pages, then embed all chunks."""
        print(f"Chunking {len(pages)} pages (size={chunk_size}, overlap={overlap})...")
        self.chunks, self.chunk_metadata = chunk_pages(pages, chunk_size, overlap)
        print(f"Created {len(self.chunks)} chunks. Embedding...")

        ### TODO: Embed all chunks
        
        if len(self.chunks) == 0:
            raise ValueError(
                "No chunks were created. Check PDF extraction output (pages/text may be empty)."
            )

        self.chunk_embeddings = self.embed_texts(self.chunks)

        print("Indexing complete.")

    # ---- Phase 2: Retrieval ----

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Strategy A: Simple vector search for Top-K chunks."""
        if self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
            raise RuntimeError("Index is empty. Run index_pages(...) first to build embeddings.")
        
        query_emb = self.embed_texts([query])[0]

        ### TODO (Task 3): Implement Strategy A (Simple Vector Search)
        # 1. Use vector_search to find the top-K matches for query_emb.
        # 2. Return a list of dicts: [{'text': ..., 'pdf_name': ..., 'page_no': ...}, ...]
        
        # Use the vector_search function to get the indices and scores of the top K most similar chunks
        top_indices, top_scores = vector_search(query_emb, self.chunk_embeddings, k=k)

        results = []
        for idx, score in zip(top_indices, top_scores):
            idx = int(idx) # ensure idx is a standard Python int for indexing
            results.append(
                {
                    "text": self.chunks[idx],
                    "pdf_name": self.chunk_metadata[idx]["pdf_name"],
                    "page_no": self.chunk_metadata[idx]["page_no"],
                    "score": float(score), # include similarity score for reference
                    "chunk_index": idx, # include chunk index for reference
                }
            )

        return results

    def retrieve_rerank(self, query: str, k: int = 5) -> List[Dict]:
        """Strategy B (Bonus): Retrieve Top-50, then rerank to Top-K."""
        self._load_reranking_model()
        query_emb = self.embed_texts([query])[0]

        ### TODO (Bonus Task 4): Implement Strategy B (Reranked Search)
        # You can refer to the official code https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
        # 1. Search for the Top-50 most similar chunks using vector_search.
        # 2. Format inputs according to Qwen3-Reranker standards:
        #    - Pair format: "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        #    - Wrap with RAGConfig.RERANK_PREFIX / RAGConfig.RERANK_SUFFIX (use Our Prefix/Suffix in RAGConfig)
        # 3. Tokenize, prepend prefix_tokens, append suffix_tokens to each input_ids.
        # 4. Pad inputs and move to device.
        # 5. Extract last-token logits, softmax over "yes"/"no" token ids.
        # 6. Sort by relevance and return top-K chunks.

        candidate_k = min(50, len(self.chunks)) # number of candidates to consider for reranking
        top_indices, top_scores = vector_search(query_emb, self.chunk_embeddings, k=candidate_k)

        if len(top_indices) == 0:
            return [] # return empty list if no candidates found
        
        pair_texts = []
        for idx in top_indices:
            doc_text = self.chunks[int(idx)]
            pair_text = (
                f"<Instruct>: {RAGConfig.RERANK_INSTRUCTION}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc_text}"
            )

            full_text = f"{RAGConfig.RERANK_PREFIX}{pair_text}{RAGConfig.RERANK_SUFFIX}"
            pair_texts.append(full_text)

        batch = self.rerank_tokenizer(
            pair_texts,
            padding=True,
            truncation=True, # ensure we truncate to fit model's max length
            max_length=2048,
            return_tensors="pt",
        )

        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.rerank_model(**batch, return_dict=True)
        logits = outputs.logits # shape: (batch_size, seq_len, vocab_size)

        last_positions = batch["attention_mask"].sum(dim=1).long() - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_indices, last_positions, :]

        yes_id = self.rerank_tokenizer.encode("yes", add_special_tokens=False)[-1]
        no_id = self.rerank_tokenizer.encode("no", add_special_tokens=False)[-1]

        choice_logits = last_logits[:, [no_id, yes_id]]
        choice_probs = torch.softmax(choice_logits, dim=-1)

        rerank_scores = choice_probs[:, 1].detach().float().cpu().numpy() # probability of "yes" as relevance score

        sorted_order = np.argsort(rerank_scores)[::-1]
        sorted_order = sorted_order[:k]

        results = []
        for order_idx in sorted_order:
            original_idx = int(top_indices[order_idx])
            results.append(
                {
                    "text": self.chunks[original_idx],
                    "pdf_name": self.chunk_metadata[original_idx]["pdf_name"],
                    "page_no": self.chunk_metadata[original_idx]["page_no"],
                    "score": float(rerank_scores[order_idx]),
                    "chunk_index": original_idx,
                }
            )

        return results

    # ---- Phase 3: Generation ----

    def generate_answer(
        self, query: str, k: int = 5, use_reranker: bool = False
    ) -> Tuple[str, List[Dict], str]:
        """Retrieve context, build prompt, generate answer."""
        self._load_generation_model()

        ### TODO (Task 3): Implement Generation
        # 1. Call self.retrieve() or self.retrieve_rerank() based on use_reranker.
        # 2. Build prompt via build_rag_prompt(query, [item['text'] for item in retrieved_items]).
        # 3. Wrap in chat template: messages = [{"role": "user", "content": prompt}]
        # 4. Tokenize via self.gen_tokenizer.apply_chat_template(..., return_dict=True)
        # 5. Generate with self.gen_model.generate(**inputs, max_new_tokens=256)
        # 6. Decode only new tokens and return (answer, retrieved_items, prompt).
        
        if use_reranker:
            retrieved_items = self.retrieve_rerank(query, k=k)
        else:
            retrieved_items = self.retrieve(query, k=k)

        contexts = [item["text"] for item in retrieved_items]
        prompt = build_rag_prompt(query, contexts)

        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.gen_tokenizer, "apply_chat_template"):
            inputs = self.gen_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            inputs = self.gen_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.gen_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.gen_tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][input_len:]
        answer = self.gen_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return answer, retrieved_items, prompt


# ==========================================
# Display Helpers
# ==========================================
def print_result(
    query: str, answer: str, references: List[Dict], prompt: str, label: str = ""
):
    """Pretty-print RAG pipeline input/output using rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    title = f"Result ({label})" if label else "Result"

    # 1. User Query
    console.print()
    console.rule(f"[bold cyan]{title}[/]")
    console.print(f"\n[bold]📝 User Query:[/] {query}")

    # 2. Retrieved Context
    table = Table(title="📚 Retrieved Context", show_lines=True, title_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Source", style="cyan", width=28)
    table.add_column("Preview", ratio=1)
    for i, ref in enumerate(references, 1):
        preview = ref["text"][:150].replace("\n", " ")
        if len(ref["text"]) > 150:
            preview += "..."
        table.add_row(str(i), f"{ref['pdf_name']} p.{ref['page_no']}", preview)
    console.print(table)

    # 3. Full Prompt to LLM
    console.print(
        Panel(prompt, title="🤖 Full Prompt to LLM", border_style="yellow", expand=True)
    )

    # 4. Generated Answer
    console.print(
        Panel(answer, title="✅ Generated Answer", border_style="green", expand=True)
    )


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("Lab 11: Task 3 & 4 - End-to-End Course QA System")
    print("=" * 60)

    # 1. Setup
    handout_dir = "coe201_handout"
    rag = CourseQASystem()
    query = "What is the purpose of LoRA?"

    # 2. Load Documents
    all_pages = load_course_materials(handout_dir)

    try:
        # 3. Index (Chunk + Embed)
        rag.index_pages(all_pages, chunk_size=256, overlap=64)

        # 4. Retrieve + Generate
        print("\nGenerating Answer (without reranker)...")
        answer_base, references_base, prompt_base = rag.generate_answer(
            query, use_reranker=False
        )
        print_result(query, answer_base, references_base, prompt_base, label="Baseline")

    except Exception as e:
        print(f"\nTask 3 Not Implemented or Error: {e}")

    # 5. Bonus: Reranking
    print("\n" + "=" * 60)
    print("Testing Bonus Task 4: Reranking")
    print("=" * 60)

    try:
        print("Generating Answer (with reranker)...")
        answer_rerank, references_rerank, prompt_rerank = rag.generate_answer(
            query, use_reranker=True
        )
        print_result(
            query, answer_rerank, references_rerank, prompt_rerank, label="Reranked"
        )

    except NotImplementedError:
        print("Bonus task 'retrieve_rerank' not implemented. Skipping Bonus Task 4.")
    except Exception as e:
        print(f"Failed to run Reranker: {e}")
