"""src/policy_rag.py -- Document chunking, TF-IDF vector retrieval, grounded policy Q&A assistant, and evaluation benchmarks.
"""
import os
import sys
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath("."))
import mock_llm

POLICIES_DIR = os.path.join("data", "policies")
RAW_DIR = os.path.join("data", "raw")

def chunk_documents(overlap=10, chunk_size=40):
    """Chunk all 5 policy documents with sliding window."""
    policy_files = sorted(glob.glob(os.path.join(POLICIES_DIR, "*.txt")))
    chunks = []

    for file_path in policy_files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        words = text.split()

        step = chunk_size - overlap
        i = 0
        file_chunks = []
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            file_chunks.append({
                "source": filename,
                "text": chunk_text
            })
            if i + chunk_size >= len(words):
                break
            i += step

        chunks.extend(file_chunks)

    return chunks

class PolicyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        self.chunk_texts = [c["text"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def retrieve(self, query, k=3):
        """Retrieve top k chunks ranked by cosine similarity."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Rank indices descending
        ranked_indices = similarities.argsort()[::-1][:k]

        results = []
        for idx in ranked_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(similarities[idx]),
                "text": self.chunks[idx]["text"]
            })
        return results

def answer_policy_question(question, retriever, k=3):
    """Augment with top k chunks and answer using mock_llm POLICY ASSISTANT."""
    retrieved_chunks = retriever.retrieve(question, k=k)

    # Build context string
    # Context format:
    # CONTEXT:\n[<source>]\n<chunk text>\n...
    context_lines = []
    for c in retrieved_chunks:
        context_lines.append(f"[{c['source']}]\n{c['text']}")
    context_str = "\n".join(context_lines)

    user_message = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
    system_message = (
        "You are an Instamart POLICY ASSISTANT. Answer only from the supplied context. "
        "If the answer cannot be found in the context, say you could not find the information."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    resp = mock_llm.chat(messages)
    answer_text = resp.choices[0].message.content
    sources = [c["source"] for c in retrieved_chunks]

    return answer_text, sources, retrieved_chunks

# Standalone functions matching Task 2 and Task 3 exact specifications
_DEFAULT_RETRIEVER = None

def get_default_retriever():
    """Lazy loader for default PolicyRetriever with overlap 10."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        chunks = chunk_documents(overlap=10, chunk_size=40)
        _DEFAULT_RETRIEVER = PolicyRetriever(chunks)
    return _DEFAULT_RETRIEVER

def retrieve(query, k=3):
    """Retrieve top k policy chunks as a list of dicts with source, score, text."""
    return get_default_retriever().retrieve(query, k=k)

def answer(question):
    """Answer question using top 3 retrieved policy chunks and mock_llm POLICY ASSISTANT."""
    ans, sources, _ = answer_policy_question(question, get_default_retriever(), k=3)
    return ans, sources

def evaluate_policy_assistant():
    print("================ PART D: POLICY RAG ASSISTANT ================\n")

    # 1. Chunk documents (overlap 10)
    chunks = chunk_documents(overlap=10, chunk_size=40)
    print(f"Total chunks created: {len(chunks)} (Reference: 15)")

    counts_by_doc = pd.Series([c["source"] for c in chunks]).value_counts()
    print("Chunks by document:")
    for doc, cnt in counts_by_doc.items():
        print(f"  {doc}: {cnt}")
    print()

    # Verify query top 3 chunks
    retriever = PolicyRetriever(chunks)
    test_query = "refund policy late delivery"
    top3_test = retriever.retrieve(test_query, k=3)
    top3_sources = [c["source"] for c in top3_test]
    print(f"Top 3 chunks for '{test_query}': {top3_sources} (Ref: all refund_policy.txt)")
    assert all(s == "refund_policy.txt" for s in top3_sources), "Not all top 3 chunks are from refund_policy.txt"

    # 2. Evaluate on policy_eval.csv
    eval_csv_path = os.path.join(RAW_DIR, "policy_eval.csv")
    df_eval = pd.read_csv(eval_csv_path)

    print("\n--- Evaluating Standard Chunker (Overlap=10) on policy_eval.csv ---")
    results = []
    for _, row in df_eval.iterrows():
        q = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = str(row["expected_fact"]) if pd.notna(row["expected_fact"]) else ""

        ans, sources, _ = answer_policy_question(q, retriever, k=3)

        if expected_doc == "NONE":
            # Unanswerable question
            doc_match = True
            fact_match = ans.startswith("I could not find")
        else:
            doc_match = expected_doc in sources
            fact_match = expected_fact in ans if expected_fact else True

        results.append({
            "question": q,
            "expected_doc": expected_doc,
            "doc_match": doc_match,
            "fact_match": fact_match,
            "answer": ans
        })

    eval_df = pd.DataFrame(results)

    answerable = eval_df[eval_df["expected_doc"] != "NONE"]
    unanswerable = eval_df[eval_df["expected_doc"] == "NONE"]

    doc_matches = answerable["doc_match"].sum()
    fact_matches = answerable["fact_match"].sum()
    fallback_matches = unanswerable["fact_match"].sum()

    print(f"Answerable Questions Expected Doc Match: {doc_matches}/{len(answerable)} (Reference: 12/12)")
    print(f"Answerable Questions Expected Fact Match: {fact_matches}/{len(answerable)} (Reference: 12/12)")
    print(f"Unanswerable Fallback Match: {fallback_matches}/{len(unanswerable)} (Reference: 2/2)")
    print()

    # 3. Ablation: Zero-Overlap Evaluation
    print("--- Evaluating Zero-Overlap Chunker (Overlap=0) ---")
    zero_chunks = chunk_documents(overlap=0, chunk_size=40)
    print(f"Total zero-overlap chunks: {len(zero_chunks)}")
    zero_retriever = PolicyRetriever(zero_chunks)

    zero_results = []
    failed_questions = []

    for _, row in df_eval.iterrows():
        q = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = str(row["expected_fact"]) if pd.notna(row["expected_fact"]) else ""

        ans, sources, _ = answer_policy_question(q, zero_retriever, k=3)

        if expected_doc == "NONE":
            doc_match = True
            fact_match = ans.startswith("I could not find")
        else:
            doc_match = expected_doc in sources
            fact_match = expected_fact in ans if expected_fact else True

        if not (doc_match and fact_match):
            failed_questions.append({
                "question": q,
                "expected_doc": expected_doc,
                "expected_fact": expected_fact,
                "answer": ans,
                "reason": "Fact or context boundary split across zero-overlap chunks"
            })

        zero_results.append({
            "question": q,
            "expected_doc": expected_doc,
            "doc_match": doc_match,
            "fact_match": fact_match
        })

    zero_df = pd.DataFrame(zero_results)
    zero_ans = zero_df[zero_df["expected_doc"] != "NONE"]
    zero_unans = zero_df[zero_df["expected_doc"] == "NONE"]

    print(f"Zero-Overlap Doc Match: {zero_ans['doc_match'].sum()}/{len(zero_ans)}")
    print(f"Zero-Overlap Fact Match: {zero_ans['fact_match'].sum()}/{len(zero_ans)}")
    print(f"Zero-Overlap Fallback Match: {zero_unans['fact_match'].sum()}/{len(zero_unans)}")
    print(f"Failed Questions in Zero-Overlap: {len(failed_questions)}")
    for f in failed_questions:
        print(f"  Failed: '{f['question']}'")
        print(f"  Expected Fact: '{f['expected_fact']}', Got: '{f['answer']}'")

    return retriever, eval_df, failed_questions

if __name__ == "__main__":
    evaluate_policy_assistant()
