#chunking symbol

import fitz
import numpy as np
import faiss
import json
from langchain_community.embeddings import OllamaEmbeddings
from transformers import AutoTokenizer
import subprocess #installed standart library
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #


tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")  # works well with phi3 ; loads a pre-trained tokenizer

def split_by_tokens(text, max_tokens=500): #max_tokens: the max number of tokens allowed per chunk
    tokens = tokenizer.encode(text, add_special_tokens=False) # converts input text to a list of tokens
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens) # converts this token slice back to text
        chunks.append(chunk_text)
    return chunks
 
def extractPDF(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text += page.get_text("text")  # Layout-preserving
    doc.close()
    return full_text


def split_by_marker_and_tokens(pdf_path, marker="$$$", max_tokens=500):
    full_text = extractPDF(pdf_path)
    
    # Split the text into sections based on the marker
    raw_sections = full_text.split(marker)
    
    chunks = {}
    for i, section in enumerate(raw_sections):
        section = section.strip()
        if not section:
            continue

        # Use first line of section as title (fallback title if not found)
        lines = section.split('\n')
        title = lines[0].strip() if lines else f"Section {i+1}"
        content = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        if not content:
            continue

        # If token count is low, just keep as is
        token_count = len(tokenizer.encode(content, add_special_tokens=False))
        if token_count <= max_tokens:
            chunks[title] = content
        else:
            # Split long section into token-based parts
            parts = split_by_tokens(content, max_tokens=max_tokens)
            for j, part in enumerate(parts, start=1):
                chunks[f"{title} - Part {j}"] = part

    return chunks


    
# 2. Embed and index using Ollama
def embed_and_index(chunks, index_path="ollama_faiss.index", metadata_path='faiss_metadata.json'):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    all_texts = []
    all_titles = []
    metadata = []

    for title, content in chunks.items():
        subchunks = split_by_tokens(content, max_tokens=500)
        for i, chunk in enumerate(subchunks):
            all_texts.append(chunk)
            all_titles.append(title)
            metadata.append({
                "chapter": title,
                "base_chapter": title.split(" - Part")[0],
                "subchunk_index": i + 1,
                "total_subchunks": len(subchunks),
                "text": chunk
            })


    # Embed all chunks
    vectors = [embedding_model.embed_query(t) for t in all_texts]
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, index_path)

    # Save metadata (title + text together)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved metadata to {metadata_path}")
    return metadata  # Return list of dicts

# 3. Search with query
def search_index(query, metadata, index_path="ollama_faiss.index", k=3):

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    query_vec = embedding_model.embed_query(query)

    index = faiss.read_index(index_path)
    D, I = index.search(np.array([query_vec]).astype("float32"), k)

    top_chunks = []
    for i in I[0]:
        entry = metadata[i]
        top_chunks.append(entry)
    return top_chunks

"""4. View the full FAISS database"""
def view_faiss_dataset(index_path='ollama_faiss.index', metadata_path='faiss_metadata.json', export_path='faiss_dataset_output.json'):
    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vectors = index.reconstruct_n(0, index.ntotal)

    print(f"\n FAISS contains {index.ntotal} vectors of dimension {index.d}.\n")

    for i in range(index.ntotal):
        entry = metadata[i]
        print(f"--- Chunk {i} ---")
        print(f"Chapter Title: {entry['chapter']}")
        print(f"Subchunk: {entry['subchunk_index']} / {entry['total_subchunks']}")
        print("Text:", entry['text'][:300].replace("\n", " ") + "...")
        print("Vector (first 5 dims):", vectors[i][:5])
        print()

    # Optionally export
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Exported dataset to {export_path}")

    # Export FAISS Dataset to File (actually it doesn't affect so it just optional)
    export_data = [{"index": i, "text": texts[i], "vector": vectors[i].tolist()} for i in range(index.ntotal)]

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Exported full FAISS dataset to {export_path}")

def ask_ollama(prompt, model="phi3"): # 1B-class model
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        stdout, stderr = process.communicate(input=(prompt + "\n").encode("utf-8"), timeout=60)

        if stderr:
            err = stderr.decode("utf-8").strip()
            if err:
                print("Error from Ollama:", err)

        return stdout.decode("utf-8").strip()

    except subprocess.TimeoutExpired:
        process.kill()
        return "LLM timed out after 60 seconds."

def rag_query(query, metadata, index_path="ollama_faiss.index", model="phi3", k=3):
    # get top-k most relevant subchunks
    top_chunks = search_index(query, metadata, index_path=index_path, k=k)
    # Step 2: Get unique chapter names from those top subchunks
    top_chapter_names = list({entry['base_chapter'] for entry in top_chunks})  # using set to deduplicate

    # Step 3: For each chapter, gather all subchunks (from metadata)
    chapter_to_subchunks = {ch: [] for ch in top_chapter_names}
    for entry in metadata:
        if entry['base_chapter'] in chapter_to_subchunks:
            chapter_to_subchunks[entry['base_chapter']].append(entry)

    # Step 4: Sort subchunks by subchunk index within each chapter
    for ch in chapter_to_subchunks:
        chapter_to_subchunks[ch].sort(key=lambda x: x['subchunk_index'])

    # Step 5: Build context from entire chapter content
    context_parts = []
    for chapter, entries in chapter_to_subchunks.items():
        
        for entry in entries:
            context_parts.append(
                f"Chapter: {entry['chapter']} (Part {entry['subchunk_index']}/{entry['total_subchunks']})\n{entry['text']}"
            )

    context = "\n\n---\n\n".join(context_parts)

    final_prompt = f"""You are an expert assistant. Answer the user's question using **only** the information provided in the context below.

If the answer is not found in the context, reply: "I don't have enough information to answer that."


Context:
{context}

Question: {query}

Answer:"""
    print(context)
    answer = ask_ollama(final_prompt, model=model)
    print("\nLLM Answer:\n", answer)




def view_faiss_dataset_vectors(index_path='ollama_faiss.index', metadata_path='faiss_metadata.json'):
    # load FAISS index
    index = faiss.read_index(index_path)

    # load metadata (chapter name + subchunk + text)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # extract the raw vectors from FAISS
    vectors = index.reconstruct_n(0, index.ntotal)

    print(f"FAISS contains {index.ntotal} vectors of dimension {index.d}\n")

    # print each vector with its text and chapter
    for i in range(index.ntotal):
        entry = metadata[i]
        print(f"--- Chunk {i} ---")
        print(f"Chapter: {entry['chapter']} (Part {entry['subchunk_index']}/{entry['total_subchunks']})")
        print("Text Preview:", entry['text'][:120].replace("\n", " ") + "...")
        print("Vector (first 10 numbers):", np.round(vectors[i][:10], 4))  # truncate for readability
        print()


if __name__ == "__main__":
    pdf = "somepdf.pdf"
    chunks = split_by_marker_and_tokens(pdf)
    texts = embed_and_index(chunks)
    
    user_input = input('Do you want to view the full FAISS database? (y/n): ')
    if user_input.strip().lower() == 'y':
        view_faiss_dataset()

    while True:
        user_query = input("Enter a search query: (press 'q' to quit) ")
        if user_query == 'q':
            break
        rag_query(user_query, texts)

