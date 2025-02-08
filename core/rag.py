import google.generativeai as genai
import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAG:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=512, chunk_overlap=100,
                 index_path="faiss_index", metadata_path="chunks.json"):
        """
        Initialize RAG pipeline with FAISS and text chunking.
        """
        self.embedder = SentenceTransformer(model_name)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()

        # Load or create FAISS index
        if os.path.exists(self.index_path + ".index"):
            self.index = faiss.read_index(self.index_path + ".index")
            self._load_metadata()
        else:
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.transcript_chunks = {}

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def _load_metadata(self):
        """Loads existing transcript chunks from a JSON metadata file."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.transcript_chunks = json.load(f)
        else:
            self.transcript_chunks = {}

    def _save_metadata(self):
        """Saves transcript chunks to a JSON file."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.transcript_chunks, f, indent=4)

    def add_transcript(self, transcript_text):
        """
        Processes and stores a Zoom transcript in FAISS after chunking.
        """
        chunks = self.text_splitter.split_text(transcript_text)
        chunk_vectors = self.embedder.encode(chunks, convert_to_numpy=True)

        # Track new chunk indices
        start_idx = len(self.transcript_chunks)
        for i, chunk in enumerate(chunks):
            self.transcript_chunks[start_idx + i] = chunk

        # Add vectors to FAISS
        self.index.add(chunk_vectors)

        # Save FAISS index and metadata
        faiss.write_index(self.index, self.index_path + ".index")
        self._save_metadata()

        print(f"Stored {len(chunks)} transcript chunks in FAISS.")

    def retrieve_relevant_snippets(self, query, top_k=4):
        """
        Retrieves top-K most relevant transcript snippets.
        """
        query_vector = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, top_k)

        retrieved_chunks = [self.transcript_chunks[str(i)] for i in indices[0] if str(i) in self.transcript_chunks]

        return retrieved_chunks

    def generate_response(self, query, relevant_snippets):
        """
        Uses Google Gemini Pro to generate a response based on retrieved transcript snippets.
        """

        context = "\n\n".join(relevant_snippets)
        prompt = f"Based on the following transcript snippets, answer the question:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Call Gemini Pro
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        return response.text if response else "No response generated."

    def get_output(self, query):
      """
      Returns the output of the RAG pipeline.
      """
      relevant_snippets = self.retrieve_relevant_snippets(query)
      response = self.generate_response(query, relevant_snippets)
      return {
          "relevant_snippets": relevant_snippets,
          "answer": response
      }

# Example Usage
if __name__ == "__main__":
    rag_pipeline = RAG()

    transcript1 = "In today's meeting, we discussed the new product roadmap and launch timeline."
    transcript2 = "The marketing strategy will focus on digital ads and influencer partnerships."

    # Add transcripts to FAISS
    rag_pipeline.add_transcript(transcript1)
    rag_pipeline.add_transcript(transcript2)

    # Query FAISS
    query = "What is the discussion about?"
    response = rag_pipeline.generate_response(query)
    print("\n🤖 AI Response:\n", response)
