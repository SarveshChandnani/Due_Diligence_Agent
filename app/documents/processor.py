# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
# from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain_core.stores import InMemoryStore
# import pandas as pd
# from pathlib import Path
# from typing import List, Dict
# import uuid
# from datetime import datetime
# # from langchain_core.stores import LocalFileStore


# class DocumentProcessor:
#     def __init__(self, session_id: str = "default", persist_directory: str = "chroma_db"):
#         self.session_id = session_id
#         self.collection_name = f"due_diligence_{session_id}".lower().replace(" ", "_")
#         self.persist_directory = persist_directory
#         self.docstore = InMemoryStore()
#         # self.docstore = LocalFileStore(
#         #  f"./parent_docs/{self.session_id}"
#         # )

#         self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#         # ==================== CHUNKING STRATEGIES ====================
#         # Child chunks: Small & precise (for retrieval)
#         self.child_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=600,      # Smaller for better retrieval precision
#             chunk_overlap=100,
#             separators=["\n\n", "\n", ".", " ", ""]
#         )

#         # Parent chunks: Larger & richer (for final context)
#         self.parent_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1800,     # Larger for better context
#             chunk_overlap=300,
#             separators=["\n\n", "\n", ".", " ", ""]
#         )

#         self.vector_store = self._get_or_create_vector_store()
#         self.retriever = self._create_parent_retriever()

#     def _get_or_create_vector_store(self):
#         """Create or load Chroma vector store"""
#         Path(self.persist_directory).mkdir(exist_ok=True)
        
#         return Chroma(
#             persist_directory=self.persist_directory,
#             embedding_function=self.embeddings,
#             collection_name=self.collection_name
#         )

#     def _create_parent_retriever(self):
#         """Create Parent-Document Retriever"""
#         return ParentDocumentRetriever(
#             vectorstore=self.vector_store,
#             child_splitter=self.child_splitter,
#             docstore=self.docstore,
#             parent_splitter=self.parent_splitter,
#             # You can add more configs here if needed
#         )

#     def ingest_multiple_documents(self, file_paths: List[str]) -> Dict:
#         """Ingest documents using Parent-Child chunking"""
#         all_docs = []
#         summary = []

#         for file_path in file_paths:
#             file_path = Path(file_path)
#             result = self._process_single_file(file_path)
            
#             if result["status"] == "success":
#                 all_docs.extend(result["documents"])
#                 summary.append({
#                     "file": file_path.name,
#                     "status": "success"
#                 })
#             else:
#                 summary.append({
#                     "file": file_path.name,
#                     "status": "failed",
#                     "error": result.get("error")
#                 })

#         if all_docs:
#             # This automatically creates both parent and child chunks

#             self.retriever.add_documents(all_docs)
#             print(f"✅ Parent-Child Chunking applied successfully for session: {self.session_id}")

#         return {
#             "session_id": self.session_id,
#             "total_files": len(file_paths),
#             "successful": len([s for s in summary if s["status"] == "success"]),
#             "summary": summary,
#             "total_chunks": len(all_docs),
#         }

#     def _process_single_file(self, file_path: Path) -> Dict:
#         """Process single file and add metadata"""
#         try:
#             if file_path.suffix.lower() == ".pdf":
#                 loader = PyPDFLoader(str(file_path))
#                 docs = loader.load()

#             elif file_path.suffix.lower() in [".md", ".markdown"]:
#                 loader = UnstructuredMarkdownLoader(str(file_path))
#                 docs = loader.load()

#             elif file_path.suffix.lower() in [".xlsx", ".xls"]:
#                 df = pd.read_excel(file_path)
#                 text = f"Financial Model: {file_path.name}\n\n" + df.to_string(index=False)
#                 docs = [Document(page_content=text, metadata={"source": file_path.name})]

#             else:
#                 return {"status": "failed", "error": f"Unsupported format: {file_path.suffix}"}

#             # Add rich metadata
#             for doc in docs:
#                 doc.metadata.update({
#                     "session_id": self.session_id,
#                     "file_name": file_path.name,
#                     "upload_date": datetime.now().isoformat(),
#                     "document_type": file_path.suffix.lower()[1:],
#                     "chunk_id": str(uuid.uuid4())
#                 })

#             return {"status": "success", "documents": docs}

#         except Exception as e:
#             return {"status": "failed", "error": str(e)}

#     def get_retriever(self, k: int = 8):
#         """Return the Parent-Document Retriever"""
#         return self.retriever

#     def similarity_search(self, query: str, k: int = 6):
#         """Direct similarity search with session filter"""
#         return self.vector_store.similarity_search(
#             query, 
#             k=k, 
#             filter={"session_id": self.session_id}
#         )
    
#     def get_storage_info(self):
#         """Debug method to see what's stored in Chroma"""
#         try:
#             count = self.vector_store._collection.count()
#             return {
#                 "collection_name": self.collection_name,
#                 "total_child_chunks_in_chroma": count,
#                 "session_id": self.session_id
#             }
#         except:
#             return {"error": "Could not fetch count"}

# app/documents/processor.py
from langchain_qdrant import QdrantVectorStore,FastEmbedSparse,RetrievalMode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

from qdrant_client import QdrantClient,models
from qdrant_client.http.models import Distance, VectorParams,SparseVectorParams
from rank_bm25 import BM25Okapi

import pandas as pd
from pathlib import Path
from typing import List, Dict
import uuid
from datetime import datetime

from evaluation.models import RAGResponse


class DocumentProcessor:
    def __init__(self, session_id: str = "default", persist_directory: str = "qdrant_db"):
        self.session_id = session_id
        self.collection_name = f"due_diligence_{session_id}".lower().replace(" ", "_")
        self.persist_directory = persist_directory
        self.docstore = InMemoryStore()

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        # ==================== CHUNKING STRATEGIES ====================
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.vector_store = self._get_or_create_vector_store()
        self.retriever = self._create_parent_retriever()

        self.bm25 = None
        self.all_docs_for_bm25 = []   # For keyword search

    def _get_or_create_vector_store(self):
        """Create or load Qdrant vector store"""
        Path(self.persist_directory).mkdir(exist_ok=True)
        
        client = QdrantClient(path=self.persist_directory)  # Local persistent storage

        # Create collection if it doesn't exist
        if not client.collection_exists(self.collection_name):
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            #     sparse_vectors_config={
            #        "sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
            #    },
            )

        return QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            # sparse_embedding=self.sparse_embeddings,
            # retrieval_mode=RetrievalMode.HYBRID,
            # vector_name="dense",
            # sparse_vector_name="sparse",
        )

    def _create_parent_retriever(self):
        """Create Parent-Document Retriever"""
        return ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def ingest_multiple_documents(self, file_paths: List[str]) -> Dict:
        """Ingest documents using Parent-Child chunking"""
        all_docs = []
        summary = []

        for file_path in file_paths:
            file_path = Path(file_path)
            result = self._process_single_file(file_path)
            
            if result["status"] == "success":
                all_docs.extend(result["documents"])
                summary.append({
                    "file": file_path.name,
                    "status": "success"
                })
            else:
                summary.append({
                    "file": file_path.name,
                    "status": "failed",
                    "error": result.get("error")
                })

        if all_docs:
            self.retriever.add_documents(all_docs)
            self.all_docs_for_bm25.extend(all_docs)
            self._build_bm25_index()
            print(f"✅ Qdrant + Parent-Child Chunking applied successfully for session: {self.session_id}")

        return {
            "session_id": self.session_id,
            "collection_name": self.collection_name,
            "total_files": len(file_paths),
            "successful": len([s for s in summary if s["status"] == "success"]),
            "summary": summary,
            "total_chunks": len(all_docs),
        }

    def _process_single_file(self, file_path: Path) -> Dict:
        """Process single file and add metadata"""
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

            elif file_path.suffix.lower() in [".md", ".markdown"]:
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()

            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
                text = f"Financial Model: {file_path.name}\n\n" + df.to_string(index=False)
                docs = [Document(page_content=text, metadata={"source": file_path.name})]

            else:
                return {"status": "failed", "error": f"Unsupported format: {file_path.suffix}"}

            # Add rich metadata
            for doc in docs:
                doc.metadata.update({
                    "session_id": self.session_id,
                    "file_name": file_path.name,
                    "upload_date": datetime.now().isoformat(),
                    "document_type": file_path.suffix.lower()[1:],
                    "chunk_id": str(uuid.uuid4())
                })

            return {"status": "success", "documents": docs}

        except Exception as e:
            return {"status": "failed", "error": str(e)}


    def _build_bm25_index(self):
        if self.all_docs_for_bm25:
            tokenized = [doc.page_content.lower().split() for doc in self.all_docs_for_bm25]
            self.bm25 = BM25Okapi(tokenized)

    def get_retriever(self, k: int = 8):
        """Return the Parent-Document Retriever"""
        return self.retriever

    def similarity_search(self, query: str, k: int = 6):
        """Direct similarity search with session filter"""
        return self.vector_store.similarity_search(
            query, 
            k=k, 
            filter={"session_id": self.session_id}
        )

    def get_storage_info(self):
        """Debug method"""
        try:
            count = self.vector_store.client.count(collection_name=self.collection_name).count
            return {
                "collection_name": self.collection_name,
                "total_vectors_in_qdrant": count,
                "session_id": self.session_id
            }
        except:
            return {"error": "Could not fetch count"}
        
    def hybrid_search(self, query: str, k: int = 8, rrf_k: int = 60):
        """
        Hybrid Search using ParentDocumentRetriever + BM25 boost
        """
        # 1. Parent-Child Retrieval (Main Semantic Search)
        parent_results = self.retriever.invoke(query)[:k*2]

        # 2. BM25 Keyword Search (for term precision)
        bm25_results = []
        if self.bm25 and self.all_docs_for_bm25:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm25_results = [self.all_docs_for_bm25[i] for i in top_indices]

        # 3. Simple Fusion (can be improved with RRF later)
        rrf_scores = {}

        # Score from Parent-Child Retriever
        for rank, doc in enumerate(parent_results):
            key = doc.page_content[:200]  # Use longer key for better deduplication
            if key not in rrf_scores:
                rrf_scores[key] = {
                    "doc": doc,
                    "score": 0.0
                }
            rrf_scores[key]["score"] += 1.0 / (rank + rrf_k)

        # Score from BM25
        for rank, doc in enumerate(bm25_results):
            key = doc.page_content[:200]
            if key not in rrf_scores:
                rrf_scores[key] = {
                    "doc": doc,
                    "score": 0.0
                }
            rrf_scores[key]["score"] += 1.0 / (rank + rrf_k)

        # Sort by RRF score and return top k
        sorted_results = sorted(
            rrf_scores.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )

        final_results = [item["doc"] for item in sorted_results[:k]]
        
        return final_results


    def ask(self, question: str, k: int = 5):
            """
            Simple RAG interface for evaluation.

            Returns:
            {
                "question": ...,
                "answer": ...,
                "contexts": [...]
            }
            """

            docs = self.hybrid_search(
                query=question,
                k=k
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0
            )

            prompt = f"""
                You are a due diligence assistant.

                Answer ONLY from the provided context.

                If the answer is not present in the context, reply:

                "I don't have enough information."

                Context:
                {context}

                Question:
                {question}
                """

            response = llm.invoke(prompt)

            return RAGResponse(
                question=question,
                answer=response.content,
                contexts=[doc.page_content for doc in docs],
                sources=[
                    doc.metadata
                    for doc in docs
                ]
            )

    def close(self) -> None:
        """
        Gracefully close all resources used by the DocumentProcessor.
        """

        try:
            if hasattr(self, "vector_store") and self.vector_store is not None:
                client = getattr(self.vector_store, "client", None)

                if client is not None:
                    client.close()

        except Exception as e:
            print(f"Warning while closing Qdrant client: {e}")