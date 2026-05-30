from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
import pandas as pd
from pathlib import Path
from typing import List, Dict
import uuid
from datetime import datetime
# from langchain_core.stores import LocalFileStore


class DocumentProcessor:
    def __init__(self, session_id: str = "default", persist_directory: str = "chroma_db"):
        self.session_id = session_id
        self.collection_name = f"due_diligence_{session_id}".lower().replace(" ", "_")
        self.persist_directory = persist_directory
        self.docstore = InMemoryStore()
        # self.docstore = LocalFileStore(
        #  f"./parent_docs/{self.session_id}"
        # )

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # ==================== CHUNKING STRATEGIES ====================
        # Child chunks: Small & precise (for retrieval)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,      # Smaller for better retrieval precision
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Parent chunks: Larger & richer (for final context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,     # Larger for better context
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.vector_store = self._get_or_create_vector_store()
        self.retriever = self._create_parent_retriever()

    def _get_or_create_vector_store(self):
        """Create or load Chroma vector store"""
        Path(self.persist_directory).mkdir(exist_ok=True)
        
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )

    def _create_parent_retriever(self):
        """Create Parent-Document Retriever"""
        return ParentDocumentRetriever(
            vectorstore=self.vector_store,
            child_splitter=self.child_splitter,
            docstore=self.docstore,
            parent_splitter=self.parent_splitter,
            # You can add more configs here if needed
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
            # This automatically creates both parent and child chunks

            self.retriever.add_documents(all_docs)
            print(f"✅ Parent-Child Chunking applied successfully for session: {self.session_id}")

        return {
            "session_id": self.session_id,
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
        """Debug method to see what's stored in Chroma"""
        try:
            count = self.vector_store._collection.count()
            return {
                "collection_name": self.collection_name,
                "total_child_chunks_in_chroma": count,
                "session_id": self.session_id
            }
        except:
            return {"error": "Could not fetch count"}