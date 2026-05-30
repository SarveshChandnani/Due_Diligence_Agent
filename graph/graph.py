# app/graph/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from graph.state import AgentState
from documents.processor import DocumentProcessor


def create_due_diligence_graph(processor_provider):
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # # Define Nodes
    # def retrieve_documents(state: AgentState):
    #     """Retrieve relevant documents for current session"""
    #     processor = DocumentProcessor(session_id=state["session_id"])
    #     retriever = processor.get_retriever(k=8)
        
    #     last_message = state["messages"][-1].content
    #     docs = retriever.invoke(last_message)
        
    #     context = "\n\n".join([doc.page_content for doc in docs])
        
    #     return {
    #         "retrieved_context": context,
    #         "sources": [{"file": doc.metadata.get("file_name", "unknown")} for doc in docs]
    #     }

    # def generate_response(state: AgentState):
    #     """Generate response using retrieved context"""
    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", """You are a helpful VC Due Diligence Analyst.
    #         Use the provided context from uploaded documents to answer accurately.
    #         Always be professional and mention sources when possible."""),
    #         ("user", """Context: {context}
            
    #         Question: {question}""")
    #     ])
        
    #     context = state.get("retrieved_context", "No documents available.")
    #     question = state["messages"][-1].content
        
    #     response = llm.invoke(
    #         prompt.format_messages(context=context, question=question)
    #     )
        
    #     return {
    #         "messages": [AIMessage(content=response.content)],
    #         "current_step": "response_generated"
    #     }

    # # Build Graph
    # workflow = StateGraph(AgentState)
    
    # workflow.add_node("retrieve", retrieve_documents)
    # workflow.add_node("generate", generate_response)
    
    # workflow.add_edge(START, "retrieve")
    # workflow.add_edge("retrieve", "generate")
    # workflow.add_edge("generate", END)
    
    # # Add persistence (MemorySaver)
    # memory = MemorySaver()
    
    # app = workflow.compile(checkpointer=memory)
    # return app

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    # ====================== NODES ======================

    def planner(state: AgentState):
        """Create a Due Diligence specific plan"""
        prompt = ChatPromptTemplate.from_template(
            """You are a professional VC Due Diligence Analyst.
            Break down the user's question into key due diligence areas.

            User Question: {question}

            Create a clear plan with 4-6 steps focusing on:
            Team, Market, Product/Technology, Traction, Business Model, Risks, etc."""
        )

        response = llm.invoke(prompt.format_messages(question=state["messages"][-1].content))
        
        return {
            "plan": [line.strip() for line in response.content.split("\n") if line.strip()],
            "next": "retrieve"
        }

    def retrieve(state: AgentState):
        """Retrieve relevant information from uploaded documents"""
        # processor = DocumentProcessor(session_id=state["session_id"])
        processor = processor_provider(state["session_id"])
        retriever = processor.get_retriever(k=10)
        
        query = state["messages"][-1].content
        docs = retriever.invoke(query)
        
        print(f"Retrieved {len(docs)} documents for query: '{query}'")  # Debug log

        storage_info = processor.get_storage_info()
        print(f"Storage info: {storage_info}")  # Debug log
        
        context = "\n\n".join([f"Source: {doc.metadata.get('file_name')}\n{doc.page_content}" 
                              for doc in docs])
        print(f"context retrieved: {context}")  # Debug log
        return {
            "retrieved_context": context,
            "sources": [{"file": doc.metadata.get("file_name"), "type": doc.metadata.get("document_type")} 
                       for doc in docs],
            "next": "analyze"
        }

    def analyze(state: AgentState):
        """Deep analysis using Due Diligence lens"""
        prompt = ChatPromptTemplate.from_template(
            """You are an experienced VC analyst performing due diligence.

            Company: {company_name}
            Question: {question}
            Context from documents: {context}

            Provide structured analysis focusing on:
            - Team strength
            - Market opportunity
            - Product/Technology
            - Traction & Metrics
            - Risks & Red flags
            """
        )

        response = llm.invoke(prompt.format_messages(
            company_name=state["company_name"],
            question=state["messages"][-1].content,
            context=state.get("retrieved_context", "No documents found.")
        ))

        return {
            "analysis_per_section": {"detailed_analysis": response.content},
            "next": "synthesize"
        }

    def synthesize(state: AgentState):
        """Synthesize into a professional response"""
        prompt = ChatPromptTemplate.from_template(
            """Create a professional, structured response as a VC Due Diligence Analyst.

            Company: {company_name}
            User Question: {question}
            Analysis: {analysis}

            Be insightful, balanced, and mention key risks."""
        )

        response = llm.invoke(prompt.format_messages(
            company_name=state["company_name"],
            question=state["messages"][-1].content,
            analysis=state.get("analysis_per_section", {})
        ))

        return {
            "messages": [AIMessage(content=response.content)],
            "final_memo": response.content,
            "next": END
        }

    # ====================== BUILD GRAPH ======================
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("analyze", analyze)
    workflow.add_node("synthesize", synthesize)

    # Define flow
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "retrieve")
    workflow.add_edge("retrieve", "analyze")
    workflow.add_edge("analyze", "synthesize")
    workflow.add_edge("synthesize", END)

    # Persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app