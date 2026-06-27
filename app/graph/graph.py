# app/graph/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from app.graph.state import AgentState
from app.documents.processor import DocumentProcessor
from app.tools.web_search import web_search
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import create_agent


def create_due_diligence_graph(processor_provider):
#     llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

#     # ====================== NODES ======================





#     def planner(state: AgentState):
#         """Create a Due Diligence specific plan"""
#         prompt = ChatPromptTemplate.from_template(
#             """You are a professional VC Due Diligence Analyst.
#             Break down the user's question into key due diligence areas.

#             User Question: {question}

#             Create a clear plan with 4-6 steps focusing on:
#             Team, Market, Product/Technology, Traction, Business Model, Risks, etc."""
#         )

#         response = llm.invoke(prompt.format_messages(question=state["messages"][-1].content))
        
#         return {
#             "plan": [line.strip() for line in response.content.split("\n") if line.strip()],
#             "next": "retrieve"
#         }

#     def retrieve(state: AgentState):
#         """Retrieve relevant information from uploaded documents"""
#         # processor = DocumentProcessor(session_id=state["session_id"])
#         processor = processor_provider(state["session_id"])
#         final_results = processor.hybrid_search(query=state["messages"][-1].content, k=10)
#         # retriever = processor.get_retriever(k=10)
        
#         # query = state["messages"][-1].content
#         # docs = retriever.invoke(query)
        
#         print(f"Retrieved {len(final_results)} documents for query: '{final_results}'")  # Debug log

#         storage_info = processor.get_storage_info()
#         print(f"Storage info: {storage_info}")  # Debug log
        
#         context = "\n\n".join([f"Source: {doc.metadata.get('file_name')}\n{doc.page_content}" 
#                               for doc in final_results])
#         print(f"context retrieved: {context}")  # Debug log
#         return {
#             "retrieved_context": context,
#             "sources": [{"file": doc.metadata.get("file_name"), "type": doc.metadata.get("document_type")} 
#                        for doc in final_results],
#             "next": "analyze"
#         }

#     def analyze(state: AgentState):
#         """Deep analysis using Due Diligence lens"""
#         prompt = ChatPromptTemplate.from_template(
#             """You are an experienced VC analyst performing due diligence, answer the question using context as a VC analyst would answer.

#             Company: {company_name}
#             Question: {question}
#             Context from documents: {context}
#             """
#         )

#         response = llm.invoke(prompt.format_messages(
#             company_name=state["company_name"],
#             question=state["messages"][-1].content,
#             context=state.get("retrieved_context", "No documents found.")
#         ))

#         return {
#             "messages": [AIMessage(content=response.content)],
#             "analysis_per_section": {"detailed_analysis": response.content},
#             "final_memo": response.content,
#             "next": "synthesize"
#         }

#     def synthesize(state: AgentState):
#         """Synthesize into a professional response"""
#         prompt = ChatPromptTemplate.from_template(
#             """Create a professional, structured response as a VC Due Diligence Analyst.

#             Company: {company_name}
#             User Question: {question}
#             Analysis: {analysis}
# """
#         )

#         response = llm.invoke(prompt.format_messages(
#             company_name=state["company_name"],
#             question=state["messages"][-1].content,
#             analysis=state.get("analysis_per_section", {})
#         ))

#         return {
#             "messages": [AIMessage(content=response.content)],
#             "final_memo": response.content,
#             "next": END
#         }

#     # ====================== BUILD GRAPH ======================
#     workflow = StateGraph(AgentState)

#     # workflow.add_node("planner", planner)
#     workflow.add_node("retrieve", retrieve)
#     workflow.add_node("analyze", analyze)
#     # workflow.add_node("synthesize", synthesize)

#     # Define flow
#     workflow.add_edge(START, "retrieve")
#     # workflow.add_edge("planner", "retrieve")
#     workflow.add_edge("retrieve", "analyze")
#     workflow.add_edge("analyze", END)
#     # workflow.add_edge("synthesize", END)

#     # Persistence
#     memory = MemorySaver()
#     app = workflow.compile(checkpointer=memory)

#     return app

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.15)
        tools = [web_search]
        llm_with_tools = llm.bind_tools(tools)

        team_agent = create_agent(
            model=llm,
            tools=[web_search],
        )

        market_agent = create_agent(
            model=llm,
            tools=[web_search],
        )

        # ====================== NODES ======================

        # def planner(state: AgentState):
        #     return {"next": "team_analyst"}

        def team_analyst(state: AgentState):
            """Detailed Team Analysis"""

            print(f"team_analyst called with: {state['messages'][-1].content[0:50]}")  # Debug log
            processor = processor_provider(state["session_id"])
            
            # More targeted and detailed query
            docs = processor.hybrid_search(
                query="founders background experience track record previous companies leadership team gaps red flags",
                k=2
            )
            
            prompt = f"""
                You are a Senior VC Due Diligence Analyst.

                Company: {state['company_name']}

                Use the following document context as your PRIMARY source.

                Context:
                {docs}

                Only use web_search if:
                - founder information is missing
                - recent developments are required
                - information needs verification

                Focus on:
                - founders' backgrounds
                - previous successes/failures
                - complementary skills
                - leadership gaps
                - red flags

                Produce a detailed assessment.
                """

            result = team_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=prompt)
                    ]
                }
            )

            return {
                "team_assessment": result["messages"][-1].content
            }

        def market_analyst(state: AgentState):
            print(f"market_analyst called with: {state['messages'][-1].content[0:50]}")  # Debug log
            """Detailed Market & Competitive Analysis"""
            processor = processor_provider(state["session_id"])
            
            docs = processor.hybrid_search(
                query="market opportunity TAM SAM SOM growth trends customer pain points go-to-market strategy competitors differentiation",
                k=2
            )
            
            prompt = f"""
                You are a Senior VC Due Diligence Analyst.

                Company: {state['company_name']}

                Use the following context as your PRIMARY source.

                Context:
                {docs}

                Only use web_search if:
                - market data is missing
                - competitor information is outdated
                - recent market developments are needed

                Focus on:
                - TAM/SAM/SOM
                - competitors
                - market timing
                - customer pain points
                - moat
                """

            result = market_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=prompt)
                    ]
                }
            )

            return {
                "market_analysis": result["messages"][-1].content
            }

        def financial_analyst(state: AgentState):
            print(f"financial_analyst called with: {state['messages'][-1].content[0:50]}")  # Debug log
            """Detailed Financial Analysis"""
            processor = processor_provider(state["session_id"])
            
            docs = processor.hybrid_search(
                query="financials ARR revenue growth burn rate runway unit economics projections assumptions cap table",
                k=5
            )
            
            prompt = f"""You are a senior VC Due Diligence Analyst.
            Perform detailed financial analysis and projection review.
            
            Company: {state['company_name']}
            Context from documents: {docs}

            Focus on:
            - Current financial health and key metrics
            - Unit economics and revenue model
            - Burn rate, runway, and capital efficiency
            - Reasonableness of future projections"""

            response = llm.invoke(prompt)
            return {"financial_analysis": response.content,  "messages": [response]}

        def legal_analyst(state: AgentState):
            print(f"legal_analyst called with: {state['messages'][-1].content[0:50]}")  # Debug log
            """Detailed Legal, IP & Compliance"""
            processor = processor_provider(state["session_id"])
            
            docs = processor.hybrid_search(
                query="legal structure cap table IP ownership patents compliance regulatory risks litigation",
                k=5
            )
            
            prompt = f"""You are a senior VC Due Diligence Analyst.
            Review legal, IP, ownership, and compliance risks.
    
            Company: {state['company_name']}
            Context from documents: {docs}

            Focus on:
            - Corporate structure and cap table cleanliness
            - IP ownership and protection
            - Regulatory and compliance risks
            - Any potential legal red flags"""

            response = llm.invoke(prompt)
            return {"legal_ip_review": response.content,  "messages": [response]}

        def product_analyst(state: AgentState):
            print(f"product_analyst called with: {state['messages'][-1].content[0:50]}")  # Debug log
            """Detailed Product, Technology & Risks"""
            processor = processor_provider(state["session_id"])
            
            docs = processor.hybrid_search(
                query="product technology differentiation roadmap scalability technical risks operations key partnerships",
                k=5
            )
            
            prompt = f"""You are a senior VC Due Diligence Analyst.
            Evaluate product, technology, operations, and associated risks.
       
            Company: {state['company_name']}
            Context from documents: {docs}

            Focus on:
            - Product viability and customer validation
            - Technical differentiation and moat
            - Scalability and operational setup
            - Major execution and technology risks"""

            response = llm.invoke(prompt)
            return {"product_tech_risks": response.content,  "messages": [response]}

        def synthesizer(state: AgentState):
            print(f"synthesizer called with: {state['messages'][-1].content[0:50]}")  # Debug log
            """Final Report Synthesis"""
            prompt = f"""You are a senior VC partner writing a final due diligence report.

            Company: {state['company_name']}

            1. Team and Leadership Assessment:
            {state.get('team_assessment', 'No data')}

            2. Market Opportunity and Competitive Analysis:
            {state.get('market_analysis', 'No data')}

            3. Financial Health and Projections:
            {state.get('financial_analysis', 'No data')}

            4. Legal, IP, and Compliance Review:
            {state.get('legal_ip_review', 'No data')}

            5. Product/Technology, Operations, and Risks:
            {state.get('product_tech_risks', 'No data')}

            Write a professional, balanced, and insightful final report."""

            response = llm.invoke(prompt)
            
            return {
                "final_report": response.content,
                "messages": [AIMessage(content=response.content)],
            }

        # ====================== BUILD GRAPH ======================
        workflow = StateGraph(AgentState)

        workflow.add_node("team_analyst", team_analyst)
        workflow.add_node("market_analyst", market_analyst)
        workflow.add_node("financial_analyst", financial_analyst)
        workflow.add_node("legal_analyst", legal_analyst)
        workflow.add_node("product_analyst", product_analyst)
        workflow.add_node("synthesizer", synthesizer)

        # team_tools = ToolNode(tools=[web_search])
        # market_tools = ToolNode(tools=[web_search])
        # workflow.add_node("team_tools", team_tools)
        # workflow.add_node("market_tools", market_tools)

        # Edges
        workflow.add_edge(START, "team_analyst")
        # workflow.add_edge("planner", "team_analyst")
        workflow.add_edge("team_analyst", "market_analyst")
        workflow.add_edge("market_analyst", "financial_analyst")
        workflow.add_edge("financial_analyst", "legal_analyst")
        workflow.add_edge("legal_analyst", "product_analyst")
        workflow.add_edge("product_analyst", "synthesizer")
        workflow.add_edge("synthesizer", END)

        # Tool conditional edges
        # workflow.add_conditional_edges("market_analyst", tools_condition, {"tools": "market_tools", END: "financial_analyst"})
        # workflow.add_conditional_edges("team_analyst", tools_condition, {"tools": "team_tools", END: "market_analyst"})
      

        # workflow.add_edge("team_tools", "team_analyst")   
        # workflow.add_edge("market_tools", "market_analyst")

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
