# app/tools/web_search.py
from langchain_core.tools import tool
from langchain_tavily import TavilySearch 

@tool
def web_search(query: str, max_results: int = 2) -> str:
    """
    Search the internet for up-to-date information.
    
    Use this tool when you need:
    - Latest news about the company or founders
    - Market trends and TAM validation
    - Competitor information
    - Founder background checks or controversies
    - Regulatory or legal updates
    
    Do NOT use this tool for information already available in the uploaded documents.
    Always make the query specific and targeted.
    """
    try:
        search = TavilySearch(
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
        )

        result = search.invoke({"query": query})

        return result

    except Exception as e:
        print(f"Error in web_search tool | query='{query}' | error={e}")

        raise