# extractor.py

from app.graph.state import DDState


EXTRACTION_QUERIES = {

    "company_name":
        "What is the company name?",

    "company_overview":
        "What problem does the company solve?",

    "industry":
        "Which industry does this startup operate in?",

    "founders":
        "Who are the founders and their backgrounds?",

    "product":
        "Describe the product and its differentiators.",

    "financials":
        "Extract ARR, MRR, revenue, burn rate and growth metrics."
}


def extractor_node(state:DDState, processor_provider, llm):

    extracted = {}

    for key, question in EXTRACTION_QUERIES.items():

        docs = processor_provider(state["session_id"]).hybrid_search(
            query=question,
            k=5
        )

        extracted[key] = llm.invoke(
            f"""
   Question:
   {question}

   Context:
   {docs}

Answer concisely.
If unavailable, say Not Found.
"""
        ).content

    return {
        "extracted_data": extracted
    }