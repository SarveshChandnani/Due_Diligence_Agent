from pathlib import Path
from dotenv import load_dotenv
from evaluation.runners.rag_runner import RAGRunner

load_dotenv()

documents = [
    Path("evaluation/docs/Zepto_pitchdeck.pdf")
]

with RAGRunner(documents) as runner:

    response = runner.ask(
        "Who founded Zepto?"
    )

    print(response)