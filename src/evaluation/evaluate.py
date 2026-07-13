"""
RAGAS Evaluation Script
Measures RAG quality: faithfulness, answer relevancy, context precision, context recall
"""
import json
import os
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()


GOLDEN_DATASET = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "The main topic is described in the introduction section of the document."
    },
    {
        "question": "What are the key findings mentioned?",
        "ground_truth": "The key findings are outlined in the results or conclusions section."
    },
    {
        "question": "Who are the intended users of this system?",
        "ground_truth": "The intended users are described in the scope or audience section."
    },
    {
        "question": "What methodology was used?",
        "ground_truth": "The methodology is described in the methods or approach section."
    },
    {
        "question": "What are the limitations mentioned?",
        "ground_truth": "The limitations are discussed in the limitations or future work section."
    },
]


def run_evaluation(retriever, gateway):
    """
    Run RAGAS evaluation on the golden dataset
    Requires: ragas, datasets packages
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("Install ragas: pip install ragas datasets")
        return None

    print(f"Running RAGAS evaluation on {len(GOLDEN_DATASET)} golden questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in GOLDEN_DATASET:
        query = item["question"]
        gt = item["ground_truth"]

        # Retrieve
        docs = retriever.retrieve(query)
        context_texts = [d.page_content for d in docs[:5]]

        # Generate
        context_str = "\n\n".join(context_texts)
        response = gateway.generate(query, context_str)

        questions.append(query)
        answers.append(response.text)
        contexts.append(context_texts)
        ground_truths.append(gt)

        print(f"  Evaluated: {query[:50]}...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:      {results['faithfulness']:.4f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.4f}")
    print(f"Context Precision: {results['context_precision']:.4f}")
    print(f"Context Recall:    {results['context_recall']:.4f}")
    print("="*50)

    output = {
        "faithfulness": float(results["faithfulness"]),
        "answer_relevancy": float(results["answer_relevancy"]),
        "context_precision": float(results["context_precision"]),
        "context_recall": float(results["context_recall"]),
        "num_questions": len(GOLDEN_DATASET),
    }

    with open("ragas_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved to ragas_results.json")
    return output


if __name__ == "__main__":
    from src.retrieval.retriever import HybridRetriever
    from src.gateway.llm_gateway import LLMGateway

    retriever = HybridRetriever()
    gateway = LLMGateway()

    print("Note: Build retriever with documents first before running evaluation")
    print("Usage: from src.evaluation.evaluate import run_evaluation")
    print("Then: run_evaluation(retriever, gateway)")