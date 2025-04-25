#!/usr/bin/env python3
"""
Deep Evaluation of RAG Systems using deepeval

This script demonstrates the use of the deepeval library to perform comprehensive
evaluations of Retrieval-Augmented Generation (RAG) systems. It covers various 
evaluation metrics and provides a framework for creating and running test cases.

Key Components:
1. Correctness Evaluation
2. Faithfulness Evaluation
3. Contextual Relevancy Evaluation
4. Combined Evaluation of Multiple Metrics
5. Batch Test Case Creation

Evaluation Metrics:
1. Correctness (GEval): Evaluates whether the actual output is factually correct
2. Faithfulness (FaithfulnessMetric): Assesses answer faithfulness to the context
3. Contextual Relevancy: Evaluates relevance of retrieved context to query and answer
"""

from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def test_correctness():
    """Test correctness of LLM output against expected output"""
    print("\n=== Testing Correctness ===")
    
    correctness_metric = GEval(
        name="Correctness",
        model="gpt-4o",
        evaluation_params=[
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        evaluation_steps=[
            "Determine whether the actual output is factually correct based on the expected output."
        ],
    )

    gt_answer = "Madrid is the capital of Spain."
    pred_answer = "MadriD."

    test_case_correctness = LLMTestCase(
        input="What is the capital of Spain?",
        expected_output=gt_answer,
        actual_output=pred_answer,
    )

    correctness_metric.measure(test_case_correctness)
    print(f"Correctness score: {correctness_metric.score}")
    return test_case_correctness, correctness_metric


def test_faithfulness():
    """Test faithfulness of LLM output to the provided context"""
    print("\n=== Testing Faithfulness ===")
    
    question = "what is 3+3?"
    context = ["6"]
    generated_answer = "6"

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        retrieval_context=context
    )

    faithfulness_metric.measure(test_case)
    print(f"Faithfulness score: {faithfulness_metric.score}")
    print(f"Reason: {faithfulness_metric.reason}")
    return test_case, faithfulness_metric


def test_contextual_relevancy():
    """Test contextual relevancy of retrieved documents"""
    print("\n=== Testing Contextual Relevancy ===")
    
    actual_output = "then go somewhere else."
    retrieval_context = [
        "this is a test context",
        "mike is a cat",
        "if the shoes don't fit, then go somewhere else."
    ]
    gt_answer = "if the shoes don't fit, then go somewhere else."

    relevance_metric = ContextualRelevancyMetric(
        threshold=1,
        model="gpt-4o",
        include_reason=True
    )
    
    relevance_test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=gt_answer,
    )

    relevance_metric.measure(relevance_test_case)
    print(f"Contextual relevancy score: {relevance_metric.score}")
    print(f"Reason: {relevance_metric.reason}")
    return relevance_test_case, relevance_metric


def create_deep_eval_test_cases(questions, gt_answers, generated_answers, retrieved_documents):
    """
    Create multiple LLMTestCases based on four lists:
    - Questions
    - Ground Truth Answers
    - Generated Answers  
    - Retrieved Documents (each element is a list)
    """
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]


def run_combined_evaluation(test_cases, metrics):
    """Run evaluation with multiple test cases and metrics"""
    print("\n=== Running Combined Evaluation ===")
    evaluate(
        test_cases=test_cases,
        metrics=metrics
    )


def main():
    # Run individual tests
    _, correctness_metric = test_correctness()
    _, faithfulness_metric = test_faithfulness()
    relevance_test_case, relevance_metric = test_contextual_relevancy()
    
    # Create an additional test case
    spain_test_case = LLMTestCase(
        input="What is the capital of Spain?",
        expected_output="Madrid is the capital of Spain.",
        actual_output="MadriD.",
        retrieval_context=["Madrid is the capital of Spain."]
    )
    
    # Run combined evaluation
    run_combined_evaluation(
        test_cases=[relevance_test_case, spain_test_case],
        metrics=[correctness_metric, faithfulness_metric, relevance_metric]
    )
    
    # Example of batch test case creation
    print("\n=== Example Batch Test Case Creation ===")
    questions = ["What is 2+2?", "What is the capital of France?"]
    gt_answers = ["4", "Paris is the capital of France."]
    generated_answers = ["The answer is 4.", "Paris."]
    retrieved_documents = [
        ["2+2=4", "Basic arithmetic"],
        ["Paris is the capital of France.", "France is a country in Europe."]
    ]
    
    batch_test_cases = create_deep_eval_test_cases(
        questions, gt_answers, generated_answers, retrieved_documents
    )
    print(f"Created {len(batch_test_cases)} test cases")


if __name__ == "__main__":
    main() 