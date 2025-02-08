from call_analysis import CallAnalysis
from rag import RAG
from utils.eval import calculate_rouge, calculate_bertscore, gpt_self_evaluation
from utils.helper import load_file

def run_test():
    """
    Runs all provided tasks for assignment
    :return:
    """
    transcript = load_file('docs/1f9594ae-ed7a-465b-a0a8-3946095bab1d.vtt')
    task1 = CallAnalysis(transcript)
    task_1_output = task1.get_output()

    print("Output from Task 1: ", task_1_output)

    # Initialize faiss indexes and its corresponding chunks stored in the system
    task2 = RAG()
    task2.add_transcript(transcript)
    question = "What is brain trust?"
    task_2_output = task2.get_output(question)
    print(f"Output from Task 2: Question->{question} \n Answer -> {task_2_output['answer']} "
          f"\n Extracted snippets -> {task_2_output['relevant_snippets']}")

    # Example input
    gold_summary = {
        "key_decisions": ["Finalize roadmap by Friday."],
        "action_items": ["Confirm pricing strategy."],
        "unresolved_questions": ["Are we short on engineering bandwidth?"]
    }

    ai_summary = {
        "key_decisions": ["Finalize the product roadmap."],
        "action_items": ["Define pricing strategy."],
        "unresolved_questions": ["Do we have enough engineers?"]
    }

    # Compute evaluation metrics
    rouge_score = calculate_rouge(gold_summary, ai_summary)
    bertscore = calculate_bertscore(gold_summary, ai_summary)
    gpt_score = gpt_self_evaluation(gold_summary, ai_summary)  # Uncomment if GPT-4 API is available
    print("Output of task three")

    print("ROUGE Score:", rouge_score)
    print("BERTScore:", bertscore)
    print("GPT Self-Evaluation:", gpt_score, " out of 5")


if __name__ == "__main__":
    run_test()