from rouge_score import rouge_scorer
from bert_score import score
import openai


def calculate_rouge(gold_summary, ai_summary):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    all_scores = [scorer.score(" ".join(gold_summary[key]), " ".join(ai_summary[key]))['rougeL'].fmeasure for key in
                  gold_summary]
    return sum(all_scores) / len(all_scores)  # Average ROUGE score


def calculate_bertscore(gold_summary, ai_summary):
    references = [" ".join(gold_summary[key]) for key in gold_summary]
    candidates = [" ".join(ai_summary[key]) for key in ai_summary]
    P, R, F1 = score(candidates, references, model_type="bert-base-uncased")
    return F1.mean().item()


def gpt_self_evaluation(gold_summary, ai_summary):
    prompt = f"""
    Evaluate the following AI-generated summary against the gold standard summary.

    Gold Summary:
    {gold_summary}

    AI-Generated Summary:
    {ai_summary}

    Score the AI generated summary using gold summary from 1 to 5. Only output a number between 1 and 5.
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful AI evaluator."},
                  {"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response.choices[0].message.content

