"""
Local LLM interface for ADA.

Python is the source of truth for calculations. The local model explains
the supplied profile and must not invent statistics or claim it performed
calculations that were not supplied.
"""

import ollama


MODEL_NAME = "llama3.2:1b"


def generate_response(prompt, system_prompt=None, num_predict=320):
    """Generate a response from the locally running Ollama model."""

    if not isinstance(prompt, str) or not prompt.strip():
        return "Please enter a question."

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt.strip()
        })

    messages.append({
        "role": "user",
        "content": prompt.strip()
    })

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.1,
                "num_predict": num_predict,
                "num_ctx": 4096
            }
        )

        answer = response.get("message", {}).get("content", "").strip()

        return answer or (
            "I couldn't form an answer. Please try rephrasing your question."
        )

    except Exception:
        return (
            "I couldn't connect to the local language model. "
            "Please check that Ollama is running and that the "
            "required model is available. Try again later."
        )


def generate_dataset_summary(dataset_information):
    """Generate a dataset-specific explanation from Python-calculated facts."""

    system_prompt = """
You are ADA, an assistant inside an AI Data Analyst application.

Your task is to EXPLAIN the supplied dataset evidence, not calculate new
statistics. Python has already calculated the facts in the profile.

Rules:
- Discuss actual columns and patterns present in the supplied profile.
- Never invent counts, percentages, correlations, causes, or column names.
- Do not claim that data is accurate, reliable, representative, or unbiased.
- Distinguish potential outliers from errors.
- Missing values do not automatically mean rows should be deleted.
- Correlation does not establish causation.
- Do not claim that preprocessing or cleaning was performed unless the
  evidence explicitly says it was performed.
- If a fact is not supplied, say it was not assessed.
- You may repeat numeric values only when they appear in the supplied
  evidence. Do not calculate or estimate new numbers.
- Use clear headings and beginner-friendly language.
- Prioritize the most useful findings; avoid generic filler.
"""

    prompt = f"""
Explain the most important findings in this dataset.

Use these sections:
1. Dataset overview
2. Important patterns
3. Data-quality findings
4. Recommended next steps

PYTHON-CALCULATED DATASET EVIDENCE:
<dataset_evidence>
{str(dataset_information)[:16000]}
</dataset_evidence>

Write a clear explanation grounded only in this evidence.
"""

    return generate_response(prompt, system_prompt, num_predict=380)


def ask_dataset_question(dataset_information, question):
    """Answer a question using the supplied dataset profile as evidence."""

    if not isinstance(question, str) or not question.strip():
        return "Please enter a question."

    system_prompt = """
You are ADA, a helpful conversational data analyst.

Answer the user's question directly using only the supplied dataset evidence.

Rules:
- Use the actual column names and statistics when relevant.
- You may repeat numeric values that appear in the evidence.
- Never invent a statistic or calculate a new one from partial summaries.
- If the evidence does not contain enough information, say what is missing.
- Explain findings in plain language.
- Do not claim that correlation proves causation.
- Treat IQR flags as potential outliers, not automatically errors.
- Do not say that missing values were filled, rows were deleted, or data
  was transformed unless the evidence explicitly confirms that action.
- Avoid generic disclaimers and unnecessary follow-up questions.
- If the question is ambiguous, state the interpretation you are using.
"""

    prompt = f"""
PYTHON-CALCULATED DATASET EVIDENCE:
<dataset_evidence>
{str(dataset_information)[:16000]}
</dataset_evidence>

USER QUESTION:
<question>
{question.strip()}
</question>

Answer the question directly. Do not go beyond the evidence.
"""

    return generate_response(prompt, system_prompt, num_predict=420)
