# Structured prompts for LLM interactions

# Reranking prompt for relevance scoring
# TODO: Refine this prompt
RERANKING_TASK_PROMPT = """
Given User's query retrieve related memory fragments/snippets that can be used to help an Assistant answer the question.
You are helping the Assistant to remember details about the User
"""
