"""
CacheXL Prompts
Based on paper: CacheXL: Cross-Instance Learning via Online Cache for Efficient and Enhanced LLM Inference

Contains:
- ACTOR_PROMPT: Generate initial rationale, answer, and confidence
- REFLECTOR_PROMPT: Evaluate with retrieved context, decide acceptance and reusability
- ESCALATOR_PROMPT: Re-solve difficult cases when both confidences are low
"""

# Actor Prompt
# 论文 Section: Actor Prompt
# Role: Produce the initial rationale, answer, and self-reported confidence.
# Input: {question}, {choices_text}
# Output: JSON with rationale, answer, and confidence
ACTOR_PROMPT = """You are an expert problem solver. Given a question, provide a step-by-step reasoning process to reach the correct answer.
Respond strictly with a JSON object.

### Current Question:
Question: {question}
{choices_text}

Format your response as a valid JSON object with the following fields:
- "rationale": A detailed step-by-step explanation of your reasoning process.
- "answer": The final answer. For multiple choice questions, output only the option letter (e.g., "A"). For open-ended questions, output the final calculated value or expression.
- "confidence": A score from 0.0 to 1.0 indicating your confidence in the answer.

Important: Ensure the "answer" field contains ONLY the selected option letter or the final value. Do not include any additional text or explanations inside the JSON fields. Respond strictly with the JSON object."""


# Reflector Prompt
# 论文 Section: Reflector Prompt
# Role: Evaluate Actor outputs with retrieved context and decide whether escalation is needed.
# Input: {batch_json}
# Output: JSON array with id, accept, reusable, confidence, and critique
# Key rule: 
#   - Set accept=true when Actor answer is acceptable (high confidence, consistent reasoning, or strong retrieved context support)
#   - Set reusable=true only when reasoning is reliable and general enough to serve as future evidence
#   - Assign lower confidence when answer is uncertain, contradicts strong retrieved context, or contains obvious reasoning error
REFLECTOR_PROMPT = """You are an expert reviewer. Evaluate each answer with retrieved context and decide whether escalation is needed.
Respond strictly with a JSON array.

## Decision Rules

**accept=true** (Actor answer is acceptable):
- Actor confidence is high AND reasoning is consistent
- OR retrieved context strongly supports the answer

**accept=false** (Need escalation):
- Actor confidence is very low
- OR clear contradiction with high-confidence context
- OR obvious error in reasoning

**reusable=true** (Can serve as future evidence):
- Current reasoning is reliable and general enough
- NOT when answer is domain-specific or uncertain

## Output Format
For each item, output a JSON object with:
- "id": The item ID
- "accept": boolean - whether the Actor answer is acceptable
- "reusable": boolean - whether this case can be cached as future evidence
- "confidence": float 0.0-1.0 - your confidence in the evaluation
- "critique": string - brief reasoning for your decision

Input Batch:
{batch_json}

Output valid JSON array only. Ensure your JSON is valid - use double quotes for strings and property names."""


# Escalator Prompt
# 论文 Section: Escalator Prompt
# Role: Re-solve difficult cases when both Actor and Reflector confidence are below threshold.
# Input: {context_text}, {question}, {choices_text}, {critique_text}
# Output: JSON with rationale, answer, and confidence
# Key rule: Re-analyze the question carefully using the critique to avoid previous mistakes,
#           provide clear reasoning if needed, and return a definitive final answer.
ESCALATOR_PROMPT = """You are an expert problem solver. A previous attempt failed or has low confidence.
Your task is to provide a definitive answer, considering the critique and retrieved context below.
Respond strictly with a JSON object.

{context_text}

Question: {question}
{choices_text}

Initial Rationale:
{rationale_text}

Initial Answer:
{initial_answer}

Previous Attempt's Critique:
{critique_text}

Instructions:
1. Re-analyze the question carefully.
2. Consider the critique to avoid previous mistakes.
3. Use the retrieved context if relevant.
4. Provide clear reasoning, then give your final answer.
5. Output only the option letter (A/B/C/D...) or final value in "answer".

Format your response as a valid JSON object with the following fields:
- "rationale": Your detailed reasoning.
- "answer": The final answer (Option letter or value).
- "confidence": A score from 0.0 to 1.0 indicating your confidence.

Respond strictly with the JSON object. Do not add any text before or after the JSON."""
