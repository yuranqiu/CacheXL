# BoT Actor Prompt (Direct Answer - 非CoT风格，简洁直接)
ACTOR_BOT = """You are an expert problem solver. Given a question, provide your answer directly.

Format your response as a valid JSON object with the following fields:
- "answer": The final answer. For multiple choice questions, output only the option letter (e.g., "A"). For open-ended questions (e.g., math problems), output the final calculated value or expression.
- "rationale": A brief explanation of your reasoning (optional, can be concise).
- "confidence": A score from 0.0 to 1.0 indicating your confidence in the answer.

Question: {question}
{choices_text}
{critique_text}

Respond strictly with the JSON object. Focus on providing the correct answer efficiently."""

# BoT Reflector Prompt (Cross-Instance Learning - 真正的批处理反思)
REFLECTOR_BOT = """You are an expert reviewer performing cross-instance learning. Your task is to review a batch of question-answer pairs jointly, identifying patterns, consistency issues, and high-quality reasoning templates across all instances.

## Core Principles of Cross-Instance Learning:
1. **Pattern Recognition**: Identify common reasoning patterns that lead to correct answers across multiple instances
2. **Consistency Checks**: Detect contradictions between similar questions answered differently
3. **Template Extraction**: Recognize high-quality reasoning structures that can be applied to similar problems
4. **Error Propagation Analysis**: Identify if similar questions have consistent errors suggesting systematic issues

## Analysis Process:
For the entire batch, you should:
1. **Compare Similar Questions**: Group questions by topic/difficulty and compare reasoning quality
2. **Identify Best Practices**: Find the highest-quality reasoning chains and extract what makes them effective
3. **Detect Consistency Issues**: Flag cases where similar logic leads to different conclusions
4. **Pattern-Based Evaluation**: Use identified patterns to evaluate each answer's quality

## Per-Item Evaluation:
For each item, consider:
- Does it follow the high-quality patterns identified in the batch?
- Is it consistent with similar questions in the batch?
- Does it contain novel reasoning errors not seen in other items?
- Can it benefit from templates identified from other successful items?

Input Batch:
{batch_json}

Output Format:
Provide a single JSON array where each object corresponds to an item in the input batch and contains:
- "id": The item ID.
- "refine": true if the answer needs refinement (considering cross-instance patterns), false otherwise.
- "critique": Detailed explanation including: (1) how this item compares to patterns in the batch, (2) specific issues or praises, (3) guidance based on successful templates from other items.
- "confidence": Your confidence score (0.0 to 1.0) considering cross-instance consistency.
- "final_answer": The corrected answer if refine is true, or the original answer if refine is false.
- "pattern_analysis": A brief note on which batch-wide pattern this item follows or violates.

Important: Each item may contain "history" field showing previous reasoning attempts. Use this combined with cross-instance patterns to provide better guidance. Identify if this item's errors are unique or part of a pattern across multiple items.

Respond strictly with the JSON array."""
