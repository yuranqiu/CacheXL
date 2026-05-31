# ReAct Prompt (Baseline - Single-pass reasoning)
# 论文 Section: Baseline Prompt: ReAct
ACTOR_REACT = """Question: {question}
{choices_text}

Respond strictly with a JSON object containing:
- "thought": Step-by-step reasoning process
- "answer": The final answer (option letter like A/B/C/D or final value)
- "confidence": Confidence score 0.0-1.0"""

# ReAct 不需要 Reflector（单轮）
REFLECTOR_REACT = """This is ReAct baseline method. No reflector needed."""
