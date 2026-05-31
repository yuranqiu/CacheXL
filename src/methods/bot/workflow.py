import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.llm import LLMClient
from core.config import ExperimentConfig
from methods.react.actor import build_actor_messages, parse_actor_response
from methods.react.reflector import (
    build_reflector_messages,
    parse_reflector_response,
)

logger = logging.getLogger(__name__)


def run_actor_for_item(
    item: dict,
    llm: LLMClient,
    model_name: str,
    dataset_name: str,
    temperature: float = 0.1,
):
    """Run Actor for a single item."""
    try:
        msgs = build_actor_messages(item, dataset_name, method="bot")
        # Actor usually needs some temperature for diversity/creativity
        # Use provided temperature
        resp = llm.chat(model_name, msgs, temperature=temperature)
        text = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})
        parsed = parse_actor_response(text)

        # Update item with new answer
        item["answer"] = parsed.get("answer")
        item["rationale"] = parsed.get("rationale")
        item["confidence"] = parsed.get("confidence")

        # Accumulate tokens
        if "actor_tokens" not in item:
            item["actor_tokens"] = {"prompt": 0, "completion": 0, "total": 0}

        item["actor_tokens"]["prompt"] += usage.get("prompt_tokens", 0)
        item["actor_tokens"]["completion"] += usage.get("completion_tokens", 0)
        item["actor_tokens"]["total"] += usage.get("total_tokens", 0)

    except Exception as e:
        logger.error(f"Actor failed for item {item.get('id')}: {e}")
        # Keep old answer or mark error?
        pass


def run_bot_batch(
    items: list[dict],
    llm: LLMClient,
    model_name: str,
    config: ExperimentConfig,
    dataset_name: str = "default",
    temperature: float = 0.1,
) -> list[dict]:
    """
    Execute Batch-of-Thought (BoT) workflow for a batch of items.

    1. Initial Actor pass.
    2. Loop (Reflector -> Actor) for max_rounds.
    """

    # Initialize status
    for item in items:
        # Preserve ground truth if not already present
        if "label" not in item:
            item["label"] = item.get("answer") or item.get("gold")

        item["status"] = "pending"
        item["history"] = []
        item["critique"] = ""  # Initialize critique
        item["actor_tokens"] = {"prompt": 0, "completion": 0, "total": 0}

    # Round 0: Initial Actor Pass
    # Run in parallel
    with ThreadPoolExecutor(max_workers=len(items)) as executor:
        futures = [
            executor.submit(
                run_actor_for_item, item, llm, model_name, dataset_name, temperature
            )
            for item in items
        ]
        for f in as_completed(futures):
            pass  # Exceptions logged in run_actor_for_item

    # Refinement Loops
    # If max_rounds=3, we do:
    # Round 1: Reflector -> Actor
    # Round 2: Reflector -> Actor
    # ...

    # Ensure max_rounds is at least 1
    max_rounds = max(1, config.max_rounds)

    for round_idx in range(1, max_rounds):
        # 1. Identify items needing refinement
        # (For the first reflector pass, we check all. Subsequently, only those marked refine=True)
        active_items = [item for item in items if item.get("status") != "done"]
        if not active_items:
            break

        # 2. Run Reflector on active items
        # Reflector takes a list of items.
        ref_msgs = build_reflector_messages(active_items, dataset_name, method="bot")
        try:
            # Reflector is usually deterministic (temp=0.1 or 0)
            ref_resp = llm.chat(model_name, ref_msgs, temperature=0.0)
            ref_text = ref_resp["choices"][0]["message"]["content"]
            ref_usage = ref_resp.get("usage", {})
            critiques_list = parse_reflector_response(ref_text)

            # Distribute Reflector tokens to active items
            # We split evenly among active items for cost tracking
            if active_items:
                prompt_per_item = ref_usage.get("prompt_tokens", 0) / len(active_items)
                completion_per_item = ref_usage.get("completion_tokens", 0) / len(
                    active_items
                )
                total_per_item = ref_usage.get("total_tokens", 0) / len(active_items)

                for item in active_items:
                    item["actor_tokens"]["prompt"] += prompt_per_item
                    item["actor_tokens"]["completion"] += completion_per_item
                    item["actor_tokens"]["total"] += total_per_item

        except Exception as e:
            logger.error(f"Reflector failed in round {round_idx}: {e}")
            break  # Stop refinement if reflector fails

        # 3. Process Critiques
        critique_map = {str(c.get("id")): c for c in critiques_list}

        items_to_refine = []

        for item in active_items:
            c = critique_map.get(str(item["id"]))
            if c:
                # Update item history with current state before overwriting
                item["history"].append(
                    {
                        "round": round_idx - 1,
                        "answer": item.get("answer"),
                        "rationale": item.get("rationale"),
                        "critique": c.get("critique"),
                        "refine": c.get("refine"),
                    }
                )

                if c.get("refine") is True:
                    item["critique"] = c.get("critique")
                    item["status"] = "refining"
                    items_to_refine.append(item)
                else:
                    item["status"] = "done"
                    # If Reflector says correct, we keep the current answer.
            else:
                # If Reflector missed this item, assume done to avoid infinite loops on missing IDs.
                item["status"] = "done"

        if not items_to_refine:
            break

        # 4. Run Actor on items_to_refine
        with ThreadPoolExecutor(max_workers=len(items_to_refine)) as executor:
            futures = [
                executor.submit(
                    run_actor_for_item, item, llm, model_name, dataset_name, temperature
                )
                for item in items_to_refine
            ]
            for f in as_completed(futures):
                pass

    return items
