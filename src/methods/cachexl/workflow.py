import sys
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .cache import EvidenceCache
from .prompts import (
    ACTOR_PROMPT,
    ESCALATOR_PROMPT,
    REFLECTOR_PROMPT,
)
from methods.react.eval import approx_tokens
from methods.react.actor import parse_actor_response
from methods.react.reflector import parse_reflector_response


def run_cachexl_batch(
    client,
    model_name: str,
    model_params: dict,
    batch: list[dict],
    cache: EvidenceCache,
    dataset_name: str = "default",
    concurrency: int = 1,
    tau_l: float = 0.5,
    tau_h: float = 0.8,
) -> list[dict]:
    """
    CacheXL 方法工作流 (论文 Algorithm 1):
    1. T_q ← RetrieveAsync(C, q, k)  -- 异步检索
    2. (r_q, a_q, c_q) ← Actor(q)  -- Actor 推理
    3. R(q) ← Synchronize(T_q)  -- 同步检索结果
    4. (f_q, ρ_q, u_q, v_q) ← Reflector(q, r_q, a_q, c_q, R(q))  -- Reflector 评估
    5. if c_q < τ_l and ρ_q < τ_l then y_q ← Escalator(...)  -- 升级
    6. if u_q=1 and v_q=1 and c_q≥τ_h and ρ_q≥τ_h then C ← Update(C, q, r_q, y_q, f_q)  -- 缓存准入
    """

    def run_llm(messages, parse_fn, retry_hint, temp=0.1):
        for attempt in range(3):
            try:
                send_msgs = messages
                if attempt > 0:
                    send_msgs = [
                        {
                            "role": "user",
                            "content": messages[0]["content"] + f"\n{retry_hint}",
                        }
                    ]

                resp = client.chat(model_name, send_msgs, temp, model_params)
                text = resp["choices"][0]["message"]["content"]
                parsed = parse_fn(text)
                return parsed, resp, text, send_msgs
            except Exception as e:
                print(f"[CacheXL] Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise e
                continue
        raise ValueError("Retry exhausted")

    def track_tokens(item, stage_key, resp, text, msgs):
        usage = resp.get("usage") or {}
        pt = usage.get("prompt_tokens") or sum(
            approx_tokens(m.get("content", "")) for m in msgs
        )
        ct = usage.get("completion_tokens") or approx_tokens(text)
        tt = usage.get("total_tokens") or (pt + ct)

        current = item.get(stage_key, {"prompt": 0.0, "completion": 0.0, "total": 0.0})
        current["prompt"] += float(pt)
        current["completion"] += float(ct)
        current["total"] += float(tt)
        item[stage_key] = current

    def run_actor(item):
        """Actor: 生成初始 rationale, answer, confidence"""
        choices = item.get("choices", [])
        choices_text = "\n".join([f"{c['label']}. {c['text']}" for c in choices])

        content = ACTOR_PROMPT.format(
            question=item["question"],
            choices_text=choices_text,
        )

        msgs = [{"role": "user", "content": content}]
        temp = model_params.get("temperature", 0.1)
        parsed, resp, text, sent_msgs = run_llm(
            msgs, parse_actor_response, "Output JSON object.", temp=temp
        )

        item["answer"] = parsed.get("answer")
        item["rationale"] = parsed.get("rationale")
        item["confidence"] = parsed.get("confidence")
        track_tokens(item, "actor_tokens", resp, text, sent_msgs)
        return item

    def fetch_batch_contexts(items):
        """异步检索 Context"""
        try:
            questions = [it["question"] for it in items]
            embeddings, usage = cache.get_embeddings_batch(questions)

            tokens_per_item = float(usage) / max(len(items), 1)
            for it in items:
                it["embedding_tokens"] = tokens_per_item

            item_contexts = {}
            item_embeddings = {}

            def process_single_retrieval(idx, it):
                try:
                    emb = embeddings[idx]
                    relevant = cache.get_relevant(it["question"], query_embedding=emb)
                    ctx = cache.format_context(relevant)
                    return it["id"], ctx, emb
                except Exception as e:
                    print(f"Retrieval failed for {it.get('id')}: {e}")
                    return it["id"], "", None

            with ThreadPoolExecutor(max_workers=min(len(items), 20)) as executor:
                futures = [
                    executor.submit(process_single_retrieval, i, it)
                    for i, it in enumerate(items)
                ]
                for fut in as_completed(futures):
                    iid, ctx, emb = fut.result()
                    item_contexts[iid] = ctx
                    item_embeddings[iid] = emb

            return item_contexts, item_embeddings
        except Exception as e:
            print(f"Batch retrieval failed: {e}")
            return {}, {}

    def run_reflector_batch(items, item_contexts):
        """Reflector: 评估答案，返回 (f_q, ρ_q, u_q, v_q)"""
        batch_input = []
        for it in items:
            batch_input.append(
                {
                    "id": it["id"],
                    "question": it["question"],
                    "choices": it.get("choices", []),
                    "answer": it.get("answer"),
                    "rationale": it.get("rationale"),
                    "actor_confidence": it.get("confidence"),
                    "context": item_contexts.get(it["id"], ""),
                }
            )

        content = REFLECTOR_PROMPT.format(
            batch_json=json.dumps(batch_input, ensure_ascii=False, indent=2)
        )
        msgs = [{"role": "user", "content": content}]

        parsed_list, resp, text, sent_msgs = run_llm(
            msgs, parse_reflector_response, "Output JSON array."
        )

        usage = resp.get("usage") or {}
        pt = usage.get("prompt_tokens") or sum(
            approx_tokens(m.get("content", "")) for m in msgs
        )
        ct = usage.get("completion_tokens") or approx_tokens(text)
        tt = usage.get("total_tokens") or (pt + ct)
        denom = max(len(items), 1)

        token_share = {
            "prompt": float(pt) / denom,
            "completion": float(ct) / denom,
            "total": float(tt) / denom,
        }

        decision_map = {d["id"]: d for d in parsed_list if d.get("id") is not None}
        return decision_map, token_share

    def run_escalator(item, critique, context):
        """Escalator: 处理困难案例（当 c_q < τ_l AND ρ_q < τ_l）"""
        choices = item.get("choices", [])
        choices_text = "\n".join([f"{c['label']}. {c['text']}" for c in choices])

        content = ESCALATOR_PROMPT.format(
            context_text=context,
            question=item["question"],
            choices_text=choices_text,
            critique_text=critique,
            rationale_text=item.get("rationale", ""),
            initial_answer=item.get("answer", ""),
        )

        msgs = [{"role": "user", "content": content}]
        temp = model_params.get("temperature", 0.1)
        parsed, resp, text, sent_msgs = run_llm(
            msgs, parse_actor_response, "Output JSON object.", temp=temp
        )

        item["final_answer"] = parsed.get("answer")
        item["escalator_rationale"] = parsed.get("rationale")
        item["escalator_confidence"] = parsed.get("confidence")
        item["used_escalator"] = True

        track_tokens(item, "escalator_tokens", resp, text, sent_msgs)
        return item

    # ========== 执行流程 (Algorithm 1) ==========

    # 准备 ground truth
    for it in batch:
        if "label" not in it and "gold" not in it:
            it["label"] = it.get("answer") or it.get("gold")

    # 步骤 1-2: Actor 推理 与 异步检索 并行执行
    # 论文: T_q ← RetrieveAsync(C, q, k) 和 (r_q, a_q, c_q) ← Actor(q)
    print("[CacheXL] Actor 推理与异步检索并行执行...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Actor 推理
        if concurrency > 1:
            actor_futures = [executor.submit(run_actor, it) for it in batch]
            for fut in as_completed(actor_futures):
                fut.result()
        else:
            for it in tqdm(batch, desc="Actor", file=sys.stdout, leave=False):
                run_actor(it)

        # 异步检索
        retrieval_future = executor.submit(fetch_batch_contexts, batch)
        item_contexts, item_embeddings = retrieval_future.result()

    print("[CacheXL] Actor 和检索完成")

    # 将检索结果赋给每个 item
    for it in batch:
        it["context_text"] = item_contexts.get(it["id"], "")
        it["embedding_cache"] = item_embeddings.get(it["id"])

    # 步骤 3: Reflector 评估
    # 论文: (f_q, ρ_q, u_q, v_q) ← Reflector(q, r_q, a_q, c_q, R(q))
    decisions, token_share = run_reflector_batch(batch, item_contexts)

    # 步骤 4-5: 处理决策 + Escalator
    escalator_candidates = []

    for item in batch:
        cur_reflect = item.get(
            "reflect_tokens", {"prompt": 0.0, "completion": 0.0, "total": 0.0}
        )
        cur_reflect["prompt"] += token_share["prompt"]
        cur_reflect["completion"] += token_share["completion"]
        cur_reflect["total"] += token_share["total"]
        item["reflect_tokens"] = cur_reflect

        did = item["id"]
        if did in decisions:
            d = decisions[did]
            accept = d.get("accept")
            is_accept = accept is True or str(accept).strip().lower() in {
                "true", "1", "yes",
            }
            reusable = d.get("reusable")
            is_reusable = reusable is True or str(reusable).strip().lower() in {
                "true", "1", "yes",
            }

            item["critique"] = d.get("critique")
            item["reflect_confidence"] = d.get("confidence", 1.0)
            item["u_q"] = is_accept  # acceptance signal
            item["v_q"] = is_reusable  # reusability signal
            reflector_conf = float(item["reflect_confidence"])
            actor_conf = float(item.get("confidence", 1.0))

            # 升级条件: c_q < τ_l AND ρ_q < τ_l
            needs_escalation = actor_conf < tau_l and reflector_conf < tau_l

            if needs_escalation:
                escalator_candidates.append(item)
            else:
                # 直接使用 Actor 答案: y_q ← a_q
                item["final_answer"] = item.get("answer")
                item["used_escalator"] = False
        else:
            # 没有 decision，保守处理
            item["final_answer"] = item.get("answer")
            item["used_escalator"] = False
            item["u_q"] = False
            item["v_q"] = False

    # 步骤 5: Escalator 兜底
    # 论文: y_q ← Escalator(q, r_q, a_q, f_q, R(q))
    if escalator_candidates:
        print(f"[CacheXL] Escalator 处理 {len(escalator_candidates)} 个困难案例...")
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        run_escalator,
                        it,
                        it.get("critique", ""),
                        it.get("context_text", ""),
                    )
                    for it in escalator_candidates
                ]
                for fut in as_completed(futures):
                    fut.result()
        else:
            for it in escalator_candidates:
                run_escalator(it, it.get("critique", ""), it.get("context_text", ""))

    # 步骤 6: 缓存准入检查 (在 Escalator 之后)
    # 论文: if u_q = 1 and v_q = 1 and c_q ≥ τ_h and ρ_q ≥ τ_h then C ← Update(C, q, r_q, y_q, f_q)
    for item in batch:
        u_q = item.get("u_q", False)
        v_q = item.get("v_q", False)
        actor_conf = float(item.get("confidence", 0.0))
        reflector_conf = float(item.get("reflect_confidence", 0.0))

        if u_q and v_q and actor_conf >= tau_h and reflector_conf >= tau_h:
            try:
                cache.add(item, embedding=item.get("embedding_cache"))
            except:
                pass

    return batch
