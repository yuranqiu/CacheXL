from core.utils import count_tokens

def simple_accuracy(items: list[dict]) -> float | None:
    # 计算简单准确率
    total = 0
    correct = 0
    for it in items:
        gold = it.get("label")
        pred = it.get("final_answer", it.get("answer"))
        if gold is None or pred is None:
            continue
        total += 1
        if str(gold).strip() == str(pred).strip():
            correct += 1
    if total == 0:
        return None
    return correct / total


def approx_tokens(text: str) -> int:
    # 使用 DeepSeek Tokenizer 精确计算 Token 数量
    return count_tokens(text)


def ks_statistic(conf_correct: list[float], conf_incorrect: list[float]) -> float | None:
    # 计算KS统计量 (衡量置信度分布差异)
    if not conf_correct or not conf_incorrect:
        return None
    a = sorted(conf_correct)
    b = sorted(conf_incorrect)
    i = 0
    j = 0
    na = len(a)
    nb = len(b)
    max_diff = 0.0
    while i < na or j < nb:
        if j == nb or (i < na and a[i] <= b[j]):
            v = a[i]
        else:
            v = b[j]
        while i < na and a[i] <= v:
            i += 1
        while j < nb and b[j] <= v:
            j += 1
        cdf_a = i / na
        cdf_b = j / nb
        diff = abs(cdf_a - cdf_b)
        if diff > max_diff:
            max_diff = diff
    return max_diff


def expected_calibration_error(confidences: list[float], correct_flags: list[bool], bins: int = 10) -> float | None:
    # 计算期望校准误差 (ECE)
    if not confidences:
        return None
    n = len(confidences)
    bin_totals = [0] * bins
    bin_correct = [0] * bins
    bin_conf_sum = [0.0] * bins
    for conf, ok in zip(confidences, correct_flags):
        idx = int(min(bins - 1, max(0, conf * bins)))
        bin_totals[idx] += 1
        bin_correct[idx] += 1 if ok else 0
        bin_conf_sum[idx] += conf
    ece = 0.0
    for i in range(bins):
        if bin_totals[i] == 0:
            continue
        acc = bin_correct[i] / bin_totals[i]
        avg_conf = bin_conf_sum[i] / bin_totals[i]
        ece += (bin_totals[i] / n) * abs(acc - avg_conf)
    return ece
