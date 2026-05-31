def make_batches(items: list[dict], size: int) -> list[list[dict]]:
    # 将数据列表按指定大小分批
    batches = []
    for i in range(0, len(items), size):
        batches.append(items[i : i + size])
    return batches
