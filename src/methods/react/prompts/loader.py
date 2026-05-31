from .react import ACTOR_REACT, REFLECTOR_REACT
from methods.bot.prompts import ACTOR_BOT, REFLECTOR_BOT


def get_actor_prompt(method: str = "react", dataset_name: str = "default") -> str:
    """获取指定方法的 Actor Prompt。

    Args:
        method: 方法类型 "react" 或 "bot"
        dataset_name: 数据集名称（预留）
    """
    if method == "bot":
        return ACTOR_BOT
    return ACTOR_REACT


def get_reflector_prompt(
    method: str = "react", dataset_name: str = "default"
) -> str:
    """获取指定方法的 Reflector Prompt。

    Args:
        method: 方法类型 "react" 或 "bot"
        dataset_name: 数据集名称（预留）
    """
    if method == "bot":
        return REFLECTOR_BOT
    return REFLECTOR_REACT
