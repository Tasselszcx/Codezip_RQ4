import os
import json
import base64
import re
import time
import torch
import numpy as np
import tiktoken
from openai import OpenAI, AzureOpenAI
from tree_sitter_languages import get_language, get_parser
from repoqa.utility import COMMENT_QUERY, FUNCTION_QUERY
from typing import Tuple, Dict, List, Optional, Union

try:
    import transformers

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# API配置
API_BASE_URL = "https://aihubmix.com/v1"
RETRY = 20
# 尝试加载 config.json


def get_config() -> Dict:
    return json.load(open("config.json", "r"))


try:
    config = get_config()
    if config and isinstance(config, dict):
        API_KEY = config.get("api_key")
        API_BASE_URL = config.get("base_url")
        AZURE_ENDPOINT = config.get("azure_endpoint")
        AZURE_API_VERSION = config.get("azure_api_version")
        CLIENT_TYPE = "Azure" if config.get("azure_endpoint") else "OpenAI"
    else:
        API_KEY = os.environ.get("OPENAI_API_KEY", "")
        AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "")
except Exception:
    API_KEY = ""
    AZURE_ENDPOINT = ""
    AZURE_API_VERSION = ""

# 检测员模型映射
EVALUATOR_MODELS = {
    "qwen3-vl-235b-a22b-instruct": "gpt-5.1-mini",
    "gpt-5.1": "qwen3-235b-a22b-instruct-2507",
    "gemini-2.5-pro": "qwen3-235b-a22b-instruct-2507",
    "claude-sonnet-4-5": "qwen3-235b-a22b-instruct-2507",
    "glm-4.5v": "qwen3-235b-a22b-instruct-2507",
    "DeepSeek-OCR": "qwen3-235b-a22b-instruct-2507",
    "gpt-5-mini": "qwen3-235b-a22b-instruct-2507",
    "gpt-5-mini-2025-08-07": "qwen3-235b-a22b-instruct-2507",
    "gpt-5-2025-08-07": "qwen3-235b-a22b-instruct-2507",
}


def create_client(client_type: str = "OpenAI", **kwargs) -> Union[OpenAI, AzureOpenAI]:
    """
    创建 OpenAI 或 AzureOpenAI 客户端

    Args:
        client_type: 客户端类型，"OpenAI" 或 "Azure"
        **kwargs: 其他参数，用于覆盖默认配置

    Returns:
        OpenAI 或 AzureOpenAI 客户端实例
    """
    if client_type == "Azure":
        return AzureOpenAI(
            api_key=kwargs.get("api_key", API_KEY),
            api_version=kwargs.get("api_version", AZURE_API_VERSION),
            azure_endpoint=kwargs.get("azure_endpoint", AZURE_ENDPOINT),
        )
    else:
        return OpenAI(
            base_url=kwargs.get("base_url", API_BASE_URL),
            api_key=kwargs.get("api_key", API_KEY),
        )


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_pil_image_to_base64(pil_image) -> str:
    """将PIL Image对象编码为base64"""
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_prompt(prompt_path: str) -> Dict:
    """加载prompt文件"""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_comments(source_code: str, lang: str) -> str:
    """移除代码中的注释"""
    try:
        source_bytes = bytes(source_code, "utf8")
        parser = get_parser(lang)
        tree = parser.parse(source_bytes)
        root_node = tree.root_node

        capture_list = []
        if lang in COMMENT_QUERY:
            for query_str in COMMENT_QUERY[lang]:
                comment_query = get_language(lang).query(query_str)
                capture_list += comment_query.captures(root_node)

        capture_list.sort(key=lambda cap: cap[0].start_byte, reverse=True)

        for node, _ in capture_list:
            source_bytes = (
                source_bytes[: node.start_byte] + source_bytes[node.end_byte:]
            )

        return source_bytes.decode("utf-8")
    except Exception as e:
        print(f"Warning: remove_comments failed: {e}")
        return source_code


def sanitize_output(model_output: str, lang: str) -> str:
    """清理和提取模型输出中的代码"""
    model_output = model_output.strip()
    search_pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    code_blocks = re.findall(
        search_pattern, model_output, re.DOTALL | re.MULTILINE)

    if not code_blocks:
        return model_output

    try:
        parser = get_parser(lang)
        fn_query = get_language(lang).query(FUNCTION_QUERY[lang])

        functions = []
        for code_block in code_blocks:
            source_bytes = bytes(code_block, "utf8")
            tree = parser.parse(source_bytes)
            root_node = tree.root_node
            captures = fn_query.captures(root_node)
            for node, _ in captures:
                functions.append(
                    source_bytes[node.start_byte: node.end_byte].decode(
                        "utf-8")
                )

        return "\n".join(functions) if functions else model_output
    except Exception as e:
        print(f"Warning: sanitize_output failed parsing: {e}")
        return model_output


def call_llm_with_images(
    client: Union[OpenAI, AzureOpenAI],
    model_name: str,
    images: Union[List[str], List],
    system_prompt: str,
    user_prompt: str,
    retry_on_empty: bool = True,
    client_type: str = "OpenAI",
    max_tokens: int = 6144,
    data_id: Optional[str] = None,
) -> Tuple[str, Dict]:
    """调用LLM API进行图片识别

    Args:
        client: OpenAI 或 AzureOpenAI 客户端
        model_name: 模型名称
        images: 图片路径列表或PIL Image对象列表
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        retry_on_empty: 如果返回空，是否重试一次

    Returns:
        (generated_text, token_info): 生成的文本和token信息
    """
    # 检查是否为PIL Image对象
    from PIL import Image as PIL_Image
    base64_images = []
    for img in images:
        if isinstance(img, PIL_Image.Image):
            base64_images.append(encode_pil_image_to_base64(img))
        elif isinstance(img, str):
            base64_images.append(encode_image_to_base64(img))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": user_prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    }
                    for image in base64_images
                ]
            ),
        },
    ]
    config = get_config()
    if config.get("qwen_api_key") and config.get("qwen_base_url") and model_name.startswith("qwen"):
        client = OpenAI(
            base_url=config.get("qwen_base_url"),
            api_key=config.get("qwen_api_key"),
        )

    for attempt in range(RETRY if retry_on_empty else 1):
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
            }
            if not model_name.startswith("gpt-5"):
                kwargs["temperature"] = 0.0
                kwargs["stream"] = False
                kwargs["max_tokens"] = max_tokens
            else:
                kwargs["max_completion_tokens"] = max_tokens
            if model_name.startswith("DeepSeek"):
                kwargs["max_tokens"] = min(4096, max_tokens)
            if model_name.startswith("qwen"):
                kwargs["extra_body"] = {
                    "thinking_budget": 1024
                }
            if model_name.startswith("glm"):
                kwargs["max_tokens"] = min(4096, max_tokens)
            if CLIENT_TYPE == "Azure":
                kwargs["extra_headers"] = {"X-TT-LOGID": "${your_logid}"}
                kwargs["extra_body"] = {
                    "thinking": {"include_thoughts": False, "budget_tokens": 1024}
                }
            response = client.chat.completions.create(**kwargs)

            generated_text = response.choices[0].message.content
            usage = response.usage

            # 如果返回空且允许重试，进行重试
            if not generated_text or not generated_text.strip():
                if retry_on_empty and attempt < RETRY - 1:
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}警告: LLM 返回空响应，正在重试(已尝试{attempt + 1}次)...")
                    time.sleep(1)
                    continue
                else:
                    # 重试后仍为空，返回空字符串
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}警告: {attempt}次后LLM 仍然返回空响应，返回空字符串")
                    kwargs.pop("messages", None)
                    return "", {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                        "api_kwargs": kwargs
                    }
            kwargs.pop("messages", None)
            return generated_text, {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "api_kwargs": kwargs
            }
        except Exception as e:
            if attempt < RETRY - 1 and retry_on_empty:
                sleep_time = 2
                prefix = f"[{data_id}] " if data_id else ""
                print(f"  {prefix}警告: 调用模型 {model_name} 时出错，{sleep_time}s后重试一次: {e}，已请求{attempt+1}次")
                time.sleep(sleep_time)
                continue
            else:
                prefix = f"[{data_id}] " if data_id else ""
                print(f"{prefix}调用模型 {model_name} 时出错: {e}，已请求{attempt+1}次")
                kwargs.pop("messages", None)
                return "", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_kwargs": kwargs
                }

    # 如果所有尝试都失败，返回空
    kwargs.pop("messages", None)
    return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_kwargs": kwargs}


def call_llm_with_text_only(
    client: Union[OpenAI, AzureOpenAI],
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    retry_on_empty: bool = True,
    client_type: str = "OpenAI",
    max_tokens: int = 6144,
    data_id: Optional[str] = None,
) -> Tuple[str, Dict]:
    """调用LLM API，只使用文本（不使用图片）

    Args:
        client: OpenAI 或 AzureOpenAI 客户端
        model_name: 模型名称
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        retry_on_empty: 如果返回空，是否重试一次

    Returns:
        (generated_text, token_info): 生成的文本和token信息
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    config = get_config()
    if config.get("qwen_api_key") and config.get("qwen_base_url") and model_name.startswith("qwen"):
        client = OpenAI(
            base_url=config.get("qwen_base_url"),
            api_key=config.get("qwen_api_key"),
        )

    for attempt in range(RETRY if retry_on_empty else 1):
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
            }
            if not model_name.startswith("gpt-5"):
                kwargs["temperature"] = 0.0
                kwargs["stream"] = False
                kwargs["max_tokens"] = max_tokens
            else:
                kwargs["max_completion_tokens"] = max_tokens
            if model_name.startswith("DeepSeek"):
                kwargs["max_tokens"] = min(4096, max_tokens)  # ds 模型限制
            if model_name.startswith("qwen"):
                kwargs["extra_body"] = {
                    "thinking_budget": 1024
                }
            if model_name.startswith("glm"):
                kwargs["max_tokens"] = min(4096, max_tokens)
            if CLIENT_TYPE == "Azure":
                kwargs["extra_headers"] = {"X-TT-LOGID": "${your_logid}"}
                kwargs["extra_body"] = {
                    "thinking": {"include_thoughts": False, "budget_tokens": 1024}
                }
            response = client.chat.completions.create(**kwargs)

            generated_text = response.choices[0].message.content
            usage = response.usage

            # 如果返回空且允许重试，进行重试
            if not generated_text or not generated_text.strip():
                if retry_on_empty and attempt < RETRY - 1:
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}警告: LLM 返回空响应，正在重试(已重试{attempt+1}次)...")
                    time.sleep(1)
                    continue
                else:
                    # 重试后仍为空，返回空字符串
                    prefix = f"[{data_id}] " if data_id else ""
                    print(f"  {prefix}警告: {attempt}次后LLM 仍然返回空响应，返回空字符串")
                    return "", {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                        "api_kwargs": kwargs
                    }

            return generated_text, {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
                "api_kwargs": kwargs
            }
        except Exception as e:
            if attempt < RETRY - 1 and retry_on_empty:
                sleep_time = 2
                prefix = f"[{data_id}] " if data_id else ""
                print(f"  {prefix}警告: 调用模型 {model_name} 时出错，{sleep_time}s后重试一次: {e}，已请求{attempt+1}次")
                time.sleep(sleep_time)
                continue
            else:
                prefix = f"[{data_id}] " if data_id else ""
                print(f"{prefix}调用模型 {model_name} 时出错: {e}，已请求{attempt+1}次")
                return "", {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_kwargs": kwargs
                }


def get_text_tokens(text: str) -> int:
    """计算文本的token数量（使用tiktoken）"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def get_text_tokens_qwen(text: str, tokenizer=None) -> Optional[int]:
    """使用Qwen tokenizer计算文本的token数量"""
    if tokenizer is None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True
            )
        except ImportError:
            return None
        except Exception:
            return None

    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"  警告: 使用Qwen tokenizer计算token失败: {e}")
        return None


def call_llm_with_logit_bias(client, eval_model, query: str, options: list[str]):
    try:
        tokenizer_model = (
            "gemini-2.5-pro" if "gemini-2.5-pro" in eval_model else "gpt-4"
        )
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)
    except KeyError:
        print(
            f"警告: 无法自动映射{tokenizer_model}到分词器。使用默认的 cl100k_base 分词器。"
        )
        tokenizer = tiktoken.get_encoding("cl100k_base")

    logit_bias = dict()
    for opt in options:
        tok_ids = tokenizer.encode(opt)
        assert len(tok_ids) == 1, "Only single token options are supported"
        logit_bias[tok_ids[0]] = 100
    kwargs = {
        "model": eval_model,
        "messages": [
            {"role": "system", "content": "You are a code quality assesing engine."},
            {"role": "user", "content": query},
        ],
        "max_tokens": 1,
        "temperature": 0.3,
        "n": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "logit_bias": logit_bias,
    }
    if isinstance(eval_model, str) and "deepseek" in eval_model.lower():
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    completion = client.chat.completions.create(**kwargs)

    logprobs = np.full(2, np.nan)
    choice = completion.choices[0]
    opt_to_idx = {t: n for n, t in enumerate(options)}
    min_lp = 0
    for logprob_item in choice.logprobs.content[0].top_logprobs:
        tok = logprob_item.token
        lp = logprob_item.logprob
        min_lp = min(min_lp, lp)
        if tok in opt_to_idx:
            logprobs[opt_to_idx[tok]] = lp
    logprobs[np.isnan(logprobs)] = (
        min_lp - 2.3
    )  # approximately 10 times less than the minimal one
    usage = completion.usage
    assert not np.isnan(logprobs).any()
    return torch.from_numpy(logprobs), {
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }


def build_folder(base_dir: str, folder_parts: list[str], **kwargs) -> str:
    """根据基础目录和文件夹部分构建完整路径"""
    if kwargs.get('enable_syntax_highlight'):
        folder_parts.append("hl")
    if kwargs.get('preserve_newlines'):
        folder_parts.append("nl")
    if kwargs.get('enable_bold'):
        folder_parts.append("bold")
    return os.path.join(base_dir, "_".join(folder_parts))
