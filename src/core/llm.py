import os
import time
from typing import Any
import httpx


class LLMClient:
    # LLM客户端封装，支持OpenAI兼容API
    def __init__(self, base_url: str, api_key: str, config: dict | None = None):
        # 初始化客户端，处理API Key、Mock模式及SSL配置
        self.mock = False
        if not api_key:
            self.mock = True
            self.base_url = ""
            self.api_key = ""
        else:
            self.base_url = base_url.rstrip("/")
            self.api_key = api_key
        self.config = config or {}

    def chat(self, model: str, messages: list[dict], temperature: float, extra: dict | None = None) -> dict[str, Any]:
        # 发送对话请求，支持Mock、重试、超时控制及JSON fallback
        if self.mock:
            if any(("反思器" in m.get("content", "")) or ("Reflector" in m.get("content", "")) for m in messages):
                content = '[{"id":"0","refine":0,"confidence":0.5,"critique":"mock","final_answer":"A"}]'
            else:
                content = '{"answer":"A","rationale":"mock","confidence":0.5}'
            return {"choices": [{"message": {"content": content}}]}
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": model, "messages": messages, "temperature": temperature}
        if extra:
            payload.update(extra)
        verify_env = os.getenv("LLM_SSL_VERIFY", "1").strip().lower()
        verify_ssl = verify_env not in {"0", "false", "no"}
        trust_env_val = os.getenv("LLM_HTTPX_TRUST_ENV", "1").strip().lower()
        trust_env = trust_env_val not in {"0", "false", "no"}
        timeout_val = os.getenv("LLM_HTTP_TIMEOUT", "120").strip()
        connect_val = os.getenv("LLM_HTTP_CONNECT_TIMEOUT", "30").strip()
        read_val = os.getenv("LLM_HTTP_READ_TIMEOUT", timeout_val).strip()
        write_val = os.getenv("LLM_HTTP_WRITE_TIMEOUT", "30").strip()
        pool_val = os.getenv("LLM_HTTP_POOL_TIMEOUT", "30").strip()
        retries_val = os.getenv("LLM_RETRY_ATTEMPTS", "2").strip()
        backoff_val = os.getenv("LLM_RETRY_BACKOFF", "2.0").strip()
        verbose = os.getenv("LLM_VERBOSE", "1").strip() == "1"
        if self.config:
            verify_ssl = self.config.get("ssl_verify", verify_ssl)
            trust_env = self.config.get("httpx_trust_env", trust_env)
            timeout_val = str(self.config.get("http_timeout", timeout_val))
            connect_val = str(self.config.get("http_connect_timeout", connect_val))
            read_val = str(self.config.get("http_read_timeout", read_val))
            write_val = str(self.config.get("http_write_timeout", write_val))
            pool_val = str(self.config.get("http_pool_timeout", pool_val))
            retries_val = str(self.config.get("retry_attempts", retries_val))
            backoff_val = str(self.config.get("retry_backoff", backoff_val))
            verbose = self.config.get("verbose", verbose)
        try:
            connect_timeout = float(connect_val)
        except ValueError:
            connect_timeout = 30
        try:
            read_timeout = float(read_val)
        except ValueError:
            read_timeout = 120
        try:
            write_timeout = float(write_val)
        except ValueError:
            write_timeout = 30
        try:
            pool_timeout = float(pool_val)
        except ValueError:
            pool_timeout = 30
        try:
            retries = int(retries_val)
        except ValueError:
            retries = 2
        try:
            backoff = float(backoff_val)
        except ValueError:
            backoff = 2.0
        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=write_timeout, pool=pool_timeout)
        role = "reflector" if any(("反思器" in m.get("content", "")) or ("Reflector" in m.get("content", "")) for m in messages) else "actor"
        for attempt in range(retries + 1):
            start = time.time()
            if verbose:
                print(f"[LLM] {role} request start attempt {attempt + 1}/{retries + 1}")
            try:
                with httpx.Client(timeout=timeout, verify=verify_ssl, trust_env=trust_env) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if not resp.is_success:
                        if verbose:
                            body = resp.text
                            if len(body) > 1000:
                                body = body[:1000]
                            print(f"[LLM] {role} response {resp.status_code}: {body}")
                        # 如果还有重试机会且是 5xx 错误或连接错误，则继续重试
                        if attempt < retries and resp.status_code >= 500:
                            print(f"[LLM] {role} 遇到错误 {resp.status_code}，将在 {backoff} 秒后重试...")
                            time.sleep(backoff)
                            continue
                        resp.raise_for_status()
                    if verbose:
                        elapsed = time.time() - start
                        print(f"[LLM] {role} request success in {elapsed:.2f}s")
                    return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as e:
                if attempt < retries:
                    print(f"[LLM] {role} network error: {e}, retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                raise e

            except (httpx.ReadTimeout, httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.NetworkError, httpx.HTTPStatusError) as e:
                # 检查是否为 HTTPStatusError 且状态码不是 429
                is_status_error = isinstance(e, httpx.HTTPStatusError)
                if is_status_error and e.response.status_code != 429:
                    raise e
                
                # 如果是 429，尝试从响应头中获取 Retry-After
                retry_after = None
                if is_status_error and e.response.status_code == 429:
                    retry_after_header = e.response.headers.get("retry-after")
                    if retry_after_header:
                        try:
                            retry_after = float(retry_after_header)
                        except ValueError:
                            pass
                
                if attempt < retries:
                    # 计算等待时间：优先使用服务器返回的 Retry-After，否则使用指数退避
                    wait_time = retry_after if retry_after is not None else backoff * (2 ** attempt)
                    
                    if verbose:
                        elapsed = time.time() - start
                        print(f"[LLM] {role} request error {type(e).__name__} after {elapsed:.2f}s, retrying in {wait_time:.2f}s")
                    
                    time.sleep(wait_time)
                    continue
                if verbose:
                    elapsed = time.time() - start
                    print(f"[LLM] {role} request failed after {elapsed:.2f}s")
                allow_fallback = os.getenv("LLM_ALLOW_FALLBACK", "0").strip() == "1"
                if self.config:
                    allow_fallback = self.config.get("allow_fallback", allow_fallback)
                if allow_fallback:
                    if role == "reflector":
                        content = '[{"id":"0","refine":0,"confidence":0.5,"critique":"mock","final_answer":"A"}]'
                    else:
                        content = '{"answer":"A","rationale":"mock","confidence":0.5}'
                    print("[LLM] fallback to mock response for this call")
                    return {"choices": [{"message": {"content": content}}]}
                raise e
            except Exception as e:
                if verbose:
                    elapsed = time.time() - start
                    print(f"[LLM] {role} request error {type(e).__name__} after {elapsed:.2f}s")
                allow_fallback = os.getenv("LLM_ALLOW_FALLBACK", "0").strip() == "1"
                if self.config:
                    allow_fallback = self.config.get("allow_fallback", allow_fallback)
                if allow_fallback:
                    if role == "reflector":
                        content = '[{"id":"0","refine":0,"confidence":0.5,"critique":"mock","final_answer":"A"}]'
                    else:
                        content = '{"answer":"A","rationale":"mock","confidence":0.5}'
                    print("[LLM] fallback to mock response for this call")
                    return {"choices": [{"message": {"content": content}}]}
                raise e
