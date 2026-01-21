import re
import urllib.request
import urllib.parse
from openai import OpenAI
import random
import json
import os
import requests
import threading
import queue

# 设置requests的默认超时时间
requests.adapters.DEFAULT_TIMEOUT = 3600
# 创建自定义的 session 来处理请求
_session = requests.Session()
# 配置重试策略
adapter = requests.adapters.HTTPAdapter(
    max_retries=5,
    # 增加连接池大小以支持高并发
    pool_connections=500,
    pool_maxsize=500
)
_session.mount('http://', adapter)
_session.mount('https://', adapter)

class ChatError(Exception):
    """聊天过程中的通用错误"""
    pass

class ChatTimeoutError(ChatError):
    """聊天超时错误"""
    pass

class ChatAPIError(ChatError):
    """API调用错误"""
    pass

os.environ['HTTP_PROXY'] = ''
os.environ['http_proxy'] = ''

os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1,10.26.0.0/16'

def post_http_request(prompt: dict, api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return response

def chat(prompt, model=None, messages=None, tools=None, system=None, json_format=False, debug=False, json_dict=False, temperature=0.3, guided_choice=None, timeout_seconds=4800):
    # if model in ["gpt-4o", "v3", "r1", "claude-3.7", "gemini"]:
    #     return chat_other(prompt, messages, tools, system, json_format, debug, json_dict, temperature, model, timeout_seconds)
    # else:
    return chat_qwen(prompt=prompt, 
                        messages=messages, 
                        tools=tools, 
                        system=system, 
                        json_format=json_format, 
                        debug=debug, 
                        json_dict=json_dict, 
                        temperature=temperature, 
                        model=model, 
                        guided_choice=guided_choice, 
                        timeout_seconds=timeout_seconds)
    
def chat_other(prompt, messages=None, tools=None, system=None, 
            json_format=False, debug=False, json_dict=False, 
            temperature=0.7, model="gpt-4o", timeout_seconds=3600):
    """
    发送聊天请求到服务器
    
    参数与服务器端的参数保持一致
    """
    payload = {
        "password": "0818",
        "prompt": prompt,
        "temperature": temperature,
        "model": model,
        "json_format": json_format,
        "debug": debug,
        "json_dict": json_dict
    }
    
    # 添加可选参数
    if messages is not None:
        payload["messages"] = messages
    if tools is not None:
        payload["tools"] = tools
    if system is not None:
        payload["system"] = system
        
    try:
        response = _session.post(
            "http://10.26.84.159:5000/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=(60, 3600)  # (连接超时, 读取超时)
        )
        
        # 检查响应状态
        if response.status_code == 200:
            return response.json()["response"]
        elif response.status_code == 401:
            raise Exception("认证失败：密码错误")
        else:
            error_msg = response.json().get("error", f"服务器错误，状态码: {response.status_code}")
            raise Exception(f"请求失败: {error_msg}")
            
    except requests.RequestException as e:
        raise Exception(f"网络错误: {str(e)}")


def chat_qwen(prompt, messages=None, tools=None, system=None, json_format=False, debug=False, json_dict=False, temperature=0.3, model = None, guided_choice=None, timeout_seconds=240):
    result_queue = queue.Queue()
    
    def _chat_thread():
        try:
            services_A100_32B = [
                {"base_url":"http://10.25.0.115:8010/v1","model":"qwen3-32b"}
            ]
            services_A100_30B = [
                {"base_url":"http://10.25.0.115:8009/v1","model":"qwen3-30b-a3b"}
            ]
            if model == "qwen3-32b":
                selected_service = random.choice(services_A100_32B)
            elif model == "qwen3-30b-a3b":
                selected_service = random.choice(services_A100_30B)
            else:
                print("model not found")
                selected_service = random.choice(services_A100_32B)
            client = OpenAI(
                base_url=selected_service["base_url"],
                api_key=selected_service.get("api_key", "token-abc123"),
            )

            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt + " /nothink"}],
            }
            data["temperature"] = temperature

            if json_dict:
                data["extra_body"]["guided_json"] = True
            if guided_choice:
                data["extra_body"]["guided_choice"] = guided_choice
            if messages is not None:
                data["messages"] = messages
            if tools is not None:
                data["tools"] = tools
            if system is not None:
                data["system"] = system
            if json_format:
                data["format"] = "json"

            message = client.chat.completions.create(**data).choices[0].message
            
            if tools is not None:
                result_queue.put(("success", message.tool_calls))
            elif json_dict:
                result_queue.put(("success", message))
            else:
                result_queue.put(("success", message.content))
                
        except Exception as e:
            result_queue.put(("error", {"type": type(e).__name__, "message": str(e)}))

    thread = threading.Thread(target=_chat_thread)
    thread.daemon = True
    thread.start()
    
    try:
        status, result = result_queue.get(timeout=timeout_seconds)
        if status == "error":
            if isinstance(result, dict) and "message" in result:
                raise ChatAPIError(f"API错误: {result['message']}")
            else:
                raise ChatAPIError(f"API错误: {result}")
        
        think_pattern = r'</think>(.*)'
        think_match = re.search(think_pattern, result, re.DOTALL)
        if think_match:
            result = think_match.group(1)

        return result
    except queue.Empty:
        raise ChatTimeoutError("Chat执行超时")

if __name__ == "__main__":
    text = """分析一下这个memory有没有错？

{'text': 'FIN7 has used application shim databases for persistence.', 'label': 'T1546', 'result': 'T1543', 'memory': {'state': 'use of shim databases for persistence', 'action': {'T1546': 'indicates use of system mechanisms for persistence', 'T1543': 'suggests creation or modification of system processes for persistence', 'T1574': 'hints at hijacking execution flow to run malicious payloads'}}, 'result_info': {'name': 'Create or Modify System Process', 'description': 'Adversaries may create or modify system-level processes to repeatedly execute malicious payloads as part of persistence. When operating systems boot up, they can start processes that perform background system functions. On Windows and Linux, these system processes are referred to as services. On macOS, launchd processes known as [Launch Daemon] and [Launch Agent] are run to finish system initialization and load user specific parameters. Adversaries may install new services, daemons, or agents that can be configured to execute at startup or a repeatable interval in order to establish persistence. Similarly, adversaries may modify existing services, daemons, or agents to achieve the same effect. Services, daemons, or agents may be created with administrator privileges but executed under root/SYSTEM privileges. Adversaries may leverage this functionality to create or modify system processes in order to escalate privileges.'}, 'label_info': {'name': 'Event Triggered Execution', 'description': 'Adversaries may establish persistence and/or elevate privileges using system mechanisms that trigger execution based on specific events. Various operating systems have means to monitor and subscribe to events such as logons or other user activity such as running specific applications/binaries. Cloud environments may also support various functions and services that monitor and can be invoked in response to specific cloud events. Adversaries may abuse these mechanisms as a means of maintaining persistent access to a victim via repeatedly executing malicious code. After gaining access to a victim system, adversaries may create/modify event triggers to point to malicious content that will be executed whenever the event trigger is invoked. Since the execution can be proxied by an account with higher permissions, such as SYSTEM or service accounts, an adversary may be able to abuse these triggered execution mechanisms to escalate their privileges.'}}"""
    ans = chat(prompt=text)
    print(ans)
