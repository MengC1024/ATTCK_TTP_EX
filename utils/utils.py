import json
import os
import re
import threading
import queue
from typing import Dict, List, Optional

import requests
from utils.chat import chat
import pandas as pd
import logging

logger = logging.getLogger("utils")
logging.getLogger("httpx").setLevel(logging.WARNING)

class ChatTimeoutError(Exception):
    pass


def load_dataset(dataset_name, dataset_type):
    dataset_path = f"./mitre-ttp-mapping/datasets/{dataset_name}/{dataset_name}_{dataset_type}.tsv"
    if os.path.exists(dataset_path):
        data_df = pd.read_csv(dataset_path, sep='\t')
        data_df["labels"] = data_df["labels"].apply(eval)
        return data_df
    else:
        logger.warning(f"数据集文件不存在: {dataset_path}，将不使用相似句子")
        return None
        
def chat_with_timeout(prompt, timeout_seconds=120):
    """使用threading.Timer实现的超时控制chat函数
    
    Args:
        prompt: 提示词
        timeout_seconds: 超时时间（秒）
        
    Returns:
        str: chat返回的结果
        
    Raises:
        ChatTimeoutError: 如果执行超时
    """
    result_queue = queue.Queue()
    
    def target():
        try:
            result = chat(prompt)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    
    try:
        status, result = result_queue.get(timeout=timeout_seconds)
        if status == "error":
            raise Exception(result)
        return result
    except queue.Empty:
        raise ChatTimeoutError("Chat执行超时")

def process_description(text):
    if not isinstance(text, str):
        return text
    
    # 递归删除所有括号及其内容
    result = ""
    skip_count = 0
    for char in text:
        if char == '(':
            skip_count += 1
        elif char == ')' and skip_count > 0:
            skip_count -= 1
        elif skip_count == 0:
            result += char
    
    # 清理可能出现的多余空格
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r'\s+\.', '.', result)  # 处理句号前的空格
    result = re.sub(r'\s+;', ';', result)   # 处理分号前的空格
    result = re.sub(r'\s+,', ',', result)   # 处理逗号前的空格
    result = result.strip()
    sentences = re.split(r'\.|\n', result)
    # sentences = re.split(r'\n', result)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return result

def get_ttp_id_to_name_and_description():
    with open("./technique_info.json", "r") as f:
        ttp_id_to_name_and_description = json.load(f)
    # 直接返回字典，其中每个条目已经包含了所需的 id、name 和 description
    return {ttp_id: {
        'name': info['name'], 
        'description': info['description']
    } 
            for ttp_id, info in ttp_id_to_name_and_description.items()}


def remember(sentences, label_list, k, service_url: str = "http://localhost:8009", allow_duplicate_tags: bool = True, sources: List[str] = None):
    """使用语义检索查找最相似的攻击描述
    
    Args:
        sentences: 要检索的文本或文本列表
        label_list: 标签列表，过滤记录
        k: 返回结果数量
        service_url: 服务地址
        allow_duplicate_tags: 是否允许返回结果中有重复的标签，默认为True
        sources: 指定要搜索的数据来源列表，可选值：
            - "official": 官方数据
            - "memory": 记忆数据
            - "procedures": Procedures数据
            - "tram": TRAM数据
            不指定则搜索所有数据源
    Returns:
        List[List[Dict]]: 每个查询的检索结果列表
    """
    try:
        # 确保输入是列表
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # 构建请求数据
        request_data = {
            "queries": sentences,
            "k": k,
            "label_list": label_list,
            "allow_duplicate_tags": allow_duplicate_tags
        }
        
        # 如果指定了数据源，添加到请求中
        if sources:
            request_data["sources"] = sources
            
        # 调用批量语义搜索接口
        response = requests.post(
            f"{service_url}/memories/search",
            json=request_data
        )
        response.raise_for_status()  # 检查响应状态
        
        results = response.json()
    
        # 格式化结果
        formatted_results = []
        for query_results in results:
            if not query_results:  # 检查是否为空列表
                formatted_results.append([{
                    "message": "没有找到相关的记录",
                    "status": "empty"
                }])
                continue
                
            query_formatted = [
                {
                    "state": result["record"]["state"],
                    "action": result["record"]["action"],
                    "similarity": float(result["score"]),
                    "tags": result["record"]["tags"],
                    "status": "success"
                }
                for result in query_results
            ]

            # for query_result in query_formatted:
            #     for technique_id in query_result["action"].keys():
            #         query_result["action"][technique_id] = query_result["action"][technique_id] + ". Official Description: " + ttp_id_to_name_and_description[technique_id]["description"]
            formatted_results.append(query_formatted)
        
        return formatted_results
        
    except requests.exceptions.RequestException as e:
        print(f"警告：调用服务时出错: {str(e)}")
        return [[{
            "message": f"服务调用失败: {str(e)}",
            "status": "error"
        }] for _ in sentences]
    except Exception as e:
        print(f"警告：处理结果时出错: {str(e)}")
        return [[{
            "message": f"处理结果时出错: {str(e)}",
            "status": "error"
        }] for _ in sentences]

def handle_same_state_memory(action1, action2):
    """
    处理相同state的memory，保持最早的记录优先
    """
    new_action = {}
    for k, v in action1.items():
        if k not in new_action:
            new_action[k] = v
    for k, v in action2.items():
        if k not in new_action:
            new_action[k] = v
    return new_action


def normalize_memory_pool(memory_pool, clean_action=True, normalize_state=True, merge_duplicates=True, verbose=True):
    """
    规范化和去重内存池
    
    Args:
        memory_pool: 内存池列表，每个元素是一个包含state和action的字典
        clean_action: 是否清理action字段，默认为True
        normalize_state: 是否规范化state字段，默认为True
        merge_duplicates: 是否合并相同state的内存，默认为True
        verbose: 是否打印详细信息，默认为True
        
    Returns:
        List[Dict]: 处理后的内存池
    """
    if verbose:
        print(f"原始内存池大小: {len(memory_pool)}")
    
    # 清理每个内存的action字段
    if clean_action:
        if verbose:
            print("正在清理action字段...")
            from tqdm import tqdm
            cleaned_memory_pool = []
            for memory in tqdm(memory_pool, desc="清理action"):
                cleaned_memory = memory.copy()
                if "action" in memory:
                    cleaned_memory["action"] = clean_memory_action(memory["action"])
                cleaned_memory_pool.append(cleaned_memory)
            memory_pool = cleaned_memory_pool
        else:
            memory_pool = [{**memory, "action": clean_memory_action(memory["action"]) if "action" in memory else {}} for memory in memory_pool]
    
    # 规范化state字段
    if normalize_state:
        if verbose:
            print("正在规范化state字段...")
            from tqdm import tqdm
            normalized_memory_pool = []
            for memory in tqdm(memory_pool, desc="规范化state"):
                normalized_memory = memory.copy()
                if "state" in memory and isinstance(memory["state"], str):
                    # 规范化state（去除多余空格、统一换行符等）
                    normalized_memory["state"] = re.sub(r'\s+', ' ', memory["state"]).strip()
                normalized_memory_pool.append(normalized_memory)
            memory_pool = normalized_memory_pool
        else:
            memory_pool = [{**memory, "state": re.sub(r'\s+', ' ', memory["state"]).strip() if "state" in memory and isinstance(memory["state"], str) else memory.get("state", "")} for memory in memory_pool]
    
    # 合并相同state的内存
    if merge_duplicates:
        # 使用state作为键建立索引
        state_to_memories = {}
        for memory in memory_pool:
            if "state" in memory and memory["state"]:
                state = memory["state"]
                if state not in state_to_memories:
                    state_to_memories[state] = []
                state_to_memories[state].append(memory)
        
        # 找出重复的state
        duplicate_states = {state: memories for state, memories in state_to_memories.items() if len(memories) > 1}
        
        if verbose:
            print(f"发现 {len(duplicate_states)} 个重复的state")
        
        # 合并重复state的内存
        merged_memory_pool = []
        processed_states = set()
        
        # 先处理重复的state
        if verbose:
            print("正在合并重复state的内存...")
            from tqdm import tqdm
            for state, memories in tqdm(duplicate_states.items(), desc="合并重复state"):
                # 合并所有相同state的memory的action，但不修改已有内容
                merged_action = {}
                
                # 按顺序处理，保持最早的记录优先
                for memory in memories:
                    if "action" in memory and isinstance(memory["action"], dict):
                        for k, v in memory["action"].items():
                            # 只有在merged_action中不存在该键时，才添加
                            if k not in merged_action:
                                merged_action[k] = v
                
                # 添加合并后的memory
                merged_memory_pool.append({"state": state, "action": merged_action})
                processed_states.add(state)
        else:
            for state, memories in duplicate_states.items():
                # 合并所有相同state的memory的action，但不修改已有内容
                merged_action = {}
                
                # 按顺序处理，保持最早的记录优先
                for memory in memories:
                    if "action" in memory and isinstance(memory["action"], dict):
                        for k, v in memory["action"].items():
                            # 只有在merged_action中不存在该键时，才添加
                            if k not in merged_action:
                                merged_action[k] = v
                
                merged_memory_pool.append({"state": state, "action": merged_action})
                processed_states.add(state)
        
        # 添加没有重复的memory
        for memory in memory_pool:
            if "state" in memory and memory["state"] not in processed_states:
                merged_memory_pool.append(memory)
        
        memory_pool = merged_memory_pool
    
    if verbose:
        print(f"处理后内存池大小: {len(memory_pool)}")
    
    return memory_pool

def clean_memory_action(action):
    """
    清理内存中的action字段，保证格式正确
    删除嵌套结构，只保留标准的术语映射
    """
    if not isinstance(action, dict):
        return {}
        
    cleaned_action = {}

    # 处理可能的嵌套结构
    def extract_attack_keys(obj, prefix=""):
        result = {}
        if isinstance(obj, dict):
            # 对于字典，递归处理其键值对
            for k, v in obj.items():
                if isinstance(k, str) and k.startswith("T") and len(k) == 5 and k[1:].isdigit():
                    # 直接是ATT&CK标签
                    if isinstance(v, str):
                        result[k] = v
                    elif isinstance(v, (int, float)):
                        result[k] = str(v)
                elif k == "action" and isinstance(v, dict):
                    # 如果键是"action"且值是字典，直接处理值
                    nested_result = extract_attack_keys(v)
                    result.update(nested_result)
                elif k == "sentence_classification" and isinstance(v, dict):
                    # 从sentence_classification中提取分类结果
                    if "classification" in v and isinstance(v["classification"], str):
                        classification = v["classification"]
                        if classification.startswith("T") and len(classification) == 5 and classification[1:].isdigit():
                            # 尝试找到相应的描述
                            if "sentence" in v and isinstance(v["sentence"], str):
                                result[classification] = v["sentence"][:100]  # 限制长度
                elif isinstance(v, dict):
                    # 对于嵌套字典，递归处理
                    nested_result = extract_attack_keys(v, prefix=k)
                    result.update(nested_result)
        return result

    # 从action中提取所有标准的ATT&CK标签
    extracted = extract_attack_keys(action)
    
    # 只保留标准格式的键值对
    for key, value in action.items():
        if isinstance(key, str) and key.startswith("T") and len(key) == 5 and key[1:].isdigit():
            if isinstance(value, str):
                cleaned_action[key] = value
            elif isinstance(value, (int, float)):
                # 将数值转为字符串
                cleaned_action[key] = str(value)
    
    # 合并提取的键值对
    cleaned_action.update(extracted)
    
    return cleaned_action

def memory_pool_stats(memory_pool):
    """
    生成内存池的统计信息
    
    Args:
        memory_pool: 内存池列表，每个元素是一个包含state和action的字典
    
    Returns:
        Dict: 统计信息字典
    """
    stats = {
        "total_memories": len(memory_pool),
        "memories_with_state": 0,
        "memories_with_action": 0,
        "empty_state_count": 0,
        "empty_action_count": 0,
        "unique_states": 0,
        "duplicate_states": 0,
        "duplicate_state_groups": 0,
        "techniques_count": {},
        "state_lengths": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        },
        "action_count_per_memory": {
            "min": float('inf'),
            "max": 0,
            "avg": 0
        }
    }
    
    # 统计基本信息
    state_counts = {}
    total_state_length = 0
    total_action_count = 0
    
    for memory in memory_pool:
        # 检查state字段
        if "state" in memory:
            stats["memories_with_state"] += 1
            state = memory["state"]
            
            if not state:  # 空state
                stats["empty_state_count"] += 1
            else:
                # 记录state长度
                state_length = len(str(state))
                stats["state_lengths"]["min"] = min(stats["state_lengths"]["min"], state_length)
                stats["state_lengths"]["max"] = max(stats["state_lengths"]["max"], state_length)
                total_state_length += state_length
                
                # 统计重复state
                if state in state_counts:
                    state_counts[state] += 1
                else:
                    state_counts[state] = 1
        
        # 检查action字段
        if "action" in memory:
            stats["memories_with_action"] += 1
            action = memory["action"]
            
            if not action:  # 空action
                stats["empty_action_count"] += 1
            else:
                # 统计action条目数
                action_count = len(action)
                stats["action_count_per_memory"]["min"] = min(stats["action_count_per_memory"]["min"], action_count)
                stats["action_count_per_memory"]["max"] = max(stats["action_count_per_memory"]["max"], action_count)
                total_action_count += action_count
                
                # 统计技术ID出现次数
                for tech_id in action.keys():
                    if tech_id in stats["techniques_count"]:
                        stats["techniques_count"][tech_id] += 1
                    else:
                        stats["techniques_count"][tech_id] = 1
    
    # 统计唯一state和重复state
    stats["unique_states"] = len(state_counts)
    duplicate_states = {state: count for state, count in state_counts.items() if count > 1}
    stats["duplicate_states"] = sum(count for count in duplicate_states.values()) - len(duplicate_states)
    stats["duplicate_state_groups"] = len(duplicate_states)
    
    # 计算平均值
    if stats["memories_with_state"] > 0:
        stats["state_lengths"]["avg"] = total_state_length / stats["memories_with_state"]
    else:
        stats["state_lengths"]["min"] = 0
        
    if stats["memories_with_action"] > 0:
        stats["action_count_per_memory"]["avg"] = total_action_count / stats["memories_with_action"]
    else:
        stats["action_count_per_memory"]["min"] = 0
    
    # 统计技术ID分布（取频率最高的10个）
    top_techniques = sorted(stats["techniques_count"].items(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_techniques"] = dict(top_techniques)
    
    return stats

def deduplicate_memory_file(input_file, output_file=None, clean_action=True, normalize_state=True, merge_duplicates=True, verbose=True):
    """
    处理内存文件，规范化和去重
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，默认为None（将在输入文件名后添加"_dedup"）
        clean_action: 是否清理action字段，默认为True
        normalize_state: 是否规范化state字段，默认为True
        merge_duplicates: 是否合并相同state的内存，默认为True
        verbose: 是否打印详细信息，默认为True
    
    Returns:
        Tuple[List[Dict], Dict]: 处理后的内存池和统计信息
    """
    # 设置默认输出文件
    if output_file is None:
        file_name, ext = os.path.splitext(input_file)
        output_file = f"{file_name}_dedup{ext}"
    
    if verbose:
        print(f"正在处理文件: {input_file}")
    
    try:
        # 读取输入文件
        with open(input_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"错误: 无法解析输入文件 {input_file} 为JSON")
                return None, None
        
        # 处理不同格式的输入
        memory_pool = None
        if isinstance(data, list):
            # 直接是memory列表
            memory_pool = data
        elif isinstance(data, dict):
            # 可能是包含memory_pool字段的字典
            if "memory_pool" in data:
                memory_pool = data["memory_pool"]
            elif "memories" in data:
                memory_pool = data["memories"]
            else:
                # 尝试找到包含多个memory的字段
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        if "state" in value[0] or "action" in value[0]:
                            memory_pool = value
                            break
        
        if memory_pool is None:
            print(f"错误: 无法从输入文件 {input_file} 中提取内存池")
            return None, None
        
        # 生成处理前的统计信息
        if verbose:
            print("处理前的统计信息:")
            stats_before = memory_pool_stats(memory_pool)
            print(f"  - 总内存数: {stats_before['total_memories']}")
            print(f"  - 唯一state数: {stats_before['unique_states']}")
            print(f"  - 重复state组数: {stats_before['duplicate_state_groups']}")
            print(f"  - 重复state数: {stats_before['duplicate_states']}")
        
        # 处理内存池
        processed_memory_pool = normalize_memory_pool(
            memory_pool, 
            clean_action=clean_action, 
            normalize_state=normalize_state, 
            merge_duplicates=merge_duplicates,
            verbose=verbose
        )
        
        # 生成处理后的统计信息
        stats_after = memory_pool_stats(processed_memory_pool)
        if verbose:
            print("\n处理后的统计信息:")
            print(f"  - 总内存数: {stats_after['total_memories']}")
            print(f"  - 唯一state数: {stats_after['unique_states']}")
            print(f"  - 重复state组数: {stats_after['duplicate_state_groups']}")
            print(f"  - 重复state数: {stats_after['duplicate_states']}")
            
            # 显示变化
            memory_reduction = stats_before['total_memories'] - stats_after['total_memories']
            if memory_reduction > 0:
                print(f"\n内存数量减少: {memory_reduction} ({memory_reduction/stats_before['total_memories']*100:.2f}%)")
        
        # 保存处理后的内存池
        with open(output_file, "w") as f:
            json.dump(processed_memory_pool, f, indent=2)
        
        if verbose:
            print(f"\n已保存处理后的内存池到: {output_file}")
        
        return processed_memory_pool, stats_after
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="内存池去重和清理工具")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument("--no-clean", action="store_true", help="不清理action字段")
    parser.add_argument("--no-normalize", action="store_true", help="不规范化state字段")
    parser.add_argument("--no-merge", action="store_true", help="不合并相同state的内存")
    parser.add_argument("--quiet", "-q", action="store_true", help="不打印详细信息")
    
    args = parser.parse_args()
    
    deduplicate_memory_file(
        args.input_file,
        args.output,
        clean_action=not args.no_clean,
        normalize_state=not args.no_normalize,
        merge_duplicates=not args.no_merge,
        verbose=not args.quiet
    )
