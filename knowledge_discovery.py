import argparse
import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from prompt import prompt_remember_old

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("remember")
logging.getLogger("httpx").setLevel(logging.WARNING)
# 常量
MEMORY_SERVICE_URL = "http://localhost:8009"
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒

class MemoryAPIClient:
    """记忆服务API客户端"""
    
    def __init__(self, base_url: str = MEMORY_SERVICE_URL):
        self.base_url = base_url
        
    def semantic_search(self, 
                       queries: List[str], 
                       k: int = 5, 
                       label_list: Optional[List[str]] = None,
                       sources: Optional[List[str]] = None) -> List[List[Dict]]:
        """执行语义搜索
        
        Args:
            queries: 查询文本列表
            k: 每个查询返回的结果数量
            label_list: 标签列表
            sources: 数据源列表，例如 ["procedures"]
            
        Returns:
            List[List[Dict]]: 搜索结果，每个查询对应一个结果列表
        """
        endpoint = f"{self.base_url}/memories/search"
        
        payload = {
            "queries": queries,
            "k": k,
            "allow_duplicate_tags": True
        }
        
        if label_list:
            payload["label_list"] = label_list
        
        if sources:
            payload["sources"] = sources
            
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"API请求失败，将在 {RETRY_DELAY}秒后重试: {str(e)}")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"API请求最终失败: {str(e)}")
                    raise
        
        # 不应该到达这里
        return []

    def get_info(self) -> Dict:
        """获取API系统信息"""
        endpoint = f"{self.base_url}/system/info"
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()


def chat(prompt: str, timeout_seconds: int = 1200, model: str = "qwen") -> str:
    """调用LLM获取响应
    
    Args:
        prompt: 提示文本
        timeout_seconds: 超时秒数
        
    Returns:
        str: LLM的响应
    """
    from utils.chatA100 import chat as chat_api
    try:
        return chat_api(prompt, timeout_seconds=timeout_seconds, model=model)
    except Exception as e:
        logger.error(f"调用LLM出错: {str(e)}")
        raise


def parse_memory_json(response: str) -> Optional[Dict]:
    """解析LLM响应中的JSON记忆数据
    
    Args:
        response: LLM的响应文本
        
    Returns:
        Optional[Dict]: 解析后的记忆字典，如果解析失败则返回None
    """
    # 尝试提取JSON代码块
    match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
    if not match:
        logger.warning("未在响应中找到JSON代码块")
        return None
        
    try:
        memory_dict = json.loads(match.group(1))
        return validate_memory_format(memory_dict)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析错误: {str(e)}")
        return None


def validate_memory_format(memory_dict: Dict) -> Optional[Dict]:
    """验证记忆条目的格式
    
    Args:
        memory_dict: 待验证的记忆条目字典
        
    Returns:
        Optional[Dict]: 验证通过返回原字典，否则返回None
    """
    try:
        # 检查必需字段
        required_fields = ["state", "action"]
        if not all(field in memory_dict for field in required_fields):
            logger.warning(f"缺少必需字段: {required_fields}")
            return None
            
        # 检查action格式
        action = memory_dict.get("action")
        if not isinstance(action, dict) or not action:
            logger.warning("action必须是非空字典")
            return None
            
        # 检查action的内容
        valid_action = {}
        for technique_id, description in action.items():
            # 检查technique_id格式
            if not re.match(r'^T\d+$', technique_id):
                logger.warning(f"跳过无效的technique_id格式: {technique_id}")
                continue
                
            # 检查description格式
            if not isinstance(description, str) or not description.strip():
                logger.warning(f"跳过无效的description格式: {description}")
                continue
                
            valid_action[technique_id] = description
            
        # 如果没有有效的action，返回None
        if not valid_action:
            logger.warning("没有有效的action")
            return None
            
        # 更新为有效的action
        memory_dict["action"] = valid_action
            
        return memory_dict
        
    except Exception as e:
        logger.error(f"验证记忆条目格式时出错: {str(e)}")
        return None


def process_similar_text(similar_data: List[Dict]) -> Dict[str, List[str]]:
    """处理相似文本数据，按技术ID组织
    
    Args:
        similar_data: 相似文本数据列表
        
    Returns:
        Dict[str, List[str]]: 按技术ID组织的相似文本
    """
    technique_sentences = {}
    
    for item in similar_data:
        record = item.get("record", {})
        state = record.get("state", "")
        
        if not state:
            continue
            
        # 获取该记录的标签/技术
        tags = record.get("tags", [])
        
        for tag in tags:
            if not tag.startswith("T"):
                continue
                
            if tag not in technique_sentences:
                technique_sentences[tag] = []
                
            technique_sentences[tag].append(state)
    
    # 对每个技术ID只保留前5个示例
    for tid in technique_sentences:
        technique_sentences[tid] = technique_sentences[tid][:5]
        
    return technique_sentences


def generate_memory(
    text: str,
    ground_truth: List[str],
    similar_data: Optional[List[Dict]] = None,
    id_to_official_descriptions: Optional[Dict[str, Dict]] = None,
    model: str = "qwen"
) -> Optional[Dict]:
    """为单条文本生成记忆
    
    Args:
        text: 需要处理的文本
        ground_truth: 正确的标签
        prediction: 当前预测结果
        memory: 现有记忆
        similar_data: 相似文本数据
        id_to_official_descriptions: 技术ID到官方描述的映射
        model: 模型名称
    Returns:
        Optional[Dict]: 生成的记忆，如果生成失败则返回None
    """
    try:
        # 收集所有相关的技术ID
        all_technique_ids = set(ground_truth)
        
        # 处理相似文本信息
        similar_str = ""
        technique_sentences = {}
        
        if similar_data:
            technique_sentences = process_similar_text(similar_data)
            
            # 添加所有技术ID
            all_technique_ids.update(technique_sentences.keys())
            
            # 格式化相似文本
            for tid, sentences in technique_sentences.items():
                similar_str += f"Technique ID: {tid}\n"
                for sent in sentences:
                    similar_str += f"- {sent}\n"
                similar_str += "\n"
        
        # 获取官方描述
        official_description_str = ""
        if id_to_official_descriptions:
            for tid in all_technique_ids:
                if tid in id_to_official_descriptions:
                    info = id_to_official_descriptions[tid]
                    official_description_str += f"Technique ID: {tid}\n"
                    official_description_str += f"Name: {info.get('name', '')}\n"
                    official_description_str += f"Description: {info.get('description', '')}\n"
                    official_description_str += "\n"
                else:
                    logger.warning(f"找不到技术ID {tid} 的官方描述")
        
        # 如果技术ID太少或太多，可能不适合生成记忆
        if len(all_technique_ids) < 2:
            logger.info(f"技术ID数量过少，跳过生成: {len(all_technique_ids)}")
            return None
            
        if len(all_technique_ids) > 20:
            logger.info(f"技术ID数量过多，跳过生成: {len(all_technique_ids)}")
            return None
        
        # 准备prompt并调用LLM
        prompt = prompt_remember_old.format(
            text=text, 
            official_description=official_description_str,
            similar=similar_str
        )
        # print(prompt)
        response = chat(prompt, timeout_seconds=120, model=model)
        
        # 解析响应
        return parse_memory_json(response)
        
    except Exception as e:
        logger.error(f"生成记忆时出错: {str(e)}")
        return None


def get_similar_sentences(data_df, api_client, similar_length=5) -> Dict[str, Dict]:
    """获取数据集中每个句子的相似句子
    
    Args:
        data_df: 包含文本和标签的DataFrame
        api_client: API客户端实例
        similar_length: 每个句子要检索的相似句子数量
        
    Returns:
        Dict[str, Dict]: 句子到相似句子信息的映射
    """
    sentence_to_similar = {}
    
    try:
        # 获取所有唯一的标签
        all_labels = set()
        for labels in data_df["labels"]:
            all_labels.update(labels)
        
        api_client = MemoryAPIClient()
        # 批量检索相似句子
        results = api_client.semantic_search(
            queries=list(data_df["text1"]),
            k=similar_length,
            label_list=list(all_labels),
            sources=["procedures"]
        )

        
        for i, (text, result) in enumerate(zip(data_df["text1"], results)):
            if not result:
                continue
                
            similar_data = {}
            # 按技术ID组织相似句子
            for item in result:
                record = item.get("record", {})
                state = record.get("state", "")
                
                if not state or state == text:  # 跳过空文本或与当前文本相同的结果
                    continue
                    
                tags = record.get("tags", [])
                
                for tag in tags:
                    if not tag.startswith("T"):
                        continue
                        
                    if tag not in similar_data:
                        similar_data[tag] = []
                        
                    if state not in similar_data[tag]:
                        similar_data[tag].append(state)
            
            # 对每个技术ID只保留前5个相似句子
            for tid in similar_data:
                similar_data[tid] = similar_data[tid][:5]
                
            # 存储相似文本信息
            if similar_data:
                similar_data["similar_labels"] = list(similar_data.keys())
                similar_data["label"] = data_df["labels"].iloc[i]
                sentence_to_similar[text] = similar_data
                
        logger.info(f"已为 {len(sentence_to_similar)} 个句子找到相似句子")
        
        return sentence_to_similar

    except Exception as e:
        logger.error(f"获取相似句子时出错: {str(e)}")
        return {}


def process_batch(
    results: List[Dict],
    sentence_to_similar: Dict[str, Dict],
    id_to_official_descriptions: Dict[str, Dict],
    model: str,
    threads: int = 4
) -> List[Dict]:
    """并行处理多条记录生成记忆
    
    Args:
        results: 记录列表
        sentence_to_similar: 句子到相似句子的映射
        id_to_official_descriptions: 技术ID到官方描述的映射
        model: 模型名称
        threads: 并行线程数
        
    Returns:
        List[Dict]: 生成的记忆列表
    """
    memories = []
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        
        for result in results:
            text = result['text']
            similar_data = None
            
            # 获取相似句子
            if text in sentence_to_similar:
                similar_info = sentence_to_similar[text]
                # 转换为API响应的格式
                similar_data = []
                for tid, sentences in similar_info.items():
                    if tid in ("label", "similar_labels"):
                        continue
                    for sentence in sentences:
                        similar_data.append({
                            "record": {
                                "state": sentence,
                                "tags": [tid]
                            },
                            "score": 0.9  # 占位分数
                        })
            
            futures.append(
                executor.submit(
                    generate_memory, 
                    text, 
                    result['labels'], 
                    similar_data,
                    id_to_official_descriptions,
                    model
                )
            )
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="生成记忆"):
            memory = future.result()
            if memory:
                memories.append(memory)
    
    return memories


def remember(dataset_name: str, suffix: str, suffix_output: str, similar_length: int = 5, threads: int = 4, data_source: str = "procedures", model: str = "qwen") -> List[Dict]:
    """主函数：加载数据、生成记忆并保存
    
    Args:
        dataset_name: 数据集名称
        suffix: 结果文件后缀
        suffix_output: 输出文件后缀
        similar_length: 每个句子要检索的相似句子数量
        threads: 并行处理的线程数
        
    Returns:
        List[Dict]: 生成的记忆列表
    """
    # 初始化API客户端
    api_client = MemoryAPIClient()
    
    # 检查API服务是否正常
    try:
        # info = api_client.get_info()
        logger.info(f"记忆服务正常运行，数据源: {data_source}")
    except Exception as e:
        logger.error(f"无法连接到记忆服务，请确保服务已启动: {str(e)}")
        return []
    
    # 检查文件路径
    results_path = f"./records/results_{dataset_name}_train_{suffix}.json"
    if not os.path.exists(results_path):
        logger.error(f"结果文件不存在: {results_path}")
        return []
        
    technique_info_path = "technique_info.json"
    if not os.path.exists(technique_info_path):
        logger.error(f"技术信息文件不存在: {technique_info_path}")
        return []
    
    # 加载数据
    try:
        with open(results_path, "r") as f:
            results = json.load(f)
            
        with open(technique_info_path, "r") as f:
            id_to_official_descriptions = json.load(f)
            
        # 加载数据集，用于获取相似句子
        dataset_path = f"./mitre-ttp-mapping/datasets/{dataset_name}/{dataset_name}_train.tsv"
        if os.path.exists(dataset_path):
            import pandas as pd
            data_df = pd.read_csv(dataset_path, sep='\t')
            data_df["labels"] = data_df["labels"].apply(eval)
            
            # 获取相似句子
            sentence_to_similar = get_similar_sentences(data_df, api_client, similar_length)
        else:
            logger.warning(f"数据集文件不存在: {dataset_path}，将不使用相似句子")
            sentence_to_similar = {}
    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        return []
    
    # 处理记忆数据
    try:
        logger.info(f"处理 {len(results['records'])} 条记录...")
        
        # 批量生成新记忆
        new_memories = process_batch(
            results['records'], 
            sentence_to_similar, 
            id_to_official_descriptions,
            model,
            threads
        )
        
        # 保存新生成的记忆
        if new_memories:
            output_path = f"./records/memories_{dataset_name}_{suffix_output}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(new_memories, f, ensure_ascii=False, indent=2)
            logger.info(f"已生成 {len(new_memories)} 条记忆，保存到 {output_path}")
        else:
            logger.warning("没有生成新的记忆")
            
        return new_memories
        
    except Exception as e:
        logger.error(f"处理记忆数据时出错: {str(e)}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成记忆')
    parser.add_argument('--dataset_name', required=True, help='数据集名称')
    parser.add_argument('--suffix', required=True, help='后缀名')
    parser.add_argument('--suffix_output', required=True, help='输出后缀名')
    parser.add_argument('--similar', type=int, default=5, help='每个句子检索的相似句子数量')
    parser.add_argument('--threads', type=int, default=96, help='并行处理的线程数')
    parser.add_argument('--data_source', type=str, default="procedures", help='数据源')
    parser.add_argument('--model', type=str, default="qwen3-32b", help='模型')
    
    args = parser.parse_args()
    remember(args.dataset_name, args.suffix, args.suffix_output, args.similar, args.threads, args.data_source, args.model)
