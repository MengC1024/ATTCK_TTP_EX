import argparse
from collections import defaultdict
import json
import re
import pandas as pd
import random
import requests
import os
import logging
from utils.chatA100 import chat
from prompt import prompt_classify_one_with_memory, prompt_classify_one_without_memory
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List
from utils.utils import remember

# 禁用httpx的日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)

def load_data(data_file):
    data = pd.read_csv(data_file, sep='\t')
    data["labels"] = data["labels"].apply(eval)
    data = data.to_dict(orient="records")
    return data

def load_datasets(dataset_name:str):
    if dataset_name == "procedures":
        train_data = load_data("./mitre-ttp-mapping/datasets/procedures/procedures_train.tsv")
        dev_data = load_data("./mitre-ttp-mapping/datasets/procedures/procedures_dev.tsv")
        test_data = load_data("./mitre-ttp-mapping/datasets/procedures/procedures_test.tsv")
    elif dataset_name == "tram":
        train_data = load_data("./mitre-ttp-mapping/datasets/tram/tram_train.tsv")
        dev_data = load_data("./mitre-ttp-mapping/datasets/tram/tram_dev.tsv")
        test_data = load_data("./mitre-ttp-mapping/datasets/tram/tram_test.tsv")
    elif dataset_name == "expert":
        train_data = load_data("./mitre-ttp-mapping/datasets/expert/expert_train.tsv")
        dev_data = load_data("./mitre-ttp-mapping/datasets/expert/expert_dev.tsv")
        test_data = load_data("./mitre-ttp-mapping/datasets/expert/expert_test.tsv")
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")
    
    # 打乱数据
    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    
    return train_data, dev_data, test_data


def get_classify_prompt_with_memory(text, memories):
    formatted_memory = ""
    choices = set()
    for memory in memories:
        formatted_memory += f"simialr memory: <memory>{memory['state']}</memory>\n"
        formatted_memory += f"additional information help to classify: <information>{memory['action']}</information>\n"
        choices.update(memory['action'].keys())
    return prompt_classify_one_with_memory.format(text=text, memory=formatted_memory), list(choices)


def classify_one_with_memory(text, memories, model, k, memory_mode):
    """使用记忆进行分类
    
    Args:
        text: 待分类文本
        memories: 记忆列表
        model: 模型名称
        k: 使用的记忆数量
        memory_mode: 记忆模式
            - "direct": 直接取前k个记忆
            - "filter": 逐个验证筛选出有效的k个记忆
    """
    try:
        if memory_mode == "filter":
            # 逐个验证记忆，筛选出有效的 k 个
            new_memories = []
            for memory in memories:
                if len(new_memories) >= k:
                    break
                choices = set(memory['action'].keys())
                prompt, _ = get_classify_prompt_with_memory(text, [memory])
                answer = chat(prompt, timeout_seconds=3600, model=model)
                pattern = r"```answer\n(.*)\n```"
                match = re.search(pattern, answer)
                if match:
                    answer = match.group(1)
                    if answer in choices:
                        new_memories.append({"state": memory['state'], "action": {answer: memory['action'][answer]}})
        else:  # memory_mode == "direct"
            # 直接取前 k 个记忆
            new_memories = memories[:k]
        choices = set(key for memory in new_memories for key in memory['action'].keys())
        prompt, _ = get_classify_prompt_with_memory(text, new_memories)
        answer = chat(prompt, timeout_seconds=3600, model=model)
        pattern = r"```answer\n(.*)\n```"
        match = re.search(pattern, answer)
        if match:
            answer = match.group(1)
        return answer, new_memories
    except Exception as e:
        print(f"警告：处理文本时出错: {str(e)}")
        return random.choice(list(choices)) if choices else None, []


def parse_raw_answer(answer):
    pattern = r"```answer\n(.*)\n```"
    match = re.search(pattern, answer)
    if match:
        answer = match.group(1)
    return answer

def classify_one_without_memory(text, model):
    try:
        prompt = prompt_classify_one_without_memory.format(text=text)
        answer = chat(prompt, timeout_seconds=3600, model=model)
        return parse_raw_answer(answer), []
    except Exception as e:
        print(f"警告：处理文本时出错: {str(e)}")
        return None, []

def classify_batch(sentences, memorys, remember_k, max_workers=None, timeout=150, memory_use=True, model=None, memory_mode="direct"):
    """并发处理批量分类任务
    
    Args:
        sentences: 待分类的文本列表
        memorys: 对应的记忆列表
        max_workers: 最大工作线程数，默认为None，会自动根据CPU核心数进行优化设置
        timeout: 单个任务的超时时间(秒)，超过后会跳过该任务
        memory_use: 是否使用记忆进行分类
        model: 使用的模型名称
        remember_k: 筛选后保留的记忆数量
        memory_mode: 记忆模式 ("direct" 或 "filter")
    """
    results = [None] * len(sentences)  # 预分配结果列表
    new_memories_list = [None] * len(sentences)  # 预分配new_memories列表
    if max_workers is None:
        max_workers = min(os.cpu_count() * 2, len(sentences))
    print(f"使用 {max_workers} 个工作线程进行并行处理")
    
    # 创建进度条
    pbar = tqdm(total=len(sentences), desc="分类进度")
    
    def task_done_callback(future):
        pbar.update(1)
        try:
            idx = future_to_index[future]
            answer, new_memories = future.result()
            results[idx] = answer
            new_memories_list[idx] = new_memories
        except Exception as e:
            print(f"处理任务时出错: {str(e)}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for idx, sentence in enumerate(sentences):
            if memory_use:
                future = executor.submit(classify_one_with_memory, sentence, memorys[idx], model, remember_k, memory_mode)
            else:
                future = executor.submit(classify_one_without_memory, sentence, model)
            future_to_index[future] = idx
            future.add_done_callback(task_done_callback)
        
        import concurrent.futures
        concurrent.futures.wait(future_to_index)
    
    pbar.close()
    return results, new_memories_list

def classify_dataset(dataset, dataset_name:str, remember_k:int, max_workers=None, memory_use=True, data_source:list[str]=["memory"], model=None, memory_mode="direct"):
    # 检查结果文件是否已存在
    output_file = f"./records/results_{dataset_name}.json"
    if os.path.exists(output_file):
        print(f"✓ 结果文件已存在，跳过处理: {output_file}")
        # 读取并返回已有结果
        with open(output_file, "r") as f:
            existing_data = json.load(f)
            results = [record["result"] for record in existing_data["records"]]
            return results
    
    print(f"开始处理: {dataset_name}")
    sentences = [data["text1"] for data in dataset]
    labels = [data["labels"] for data in dataset]
    unique_label_list = list(set([label for labels in labels for label in labels]))
    # 只搜索memory来源的数据，获取10倍的记忆数量
    if memory_use:  
        memories = remember(sentences=sentences, label_list=unique_label_list, k=remember_k*10, 
                            service_url="http://localhost:8009", 
                            allow_duplicate_tags=True, 
                            sources=data_source)
    else:
        memories = []
    results, new_memories_list = classify_batch(sentences, memories, max_workers=max_workers, memory_use=memory_use, model=model, remember_k=remember_k, memory_mode=memory_mode)
    
    # 构建结果结构
    output = {
        "records": []  # 存储分类记录
    }
    
    for i in range(len(results)):
        output["records"].append({
            "text": sentences[i],
            "new_memories": new_memories_list[i],  # 直接存储new_memories
            "result": results[i],
            "labels": labels[i]
        })

    # 确保records目录存在
    os.makedirs("./records", exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"✓ 结果已保存到: {output_file}")
    return results

def classify(dataset_name, suffix, remember_k, max_workers=None, dataset_type=['train', 'dev', 'test'], memory_use=True, data_source=None, model=None, memory_mode="direct"):
    """批量分类数据集
    
    Args:
        dataset_name: 数据集名称
        suffix: 输出文件后缀
        remember_k: 每个查询使用的记忆数量
        max_workers: 最大工作线程数
        dataset_type: 要处理的数据集类型列表
        memory_use: 是否使用记忆进行分类
        data_source: 数据源列表，可选值：["official", "memory", "procedures", "tram"]，
                    可以是字符串（用逗号分隔）或列表
        model: 使用的模型名称
        memory_mode: 记忆模式
            - "direct": 直接取前k个记忆（对应原Document 1的行为）
            - "filter": 逐个验证筛选出有效的k个记忆（对应原Document 2的行为）
    """
    # 确保remember_k是整数类型
    remember_k = int(remember_k)
    max_workers = int(max_workers)
    
    # 处理memory_use参数
    if isinstance(memory_use, str):
        if memory_use.lower() == "true":
            memory_use = True
        elif memory_use.lower() == "false":
            memory_use = False
        else:
            raise ValueError(f"无效的memory_use值: {memory_use}")
    
    # 处理memory_mode参数
    valid_modes = {"direct", "filter"}
    if isinstance(memory_mode, str):
        memory_mode = memory_mode.lower()
    if memory_mode not in valid_modes:
        raise ValueError(f"无效的memory_mode值: {memory_mode}。有效值为: {valid_modes}")
            
    # 处理data_source参数
    valid_sources = {"official_full", "memory", "procedures", "official_first_line"}
    if data_source is None:
        data_source = ["memory"]  # 默认值
    elif isinstance(data_source, str):
        # 处理字符串输入，支持逗号分隔的多个值
        data_source = [s.strip().lower() for s in data_source.split(",")]
    elif isinstance(data_source, list):
        # 确保列表中的所有元素都是字符串
        data_source = [str(s).strip().lower() for s in data_source]
    else:
        raise ValueError("data_source必须是字符串（逗号分隔）或字符串列表")
    
    # 验证数据源的有效性
    invalid_sources = set(data_source) - valid_sources
    if invalid_sources:
        raise ValueError(f"无效的数据源: {invalid_sources}。有效值为: {valid_sources}")
        
    print(f"使用数据源: {data_source}")
    
    # 加载数据集
    train_data, dev_data, test_data = load_datasets(dataset_name)
    
    # print("\n" + "="*60)
    # print("开始处理数据集...")
    # print("="*60 + "\n")
    
    # 根据dataset_type处理不同的数据集
    if 'train' in dataset_type:
        classify_dataset(train_data, f"{dataset_name}_train_{suffix}", max_workers=max_workers, 
                         remember_k=remember_k, memory_use=memory_use,
                         data_source=data_source, model=model, memory_mode=memory_mode)
    if 'dev' in dataset_type:
        classify_dataset(dev_data, f"{dataset_name}_dev_{suffix}", max_workers=max_workers, 
                         remember_k=remember_k, memory_use=memory_use, 
                         data_source=data_source, model=model, memory_mode=memory_mode)
    if 'test' in dataset_type:
        classify_dataset(test_data, f"{dataset_name}_test_{suffix}", max_workers=max_workers, 
                         remember_k=remember_k, memory_use=memory_use, 
                         data_source=data_source, model=model, memory_mode=memory_mode)
    
    # print("\n" + "="*60)
    # print("所有数据集处理完成！")
    # print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分类')
    parser.add_argument('--dataset_name', default='procedures', help='数据集名称')
    parser.add_argument('--suffix', default='1', help='后缀名')
    parser.add_argument('--max_workers', default=50, help='最大工作线程数')
    parser.add_argument('--remember_k', default=5, help='记忆数量')
    parser.add_argument('--dataset_type', default=['train', 'dev', 'test'], help='数据集类型')
    parser.add_argument('--memory_use', default="True", help='是否使用记忆')
    parser.add_argument('--data_source', default="memory", 
                       help='数据源，可选值：official,memory,procedures,tram。多个值用逗号分隔，例如：memory,procedures')
    parser.add_argument('--model', default="qwen", help='模型')
    parser.add_argument('--memory_mode', default="direct", 
                       help='记忆模式：direct（直接取前k个）或 filter（逐个验证筛选）')
    
    args = parser.parse_args()
    
    classify(args.dataset_name, args.suffix, args.remember_k, args.max_workers, args.dataset_type, 
            args.memory_use, args.data_source, args.model, args.memory_mode)
