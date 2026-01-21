import json
from collections import defaultdict
import os
import random
import re
from utils.chatA100 import chat
import concurrent.futures
from tqdm import tqdm
import argparse
from prompt import prompt_reclassify
from utils.utils import remember

def reclassify(dataset_name, suffix, k, model, memory_mode):
    """重新分类
    
    Args:
        dataset_name: 数据集名称
        suffix: 后缀名
        k: 记忆数量
        memory_mode: 记忆模式
            - "direct": 直接取排序后的前k个记忆
            - "filter": 逐个验证筛选出有效的k个记忆
    """

    output_file = f"./records/results_{dataset_name}_test_{suffix}_classify2_top{k}_{memory_mode}.json"

    if os.path.exists(output_file):
        print(f"结果文件已存在，请勿重复处理: {output_file}")
        return


    k = int(k)
    
    # 验证memory_mode
    valid_modes = {"direct", "filter"}
    if isinstance(memory_mode, str):
        memory_mode = memory_mode.lower()
    if memory_mode not in valid_modes:
        raise ValueError(f"无效的memory_mode值: {memory_mode}。有效值为: {valid_modes}")
    print(f"使用记忆模式: {memory_mode}")
    
    path = f"./records/results_{dataset_name}_test_{suffix}.json"
    with open(path, "r") as f:
        test_results = json.load(f)

    records = test_results['records']
    label_list = list(set([item['labels'][0] for item in records]))

    # memories_result = memory_service.search_semantic([item['text'] for item in records], 10*k, label_list, True, ["memory"])
    memories_result = remember(sentences=[item['text'] for item in records], 
                               label_list=label_list, k=k*10, 
                            service_url="http://localhost:8009", 
                            allow_duplicate_tags=True, 
                            sources=["memory"])
    memories = []

    for single_query_result in tqdm(memories_result, desc="处理查询结果"):
        query_formatted = []
        for record in single_query_result:
            record_copy = dict(record)
            query_formatted.append(record_copy)
        memories.append(query_formatted)

    tasks = []

    text_to_memories = defaultdict(list)
    for i in range(len(records)):
        memories_now = memories[i]
        result_now = records[i]['result']
        
        # 先按原逻辑排序，优先选择包含当前结果的记忆
        memory_topk = []
        for memory in memories_now:
            if len(memory_topk) < k and result_now in memory['action'].keys():
                memory_topk.append(memory)
        for memory in memories_now:
            if len(memory_topk) < k:
                if result_now not in memory['action'].keys():
                    memory_topk.append(memory)
        
        text_to_memories[records[i]['text']] = memory_topk

        tasks.append({
            "text": records[i]['text'],
            "result": records[i]['result'],
            "labels": records[i]['labels'],
            "memories": memory_topk if memory_mode == "direct" else memories_now,  # filter模式传入更多记忆供筛选
            "memory_mode": memory_mode,
            "k": k,
            "model": model
        })
    
    # 执行并发任务处理
    all_results = process_tasks_with_progress(tasks)
    

    with open(output_file, "w") as f:
        json.dump({"records": all_results}, f, ensure_ascii=False, indent=2)

    print(f"处理完成，共处理 {len(all_results)} 个任务")
    print(f"结果已保存至: {output_file}")


def filter_single_memory(text, memory,model):
    """验证单个记忆是否有效，返回筛选后的记忆"""
    choices = set(memory['action'].keys())
    if not choices:
        return None
    
    # 构建单个记忆的prompt进行验证
    # different_info = f"The different between the original one and the similar one are: {memory['action']}"
    # formatted_memories = f"1. {different_info}\n"
    unique_labels = list(choices)
    
    prompt = prompt_reclassify.format(
        text=text, 
        result="",  # filter模式下不提供之前的结果
        memory=memory['action'],
        Techniques=unique_labels,
        Single_Technique=unique_labels[0]
    )
    
    try:
        answer = chat(prompt, model=model, timeout_seconds=3600)
        pattern = r"```answer\n(.*)\n```"
        match = re.search(pattern, answer)
        if match:
            answer = match.group(1)
            if answer in choices:
                # 返回只包含选中标签的记忆
                return {"state": memory.get('state', memory.get('text', '')), 
                        "action": {answer: memory['action'][answer]}}
    except Exception as e:
        print(f"验证记忆时出错: {str(e)}")
    
    return None


def classify_task(task):
    text = task['text']
    result = task['result']
    memories = task['memories']
    memory_mode = task.get('memory_mode', 'direct')
    model = task.get('model', 'gpt-4')
    k = task.get('k', 5)
    if memory_mode == "filter":
        # filter模式：逐个验证记忆，筛选出有效的k个
        filtered_memories = []
        for memory in memories:
            if len(filtered_memories) >= k:
                break
            filtered_memory = filter_single_memory(text, memory, model)
            if filtered_memory is not None:
                filtered_memories.append(filtered_memory)
        
        # 使用筛选后的记忆进行最终分类
        if not filtered_memories:
            # 如果没有筛选出有效记忆，回退到直接使用前k个
            filtered_memories = memories[:k]
        
        memories_for_classify = filtered_memories
    else:
        # direct模式：直接使用传入的记忆
        memories_for_classify = memories
    
    # 构建最终分类的prompt
    unique_labels = []
    for memory in memories_for_classify:
        unique_labels.extend(list(memory["action"].keys()))
    unique_labels = list(set(unique_labels))
    if len(unique_labels) == 0:
        print(f"No unique labels found for {text}")
        return {
            "text": text,
            "result": result,  # 保持原结果
            "old_result": result,
            "labels": task['labels'],
            "memories": memories_for_classify,
            "memory_mode": memory_mode
        }
    
    different_between_memories = []
    for i in range(len(memories_for_classify)):
        memory = memories_for_classify[i]
        different_between_memories.append(memory['action'])
    
    formatted_memories = ""
    for i in range(len(different_between_memories)):
        formatted_memories += f"{i+1}. {different_between_memories[i]}\n"
    prompt = prompt_reclassify.format(
        text=text, 
        result=result, 
        memory=formatted_memories,
        Techniques=unique_labels,
        Single_Technique=unique_labels[0]
    )    
    answer = chat(prompt, model=model, timeout_seconds=3600)
    pattern = r"```answer\n(.*)\n```"
    match = re.search(pattern, answer)
    if match:
        answer = match.group(1)
    else:
        answer = random.choice(unique_labels)
    
    return {
        "text": text,
        "result": answer,
        "old_result": result,
        "labels": task['labels'],
        "memories": memories_for_classify,
        "memory_mode": memory_mode
    }


def process_tasks_with_progress(tasks, max_workers=96):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(classify_task, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="处理任务") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    task = future_to_task[future]
                    print(f'任务 {task["text"][:30]}... 生成异常: {exc}')
                pbar.update(1)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分类')
    parser.add_argument('--dataset_name', default='procedures', help='数据集名称')
    parser.add_argument('--suffix', default='1', help='后缀名')
    parser.add_argument('--remember_k', default=5, help='记忆数量')
    parser.add_argument('--model', default="gpt-4", help='模型名称')
    parser.add_argument('--memory_mode', default="direct", 
                       help='记忆模式：direct（直接取前k个）或 filter（逐个验证筛选）')

    args = parser.parse_args()
    
    reclassify(args.dataset_name, args.suffix, args.remember_k, args.model, args.memory_mode)
