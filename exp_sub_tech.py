import argparse
import json
import re
import pandas as pd
import random
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

# 假设 utils.chatA100 是你本地的模块
try:
    from utils.chatA100 import chat
except ImportError:
    print("Error: 无法导入 utils.chatA100，请确保文件存在。")
    # 仅供测试用的假函数
    def chat(prompt, timeout_seconds, model):
        return "```answer\nT1059.001\n```"

# 禁用httpx日志
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# 1. 动态 Prompt 模板
# ==========================================
CODE_BLOCK = "```"

# 这是一个动态模板，需要填入 parent_info 和 candidate_list
PROMPT_HIERARCHICAL = """You are a cybersecurity expert specializing in the MITRE ATT&CK framework.
I have identified that the following threat intelligence text involves the technique: **{parent_info}**.
Your task is to identify the specific **Sub-technique** that best matches the description.

Here are the candidate sub-techniques for {parent_id}:
{candidate_list}

Input Text:
"{text}"

Requirements:
1. Analyze the text and match it against the provided candidate descriptions.
2. Select the MOST accurate Sub-technique ID.
3. If the text generally describes the parent technique but lacks detail for a specific sub-technique, or fits none, you may answer with the Parent ID itself ({parent_id}).
4. Strictly follow the output format.

Output Format:
{CODE_BLOCK}answer
Txxxx.yyy
{CODE_BLOCK}
"""

def parse_raw_answer(answer):
    """提取最后一个符合 Txxxx.yyy 格式的 ID"""
    # 优先找子技术 Txxxx.yyy
    pattern = r"(T\d{4}\.\d{3})"
    matches = re.findall(pattern, answer)
    if matches:
        return matches[-1]
    
    # 兜底：找父技术 Txxxx
    parent_pattern = r"(T\d{4})"
    parent_matches = re.findall(parent_pattern, answer)
    if parent_matches:
        return parent_matches[-1]
        
    return "Unknown"

# ==========================================
# 2. MITRE 知识库加载
# ==========================================
import pandas as pd
import os

def load_mitre_knowledge_base(file_path):
    """
    智能读取 MITRE 知识库文件
    支持读取原本的 .csv 导出文件，也支持直接读取原始 .xlsx 文件
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MITRE 数据文件未找到: {file_path}")

    print(f"Loading knowledge base from: {file_path}")

    # 1. 自动判断文件类型
    if file_path.endswith('.csv'):
        # 您当前上传的文件是这种情况
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        # 如果您将来直接用 MITRE 的原始 Excel 文件
        # 通常 techniques 信息在 'techniques' 这个 sheet 里
        try:
            df = pd.read_excel(file_path, sheet_name='techniques')
        except ValueError:
            # 如果找不到 sheet，尝试直接读取（或者是其他 sheet 名）
            print("Warning: 'techniques' sheet not found, loading first sheet.")
            df = pd.read_excel(file_path)
    else:
        raise ValueError("不支持的文件格式，请提供 .csv 或 .xlsx 文件")

    # 2. 清洗列名 (去除可能存在的空格)
    df.columns = [c.strip() for c in df.columns]
    
    # 3. 验证关键列是否存在
    required_cols = ['ID', 'name', 'description', 'sub-technique of']
    for col in required_cols:
        if col not in df.columns:
            # 尝试模糊匹配（有时候列名会有大小写差异）
            raise ValueError(f"文件缺少关键列: {col}。现有列: {list(df.columns)}")

    parent_map = {}
    name_map = {}

    # 建立 ID -> Name 映射
    for _, row in df.iterrows():
        if pd.notna(row['ID']) and pd.notna(row['name']):
            name_map[row['ID']] = row['name']

    # 建立 Parent -> Children 映射
    # 过滤出是子技术的行
    # 在 csv 中该列通常叫 'sub-technique of'，确保该列不为空
    if 'sub-technique of' in df.columns:
        sub_techniques = df[pd.notna(df['sub-technique of'])]
        
        for _, row in sub_techniques.iterrows():
            parent_id = row['sub-technique of']
            sub_id = row['ID']
            name = row['name']
            
            # 清洗描述
            desc = str(row['description'])
            if len(desc) > 500:
                desc = desc[:500] + "..."
            desc = desc.replace('\n', ' ')
            
            if parent_id not in parent_map:
                parent_map[parent_id] = []
            
            parent_map[parent_id].append({
                "id": sub_id,
                "name": name,
                "description": desc
            })
    
    return parent_map, name_map


# ==========================================
# 3. 数据加载与处理
# ==========================================
def load_data(data_file):
    if not os.path.exists(data_file):
        print(f"警告: 文件不存在 {data_file}")
        return []
    data = pd.read_csv(data_file, sep='\t')
    data["labels"] = data["labels"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    return data.to_dict(orient="records")

def load_datasets(dataset_name):
    base_path = f"./mitre-ttp-mapping/datasets/{dataset_name}/old"
    # 简单的路径拼接，可根据实际情况调整
    train = load_data(f"{base_path}/{dataset_name}_train.tsv")
    dev = load_data(f"{base_path}/{dataset_name}_dev.tsv")
    test = load_data(f"{base_path}/{dataset_name}_test.tsv")
    return train, dev, test

def generate_candidates_text(parent_id, candidates):
    """生成 Prompt 中的候选列表文本"""
    text_list = []
    for idx, item in enumerate(candidates):
        text_list.append(f"{idx+1}. ID: {item['id']} | Name: {item['name']}\n   Description: {item['description']}")
    return "\n".join(text_list)

def classify_one_hierarchical(text, parent_id, mitre_kb, model):
    """
    单次推理：
    1. 根据 parent_id 获取候选子技术。
    2. 如果没有子技术，直接返回 parent_id (不做细分)。
    3. 构建 Prompt 调用模型。
    """
    parent_map, name_map = mitre_kb
    
    # 获取该父技术下的所有子技术
    candidates = parent_map.get(parent_id, [])
    
    if not candidates:
        # 如果这个父技术没有子技术（也就是它是叶子节点），直接返回它自己
        return parent_id, "No Sub-techniques"

    parent_name = name_map.get(parent_id, "Unknown Technique")
    parent_info = f"{parent_id} ({parent_name})"
    candidate_str = generate_candidates_text(parent_id, candidates)
    
    prompt = PROMPT_HIERARCHICAL.format(
        parent_info=parent_info,
        parent_id=parent_id,
        candidate_list=candidate_str,
        text=text,
        CODE_BLOCK=CODE_BLOCK
    )
    
    try:
        answer_raw = chat(prompt, timeout_seconds=3600, model=model)
        return parse_raw_answer(answer_raw), candidates
    except Exception as e:
        print(f"Error in chat: {e}")
        return "Error", candidates

def process_dataset(dataset, dataset_name, suffix, mitre_kb, max_workers, model):
    """处理整个数据集"""
    output_file = f"./records/results_{dataset_name}_{suffix}_hierarchical.json"
    
    if os.path.exists(output_file):
        print(f"结果已存在: {output_file}")
        return

    print(f"\nProcessing {dataset_name} ({suffix})...")
    
    # 1. 任务构建
    # 将每个样本拆解为 (text, parent_id, target_sub_id) 的任务
    # 如果一条数据有多个 sub-technique 属于同一个 parent，通常我们只需要做一次 parent query
    # 但为了评估 F1，我们将每个 (Text, Parent) 视为一个分类问题
    
    tasks = []
    
    for item in dataset:
        text = item.get('text1', '')
        labels = item.get('labels', [])
        
        # 提取该样本中所有涉及的父技术ID
        # 仅处理包含子技术标签的样本
        # 逻辑：对于标签 T1059.001，Parent 是 T1059。
        # 我们构建一个任务：已知 Parent T1059，预测 Sub。期望结果是 T1059.001。
        
        sub_tech_labels = [l for l in labels if '.' in l]
        if not sub_tech_labels:
            continue
            
        # 按 Parent 分组
        parent_to_targets = {}
        for label in sub_tech_labels:
            parent = label.split('.')[0]
            if parent not in parent_to_targets:
                parent_to_targets[parent] = []
            parent_to_targets[parent].append(label)
        
        for parent, targets in parent_to_targets.items():
            tasks.append({
                "text": text,
                "parent_id": parent,
                "ground_truth_subs": targets, # 该父类下真实的子技术列表
                "original_labels": labels
            })
    
    print(f"构建了 {len(tasks)} 个子技术分类任务 (从 {len(dataset)} 条原始数据)")
    
    if not tasks:
        return

    # 2. 并发执行
    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(classify_one_hierarchical, t['text'], t['parent_id'], mitre_kb, model): i
            for i, t in enumerate(tasks)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="Inference"):
            idx = future_to_idx[future]
            try:
                pred, candidates = future.result()
                results[idx] = {
                    "answer": pred,
                    "candidates": [c['id'] for c in candidates] # 记录当时给了哪些选项
                }
            except Exception as e:
                results[idx] = {"answer": "Error", "candidates": []}

    # 3. 整合结果与评估
    final_records = []
    y_true_all = []
    y_pred_all = []
    
    for i, task in enumerate(tasks):
        res = results[i]
        pred = res['answer']
        gt_subs = task['ground_truth_subs']
        
        # F1 计算逻辑：
        # 这是一个多分类问题（在该 Parent 的候选集中选一个）。
        # 如果 pred 在 gt_subs 中，算对 (Match)。
        # 如果 pred 不在，取 gt_subs[0] 作为 Target，pred 作为 Prediction。
        
        if pred in gt_subs:
            y_true_all.append(pred)
            y_pred_all.append(pred)
            is_correct = True
        else:
            y_true_all.append(gt_subs[0])
            y_pred_all.append(pred)
            is_correct = False
            
        final_records.append({
            "text": task['text'],
            "parent_id": task['parent_id'],
            "ground_truth": gt_subs,
            "prediction": pred,
            "is_correct": is_correct,
            "options_provided": res['candidates']
        })

    # 4. 计算指标
    acc = accuracy_score(y_true_all, y_pred_all)
    macro_f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    
    print("\n" + "="*40)
    print(f"Hierarchical Classification Results ({suffix})")
    print("="*40)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print("="*40 + "\n")
    
    # 保存
    os.makedirs("./records", exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump({
            "metrics": {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1},
            "records": final_records
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='procedures')
    parser.add_argument('--suffix', default='1')
    parser.add_argument('--max_workers', default=50, type=int)
    parser.add_argument('--model', default="gpt-4")
    parser.add_argument('--mitre_file', default="enterprise-attack-v16.1.xlsx", help="Path to MITRE techniques CSV")
    
    args = parser.parse_args()
    
    # 加载 MITRE 知识库
    try:
        print("Loading MITRE Knowledge Base...")
        mitre_kb = load_mitre_knowledge_base(args.mitre_file)
        print(f"✓ Loaded. Found {len(mitre_kb[0])} parent techniques with sub-techniques.")
    except Exception as e:
        print(f"Failed to load MITRE file: {e}")
        return

    # 加载数据
    train, dev, test = load_datasets(args.dataset_name)
    
    # 这里默认只跑 test，如需跑其他请取消注释
    if test:
        process_dataset(test, args.dataset_name, f"test_{args.suffix}", mitre_kb, args.max_workers, args.model)

if __name__ == "__main__":
    main()
