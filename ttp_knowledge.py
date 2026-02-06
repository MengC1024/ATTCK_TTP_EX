import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union, Set, Protocol, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from pydantic import BaseModel, Field
from enum import Enum, auto
from abc import ABC, abstractmethod


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TextMemory")


def process_description(text: str, mode: Literal["first_line", "first_paragraph", "full"] = "full") -> str:
    """处理描述文本，支持多种提取模式
    
    Args:
        text: 输入文本
        mode: 提取模式
            - "first_line": 返回第一行（以换行符或句号分割）
            - "first_paragraph": 返回第一段（以双换行符分割）
            - "full": 返回全部内容（默认）
    
    Returns:
        str: 处理后的文本
    """
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
    
    # 根据模式返回不同内容
    if mode == "first_line":
        # 按句号或换行符分割，返回第一句
        sentences = re.split(r'\.|\n', result)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences[0] + '.' if sentences else result
    
    elif mode == "first_paragraph":
        # 按双换行符或多个句号分割，返回第一段
        paragraphs = re.split(r'\n\n+', result)
        paragraphs = [para.strip() for para in paragraphs if para.strip()]
        return paragraphs[0] if paragraphs else result
    
    else:  # mode == "full"
        return result


class DataSource(str, Enum):
    """数据来源枚举（使用字符串枚举以便于序列化和使用）"""
    OFFICIAL_FIRST_LINE = "official_first_line"  # 官方数据 - 第一行
    OFFICIAL_FIRST_PARAGRAPH = "official_first_paragraph"  # 官方数据 - 第一段
    OFFICIAL_FULL = "official_full"  # 官方数据 - 完整内容
    MEMORY = "memory"  # 用户输入
    PROCEDURES = "procedures"  # 系统生成
    TRAM = "tram"  # TRAM数据
    
    @classmethod
    def from_string(cls, source_str: str) -> Optional['DataSource']:
        """从字符串创建DataSource枚举实例
        
        Args:
            source_str: 数据源字符串
            
        Returns:
            Optional[DataSource]: 如果字符串有效则返回DataSource实例，否则返回None
        """
        try:
            return cls(source_str.lower())
        except ValueError:
            return None
    
    @classmethod
    def get_all_sources(cls) -> List[str]:
        """获取所有数据源的字符串表示
        
        Returns:
            List[str]: 所有数据源的字符串值列表
        """
        return [source.value for source in cls]
    
    @classmethod
    def get_official_sources(cls) -> List['DataSource']:
        """获取所有官方数据源
        
        Returns:
            List[DataSource]: 官方数据源列表
        """
        return [cls.OFFICIAL_FIRST_LINE, cls.OFFICIAL_FIRST_PARAGRAPH, cls.OFFICIAL_FULL]


class MemoryRecord(BaseModel):
    """记忆条目的模型"""
    id: str
    state: str
    action: Dict[str, str]
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    source: DataSource = DataSource.MEMORY  # 数据来源，默认为用户输入

    @classmethod
    def create(cls, id: str, state: str, action: Dict[str, str], 
               tags: Optional[List[str]] = None, metadata: Optional[Dict] = None,
               source: DataSource = DataSource.MEMORY) -> 'MemoryRecord':
        """创建一个新的记忆条目"""
        return cls(
            id=id,
            state=state,
            action=action,
            tags=tags or [],
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
            source=source
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = self.dict()
        # 将枚举转换为字符串
        if "source" in data and isinstance(data["source"], DataSource):
            data["source"] = data["source"].value
        return data


class UpdateRecord(BaseModel):
    """更新记忆条目的模型"""
    id: Optional[str] = None
    state: Optional[str] = None
    action: Optional[Dict[str, str]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    content: Optional[Dict[str, Any]] = None


class MemoryStorage(ABC):
    """记忆存储抽象基类"""
    
    @abstractmethod
    def save(self, records: List[MemoryRecord]) -> bool:
        """保存记录到存储
        
        Args:
            records: 记忆记录列表
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    def load(self) -> List[MemoryRecord]:
        """从存储加载记录
        
        Returns:
            List[MemoryRecord]: 加载的记忆记录列表
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """清空存储
        
        Returns:
            bool: 清空是否成功
        """
        pass


class JsonFileStorage(MemoryStorage):
    """JSON文件存储实现"""
    
    def __init__(self, filepath: str):
        """初始化JSON文件存储
        
        Args:
            filepath: 存储文件的路径
        """
        self.filepath = filepath
        
    def save(self, records: List[MemoryRecord]) -> bool:
        """保存记录到JSON文件
        
        Args:
            records: 记忆记录列表
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
            
            # 保存为JSON
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump([record.to_dict() for record in records], f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存数据到JSON文件时出错: {str(e)}")
            return False
    
    def load(self) -> List[MemoryRecord]:
        """从JSON文件加载记录
        
        Returns:
            List[MemoryRecord]: 加载的记忆记录列表
        """
        if not os.path.exists(self.filepath):
            logger.info(f"存储文件 {self.filepath} 不存在，将返回空列表")
            return []
            
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                records_data = json.load(f)
                return [MemoryRecord(**record) for record in records_data]
        except Exception as e:
            logger.error(f"从JSON文件加载数据时出错: {str(e)}")
            return []
    
    def clear(self) -> bool:
        """删除JSON文件
        
        Returns:
            bool: 删除是否成功
        """
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
                logger.info(f"已删除存储文件 {self.filepath}")
                return True
            except Exception as e:
                logger.error(f"删除存储文件时出错: {str(e)}")
                return False
        return True


class EmbeddingModel(ABC):
    """文本嵌入模型抽象基类"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本编码为向量
        
        Args:
            texts: 待编码的文本列表
            
        Returns:
            np.ndarray: 文本向量，每行对应一个文本
        """
        pass
    
    @abstractmethod
    def get_similarity(self, query_embeddings: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """计算查询向量和文档向量的相似度
        
        Args:
            query_embeddings: 查询向量，形状为(n_queries, dim)
            document_embeddings: 文档向量，形状为(n_docs, dim)
            
        Returns:
            np.ndarray: 相似度矩阵，形状为(n_queries, n_docs)
        """
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """基于SentenceTransformer的文本嵌入实现"""
    
    def __init__(self, model_name_or_path: str):
        """初始化SentenceTransformer模型
        
        Args:
            model_name_or_path: 模型名称或路径
        """
        self.model_name = model_name_or_path
        self._model = None
        
    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                import torch
                # 检查可用的 GPU
                if torch.cuda.is_available():
                    # 检查每个 GPU 的内存使用情况
                    gpu_count = torch.cuda.device_count()
                    logger.info(f"发现 {gpu_count} 个 GPU")
                    
                    # 找到内存最空闲的 GPU
                    best_gpu = 0
                    max_free_memory = 0
                    
                    for i in range(gpu_count):
                        # 使用 mem_get_info 获取实际可用内存（包括其他进程）
                        free_memory, total_memory = torch.cuda.mem_get_info(i)
                        logger.info(f"GPU {i}: {free_memory / 1024**3:.2f} GB 可用 / {total_memory / 1024**3:.2f} GB 总计")
                        if free_memory > max_free_memory:
                            max_free_memory = free_memory
                            best_gpu = i
                    
                    device = f"cuda:{best_gpu}"
                    logger.info(f"选择 GPU {best_gpu}（{max_free_memory / 1024**3:.2f} GB 可用）")
                else:
                    device = "cpu"
                    logger.info("未检测到 GPU，使用 CPU")
                
                logger.info(f"加载SentenceTransformer模型: {self.model_name} on {device}")
                self._model = SentenceTransformer(self.model_name, device=device)
                logger.info(f"模型加载完成，使用设备: {device}")
            except Exception as e:
                logger.error(f"加载SentenceTransformer模型时出错: {str(e)}")
                raise
                
    def encode(self, texts: List[str]) -> np.ndarray:
        """编码文本列表
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 文本嵌入向量，形状为(len(texts), dim)
        """
        if not texts:
            return np.array([])
            
        self._load_model()
        try:
            return self._model.encode(texts)
        except Exception as e:
            logger.error(f"编码文本时出错: {str(e)}")
            raise
    
    def get_similarity(self, query_embeddings: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """使用余弦相似度计算查询向量和文档向量的相似度
        
        Args:
            query_embeddings: 查询向量，形状为(n_queries, dim)
            document_embeddings: 文档向量，形状为(n_docs, dim)
            
        Returns:
            np.ndarray: 相似度矩阵，形状为(n_queries, n_docs)
        """
        if query_embeddings.size == 0 or document_embeddings.size == 0:
            return np.array([])
            
        try:
            return cosine_similarity(query_embeddings, document_embeddings)
        except Exception as e:
            logger.error(f"计算相似度时出错: {str(e)}")
            raise
            
    def release_memory(self):
        """释放模型占用的GPU内存"""
        if self._model is not None:
            try:
                import torch
                import gc
                
                logger.info("开始释放模型占用的GPU内存")
                # 删除模型实例
                del self._model
                self._model = None
                
                # 清理PyTorch缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 强制执行垃圾回收
                gc.collect()
                
                logger.info("GPU内存释放完成")
            except Exception as e:
                logger.error(f"释放GPU内存时出错: {str(e)}")


class DataLoader(ABC):
    """数据加载器抽象基类"""
    
    @abstractmethod
    def load(self) -> List[MemoryRecord]:
        """加载数据为记忆记录
        
        Returns:
            List[MemoryRecord]: 记忆记录列表
        """
        pass


class ExcelDataLoader(DataLoader):
    """从Excel加载官方数据 - 支持三种模式，直接从Excel读取技术信息"""
    
    def __init__(self, excel_path: str):
        """初始化Excel数据加载器
        
        Args:
            excel_path: Excel文件路径
        """
        self.excel_path = excel_path
    
    def _load_technique_info_from_excel(self) -> Dict[str, Dict[str, str]]:
        """直接从Excel加载技术信息
        
        Returns:
            Dict[str, Dict[str, str]]: 技术ID到技术信息(name, description等)的映射
            
        Raises:
            FileNotFoundError: 当Excel文件不存在时抛出
            RuntimeError: 当加载失败时抛出
        """
        try:
            if not os.path.exists(self.excel_path):
                raise FileNotFoundError(f"Excel文件不存在: {self.excel_path}")
            
            logger.info(f"从 {self.excel_path} 读取技术信息")
            df = pd.read_excel(self.excel_path)
            
            # 提取需要的列
            df_info = df[["ID", "name", "description"]].copy()
            
            technique_info = {}
            for _, row in df_info.iterrows():
                # 处理 description，移除括号内容
                cleaned_desc = process_description(row['description'], mode="full")
                technique_info[row['ID']] = {
                    'name': row['name'],
                    'description': cleaned_desc
                }
            
            logger.info(f"从Excel加载了 {len(technique_info)} 个技术信息")
            return technique_info
            
        except Exception as e:
            error_msg = f"从Excel加载技术信息时出错: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load(self) -> List[MemoryRecord]:
        """从Excel加载官方数据 - 生成三个版本
        
        Returns:
            List[MemoryRecord]: 官方数据记忆记录（三个版本）
        """
        try:
            # 检查文件是否存在
            
            # 加载技术信息
            technique_info = self._load_technique_info_from_excel()
            if not os.path.exists(self.excel_path):
                logger.error(f"Excel文件不存在: {self.excel_path}")
                return []
                
            # 读取Excel文件
            logger.info(f"从Excel加载官方数据（三个版本）: {self.excel_path}")
            df = pd.read_excel(self.excel_path)
            
            # 过滤掉包含点的ID
            df = df[~df["ID"].str.contains(r"\.")]
            
            all_records = []
            base_id_counter = 1
            
            # 为每一行生成三个版本
            for i, row in df.iterrows():
                attack_id = row['ID']
                
                # 从技术信息中获取名称和描述
                tech_info = technique_info.get(attack_id)
                if not tech_info:
                    logger.warning(f"未找到ID {attack_id} 对应的技术信息，跳过")
                    continue
                
                tech_name = tech_info.get('name')
                if not tech_name:
                    logger.warning(f"ID {attack_id} 对应的技术信息中没有name字段，跳过")
                    continue
                    
                tech_desc = tech_info.get('description', '')
                
                # 构建action
                if tech_desc:
                    action = {
                        attack_id: f"this description belongs to: {attack_id}."
                    }
                else:
                    action = {
                        attack_id: f"this description belongs to: {attack_id}."
                    }
                
                # 构建标签
                tags = [attack_id]
                
                # 构建元数据
                metadata = {
                    "attack_id": attack_id,
                    "attack_name": tech_name,
                    "platform": row.get('platform', ''),
                    "permissions_required": row.get('permissions_required', ''),
                    "effective_permissions": row.get('effective_permissions', '')
                }
                
                # 生成三个版本的记录
                modes = [
                    ("first_line", DataSource.OFFICIAL_FIRST_LINE),
                    ("first_paragraph", DataSource.OFFICIAL_FIRST_PARAGRAPH),
                    ("full", DataSource.OFFICIAL_FULL)
                ]
                
                for mode_name, data_source in modes:
                    # 处理描述文本
                    cleaned_description = process_description(row['description'], mode=mode_name)
                    
                    # 创建记录
                    record = MemoryRecord.create(
                        id=f"{data_source.value}_{base_id_counter}",
                        state=cleaned_description,
                        action=action,
                        tags=tags,
                        metadata={**metadata, "description_mode": mode_name},
                        source=data_source
                    )
                    all_records.append(record)
                
                base_id_counter += 1
                
            logger.info(f"从Excel加载了 {len(all_records)} 条官方数据（{len(df)} 条原始记录 × 3种模式）")
            return all_records
                
        except Exception as e:
            logger.error(f"从Excel加载官方数据时出错: {str(e)}")
            return []


class BaseTsvDataLoader(DataLoader):
    """从TSV文件加载数据的基类"""
    
    def __init__(self, tsv_path: str, excel_path: str, id_prefix: str, data_source: DataSource):
        """初始化TSV数据加载器
        
        Args:
            tsv_path: TSV文件路径
            excel_path: Excel文件路径（用于读取技术信息）
            id_prefix: 记录ID的前缀
            data_source: 数据来源枚举值
        """
        self.tsv_path = tsv_path
        self.excel_path = excel_path
        self.id_prefix = id_prefix
        self.data_source = data_source
        
    def _load_technique_info_from_excel(self) -> Dict[str, Dict[str, str]]:
        """直接从Excel加载技术信息
        
        Returns:
            Dict[str, Dict[str, str]]: 技术ID到技术信息(name, description等)的映射
        """
        try:
            if not os.path.exists(self.excel_path):
                raise FileNotFoundError(f"Excel文件不存在: {self.excel_path}")
            
            logger.info(f"从 {self.excel_path} 读取技术信息")
            df = pd.read_excel(self.excel_path)
            
            # 提取需要的列
            df_info = df[["ID", "name", "description"]].copy()
            
            technique_info = {}
            for _, row in df_info.iterrows():
                # 处理 description，移除括号内容
                cleaned_desc = process_description(row['description'], mode="full")
                technique_info[row['ID']] = {
                    'name': row['name'],
                    'description': cleaned_desc
                }
            
            return technique_info
            
        except Exception as e:
            logger.error(f"从Excel加载技术信息时出错: {str(e)}")
            raise RuntimeError(f"加载技术信息失败: {str(e)}")
        
    def _create_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """创建元数据，子类可以重写此方法以添加特定的元数据
        
        Args:
            row: 数据行
            
        Returns:
            Dict[str, Any]: 元数据字典
        """
        return {
            "labels": row['labels']
        }
    
    def _parse_labels(self, labels_str: str) -> List[str]:
        """解析标签字符串为标签列表
        
        Args:
            labels_str: 标签字符串，可能是字符串或字符串表示的列表
            
        Returns:
            List[str]: 标签列表
        """
        try:
            if isinstance(labels_str, str):
                # 尝试解析字符串表示的列表
                if labels_str.startswith('[') and labels_str.endswith(']'):
                    # 使用 eval 解析字符串列表（确保输入是安全的）
                    labels = eval(labels_str)
                    if isinstance(labels, list):
                        return [str(label).strip() for label in labels if label]
                # 如果是单个标签
                return [labels_str.strip()]
            elif isinstance(labels_str, list):
                return [str(label).strip() for label in labels_str if label]
            else:
                logger.warning(f"无效的标签格式: {type(labels_str)}")
                return []
        except Exception as e:
            logger.error(f"解析标签时出错: {str(e)}, 标签内容: {labels_str}")
            return []
        
    def load(self) -> List[MemoryRecord]:
        """从TSV文件加载数据
        
        Returns:
            List[MemoryRecord]: 记忆记录列表
        """
        try:
            # 检查文件是否存在
            
            # 加载技术信息
            technique_info = self._load_technique_info_from_excel()
            if not os.path.exists(self.tsv_path):
                logger.error(f"TSV文件不存在: {self.tsv_path}")
                return []
                
            # 读取TSV文件
            logger.info(f"从TSV加载{self.data_source.value}数据: {self.tsv_path}")
            df = pd.read_csv(self.tsv_path, sep='\t')
            
            # 检查必要的列是否存在
            required_columns = ['text1', 'labels']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"TSV文件缺少必要的列: {missing_columns}")
            
            records = []
            invalid_records = []
            
            for i, row in df.iterrows():
                try:
                    # 解析标签
                    labels = self._parse_labels(row['labels'])
                    if not labels:
                        invalid_records.append((i, "空标签"))
                        continue
                        
                    # 处理文本（TSV数据使用full模式）
                    state = process_description(row['text1'], mode="full")
                    if not state or not state.strip():
                        invalid_records.append((i, "空文本"))
                        continue
                    
                    # 构建action（技术ID和描述的映射）
                    action = {}
                    for label in labels:
                        # 从技术信息中获取名称和描述
                        tech_info = technique_info.get(label)
                        if not tech_info:
                            error_msg = f"找不到标签 {label} 对应的技术信息"
                            logger.warning(error_msg)
                            invalid_records.append((i, error_msg))
                            continue
                        
                        tech_name = tech_info.get('name')
                        if not tech_name:
                            error_msg = f"标签 {label} 对应的技术信息中没有name字段"
                            logger.warning(error_msg)
                            invalid_records.append((i, error_msg))
                            continue
                            
                        tech_desc = tech_info.get('description', '')
                        
                        if tech_desc:
                            action[label] = f"{label}."
                        else:
                            logger.warning(f"标签 {label} 对应的技术信息中没有description字段")
                            continue
                    
                    # 如果所有标签都无效，跳过此记录
                    if not action:
                        invalid_records.append((i, "所有标签都无效"))
                        continue
                    
                    # 创建记录
                    record = MemoryRecord.create(
                        id=f"{self.id_prefix}_{i + 1}",
                        state=state,
                        action=action,
                        tags=labels,
                        metadata=self._create_metadata(row),
                        source=self.data_source
                    )
                    records.append(record)
                    
                except Exception as e:
                    invalid_records.append((i, str(e)))
                    continue
            
            # 报告加载结果
            logger.info(f"从TSV加载了 {len(records)} 条{self.data_source.value}数据")
            if invalid_records:
                logger.warning(f"发现 {len(invalid_records)} 条无效记录:")
                for i, reason in invalid_records[:5]:  # 只显示前5条
                    logger.warning(f"  行 {i+1}: {reason}")
            
            return records
                
        except Exception as e:
            logger.error(f"从TSV加载{self.data_source.value}数据时出错: {str(e)}")
            return []


class ProceduresDataLoader(BaseTsvDataLoader):
    """加载Procedures数据"""
    
    def __init__(self, tsv_path: str, excel_path: str):
        """初始化Procedures数据加载器
        
        Args:
            tsv_path: TSV文件路径
            excel_path: Excel文件路径
        """
        super().__init__(tsv_path, excel_path, "proc", DataSource.PROCEDURES)
        
    def _create_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """创建Procedures特定的元数据
        
        Args:
            row: 数据行
            
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = super()._create_metadata(row)
        metadata.update({
            "source": row.get('source', ''),
            "url": row.get('url', '')
        })
        return metadata


class TramDataLoader(BaseTsvDataLoader):
    """加载TRAM数据"""
    
    def __init__(self, tsv_path: str, excel_path: str):
        """初始化TRAM数据加载器
        
        Args:
            tsv_path: TSV文件路径
            excel_path: Excel文件路径
        """
        super().__init__(tsv_path, excel_path, "tram", DataSource.TRAM)
        
    def _create_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """创建TRAM特定的元数据
        
        Args:
            row: 数据行
            
        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = super()._create_metadata(row)
        metadata.update({
            "source": "tram"
        })
        return metadata


class MemoryConfig:
    """记忆系统配置"""
    
    def __init__(self, 
                storage_path: str = "text_memory.json", 
                model_name: str = "/home/mengcheng/models/sentence-transformers-all-mpnet-base-v2",
                excel_path: str = "./enterprise-attack-v16.1.xlsx",
                procedures_path: str = "./mitre-ttp-mapping/datasets/procedures/procedures_train.tsv",
                tram_path: str = "./mitre-ttp-mapping/datasets/tram/tram_train.tsv",
                enable_persistence: bool = True,
                auto_load_from_json: bool = False):
        """初始化记忆系统配置
        
        Args:
            storage_path: 存储文件路径
            model_name: 嵌入模型名称或路径
            excel_path: Excel数据文件路径
            procedures_path: Procedures数据文件路径
            tram_path: TRAM数据文件路径
            enable_persistence: 是否启用持久化存储（保存到JSON）
            auto_load_from_json: 是否从JSON自动加载（False表示总是从原始文件加载）
        """
        self.storage_path = storage_path
        self.model_name = model_name
        self.excel_path = excel_path
        self.procedures_path = procedures_path
        self.tram_path = tram_path
        self.enable_persistence = enable_persistence
        self.auto_load_from_json = auto_load_from_json


class TextMemory:
    """文本记忆管理器"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """初始化文本记忆管理器
        
        Args:
            config: 记忆系统配置，如果为None则使用默认配置
        """
        self.config = config or MemoryConfig()
        
        # 初始化依赖组件
        self.storage = JsonFileStorage(self.config.storage_path) if self.config.enable_persistence else None
        self.embedding_model = SentenceTransformerEmbedding(self.config.model_name)
        self.official_loader = ExcelDataLoader(self.config.excel_path)
        self.procedures_loader = ProceduresDataLoader(
            self.config.procedures_path,
            self.config.excel_path
        )
        self.tram_loader = TramDataLoader(
            self.config.tram_path,
            self.config.excel_path
        )
        
        # 内存中的记录和缓存
        self.records: List[MemoryRecord] = []
        self.embeddings_cache: Optional[np.ndarray] = None
        
        logger.info(f"记忆系统已创建 (enable_persistence={self.config.enable_persistence}, auto_load_from_json={self.config.auto_load_from_json})")
        logger.info("请手动调用 load_data() 方法加载数据")
    
    def load_data(self, sources: Optional[List[DataSource]] = None, force_reload: bool = True):
        """手动加载数据
        
        Args:
            sources: 要加载的数据源列表，如果为None则加载所有数据源
            force_reload: 是否强制从原始文件重新加载（True表示不使用JSON缓存）
        """
        # 如果不强制重新加载，且允许从JSON加载，则尝试从JSON加载
        if not force_reload and self.config.auto_load_from_json and self.storage:
            logger.info("尝试从JSON文件加载数据")
            self.records = self.storage.load()
            if self.records:
                logger.info(f"从JSON文件加载了 {len(self.records)} 条记录")
                return
            logger.info("JSON文件为空或不存在，将从原始文件加载")
        
        # 从原始文件加载数据
        all_records = []
        
        # 确定要加载的数据源
        if sources is None:
            # 默认加载所有数据源（三个官方版本 + procedures + tram）
            sources_to_load = DataSource.get_official_sources() + [DataSource.PROCEDURES, DataSource.TRAM]
        else:
            sources_to_load = sources
        
        logger.info(f"开始从原始文件加载数据源: {[s.value for s in sources_to_load]}")
        
        # 检查是否需要加载官方数据（任意一个版本）
        official_sources = set(DataSource.get_official_sources())
        need_official = bool(official_sources.intersection(set(sources_to_load)))
        
        if need_official:
            logger.info("加载官方数据（三个版本）")
            official_records = self.official_loader.load()
            # 只保留请求的版本
            for record in official_records:
                if record.source in sources_to_load:
                    all_records.append(record)
        
        if DataSource.PROCEDURES in sources_to_load:
            logger.info("加载Procedures数据")
            all_records.extend(self.procedures_loader.load())
            
        if DataSource.TRAM in sources_to_load:
            logger.info("加载TRAM数据")
            all_records.extend(self.tram_loader.load())
        
        if not all_records:
            raise RuntimeError("无法从任何数据源加载数据")
        
        self.records = all_records
        
        # 如果启用了持久化，保存数据
        if self.storage and self.config.enable_persistence:
            logger.info("保存数据到JSON文件")
            self.storage.save(self.records)
            
        logger.info(f"数据加载完成，共加载 {len(self.records)} 条记录")
    
    def add_text(self, state: str, action: Dict[str, str], 
                tags: Optional[List[str]] = None, 
                metadata: Optional[Dict] = None) -> str:
        """添加新的记忆条目
        
        Args:
            state: 状态文本
            action: 动作字典，key是标签，value是描述文本
            tags: 文本标签列表
            metadata: 额外的元数据
            
        Returns:
            str: 记录ID
            
        Raises:
            ValueError: 当参数无效时抛出
        """
        if not state or not state.strip():
            raise ValueError("state不能为空")
            
        if not isinstance(action, dict) or not action:
            raise ValueError("action必须是非空字典")
            
        # 创建新记录
        record = MemoryRecord.create(
            id=str(len(self.records) + 1),
            state=state,
            action=action,
            tags=tags,
            metadata=metadata
        )
        
        # 添加记录到内存
        self.records.append(record)
        self.embeddings_cache = None
        
        # 如果启用了持久化，保存到存储
        if self.storage and self.config.enable_persistence:
            if not self.storage.save(self.records):
                raise RuntimeError("保存记录失败")
            
        logger.info(f"添加新记忆，ID: {record.id}")
        return record.id

    def get_text(self, text_id: str) -> MemoryRecord:
        """根据ID获取记录
        
        Args:
            text_id: 记录ID
            
        Returns:
            MemoryRecord: 记录
            
        Raises:
            ValueError: 当记录不存在时抛出
        """
        for record in self.records:
            if record.id == text_id:
                return record
        raise ValueError(f"未找到ID为 {text_id} 的记录")

    def search_by_tags(self, tags: List[str]) -> List[MemoryRecord]:
        """根据标签搜索记录
        
        Args:
            tags: 标签列表
            
        Returns:
            List[MemoryRecord]: 匹配的记录列表
            
        Raises:
            ValueError: 当标签列表为空时抛出
        """
        if not tags:
            raise ValueError("标签列表不能为空")
            
        results = []
        for record in self.records:
            if any(tag in record.tags for tag in tags):
                results.append(record)
                
        if not results:
            raise ValueError(f"未找到包含标签 {tags} 的记录")
            
        return results

    def search_by_ids(self, ids: List[str]) -> List[MemoryRecord]:
        """根据ID列表搜索记录
        
        Args:
            ids: 记录ID列表
            
        Returns:
            List[MemoryRecord]: 匹配的记录列表
            
        Raises:
            ValueError: 当ID列表为空时抛出
        """
        if not ids:
            raise ValueError("ID列表不能为空")
            
        results = []
        for record in self.records:
            if record.id in ids:
                results.append(record)
                
        if not results:
            logger.warning(f"未找到ID在 {ids} 中的记录")
            
        return results

    def _filter_records_by_labels(self, label_list: List[str], records: Optional[List[MemoryRecord]] = None) -> Tuple[List[MemoryRecord], List[int]]:
        """根据标签列表过滤记录
        
        Args:
            label_list: 标签列表
            records: 要过滤的记录列表，如果为None则使用所有记录
            
        Returns:
            Tuple[List[MemoryRecord], List[int]]: 过滤后的记录列表和对应的原始索引
        """
        records_to_filter = records if records is not None else self.records
        filtered_records = []
        original_indices = []
        label_set = set(label_list)
        
        for idx, record in enumerate(records_to_filter):
            # 检查记录的标签是否与label_list有交集
            common_labels = set(record.tags) & label_set
            if common_labels:
                # 只保留交集中的action
                filtered_record = MemoryRecord(**record.dict())
                filtered_record.action = {
                    label: record.action[label] 
                    for label in common_labels 
                    if label in record.action
                }
                if filtered_record.action:  # 确保过滤后的action不为空
                    filtered_records.append(filtered_record)
                    original_indices.append(idx)
                    
        return filtered_records, original_indices

    def _get_top_k_results(self, 
                         similarities: np.ndarray,
                         records: List[MemoryRecord], 
                         k: int,
                         allow_duplicate_tags: bool = True) -> List[List[Tuple[MemoryRecord, float]]]:
        """基于相似度获取top-k个最相似的结果
        
        Args:
            similarities: 相似度矩阵，形状为(n_queries, n_records)
            records: 记录列表
            k: 每个查询返回的结果数量
            allow_duplicate_tags: 是否允许返回结果中有重复的标签
            
        Returns:
            List[List[Tuple[MemoryRecord, float]]]: 每个查询的匹配记录和相似度分数列表
        """
        # 对于每个查询构建结果
        results = []
        for query_idx, similarity_row in enumerate(similarities):
            query_results = []
            
            if allow_duplicate_tags:
                # 直接获取top-k结果
                k_actual = min(k, len(records))
                top_indices = np.argsort(similarity_row)[-k_actual:][::-1]
                
                for idx in top_indices:
                    query_results.append((records[idx], float(similarity_row[idx])))
            else:
                # 不允许重复标签时，需要特殊处理
                # 首先按相似度排序所有记录
                sorted_indices = np.argsort(similarity_row)[::-1]
                seen_tags = set()
                
                # 遍历所有记录，直到收集到k个不重复标签的记录或遍历完所有记录
                for idx in sorted_indices:
                    record = records[idx]
                    record_tags = set(record.tags)
                    
                    # 检查该记录的标签是否与已选标签有重叠
                    if not record_tags.intersection(seen_tags):
                        query_results.append((record, float(similarity_row[idx])))
                        seen_tags.update(record_tags)
                        
                        # 如果已经收集到k个结果，就停止
                        if len(query_results) >= k:
                            break
            
            results.append(query_results)
            
        return results

    def get_memories_by_source(self, data_source: DataSource) -> List[MemoryRecord]:
        """获取来自指定数据源的所有记忆记录
        
        Args:
            data_source: 数据源枚举值
            
        Returns:
            List[MemoryRecord]: 指定数据源的记录列表
        """
        return [record for record in self.records if record.source == data_source]
    
    def batch_semantic_search(self, 
                             queries: List[str], 
                             k: int = 5, 
                             label_list: Optional[List[str]] = None,
                             allow_duplicate_tags: bool = True,
                             sources: Optional[List[DataSource]] = None) -> List[List[Tuple[MemoryRecord, float]]]:
        """批量语义搜索最相似的记录
        
        Args:
            queries: 查询文本列表
            k: 每个查询返回的结果数量
            label_list: 标签列表
            allow_duplicate_tags: 是否允许返回结果中有重复的标签
            sources: 指定要搜索的数据来源列表
            
        Returns:
            List[List[Tuple[MemoryRecord, float]]]: 每个查询的匹配记录和相似度分数列表
            
        Raises:
            ValueError: 当参数无效或没有匹配记录时抛出
            RuntimeError: 当搜索过程出错时抛出
        """
        if not queries:
            raise ValueError("查询列表不能为空")
            
        if k <= 0:
            raise ValueError("k必须大于0")
            
        if not self.records:
            raise ValueError("没有可搜索的记录，请先调用 load_data() 加载数据")
            
        # 根据指定的数据来源过滤记录
        if sources:
            print(f"使用数据源{sources}")
            target_sources = set(sources)
            target_records = [record for record in self.records if record.source in target_sources]
            print(f"原始records长度: {len(self.records)}")
            print(f"使用的records长度: {len(target_records)}")
            
            # 检查记录的有效性
            valid_records = []
            invalid_records = []
            for record in target_records:
                if not record.state or not record.state.strip():
                    invalid_records.append(record)
                else:
                    valid_records.append(record)
            
            if invalid_records:
                print(f"警告：发现 {len(invalid_records)} 条无效记录（state为空）")
                print("无效记录示例：")
                for i, record in enumerate(invalid_records[:3]):
                    print(f"记录 {i+1}:")
                    print(f"  ID: {record.id}")
                    print(f"  State: '{record.state}'")
                    print(f"  Tags: {record.tags}")
                    
            target_records = valid_records
            print(f"有效records长度: {len(target_records)}")
            
            if not target_records:
                raise ValueError(f"在指定的数据源 {sources} 中没有找到任何有效记录")
        else:
            raise ValueError("未指定数据源")

        # 如果提供了label_list，先过滤内存
        if label_list:
            original_length = len(target_records)
            target_records, indices = self._filter_records_by_labels(label_list, target_records)
            print(f"标签过滤前记录数: {original_length}")
            print(f"标签过滤后记录数: {len(target_records)}")
            if not target_records:
                raise ValueError(f"使用标签 {label_list} 过滤后没有匹配的记录")
            
        # 获取状态文本
        states = []
        for i, record in enumerate(target_records):
            state = record.state.strip() if record.state else ""
            if state:
                states.append(state)
            else:
                print(f"警告：记录 {record.id} 的state为空")
                
        print(f"states长度: {len(states)}")
        if not states:
            raise ValueError("没有有效的状态文本用于编码")
            
        try:
            # 获取嵌入向量
            record_embeddings = self.embedding_model.encode(states)
            print(f"record_embeddings长度: {len(record_embeddings)}")
            if len(record_embeddings) == 0:
                raise ValueError("记录嵌入向量为空")
                
            query_embeddings = self.embedding_model.encode(queries)
            print(f"query_embeddings长度: {len(query_embeddings)}")
            
            # 计算相似度
            similarities = self.embedding_model.get_similarity(query_embeddings, record_embeddings)
            print(f"相似度矩阵形状: {similarities.shape}")
            
            # 获取最相似的结果
            results = self._get_top_k_results(similarities, target_records, k, allow_duplicate_tags)
            if not any(results):
                raise ValueError("未找到匹配的记录")
                
            # 释放GPU内存
            self.embedding_model.release_memory()
            
            return results
            
        except Exception as e:
            # 确保在发生异常时也释放GPU内存
            try:
                self.embedding_model.release_memory()
            except:
                pass
            raise RuntimeError(f"语义搜索时出错: {str(e)}")

    def delete_text(self, text_id: str) -> bool:
        """删除指定的记录
        
        Args:
            text_id: 记录ID
            
        Returns:
            bool: 是否成功删除
            
        Raises:
            ValueError: 当记录不存在时抛出
        """
        for i, record in enumerate(self.records):
            if record.id == text_id:
                del self.records[i]
                self.embeddings_cache = None
                
                # 如果启用了持久化，保存更改
                if self.storage and self.config.enable_persistence:
                    if not self.storage.save(self.records):
                        raise RuntimeError("保存更改失败")
                        
                logger.info(f"删除记忆，ID: {text_id}")
                return True
        raise ValueError(f"未找到要删除的记忆，ID: {text_id}")

    def get_all_texts(self) -> List[MemoryRecord]:
        """获取所有记录
        
        Returns:
            List[MemoryRecord]: 所有记录列表
        """
        return self.records
    
    def add_new_batch_memory(self, memory_list: List[Dict[str, Any]]) -> int:
        """批量添加新的记忆条目
        
        Args:
            memory_list: 包含state、action、tags和metadata的内存字典列表
            
        Returns:
            int: 成功添加的记录数量
            
        Raises:
            ValueError: 当参数无效时抛出
            RuntimeError: 当保存失败时抛出
        """
        if not memory_list:
            raise ValueError("记忆列表不能为空")
            
        def validate_memory(memory: Dict[str, Any]) -> bool:
            """验证记忆条目的格式"""
            if not isinstance(memory, dict):
                return False
            if "state" not in memory or "action" not in memory:
                return False
            if not isinstance(memory["action"], dict):
                return False
            return True
            
        # 验证所有记忆条目
        valid_memories = [m for m in memory_list if validate_memory(m)]
        if not valid_memories:
            raise ValueError("没有有效的记忆条目")
            
        try:
            # 添加id和timestamp
            current_time = datetime.now().isoformat()
            new_records = []
            
            for i, memory in enumerate(valid_memories):
                record = MemoryRecord(
                    id=str(len(self.records) + i + 1),
                    state=memory["state"],
                    action=memory["action"],
                    tags=memory["action"].keys(),
                    metadata=memory.get("metadata", {}),
                    timestamp=current_time,
                    source=DataSource.MEMORY
                )
                new_records.append(record)
                
            # 添加到内存
            self.records.extend(new_records)
            self.embeddings_cache = None
            
            # 如果启用了持久化，保存
            if self.storage and self.config.enable_persistence:
                if not self.storage.save(self.records):
                    raise RuntimeError("保存记录失败")
                
            logger.info(f"批量添加记忆，共 {len(new_records)} 条")
            return len(new_records)
            
        except Exception as e:
            raise RuntimeError(f"批量添加记录时出错: {str(e)}")
    
    def reinitialize(self, force_delete_storage: bool = False, from_sources: bool = True) -> bool:
        """重新初始化记忆系统
        
        Args:
            force_delete_storage: 是否强制删除存储文件
            from_sources: 是否从原始数据源重新初始化数据
            
        Returns:
            bool: 重新初始化是否成功
            
        Raises:
            RuntimeError: 当初始化失败时抛出
        """
        try:
            logger.info(f"开始重新初始化记忆系统: force_delete={force_delete_storage}, from_sources={from_sources}")
            
            # 清空现有内存和缓存
            self.records = []
            self.embeddings_cache = None
            
            # 如果需要强制删除存储文件且启用了持久化
            if force_delete_storage and self.storage and not self.storage.clear():
                raise RuntimeError("删除存储文件失败")
            
            # 从各数据源重新初始化
            if from_sources:
                self.load_data(force_reload=True)
            else:
                # 如果启用了持久化，保存空记录
                if self.storage and self.config.enable_persistence:
                    if not self.storage.save(self.records):
                        raise RuntimeError("保存空记录失败")
                logger.info("重新初始化完成，记忆系统为空")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"重新初始化失败: {str(e)}")
