from fastapi import FastAPI, HTTPException, Depends, Query, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
from memory import TextMemory, DataSource, MemoryConfig
import logging
import uvicorn
import time
from datetime import datetime
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory_service")

# 初始化API应用
app = FastAPI(
    title="文本记忆系统 API",
    description="基于语义搜索的文本记忆系统，支持多数据源检索和管理",
    version="2.0.0"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务层单例
class MemoryService:
    """记忆服务的业务逻辑层"""
    
    def __init__(self, 
                 enable_persistence: bool = True,
                 auto_load_from_json: bool = False):
        """初始化记忆服务
        
        Args:
            enable_persistence: 是否启用持久化存储（保存到JSON）
            auto_load_from_json: 是否从JSON自动加载（False表示总是从原始文件加载）
        """
        config = MemoryConfig(
            enable_persistence=enable_persistence,
            auto_load_from_json=auto_load_from_json
        )
        self.memory = TextMemory(config)
        logger.info(f"记忆服务初始化完成 (enable_persistence={enable_persistence}, auto_load_from_json={auto_load_from_json})")
        logger.info("请调用 /system/load 接口加载数据")
    
    def load_data(self, sources: Optional[List[str]] = None, force_reload: bool = True):
        """手动加载数据
        
        Args:
            sources: 要加载的数据源列表
            force_reload: 是否强制从原始文件重新加载
        """
        if sources:
            data_sources = []
            for source_str in sources:
                source = DataSource.from_string(source_str)
                if source:
                    data_sources.append(source)
            self.memory.load_data(data_sources if data_sources else None, force_reload=force_reload)
        else:
            self.memory.load_data(force_reload=force_reload)
    
    def add_memory(self, state: str, action: Dict[str, str], 
                  tags: Optional[List[str]] = None, 
                  metadata: Optional[Dict] = None) -> str:
        """添加新的记忆条目"""
        return self.memory.add_text(state, action, tags, metadata)
    
    def get_memory(self, memory_id: str):
        """获取指定ID的记忆条目"""
        return self.memory.get_text(memory_id)
    
    def search_semantic(self, queries: List[str], k: int = 5, 
                       label_list: Optional[List[str]] = None,
                       allow_duplicate_tags: bool = True,
                       sources: Optional[List[DataSource]] = None):
        """执行语义搜索"""
        return self.memory.batch_semantic_search(
            queries, k, label_list, allow_duplicate_tags, sources
        )
    
    def search_by_tags(self, tags: List[str]):
        """根据标签搜索记忆条目"""
        return self.memory.search_by_tags(tags)
    
    def search_by_ids(self, ids: List[str]):
        """根据ID列表搜索记忆条目"""
        return self.memory.search_by_ids(ids)
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除指定ID的记忆条目"""
        return self.memory.delete_text(memory_id)
    
    def get_all_memories(self):
        """获取所有记忆条目"""
        return self.memory.get_all_texts()
    
    def get_memories_by_source(self, source: DataSource):
        """获取指定数据源的记忆条目"""
        return self.memory.get_memories_by_source(source)
    
    def add_batch_memories(self, memories: List[Dict[str, Any]]) -> int:
        """批量添加记忆条目"""
        return self.memory.add_new_batch_memory(memories)
    
    def reinitialize(self, force_delete_storage: bool = False, 
                    from_sources: bool = True) -> bool:
        """重新初始化记忆系统"""
        return self.memory.reinitialize(force_delete_storage, from_sources)
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        total_memories = len(self.memory.get_all_texts())
        
        # 统计各个数据源的记录数
        by_source = {}
        for source in DataSource:
            count = len(self.memory.get_memories_by_source(source))
            if count > 0:  # 只显示有数据的源
                by_source[source.value] = count
        
        return {
            "total_memories": total_memories,
            "by_source": by_source,
            "data_sources": DataSource.get_all_sources(),
            "official_sources": [s.value for s in DataSource.get_official_sources()],
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "enable_persistence": self.memory.config.enable_persistence,
                "auto_load_from_json": self.memory.config.auto_load_from_json
            }
        }

# 创建服务实例
# enable_persistence=True: 保存到JSON文件以便查看
# auto_load_from_json=False: 每次都从原始文件重新加载，不使用JSON缓存
memory_service = MemoryService(
    enable_persistence=True,      # 保存到JSON
    auto_load_from_json=False     # 不从JSON加载，总是重新加载
)

# 请求计时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 异常处理
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"请求处理出错: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"服务器错误: {str(exc)}"}
    )

# 数据模型
class TextInput(BaseModel):
    state: str
    action: Dict[str, str]
    tags: Optional[List[str]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        if not v:
            raise ValueError("action 不能为空")
        return v

class BatchMemoryInput(BaseModel):
    memories: List[Dict[str, Any]]
    
    @field_validator('memories')
    @classmethod
    def validate_memories(cls, v):
        if not v:
            raise ValueError("memories 列表不能为空")
        return v

class ReinitializeInput(BaseModel):
    force_delete_storage: Optional[bool] = False
    from_sources: Optional[bool] = True

class LoadDataInput(BaseModel):
    sources: Optional[List[str]] = None
    force_reload: Optional[bool] = True

class SearchQuery(BaseModel):
    queries: List[str]
    k: Optional[int] = 5
    label_list: Optional[List[str]] = None
    allow_duplicate_tags: Optional[bool] = True
    sources: Optional[List[str]] = None
    
    @field_validator('queries')
    @classmethod
    def validate_queries(cls, v):
        if not v:
            raise ValueError("queries 列表不能为空")
        return v
        
    @field_validator('k')
    @classmethod
    def validate_k(cls, v):
        if v <= 0:
            raise ValueError("k 必须大于0")
        return v
    
    def get_data_sources(self) -> Optional[List[DataSource]]:
        """将字符串数据源列表转换为DataSource枚举列表"""
        if not self.sources:
            return None
            
        data_sources = []
        for source_str in self.sources:
            source = DataSource.from_string(source_str)
            if source:
                data_sources.append(source)
        
        return data_sources if data_sources else None

class SearchByIdsInput(BaseModel):
    ids: List[str]
    
    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v):
        if not v:
            raise ValueError("ids 列表不能为空")
        return v

# API端点
@app.post("/system/load", status_code=status.HTTP_200_OK)
async def load_data(input_data: Optional[LoadDataInput] = None):
    """手动加载数据
    
    从原始文件（Excel/TSV）加载数据，并可选择性保存到JSON
    
    Args:
        sources: 要加载的数据源列表，可选值：
                - official_first_line
                - official_first_paragraph
                - official_full
                - procedures
                - tram
                如果不指定，则加载所有数据源
        force_reload: 是否强制从原始文件重新加载（默认为True）
    """
    try:
        if input_data:
            sources = input_data.sources
            force_reload = input_data.force_reload if input_data.force_reload is not None else True
        else:
            sources = None
            force_reload = True
            
        logger.info(f"手动加载数据，数据源: {sources}, force_reload: {force_reload}")
        memory_service.load_data(sources, force_reload)
        
        return {
            "message": "数据加载成功",
            "total_records": len(memory_service.get_all_memories()),
            "by_source": {
                source.value: len(memory_service.get_memories_by_source(source))
                for source in DataSource
                if len(memory_service.get_memories_by_source(source)) > 0
            }
        }
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/memories", response_model=str, status_code=status.HTTP_201_CREATED)
async def add_text(text_input: TextInput):
    """添加新的文本记忆
    
    接收状态文本、动作、标签和元数据，创建新的记忆条目
    """
    try:
        text_id = memory_service.add_memory(
            state=text_input.state,
            action=text_input.action,
            tags=text_input.tags,
            metadata=text_input.metadata
        )
        logger.info(f"成功创建记忆 ID: {text_id}")
        return text_id
    except Exception as e:
        logger.error(f"创建记忆失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"添加记忆失败: {str(e)}"
        )

@app.get("/memories/{memory_id}")
async def get_memory(memory_id: str):
    """获取指定ID的记忆条目"""
    memory_record = memory_service.get_memory(memory_id)
    if memory_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"记忆 ID '{memory_id}' 不存在"
        )
    return memory_record

@app.post("/memories/search", status_code=status.HTTP_200_OK)
async def semantic_search(query: SearchQuery):
    """批量语义搜索
    
    根据查询文本列表进行语义搜索，返回最相似的记忆条目
    
    支持的数据源：
    - official_first_line: 官方数据（第一行）
    - official_first_paragraph: 官方数据（第一段）
    - official_full: 官方数据（完整内容）
    - procedures: Procedures数据
    - tram: TRAM数据
    """
    try:
        logger.info(f"执行语义搜索: {len(query.queries)} 个查询, 搜索数据源: {query.sources}")
        
        results = memory_service.search_semantic(
            queries=query.queries, 
            k=query.k,
            label_list=query.label_list,
            allow_duplicate_tags=query.allow_duplicate_tags,
            sources=query.get_data_sources()
        )
        
        # 将结果转换为可序列化的格式
        formatted_results_tqdm = []
        start_time_tqdm = time.perf_counter()

        for single_query_result in tqdm(results, desc="处理查询结果"):
            query_formatted = []
            for record, score in single_query_result:
                record_copy = dict(record)
                query_formatted.append({"record": record_copy, "score": float(score)})
            formatted_results_tqdm.append(query_formatted)

        end_time_tqdm = time.perf_counter()
        duration_tqdm = end_time_tqdm - start_time_tqdm
        print(f"使用 tqdm 监控的总时间: {duration_tqdm:.4f} 秒")

        return formatted_results_tqdm
    
    except ValueError as e:
        logger.warning(f"语义搜索参数无效: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"语义搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/memories/tags/{tag}")
async def search_by_tags(tag: str):
    """根据标签搜索记忆条目"""
    try:
        results = memory_service.search_by_tags([tag])
        if not results:
            return []
        return results
    except Exception as e:
        logger.error(f"按标签搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/memories/search_by_id", status_code=status.HTTP_200_OK)
async def search_by_ids(input_data: SearchByIdsInput):
    """根据ID列表搜索记忆条目
    
    接收一个ID列表，返回所有匹配的记忆条目
    """
    try:
        logger.info(f"按ID搜索记忆: {input_data.ids}")
        results = memory_service.search_by_ids(input_data.ids)
        return results
    except ValueError as e:
        logger.warning(f"按ID搜索参数无效: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"按ID搜索失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: str):
    """删除指定的记忆条目"""
    success = memory_service.delete_memory(memory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"记忆 ID '{memory_id}' 不存在"
        )
    logger.info(f"成功删除记忆 ID: {memory_id}")
    return None

@app.get("/memories")
async def list_memories(source: Optional[str] = None):
    """获取指定数据源的记忆条目列表
    
    Args:
        source: 数据源名称，可选值：
                - official_first_line
                - official_first_paragraph
                - official_full
                - procedures
                - tram
                - memory
    """
    try:
        if source is None:
            return memory_service.get_all_memories()
        
        data_source = DataSource.from_string(source)
        if data_source:
            return memory_service.get_memories_by_source(data_source)
        else:
            valid_sources = DataSource.get_all_sources()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"无效的数据源: {source}. 有效值: {valid_sources}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取记忆列表失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/memories/batch", status_code=status.HTTP_201_CREATED)
async def add_batch_memories(input_data: BatchMemoryInput):
    """批量添加记忆条目"""
    try:
        result = memory_service.add_batch_memories(input_data.memories)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="批量添加失败，请检查数据格式"
            )
        logger.info(f"成功批量添加 {result} 条记忆")
        return {"count": result, "message": f"成功添加 {result} 条记忆"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量添加记忆失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/system/reinitialize", status_code=status.HTTP_200_OK)
async def reinitialize_memory(input_data: ReinitializeInput):
    """重新初始化内存系统
    
    可用于重置内存状态或重新加载数据
    """
    try:
        logger.warning(f"请求重新初始化内存系统: force_delete={input_data.force_delete_storage}, from_sources={input_data.from_sources}")
        success = memory_service.reinitialize(
            force_delete_storage=input_data.force_delete_storage,
            from_sources=input_data.from_sources
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="重新初始化失败"
            )
        logger.info("内存系统已成功重新初始化")
        return {"message": "内存系统已成功重新初始化"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重新初始化内存系统失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/system/info")
async def get_system_info():
    """获取系统信息
    
    返回系统配置、数据源统计等信息
    """
    try:
        return memory_service.get_system_info()
    except Exception as e:
        logger.error(f"获取系统信息失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        "memory_service:app",
        host="0.0.0.0",
        port=8009,
        reload=False
    )
