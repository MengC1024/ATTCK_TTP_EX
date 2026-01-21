#!/usr/bin/env python3
"""
通过API加载数据到Memory系统的脚本

用法:
    python load_memory.py <file_path> [--no-clear] [--host <host>] [--port <port>]

示例:
    # 清除现有memory并加载新数据（默认）
    python load_memory.py data.json
    
    # 不清除现有memory，追加加载
    python load_memory.py data.json --no-clear
    
    # 指定API服务地址
    python load_memory.py data.json --host localhost --port 8009

文件格式:
    JSON文件，包含记忆列表，每条记忆包含以下字段：
    - state: str (必需) - 状态描述文本
    - action: dict (必需) - 动作字典，key为标签，value为描述
    - tags: list (可选) - 标签列表（如果不提供，会从action的keys中提取）
    - metadata: dict (可选) - 额外的元数据

示例JSON格式:
    [
        {
            "state": "攻击者尝试获取系统权限",
            "action": {
                "T1078": "使用有效账户进行访问",
                "T1110": "尝试暴力破解密码"
            },
            "tags": ["T1078", "T1110"],
            "metadata": {
                "severity": "high",
                "category": "credential_access"
            }
        }
    ]
"""

import argparse
import json
import logging
import os
import sys
import requests
from typing import List, Dict, Any
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LoadMemory")


class MemoryAPIClient:
    """Memory API客户端"""
    
    def __init__(self, host: str = "localhost", port: int = 8009):
        """初始化API客户端
        
        Args:
            host: API服务主机地址
            port: API服务端口
        """
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        logger.info(f"API客户端初始化完成: {self.base_url}")
    
    def check_health(self) -> bool:
        """检查API服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            response = self.session.get(f"{self.base_url}/system/info", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"连接API服务失败: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息
        
        Returns:
            Dict[str, Any]: 系统信息
            
        Raises:
            requests.RequestException: 请求失败
        """
        response = self.session.get(f"{self.base_url}/system/info")
        response.raise_for_status()
        return response.json()
    
    def reinitialize(self, force_delete_storage: bool = True, from_sources: bool = False) -> Dict[str, Any]:
        """重新初始化memory系统（清除数据）
        
        Args:
            force_delete_storage: 是否强制删除存储文件
            from_sources: 是否从数据源重新加载
            
        Returns:
            Dict[str, Any]: 响应数据
            
        Raises:
            requests.RequestException: 请求失败
        """
        data = {
            "force_delete_storage": force_delete_storage,
            "from_sources": from_sources
        }
        response = self.session.post(f"{self.base_url}/system/reinitialize", json=data)
        response.raise_for_status()
        return response.json()
    
    def batch_add_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量添加记忆
        
        Args:
            memories: 记忆列表
            
        Returns:
            Dict[str, Any]: 响应数据，包含成功添加的数量
            
        Raises:
            requests.RequestException: 请求失败
        """
        data = {"memories": memories}
        response = self.session.post(f"{self.base_url}/memories/batch", json=data)
        response.raise_for_status()
        return response.json()


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """从JSON文件加载数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        List[Dict[str, Any]]: 记忆数据列表
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
        ValueError: 数据格式不正确
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    logger.info(f"正在读取文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证数据格式
    if not isinstance(data, list):
        raise ValueError("JSON文件必须包含一个列表")
    
    if not data:
        raise ValueError("JSON文件不能为空列表")
    
    # 验证每条记录的格式
    valid_memories = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"跳过第 {i+1} 条记录：不是字典类型")
            continue
            
        if "state" not in item:
            logger.warning(f"跳过第 {i+1} 条记录：缺少 'state' 字段")
            continue
            
        if "action" not in item:
            logger.warning(f"跳过第 {i+1} 条记录：缺少 'action' 字段")
            continue
            
        if not isinstance(item["action"], dict):
            logger.warning(f"跳过第 {i+1} 条记录：'action' 必须是字典类型")
            continue
        
        if not item["action"]:
            logger.warning(f"跳过第 {i+1} 条记录：'action' 不能为空")
            continue
        
        # 如果没有提供tags，从action的keys中提取
        if "tags" not in item or not item["tags"]:
            item["tags"] = list(item["action"].keys())
        
        # 确保metadata字段存在
        if "metadata" not in item:
            item["metadata"] = {}
            
        valid_memories.append(item)
    
    if not valid_memories:
        raise ValueError("没有有效的记忆数据")
    
    logger.info(f"成功加载 {len(valid_memories)} 条有效记忆（总共 {len(data)} 条）")
    return valid_memories


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="通过API将数据加载到Memory系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "file_path",
        help="要加载的JSON文件路径"
    )
    
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="不清除现有memory，追加加载（默认会清除）"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="API服务主机地址（默认: localhost）"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8009,
        help="API服务端口（默认: 8009）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟运行，不实际写入数据"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建API客户端
        api_client = MemoryAPIClient(host=args.host, port=args.port)
        
        # 检查API服务是否可用
        logger.info("检查API服务连接...")
        if not api_client.check_health():
            raise RuntimeError(
                f"无法连接到API服务 {api_client.base_url}\n"
                f"请确保memory_service已启动: python memory_service.py"
            )
        logger.info("✓ API服务连接成功")
        
        # 获取并显示当前系统信息
        try:
            sys_info = api_client.get_system_info()
            logger.info(f"当前系统记录数: {sys_info.get('total_memories', 0)}")
            by_source = sys_info.get('by_source', {})
            if by_source:
                logger.info(f"各数据源记录数: {by_source}")
        except Exception as e:
            logger.warning(f"获取系统信息失败: {str(e)}")
        
        # 加载数据文件
        memories = load_json_file(args.file_path)
        
        if args.dry_run:
            logger.info("=== 模拟运行模式 ===")
            logger.info(f"将要添加 {len(memories)} 条记忆")
            logger.info("\n前3条记忆预览:")
            for i, mem in enumerate(memories[:3], 1):
                logger.info(f"\n记忆 {i}:")
                logger.info(f"  State: {mem['state'][:100]}...")
                logger.info(f"  Tags: {mem['tags']}")
                logger.info(f"  Action keys: {list(mem['action'].keys())}")
            logger.info("\n=== 模拟运行结束（未实际写入） ===")
            return
        
        # 是否清除现有memory
        should_clear = not args.no_clear
        
        if should_clear:
            logger.info("正在清除现有memory...")
            result = api_client.reinitialize(
                force_delete_storage=True,
                from_sources=False
            )
            logger.info(f"✓ {result.get('message', '清除完成')}")
        else:
            logger.info("保留现有memory，将追加新数据")
        
        # 批量添加记忆
        logger.info(f"正在添加 {len(memories)} 条记忆到系统...")
        result = api_client.batch_add_memories(memories)
        
        count = result.get('count', 0)
        logger.info(f"✓ 成功添加 {count} 条记忆")
        
        # 获取更新后的系统信息
        try:
            sys_info = api_client.get_system_info()
            total = sys_info.get('total_memories', 0)
            memory_count = sys_info.get('by_source', {}).get('memory', 0)
            logger.info(f"当前系统总记录数: {total}")
            logger.info(f"MEMORY数据源记录数: {memory_count}")
        except Exception as e:
            logger.warning(f"获取更新后的系统信息失败: {str(e)}")
        
    except FileNotFoundError as e:
        logger.error(f"文件错误: {str(e)}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"JSON格式错误: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"数据格式错误: {str(e)}")
        sys.exit(1)
    except requests.RequestException as e:
        logger.error(f"API请求失败: {str(e)}")
        if hasattr(e.response, 'text'):
            logger.error(f"错误详情: {e.response.text}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
