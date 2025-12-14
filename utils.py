# utils.py
import os
import logging
import yaml
from typing import Tuple, Optional, List, Any, Dict
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
import collections
import json
import time
from datetime import datetime
import hashlib

# 配置日志
def setup_logging(config_manager=None):
    """设置日志配置"""
    if config_manager:
        log_config = config_manager.get('logging')
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', '[%(asctime)s] %(levelname)s | %(message)s')
        
        # 创建日志目录
        logs_dir = config_manager.get('paths.logs_dir', './logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        log_file = os.path.join(logs_dir, config_manager.get('logging.file', 'afsim_rag.log'))
        
        # 配置根日志记录器
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
    else:
        # 默认配置
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s | %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    return logging.getLogger(__name__)

class ConfigManager:
    """配置管理器（单例模式）"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._load_config()
            return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        config_paths = [
            "config.yaml",
            os.path.join(os.path.dirname(__file__), "config.yaml"),
            os.path.expanduser("~/.afsim_rag/config.yaml"),
            "/etc/afsim_rag/config.yaml"
        ]
        
        loaded = False
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                    self.config_path = path
                    loaded = True
                    print(f"✓ 加载配置文件: {path}")
                    break
                except Exception as e:
                    print(f"❌ 加载配置文件失败 {path}: {e}")
        
        if not loaded:
            # 使用默认配置
            self.config = self._get_default_config()
            self.config_path = None
            print("⚠ 使用默认配置")
        
        # 设置环境变量
        self._setup_environment()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'system': {
                'name': 'AFSIM RAG 代码生成系统',
                'version': '1.0.0',
                'debug': True,
                'log_level': 'INFO'
            },
            'model': {
                'path': 'D:/Qwen/Qwen/Qwen3-4B',
                'embedding_model': 'BAAI/bge-small-zh-v1.5',
                'max_tokens': 1024,
                'temperature': 0.3,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1
            },
            'rag': {
                'chroma_db_path': './chroma_db',
                'collection_name': 'afsim_tutorials',
                'docs_folder': 'tutorials',
                'chunk_size': 350,
                'chunk_overlap': 100,
                'search_top_k': 5
            },
            'web': {
                'server_port': 7860,
                'server_name': '0.0.0.0'
            }
        }
    
    def _setup_environment(self):
        """设置环境变量"""
        # 设置日志级别
        log_level = self.get('system.log_level', 'INFO')
        os.environ['LOG_LEVEL'] = log_level
        
        # 设置调试模式
        debug = self.get('system.debug', False)
        if debug:
            os.environ['DEBUG'] = '1'
        
        # 设置HuggingFace缓存目录
        cache_dir = self.get('paths.cache_dir', './cache')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = cache_dir
    
    def get(self, key: str, default=None) -> Any:
        """获取配置值（支持点表示法）"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, {})
                else:
                    return default
            
            return value if value != {} else default
        except (AttributeError, KeyError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
        except Exception as e:
            print(f"设置配置失败: {e}")
            return False
    
    def save(self, path: str = None) -> bool:
        """保存配置到文件"""
        if path is None:
            if self.config_path:
                path = self.config_path
            else:
                path = "config.yaml"
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"✓ 配置已保存到: {path}")
            return True
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
            return False
    
    def reload(self) -> bool:
        """重新加载配置"""
        with self._lock:
            try:
                self._load_config()
                return True
            except Exception as e:
                print(f"❌ 重新加载配置失败: {e}")
                return False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []
        
        # 检查必要的配置项
        required_paths = [
            ('model.path', '模型路径'),
            ('rag.docs_folder', '教程文件夹'),
            ('rag.chroma_db_path', '向量数据库路径')
        ]
        
        for key, desc in required_paths:
            value = self.get(key)
            if not value:
                errors.append(f"缺少配置项: {desc} ({key})")
            elif key.endswith('_path') or key.endswith('_folder'):
                if not os.path.exists(value):
                    errors.append(f"路径不存在: {desc} ({value})")
        
        # 检查数值范围
        numeric_checks = [
            ('model.temperature', 0.0, 2.0),
            ('model.top_p', 0.0, 1.0),
            ('rag.chunk_size', 100, 2000),
            ('rag.search_top_k', 1, 20)
        ]
        
        for key, min_val, max_val in numeric_checks:
            value = self.get(key)
            if value is not None:
                if not (min_val <= value <= max_val):
                    errors.append(f"配置 {key} 值 {value} 超出范围 [{min_val}, {max_val}]")
        
        return len(errors) == 0, errors
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("配置摘要")
        print("="*60)
        
        sections = {
            '系统': ['system.name', 'system.version', 'system.debug'],
            '模型': ['model.path', 'model.embedding_model', 'model.max_tokens'],
            'RAG': ['rag.docs_folder', 'rag.chroma_db_path', 'rag.search_top_k'],
            'Web': ['web.server_port', 'web.server_name']
        }
        
        for section, keys in sections.items():
            print(f"\n{section}:")
            print("-"*40)
            for key in keys:
                value = self.get(key)
                print(f"  {key}: {value}")
        
        print("="*60)

@lru_cache(maxsize=128)
def get_config() -> ConfigManager:
    """获取配置管理器实例"""
    return ConfigManager()

class PerformanceTimer:
    """性能计时器"""
    def __init__(self, name: str = None):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.name:
            elapsed = self.end_time - self.start_time
            print(f"⏱️ {self.name} 耗时: {elapsed:.2f}秒")
    
    def elapsed(self) -> float:
        """获取耗时（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

class CacheManager:
    """缓存管理器"""
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        self.cache_dir = cache_dir
        self.ttl = ttl
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        # 创建key的哈希
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查是否过期
            if time.time() - cache_data['timestamp'] > self.ttl:
                os.remove(cache_path)
                return None
            
            return cache_data['data']
        except Exception:
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """设置缓存值"""
        cache_path = self._get_cache_path(key)
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'data': value
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"缓存设置失败: {e}")
            return False
    
    def clear(self, older_than: int = None) -> int:
        """清理缓存"""
        count = 0
        now = time.time()
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.cache_dir, filename)
                
                try:
                    if older_than:
                        mtime = os.path.getmtime(filepath)
                        if now - mtime > older_than:
                            os.remove(filepath)
                            count += 1
                    else:
                        os.remove(filepath)
                        count += 1
                except Exception:
                    continue
        
        return count

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def retryable_function(func, *args, **kwargs):
    """可重试的函数装饰器"""
    return func(*args, **kwargs)

class ThreadSafeDict:
    """线程安全的字典"""
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()
    
    def __getitem__(self, key):
        with self._lock:
            return self._data[key]
    
    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value
    
    def __delitem__(self, key):
        with self._lock:
            del self._data[key]
    
    def __contains__(self, key):
        with self._lock:
            return key in self._data
    
    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)
    
    def keys(self):
        with self._lock:
            return list(self._data.keys())
    
    def values(self):
        with self._lock:
            return list(self._data.values())
    
    def items(self):
        with self._lock:
            return list(self._data.items())
    
    def clear(self):
        with self._lock:
            self._data.clear()

def ensure_directories(config_manager: ConfigManager):
    """确保所有必要的目录都存在"""
    directories = [
        config_manager.get('paths.tutorials_dir', 'tutorials'),
        config_manager.get('paths.chroma_db_dir', './chroma_db'),
        config_manager.get('paths.logs_dir', './logs'),
        config_manager.get('paths.exports_dir', './exports'),
        config_manager.get('paths.cache_dir', './cache')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 确保目录存在: {directory}")
    
    return True