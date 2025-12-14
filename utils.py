import os
import logging
import yaml
from typing import Tuple, Optional, List
import threading

class ConfigManager:
    """配置管理器"""
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
            "/etc/afsim_coder/config.yaml"
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                logging.info(f"加载配置文件: {path}")
                return
        
        # 默认配置
        self.config = {
            'model': {
                'path': "D:/Qwen/Qwen/Qwen3-4B",
                'max_tokens': 4096,
                'temperature': 0.2
            },
            'vector_db': {
                'chunk_size': 1500,
                'chunk_overlap': 250,
                'persist_dir': "vector_db_afsim_enhanced"
            }
        }
        logging.warning("使用默认配置")
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value.get(k, {})
            return value if value != {} else default
        except AttributeError:
            return default


