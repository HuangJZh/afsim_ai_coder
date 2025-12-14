# utils.py
import os
import logging
import yaml
from typing import Tuple, Optional, List
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
import collections

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

class FileReader:
    """统一的文件读取工具类"""
    
    def __init__(self):
        self.config = ConfigManager()
    
    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def read_file_safely(file_path: str) -> Tuple[str, str]:
        """安全读取文件，自动检测编码"""
        encodings = ConfigManager().get('project.encodings', )
        max_size = ConfigManager().get('project.max_file_size_mb') * 1024 * 1024
        
        # 检查文件大小
        try:
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                raise ValueError(f"文件过大: {file_size} bytes")
        except OSError as e:
            raise ValueError(f"无法访问文件: {e}")
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                    content = f.read()
                return content, encoding
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logging.warning(f"使用编码 {encoding} 读取文件 {file_path} 失败: {e}")
                continue
        
        # 所有编码都失败，使用替换模式
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return content, 'utf-8 (with replacement)'
        except Exception as e:
            raise IOError(f"完全无法读取文件 {file_path}: {e}")
    
    @staticmethod
    def should_skip_file(file_path: str) -> bool:
        """判断是否应该跳过文件"""
        config = ConfigManager()
        filename = os.path.basename(file_path).lower()
        
        # 检查跳过模式
        skip_patterns = config.get('project.skip_patterns', [])
        if any(pattern in filename for pattern in skip_patterns):
            return True
        
        # 检查文件大小
        try:
            file_size = os.path.getsize(file_path)
            max_size = config.get('project.max_file_size_mb', 5) * 1024 * 1024
            if file_size > max_size:
                return True
        except OSError:
            return True
            
        return False

class InputStateManager:
    """输入状态管理器"""
    
    def __init__(self, max_buffer_size: int = 100):
        self._state_lock = threading.Lock()
        self._is_processing = False
        self._input_buffer = collections.deque(maxlen=max_buffer_size)
        self._processing_lock = threading.Lock()
    
    @property
    def is_processing(self) -> bool:
        """是否正在处理"""
        with self._state_lock:
            return self._is_processing
    
    @is_processing.setter
    def is_processing(self, value: bool):
        """设置处理状态"""
        with self._state_lock:
            self._is_processing = value
    
    def add_input(self, text: str) -> bool:
        """添加输入到缓冲区"""
        with self._state_lock:
            if len(self._input_buffer) >= self._input_buffer.maxlen:
                return False  # 缓冲区已满
            self._input_buffer.append(text)
            return True
    
    def get_all_inputs(self) -> List[str]:
        """获取所有输入并清空缓冲区"""
        with self._state_lock:
            inputs = list(self._input_buffer)
            self._input_buffer.clear()
            return inputs
    
    def clear(self):
        """清空缓冲区"""
        with self._state_lock:
            self._input_buffer.clear()
    
    def acquire_processing_lock(self) -> bool:
        """获取处理锁"""
        return self._processing_lock.acquire(blocking=False)
    
    def release_processing_lock(self):
        """释放处理锁"""
        if self._processing_lock.locked():
            self._processing_lock.release()

class CodeValidator:
    """代码验证器"""
    
    @staticmethod
    def validate_afsim_code(generated_code: str) -> dict:
        """验证生成的AFSIM代码"""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        code_lower = generated_code.lower()
        
        # 检查基本结构
        required_sections = ["platform", "component", "behavior"]
        found_sections = []
        
        for section in required_sections:
            if section in code_lower:
                found_sections.append(section)
        
        if not found_sections:
            validation_result["warnings"].append("代码可能缺少关键AFSIM组件定义")
        
        # 检查导入语句
        import_keywords = ["include", "import", "from", "using"]
        has_imports = any(keyword in code_lower for keyword in import_keywords)
        
        if not has_imports:
            validation_result["suggestions"].append("考虑添加必要的导入语句")
        
        # 检查语法问题
        if "{" in generated_code and "}" not in generated_code:
            validation_result["errors"].append("检测到不匹配的花括号")
            validation_result["is_valid"] = False
        
        if "(" in generated_code and ")" not in generated_code:
            validation_result["warnings"].append("检测到可能不匹配的括号")
        
        return validation_result

def setup_logging():
    """设置日志"""
    config = ConfigManager()
    log_level = getattr(logging, config.get('logging.level', 'INFO'))
    log_file = config.get('logging.file', 'afsim_coder.log')
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )