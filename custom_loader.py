import os
import chardet
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.directory import DirectoryLoader

class SmartTextLoader(TextLoader):
    """智能文本加载器，自动检测文件编码"""
    
    def __init__(self, file_path: str, **kwargs):
        super().__init__(file_path, **kwargs)
        self.file_path = file_path
    
    def load(self):
        """加载文档，自动检测编码"""
        try:
            # 首先尝试UTF-8
            return super().load()
        except UnicodeDecodeError:
            # 如果UTF-8失败，自动检测编码
            return self._load_with_encoding_detection()
    
    def _load_with_encoding_detection(self):
        """使用编码检测加载文件"""
        try:
            # 读取二进制文件检测编码
            with open(self.file_path, 'rb') as f:
                raw_data = f.read()
            
            # 检测编码
            encoding = chardet.detect(raw_data)['encoding']
            if encoding is None:
                encoding = 'utf-8'  # 默认使用UTF-8
            
            # 尝试使用检测到的编码读取
            try:
                text = raw_data.decode(encoding, errors='replace')
            except (UnicodeDecodeError, LookupError):
                # 如果检测的编码失败，尝试常见的中文编码
                for enc in ['utf-8', 'ascii', 'latin-1', 'iso-8859-1']:
                    try:
                        text = raw_data.decode(enc, errors='replace')
                        encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # 所有编码都失败，使用errors='replace'
                    text = raw_data.decode('utf-8', errors='replace')
            
            # 创建文档
            from langchain_core.documents import Document
            metadata = {"source": self.file_path, "encoding": encoding}
            return [Document(page_content=text, metadata=metadata)]
            
        except Exception as e:
            print(f"❌ 无法加载文件 {self.file_path}: {e}")
            # 返回空文档而不是抛出异常
            from langchain_core.documents import Document
            return [Document(page_content="", metadata={"source": self.file_path, "error": str(e)})]

class RobustDirectoryLoader(DirectoryLoader):
    """健壮的目录加载器，跳过无法读取的文件"""
    
    def load_file(self, file_path, path, docs, pbar):
        """加载单个文件，跳过错误文件"""
        try:
            if pbar:
                pbar.update(1)
            
            # 跳过非文本文件或损坏的文件
            if not self._is_readable_text_file(file_path):
                print(f"⚠️  跳过非文本文件: {file_path}")
                return
            
            sub_docs = self.loader_cls(str(file_path), **self.loader_kwargs).load()
            docs.extend(sub_docs)
            
        except Exception as e:
            print(f"❌ 跳过无法加载的文件: {file_path} - 错误: {e}")
    
    def _is_readable_text_file(self, file_path):
        """检查文件是否为可读的文本文件"""
        try:
            # 检查文件大小（避免读取超大文件）
            file_size = os.path.getsize(file_path)
            if file_size > 50 * 1024 * 1024:  # 50MB
                print(f"⚠️  跳过过大文件: {file_path} ({file_size} bytes)")
                return False
            
            # 检查文件扩展名
            valid_extensions = {'.py', '.js', '.java', '.c', '.cpp', '.h', '.html', 
                                '.css', '.php', '.rb', '.go', '.rs', '.ts', '.json',
                                '.xml', '.yml', '.yaml', '.md', '.txt', '.csv'}
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in valid_extensions:
                return False
            
            return True
            
        except Exception:
            return False

def get_file_encoding(file_path):
    """获取文件编码"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(4096)  # 只读取前4KB来检测编码
        return chardet.detect(raw_data)['encoding'] or 'utf-8'
    except Exception:
        return 'utf-8'