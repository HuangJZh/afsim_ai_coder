import os
import torch
from typing import List, Dict, Any
import numpy as np
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import ConfigManager

class AFSIMRAGSystem:
    def __init__(self, 
                 model_path: str = None,
                 embedding_model: str = None,
                 chroma_db_path: str = None):
        """
        初始化AFSIM RAG系统
        """
        print("正在初始化AFSIM RAG系统...")
        
        # 初始化配置管理器
        self.config = ConfigManager()
        
        # 使用配置值或参数值
        self.model_path = model_path or self.config.get('model.path')
        self.embedding_model_name = embedding_model or self.config.get('embedding.model_name')
        self.chroma_db_path = chroma_db_path or self.config.get('database.chroma_path')
        
        # 初始化组件
        self._init_embedding_model()
        self._init_vector_db()
        self._init_llm()
        
        print("系统初始化完成！")
        
    def _init_embedding_model(self):
        """初始化嵌入模型"""
        print(f"加载嵌入模型: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # 从配置获取嵌入参数
        self.normalize_embeddings = self.config.get('embedding.normalize_embeddings', True)
        self.embedding_batch_size = self.config.get('embedding.batch_size', 32)
        
        print(f"嵌入维度: {self.embedding_dim}")
        
    def _init_vector_db(self):
        """初始化向量数据库"""
        print(f"初始化Chroma数据库: {self.chroma_db_path}")
        
        # 创建数据库目录如果不存在
        os.makedirs(self.chroma_db_path, exist_ok=True)
        
        db_settings = self.config.get('database.settings', {})
        self.client = PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=db_settings.get('anonymized_telemetry', False)
            )
        )
        
        # 创建或获取集合
        collection_name = self.config.get('vector_db.collection_name', 'afsim_tutorials')
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "AFSIM教程文档向量存储"}
        )
        
        print(f"数据库文档数量: {self.collection.count()}")
        
    def _init_llm(self):
        """初始化Qwen3-4B模型"""
        print(f"加载Qwen3-4B模型: {self.model_path}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 从配置获取加载参数
            torch_dtype_str = self.config.get('system.torch_dtype', 'float16')
            load_in_4bit = self.config.get('system.load_in_4bit', True)
            device_map = self.config.get('system.device_map', 'auto')
            
            torch_dtype = getattr(torch, torch_dtype_str) if hasattr(torch, torch_dtype_str) else torch.float16
            
            # 尝试使用量化加载
            if load_in_4bit:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                        load_in_4bit=True
                    )
                    print("✓ 使用4-bit量化加载模型")
                except Exception as e:
                    print(f"⚠ 量化加载失败: {e}，尝试全精度加载")
                    load_in_4bit = False
            
            if not load_in_4bit:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device_map
                )
                print("✓ 使用全精度加载模型")
            
            # 从配置获取生成参数
            generation_config = self.config.get('model.generation', {})
            self.generation_config = {
                "max_new_tokens": generation_config.get('max_new_tokens', 512),
                "temperature": generation_config.get('temperature', 0.3),
                "top_p": generation_config.get('top_p', 0.9),
                "do_sample": generation_config.get('do_sample', True),
                "repetition_penalty": generation_config.get('repetition_penalty', 1.1),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            print("✓ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def load_documents_from_folder(self, folder_path=None):
        """
        从文件夹加载所有.md文件到向量数据库
        """
        if folder_path is None:
            folder_path = self.config.get('paths.tutorials_folder', 'tutorials')
        
        print(f"开始扫描文件夹: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"❌ 文件夹不存在: {folder_path}")
            return False
        
        if not os.path.isdir(folder_path):
            print(f"❌ 路径不是文件夹: {folder_path}")
            return False
        
        try:
            # 从配置获取支持的文件扩展名
            supported_extensions = self.config.get('document.supported_extensions', ['.md'])
            
            # 扫描所有支持的文件
            supported_files = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if any(file.endswith(ext) for ext in supported_extensions):
                        full_path = os.path.join(root, file)
                        supported_files.append(full_path)
            
            print(f"找到 {len(supported_files)} 个支持的文件")
            
            if not supported_files:
                print("⚠ 未找到任何支持的文件")
                return False
            
            documents = []
            metadatas = []
            ids = []
            
            # 从配置获取分块参数
            chunk_size = self.config.get('document.default_chunk_size', 400)
            max_file_size = self.config.get('document.max_file_size_mb', 10) * 1024 * 1024
            
            # 读取每个文件
            for file_path in supported_files:
                try:
                    # 检查文件大小
                    file_size = os.path.getsize(file_path)
                    if file_size > max_file_size:
                        print(f"⚠ 文件过大跳过: {os.path.basename(file_path)} ({file_size/1024/1024:.1f}MB)")
                        continue
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc_content = f.read()
                    
                    if not doc_content.strip():
                        print(f"⚠ 文件内容为空: {os.path.basename(file_path)}")
                        continue
                    
                    # 分割文档
                    paragraphs = self._split_into_chunks(doc_content, chunk_size=chunk_size)
                    
                    for i, para in enumerate(paragraphs):
                        if para.strip():  # 跳过空段落
                            doc_id = f"{os.path.basename(file_path)}_{i}"
                            documents.append(para)
                            metadatas.append({
                                "source": file_path,
                                "paragraph": i,
                                "filename": os.path.basename(file_path),
                                "filepath": file_path
                            })
                            ids.append(doc_id)
                    
                    print(f"✓ 已加载: {os.path.basename(file_path)} ({len(paragraphs)} 段落)")
                    
                except Exception as e:
                    print(f"❌ 读取文件失败 {file_path}: {e}")
            
            # 批量嵌入并存储
            if documents:
                print(f"正在生成 {len(documents)} 个文档块的向量...")
                embeddings = self.embedding_model.encode(
                    documents,
                    show_progress_bar=True,
                    normalize_embeddings=self.normalize_embeddings,
                    batch_size=self.embedding_batch_size,
                    convert_to_numpy=True
                )
                
                print("正在存储到向量数据库...")
                
                # 分批存储，避免内存问题
                batch_size = self.config.get('vector_db.chunk_size', 100)
                for i in range(0, len(documents), batch_size):
                    end_idx = min(i + batch_size, len(documents))
                    
                    self.collection.add(
                        embeddings=embeddings[i:end_idx].tolist(),
                        documents=documents[i:end_idx],
                        metadatas=metadatas[i:end_idx],
                        ids=ids[i:end_idx]
                    )
                    
                    print(f"  已存储 {end_idx}/{len(documents)} 个文档块")
                
                print(f"✅ 成功加载 {len(documents)} 个文档块")
                return True
            else:
                print("⚠ 未找到任何文档内容")
                return False
                
        except Exception as e:
            print(f"❌ 加载文档失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_documents_from_list(self, file_list_path: str, base_dir: str = "."):
        """
        从文件列表加载文档（备用方法）
        """
        print(f"从文件列表加载文档: {file_list_path}")
        
        if not os.path.exists(file_list_path):
            print(f"❌ 文件不存在: {file_list_path}")
            return False
        
        try:
            with open(file_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            documents = []
            metadatas = []
            ids = []
            
            for line in lines:
                line = line.strip()
                if line.endswith('.md'):
                    # 清理路径
                    file_path = line.replace('D:.\\', '').replace('D:.', '').strip()
                    file_path = file_path.replace('\\', '/')
                    
                    # 添加基础目录
                    if not os.path.isabs(file_path):
                        file_path = os.path.join(base_dir, file_path)
                    
                    if os.path.exists(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                doc_content = f.read()
                            
                            paragraphs = self._split_into_chunks(doc_content)
                            
                            for i, para in enumerate(paragraphs):
                                if para.strip():
                                    doc_id = f"{os.path.basename(file_path)}_{i}"
                                    documents.append(para)
                                    metadatas.append({
                                        "source": file_path,
                                        "paragraph": i,
                                        "filename": os.path.basename(file_path)
                                    })
                                    ids.append(doc_id)
                            
                            print(f"✓ 已加载: {os.path.basename(file_path)} ({len(paragraphs)} 段落)")
                            
                        except Exception as e:
                            print(f"❌ 读取文件失败 {file_path}: {e}")
                    else:
                        print(f"⚠ 文件不存在: {file_path}")
            
            if documents:
                print(f"正在生成 {len(documents)} 个文档块的向量...")
                embeddings = self.embedding_model.encode(
                    documents,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )
                
                print("正在存储到向量数据库...")
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                print(f"✅ 成功加载 {len(documents)} 个文档块")
                return True
            else:
                print("⚠ 未找到任何文档内容")
                return False
                
        except Exception as e:
            print(f"❌ 加载文档失败: {e}")
            return False
    
    def _split_into_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """将文本分割成块"""
        if chunk_size is None:
            chunk_size = self.config.get('document.default_chunk_size', 400)
            
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def retrieve_relevant_docs(self, query: str, n_results: int = 3) -> List[Dict]:
        """检索相关文档"""
        if self.collection.count() == 0:
            print("⚠ 向量数据库为空，请先加载文档")
            return []
        
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(
                query,
                normalize_embeddings=True
            ).tolist()
            
            # 检索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # 格式化结果
            retrieved_docs = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved_docs.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            return []
    
    def format_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """格式化提示词"""
        if not retrieved_docs:
            return f"""你是一个AFSIM（Advanced Framework for Simulation）专家助手。
请回答以下问题：

问题：{query}

回答："""
        
        # 构建上下文
        context = "以下是相关的AFSIM教程内容：\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"【文档{i}】{doc['metadata']['filename']}\n"
            context += f"{doc['content'][:800]}\n\n"
        
        # 完整提示
        prompt = f"""你是一个AFSIM专家助手。
请基于提供的教程内容直接生成AFSIM代码

问题：{query}

{context}
AFSIM代码："""
        
        return prompt
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """生成回答"""
        print(f"\n处理查询: {query[:50]}...")
        
        # 检索相关文档
        retrieved_docs = self.retrieve_relevant_docs(query)
        
        if not retrieved_docs:
            print("⚠ 未找到相关文档，将基于模型知识回答")
        
        # 构建提示
        prompt = self.format_prompt(query, retrieved_docs)
        
        try:
            # 生成回答
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的回答（去掉提示部分）
            if prompt in response:
                response = response[len(prompt):].strip()
            
            # 清理响应
            response = self._clean_response(response)
            
            # 提取来源信息
            sources = list(set([doc['metadata']['filename'] for doc in retrieved_docs]))
            
            print(f"✓ 回答生成完成，长度: {len(response)} 字符")
            
            return {
                "response": response,
                "sources": sources,
                "raw_docs": retrieved_docs
            }
            
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response": f"生成回答时出错: {str(e)}",
                "sources": [],
                "raw_docs": []
            }
    
    def _clean_response(self, text: str) -> str:
        """清理响应文本"""
        # 移除多余的空行
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # 限制最大长度
        cleaned_text = '\n'.join(cleaned_lines)
        if len(cleaned_text) > 2000:
            cleaned_text = cleaned_text[:2000] + "...\n\n(回答过长，已截断)"
        
        return cleaned_text
    
    def interactive_chat(self):
        """交互式聊天"""
        print("\n" + "="*60)
        print("AFSIM RAG 系统 - 交互模式")
        print("="*60)
        print("命令:")
        print("  'exit' 或 'quit' - 退出")
        print("  'clear' - 清空上下文")
        print("  'sources' - 显示当前来源")
        print("  'reload' - 重新加载文档")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 用户: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("再见！")
                    break
                elif user_input.lower() == 'clear':
                    print("上下文已清空")
                    continue
                elif user_input.lower() == 'sources':
                    print(f"数据库中有 {self.collection.count()} 个文档块")
                    continue
                elif user_input.lower() == 'reload':
                    print("重新加载文档...")
                    self.load_documents_from_folder("tutorials")
                    continue
                elif not user_input:
                    continue
                
                # 生成回答
                result = self.generate_response(user_input)
                
                print(f"\n🤖 AFSIM助手:")
                print("-"*40)
                print(result["response"])
                print("-"*40)
                if result["sources"]:
                    print("参考来源:")
                    for source in result["sources"]:
                        print(f"  • {source}")
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n程序已中断")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()