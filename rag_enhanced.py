import os
import warnings
import threading
import time
import logging
from typing import List, Dict, Any
from functools import lru_cache

import torch
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

from project_learner import AFSIMProjectLearner
from utils import ConfigManager, FileReader, InputStateManager, CodeValidator, setup_logging

class EnhancedRAGChatSystem:
    def __init__(self, 
                 project_root: str,
                 model_path: str = None,
                 documents_path: str = None,
                 embedding_model: str = None,
                 vector_db_dir: str = None):
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config = ConfigManager()
        self.project_root = project_root
        self.documents_path = documents_path or project_root
        self.vector_db_dir = vector_db_dir or self.config.get('vector_db.persist_dir')
        self.model_path = model_path or self.config.get('model.path')
        self.embedding_model = embedding_model or self.config.get('embedding.model')
        
        # 初始化状态管理器
        self.input_state = InputStateManager(max_buffer_size=50)
        
        self.logger.info("正在初始化AFSIM项目学习器...")
        self.project_learner = AFSIMProjectLearner(project_root)
        self.project_learner.analyze_project_structure()
        
        self.logger.info("正在加载嵌入模型...")
        self.embeddings = self._setup_embeddings()
        
        self.logger.info("正在加载大语言模型...")
        self.tokenizer, self.model = self._setup_llm()
        
        # 构建或加载向量数据库
        self.vector_db = self.build_or_load_vector_db()
        self.enhanced_qa_chain = self.create_enhanced_qa_chain()
        
        self.conversation_history = []
        
        self.logger.info("EnhancedRAGChatSystem 初始化完成")
    
    def _setup_embeddings(self):
        """设置嵌入模型"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {device}")
        
        try:
            return HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True},
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
        except Exception as e:
            self.logger.error(f"加载嵌入模型失败，回退到CPU: {e}")
            return HuggingFaceBgeEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
    
    def _setup_llm(self):
        """设置语言模型"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                padding_side='left'
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 根据可用设备选择加载方式
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    dtype=torch.float16,
                    device_map="cuda",
                    trust_remote_code=True,
                ).eval()
            
            return tokenizer, model
                
        except Exception as e:
            self.logger.error(f"加载语言模型失败: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def search_similar_documents_cached(self, query: str, top_k: int = None):
        """带缓存的文档搜索"""
        if top_k is None:
            top_k = self.config.get('vector_db.search_k', 6)
        return self.vector_db.similarity_search(query, k=top_k)
    
    def build_or_load_vector_db(self):
        """构建或加载向量数据库"""
        if os.path.exists(self.vector_db_dir):
            self.logger.info("加载已有向量数据库...")
            try:
                vector_db = Chroma(
                    persist_directory=self.vector_db_dir,
                    embedding_function=self.embeddings
                )
                # 验证数据库是否有效
                if hasattr(vector_db, '_collection') and vector_db._collection.count() > 0:
                    self.logger.info("向量数据库加载成功")
                    return vector_db
                else:
                    self.logger.warning("向量数据库为空，将重新构建")
            except Exception as e:
                self.logger.error(f"加载向量数据库失败，将重新构建: {e}")
        
        self.logger.info("构建新的向量数据库...")
        return self.build_knowledge_base()
    
    def build_knowledge_base(self):
        """构建知识库 - 修复批量大小问题"""
        self.logger.info("开始处理文档构建知识库...")
        start_time = time.time()

        try:
            # 收集所有可读的文件
            all_txt_files = []
            for root, dirs, files in os.walk(self.documents_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        if not FileReader.should_skip_file(file_path):
                            all_txt_files.append(file_path)
            
            self.logger.info(f"找到 {len(all_txt_files)} 个文本文件，开始加载...")
            
            documents = []
            skipped_files = 0
            
            # 使用进度条
            with tqdm(total=len(all_txt_files), desc="加载文件") as pbar:
                for file_path in all_txt_files:
                    try:
                        content, encoding = FileReader.read_file_safely(file_path)
                        if content and content.strip():
                            from langchain_core.documents import Document
                            doc = Document(
                                page_content=content,
                                metadata={"source": file_path, "encoding": encoding}
                            )
                            documents.append(doc)
                        else:
                            skipped_files += 1
                    except Exception as e:
                        self.logger.warning(f"跳过无法读取的文件: {file_path} - {e}")
                        skipped_files += 1
                    finally:
                        pbar.update(1)
                        pbar.set_postfix({"当前文件": os.path.basename(file_path)})
            
            if not documents:
                raise ValueError(f"在 {self.documents_path} 目录中未找到可读文档")
            
            self.logger.info(f"成功加载 {len(documents)} 个有效文档 (跳过了 {skipped_files} 个文件)")

            # 分割文档
            chunk_size = self.config.get('vector_db.chunk_size')
            chunk_overlap = self.config.get('vector_db.chunk_overlap')
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "；", " ", ""]
            )
            
            with tqdm(total=len(documents), desc="分割文档") as pbar:
                texts = []
                for doc in documents:
                    chunks = text_splitter.split_documents([doc])
                    texts.extend(chunks)
                    pbar.update(1)
            
            self.logger.info(f"文档分割完成，得到 {len(texts)} 个文本块")

            # 修复：分批添加到向量数据库
            self.logger.info("分批创建向量数据库...")
            vector_db = self._create_vector_db_in_batches(texts)
            
            end_time = time.time()
            self.logger.info(f"知识库构建完成，耗时 {(end_time - start_time)/60:.2f} 分钟")
            return vector_db
            
        except Exception as e:
            self.logger.error(f"构建知识库失败: {e}")
            raise

    def _create_vector_db_in_batches(self, texts, batch_size=4000):
        """分批创建向量数据库以避免批量大小限制"""
        from langchain_community.vectorstores import Chroma
        
        self.logger.info(f"开始分批处理 {len(texts)} 个文档块，批次大小: {batch_size}")
        
        # 第一次批次创建数据库
        first_batch = texts[:batch_size]
        self.logger.info(f"创建初始向量数据库，包含 {len(first_batch)} 个文档...")
        
        vector_db = Chroma.from_documents(
            documents=first_batch,
            embedding=self.embeddings,
            persist_directory=self.vector_db_dir
        )
        
        # 剩余批次逐步添加
        remaining_texts = texts[batch_size:]
        if remaining_texts:
            self.logger.info(f"逐步添加剩余 {len(remaining_texts)} 个文档...")
            
            for i in range(0, len(remaining_texts), batch_size):
                batch = remaining_texts[i:i + batch_size]
                self.logger.info(f"添加批次 {i//batch_size + 1}/{(len(remaining_texts)-1)//batch_size + 1}，包含 {len(batch)} 个文档")
                
                vector_db.add_documents(batch)
                
                # 及时清理内存
                del batch
        
        # 持久化最终数据库
        vector_db.persist()
        self.logger.info("向量数据库创建完成并已持久化")
        
        return vector_db
    
    def create_enhanced_qa_chain(self):
        """创建增强的QA链，包含项目上下文"""
        
        class CustomQwenLLM:
            def __init__(self, model, tokenizer, project_learner):
                self.model = model
                self.tokenizer = tokenizer
                self.project_learner = project_learner
                self.config = ConfigManager()
                self.logger = logging.getLogger(__name__)
            
            def __call__(self, prompt):
                try:
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=32000,
                        padding=True
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.config.get('model.max_tokens'),
                            do_sample=True,
                            temperature=self.config.get('model.temperature'),
                            top_p=self.config.get('model.top_p'),
                            top_k=self.config.get('model.top_k'),
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=self.config.get('model.repetition_penalty'),
                            num_return_sequences=1,
                            use_cache=True  # 启用KV缓存
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    return response
                    
                except Exception as e:
                    self.logger.error(f"生成回答时出错: {e}")
                    return f"生成回答时出错: {str(e)}"
        
        # 创建检索器
        search_k = self.config.get('vector_db.search_k')
        retriever = self.vector_db.as_retriever(search_kwargs={"k": search_k})
        
        class EnhancedCodeGenerationChain:
            def __init__(self, llm, retriever, project_learner):
                self.llm = llm
                self.retriever = retriever
                self.project_learner = project_learner
                self.validator = CodeValidator()
                self.logger = logging.getLogger(__name__)
            
            def run(self, query):
                try:
                    # 检索相关文档（使用缓存）
                    docs = self.retriever.get_relevant_documents(query)
                    
                    # 获取项目上下文
                    project_context = self.project_learner.generate_context_prompt(query)
                    
                    # 构建增强的提示词
                    context = self._build_context_prompt(project_context, docs)
                    prompt = self._build_generation_prompt(context, query)
                    
                    # 生成回答
                    response = self.llm(prompt)
                    
                    # 验证生成的代码
                    validation_result = self.validator.validate_afsim_code(response)
                    
                    return {
                        "result": response,
                        "source_documents": docs,
                        "project_context": project_context,
                        "validation": validation_result
                    }
                    
                except Exception as e:
                    self.logger.error(f"运行增强代码生成链时出错: {e}")
                    return {
                        "result": f"生成代码时出错: {str(e)}",
                        "source_documents": [],
                        "project_context": "",
                        "validation": {"is_valid": False, "errors": [str(e)]}
                    }
            
            def _build_context_prompt(self, project_context, docs):
                """构建上下文提示词"""
                context = " AFSIM项目上下文信息:\n\n"
                context += project_context
                context += "\n\n 相关代码示例:\n\n"
                
                for i, doc in enumerate(docs, 1):
                    source_info = f"来源: {doc.metadata.get('source', '未知')}" if hasattr(doc, 'metadata') else ""
                    context += f"示例 {i} {source_info}:\n{doc.page_content}\n{'='*50}\n"
                
                return context
            
            def _clean_query(self, query: str) -> str:
                """清理查询中的重复内容"""
                import re
                
                # 移除过多的重复要求
                lines = query.split('\n')
                cleaned_lines = []
                required_keywords_seen = set()
                
                for line in lines:
                    line_clean = line.strip()
                    if not line_clean:
                        continue
                        
                    # 检测重复的"必须包含"模式
                    if "必须包含" in line_clean:
                        # 提取关键内容
                        key_content = re.sub(r'.*必须包含', '', line_clean).strip()
                        if key_content and key_content not in required_keywords_seen:
                            cleaned_lines.append(line_clean)
                            required_keywords_seen.add(key_content)
                    else:
                        cleaned_lines.append(line_clean)
                
                # 如果清理后内容太少，返回原始查询的重要部分
                if len(cleaned_lines) < 2:
                    # 提取原始查询中的关键行
                    important_lines = []
                    for line in lines[:10]:  # 只取前10行避免重复
                        line_clean = line.strip()
                        if line_clean and "必须包含" not in line_clean:
                            important_lines.append(line_clean)
                    return "\n".join(important_lines[:5])
                
                return "\n".join(cleaned_lines[:20])  # 限制长度
            
            def _build_generation_prompt(self, context, query):
                """构建生成提示词"""
                cleaned_query = self._clean_query(query)

                return f"""你是一个AFSIM代码生成专家，熟悉整个项目结构和基础库的使用。
{context}
 用户需求: {query}
请基于以上项目结构和相关代码示例，直接生成准确、完整的AFSIM代码。
 禁止:
不要添加解释性文字
不要重复相同的内容
不要输出不完整的代码块
请生成完整的AFSIM代码:"""
        
        return EnhancedCodeGenerationChain(
            llm=CustomQwenLLM(self.model, self.tokenizer, self.project_learner),
            retriever=retriever,
            project_learner=self.project_learner
        )
    
    def generate_enhanced_response(self, query: str) -> Dict[str, Any]:
        """生成增强的回答"""
        try:
            # 使用增强的QA链生成回答
            result = self.enhanced_qa_chain.run(query)
            
            # 更新对话历史
            self.conversation_history.append({
                'query': query,
                'response': result["result"],
                'sources': len(result["source_documents"]),
                'validation': result["validation"],
                'timestamp': time.time()
            })
            
            # 限制历史记录长度
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-6:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成增强响应时出错: {e}")
            return {
                "result": f"生成回答时出错: {str(e)}", 
                "source_documents": [],
                "project_context": "",
                "validation": {"is_valid": False, "errors": [str(e)]}
            }
    
    def get_vector_db_info(self):
        """获取向量数据库信息"""
        if hasattr(self.vector_db, '_collection'):
            count = self.vector_db._collection.count()
            return f"向量数据库包含 {count} 个文档片段"
        return "向量数据库信息不可用"
    
    def get_project_info(self):
        """获取项目信息"""
        return self.project_learner.get_project_summary()
    
    def search_project_files(self, keyword: str, max_results: int = 5):
        """在项目中搜索文件"""
        return self.project_learner.find_related_files(keyword, max_results)
    
    def __del__(self):
        """析构函数释放资源"""
        self.logger.info("清理系统资源...")
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'vector_db'):
            del self.vector_db
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 在rag_enhanced.py中添加以下类和方法
class StageAwareRAGSystem:
    """阶段感知的RAG系统"""
    
    def __init__(self, project_learner: AFSIMProjectLearner, vector_db, embeddings, model, tokenizer):
        self.project_learner = project_learner
        self.vector_db = vector_db
        self.embeddings = embeddings
        self.model = model
        self.tokenizer = tokenizer
        self.config = ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # 阶段特定的检索器
        self.stage_retrievers = {}
        
    def build_stage_aware_retriever(self, stage_name: str):
        """为特定阶段构建感知检索器"""
        try:
            # 获取阶段特定的上下文
            stage_context = self.project_learner.get_stage_context(stage_name)
            
            # 阶段特定的检索参数
            search_kwargs = {
                "k": self.config.get(f'stages.{stage_name}.search_k', 8),
                "score_threshold": self.config.get(f'stages.{stage_name}.score_threshold', 0.7)
            }
            
            # 创建阶段感知的检索器
            retriever = self.vector_db.as_retriever(search_kwargs=search_kwargs)
            
            # 增强检索器
            enhanced_retriever = self._enhance_retriever_for_stage(retriever, stage_name, stage_context)
            
            self.stage_retrievers[stage_name] = enhanced_retriever
            return enhanced_retriever
            
        except Exception as e:
            self.logger.error(f"构建阶段 {stage_name} 检索器失败: {e}")
            return self.vector_db.as_retriever(search_kwargs={"k": 6})
    
    def _enhance_retriever_for_stage(self, retriever, stage_name: str, stage_context: str):
        """为特定阶段增强检索器"""
        
        class StageAwareRetriever:
            def __init__(self, base_retriever, stage_name, stage_context, project_learner):
                self.base_retriever = base_retriever
                self.stage_name = stage_name
                self.stage_context = stage_context
                self.project_learner = project_learner
                self.logger = logging.getLogger(__name__)
            
            def get_relevant_documents(self, query: str):
                """获取相关文档，考虑阶段上下文"""
                try:
                    # 增强查询
                    enhanced_query = self._enhance_query_for_stage(query)
                    
                    # 执行基础检索
                    base_docs = self.base_retriever.get_relevant_documents(enhanced_query)
                    
                    # 过滤和排序文档
                    filtered_docs = self._filter_docs_by_stage(base_docs)
                    sorted_docs = self._sort_docs_by_relevance(filtered_docs, query)
                    
                    return sorted_docs
                    
                except Exception as e:
                    self.logger.error(f"检索文档失败: {e}")
                    return []
            
            def _enhance_query_for_stage(self, query: str) -> str:
                """为特定阶段增强查询"""
                stage_keywords = {
                    "project_structure": ["项目结构", "文件夹", "组织", "布局"],
                    "platforms": ["平台", "飞机", "车辆", "平台类型"],
                    "weapons": ["武器", "导弹", "发射", "战斗部"],
                    "sensors": ["传感器", "雷达", "探测", "跟踪"],
                    "processors": ["处理器", "控制", "算法", "决策"],
                    "scenarios": ["场景", "任务", "对抗", "环境"],
                    "signatures": ["特征", "雷达反射", "红外", "光学"],
                    "main_program": ["主程序", "入口", "初始化", "事件循环"]
                }
                
                keywords = stage_keywords.get(self.stage_name, [])
                enhanced = query
                
                # 添加阶段关键词
                if keywords:
                    enhanced += " " + " ".join(keywords[:3])
                
                # 添加阶段上下文中的关键词
                context_words = re.findall(r'\b\w{4,}\b', self.stage_context)
                if context_words:
                    enhanced += " " + " ".join(context_words[:5])
                
                return enhanced
            
            def _filter_docs_by_stage(self, docs):
                """根据阶段过滤文档"""
                filtered = []
                for doc in docs:
                    if self._is_relevant_to_stage(doc):
                        filtered.append(doc)
                
                # 如果过滤后文档太少，返回原始文档
                if len(filtered) < 3:
                    return docs[:5]
                
                return filtered[:8]  # 限制数量
            
            def _is_relevant_to_stage(self, doc) -> bool:
                """检查文档是否与阶段相关"""
                doc_text = doc.page_content.lower()
                
                # 阶段特定的关键词检查
                stage_keywords_map = {
                    "platforms": ["platform_type", "mover", "aircraft", "vehicle"],
                    "weapons": ["weapon_type", "missile", "launch", "warhead"],
                    "sensors": ["sensor_type", "radar", "detect", "track"],
                    "processors": ["processor_type", "tasker", "controller", "algorithm"],
                    "scenarios": ["scenario", "mission", "setup", "environment"],
                    "signatures": ["signature", "rcs", "radar_cross_section", "emission"],
                    "main_program": ["main", "include", "initialize", "event_loop"]
                }
                
                keywords = stage_keywords_map.get(self.stage_name, [])
                if keywords:
                    return any(keyword in doc_text for keyword in keywords)
                
                return True
            
            def _sort_docs_by_relevance(self, docs, query: str):
                """根据相关性排序文档"""
                if len(docs) <= 1:
                    return docs
                
                # 简单的相关性评分
                scored_docs = []
                query_lower = query.lower()
                
                for doc in docs:
                    score = 0
                    doc_text = doc.page_content.lower()
                    
                    # 查询词匹配
                    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
                    doc_words = set(re.findall(r'\b\w{3,}\b', doc_text))
                    
                    common_words = query_words.intersection(doc_words)
                    score += len(common_words) * 2
                    
                    # 阶段关键词匹配
                    if self.stage_name == "platforms" and "platform_type" in doc_text:
                        score += 5
                    elif self.stage_name == "weapons" and "weapon_type" in doc_text:
                        score += 5
                    
                    # 元数据检查
                    if hasattr(doc, 'metadata'):
                        if "source" in doc.metadata and self.stage_name in doc.metadata["source"]:
                            score += 3
                    
                    scored_docs.append((score, doc))
                
                # 按分数排序
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored_docs]
        
        return StageAwareRetriever(retriever, stage_name, stage_context, self.project_learner)

class EnhancedStageAwareRAGChatSystem(EnhancedRAGChatSystem):
    """增强的阶段感知RAG聊天系统"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化阶段感知系统
        self.stage_aware_system = StageAwareRAGSystem(
            project_learner=self.project_learner,
            vector_db=self.vector_db,
            embeddings=self.embeddings,
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        # 阶段特定的QA链
        self.stage_qa_chains = {}
        
        # 从demos中学习
        self.logger.info("从demo项目中学习阶段模式...")
        self.project_learner.learn_from_demos()
        
    def get_stage_aware_qa_chain(self, stage_name: str):
        """获取阶段感知的QA链"""
        if stage_name in self.stage_qa_chains:
            return self.stage_qa_chains[stage_name]
        
        self.logger.info(f"为阶段 {stage_name} 构建QA链...")
        
        # 获取阶段特定的检索器
        retriever = self.stage_aware_system.build_stage_aware_retriever(stage_name)
        
        # 创建阶段特定的LLM
        class StageAwareLLM:
            def __init__(self, model, tokenizer, project_learner, stage_name):
                self.model = model
                self.tokenizer = tokenizer
                self.project_learner = project_learner
                self.stage_name = stage_name
                self.config = ConfigManager()
                self.logger = logging.getLogger(__name__)
            
            def __call__(self, prompt):
                try:
                    # 阶段特定的生成参数
                    stage_params = self.config.get(f'stages.{self.stage_name}', {})
                    max_tokens = stage_params.get('max_tokens', self.config.get('model.max_tokens', 1024))
                    temperature = stage_params.get('temperature', self.config.get('model.temperature', 0.2))
                    
                    inputs = self.tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=32000,
                        padding=True
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=self.config.get('model.top_p', 0.9),
                            pad_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=self.config.get('model.repetition_penalty', 1.1)
                        )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    return response
                    
                except Exception as e:
                    self.logger.error(f"阶段 {self.stage_name} 生成回答时出错: {e}")
                    return f"生成回答时出错: {str(e)}"
        
        # 创建阶段感知的QA链
        class StageAwareQAChain:
            def __init__(self, llm, retriever, project_learner, stage_name):
                self.llm = llm
                self.retriever = retriever
                self.project_learner = project_learner
                self.stage_name = stage_name
                self.logger = logging.getLogger(__name__)
            
            def run(self, query: str, project_context: Dict = None) -> Dict[str, Any]:
                try:
                    # 获取阶段特定的学习结果
                    stage_context = self.project_learner.get_stage_context(self.stage_name, query)
                    
                    # 检索相关文档
                    docs = self.retriever.get_relevant_documents(query)
                    
                    # 构建增强提示词
                    prompt = self._build_stage_prompt(query, stage_context, docs, project_context)
                    
                    # 生成回答
                    response = self.llm(prompt)
                    
                    return {
                        "result": response,
                        "stage_name": self.stage_name,
                        "source_documents": docs,
                        "stage_context": stage_context,
                        "prompt_length": len(prompt)
                    }
                    
                except Exception as e:
                    self.logger.error(f"阶段 {self.stage_name} QA链运行失败: {e}")
                    return {
                        "result": f"生成失败: {str(e)}",
                        "stage_name": self.stage_name,
                        "source_documents": [],
                        "stage_context": "",
                        "prompt_length": 0
                    }
            
            def _build_stage_prompt(self, query: str, stage_context: str, docs: List, project_context: Dict = None) -> str:
                """构建阶段特定的提示词"""
                # 文档内容
                doc_content = ""
                for i, doc in enumerate(docs[:4], 1):  # 限制文档数量
                    doc_content += f"示例文档 {i}:\n{doc.page_content[:500]}...\n\n"
                
                # 项目上下文
                proj_context = ""
                if project_context:
                    proj_context = f"\n当前项目上下文:\n{json.dumps(project_context, indent=2, ensure_ascii=False)}\n"
                
                # 阶段特定的指令
                stage_instructions = {
                    "project_structure": "你需要分析AFSIM项目需求并规划项目结构。输出应该是一个清晰的JSON格式结构，包含必要的文件夹和文件规划。",
                    "platforms": "你需要生成AFSIM平台定义。确保包含完整的平台类型定义、物理参数、组件配置和行为定义。",
                    "weapons": "你需要生成AFSIM武器定义。包括武器类型、性能参数、制导系统和战斗部配置。",
                    "sensors": "你需要生成AFSIM传感器定义。包含传感器类型、探测参数、工作模式和数据输出格式。",
                    "processors": "你需要生成AFSIM处理器定义。包括处理器类型、输入输出接口、处理算法和配置参数。",
                    "scenarios": "你需要生成AFSIM场景定义。包含场景描述、平台配置、环境设置和事件序列。",
                    "signatures": "你需要生成AFSIM特征信号定义。包括特征类型、RCS值、角度依赖性和辐射特性。",
                    "main_program": "你需要生成AFSIM主程序。包含必要的导入、初始化、事件循环和输出配置。"
                }
                
                instruction = stage_instructions.get(self.stage_name, "请根据需求生成相应的AFSIM代码。")
                
                prompt = f"""{instruction}

阶段学习总结:
{stage_context}

相关代码示例:
{doc_content}
{proj_context}
用户需求: {query}

请基于以上信息生成完整的{self.stage_name}阶段代码。
输出要求:
1. 只输出AFSIM代码，不添加额外解释
2. 确保代码完整性和正确性
3. 遵循示例中的最佳实践

生成代码:"""
                
                return prompt
        
        # 创建LLM实例
        llm = StageAwareLLM(self.model, self.tokenizer, self.project_learner, stage_name)
        
        # 创建QA链
        qa_chain = StageAwareQAChain(llm, retriever, self.project_learner, stage_name)
        
        self.stage_qa_chains[stage_name] = qa_chain
        return qa_chain
    
    def generate_stage_response(self, stage_name: str, query: str, project_context: Dict = None) -> Dict[str, Any]:
        """生成阶段特定的响应"""
        try:
            # 获取阶段感知的QA链
            qa_chain = self.get_stage_aware_qa_chain(stage_name)
            
            # 执行生成
            result = qa_chain.run(query, project_context)
            
            # 记录到历史
            self.conversation_history.append({
                'stage': stage_name,
                'query': query,
                'response': result["result"],
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"生成阶段 {stage_name} 响应失败: {e}")
            return {
                "result": f"生成失败: {str(e)}",
                "stage_name": stage_name,
                "error": str(e)
            }
