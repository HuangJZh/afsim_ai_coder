# rag_enhanced.py (完整优化版)
import os
import warnings
import threading
import time
import logging
import collections
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

    


class EnhancedInputHandler:
    """优化版的输入处理器"""
    
    def __init__(self, chat_system: EnhancedRAGChatSystem):
        self.chat_system = chat_system
        self.termination_shortcut = "//end"
        self.help_shortcut = "//help"
        self.info_shortcut = "//info"
        self.search_shortcut = "//search"
        self.clear_shortcut = "//clear"
        self.history_shortcut = "//history"
        self.logger = logging.getLogger(__name__)
        
    def start_input_listener(self):
        """启动输入监听线程"""
        input_thread = threading.Thread(target=self._input_loop, daemon=True)
        input_thread.start()
        self.logger.info("输入监听器已启动")
        
    def _input_loop(self):
        """输入循环"""
        self._show_welcome_message()
        
        while True:
            try:
                line = input().strip()
                
                if line == self.termination_shortcut:
                    self._process_complete_query()
                elif line == self.help_shortcut:
                    self._show_help()
                elif line == self.info_shortcut:
                    self._show_system_info()
                elif line == self.clear_shortcut:
                    self._clear_input()
                elif line == self.history_shortcut:
                    self._show_conversation_history()
                elif line.startswith(f"{self.search_shortcut} "):
                    keyword = line[len(self.search_shortcut)+1:]
                    self._search_files(keyword)
                elif line:
                    # 使用状态管理器添加输入
                    if not self.chat_system.input_state.add_input(line):
                        print("⚠️  输入缓冲区已满，请先处理当前输入或使用 //clear 清空")
                else:
                    # 空行，忽略
                    continue
                    
            except KeyboardInterrupt:
                self.logger.info("用户中断程序")
                print("\n\n程序退出。")
                break
            except Exception as e:
                self.logger.error(f"输入错误: {e}")
                print(f"输入错误: {e}")
    
    def _show_welcome_message(self):
        """显示欢迎信息"""
        print(f"\n{'='*80}")
        print(" AFSIM 智能代码生成系统 (增强优化版)")
        print(f"{'='*80}")
        print(f"请输入AFSIM代码生成需求（支持多行输入）")
        print(f"输入 '{self.termination_shortcut}' 结束输入并生成代码")
        print(f"输入 '{self.help_shortcut}' 显示帮助信息")
        print(f"输入 '{self.info_shortcut}' 显示系统信息")
        print(f"输入 '{self.search_shortcut} 关键词' 搜索项目文件")
        print(f"输入 '{self.clear_shortcut}' 清空当前输入")
        print(f"输入 '{self.history_shortcut}' 显示对话历史")
        print(f"{'-'*80}")
        self._show_example_queries()
    
    def _show_example_queries(self):
        """显示示例查询"""
        print("\n 示例代码生成需求：")
        examples = [
            "使用base_types库创建一个战斗机平台，包含雷达和导弹",
            "基于weapons库配置空对空导弹系统",
            "使用sensors库添加雷达探测和跟踪功能", 
            "创建包含多个平台的复杂空战场景",
            "实现导弹的制导和控制系统",
            "配置平台的飞行行为和战术"
        ]
        for i, example in enumerate(examples, 1):
            print(f"  {i}. {example}")
        print(f"{'-'*80}\n")
    
    def _show_help(self):
        """显示帮助信息"""
        print(f"\n{'='*50}")
        print("帮助信息:")
        print(f"{'='*50}")
        print(f"  {self.termination_shortcut} - 结束输入并生成代码")
        print(f"  {self.help_shortcut} - 显示此帮助信息") 
        print(f"  {self.info_shortcut} - 显示系统信息")
        print(f"  {self.search_shortcut} 关键词 - 搜索项目中的文件")
        print(f"  {self.clear_shortcut} - 清空当前输入缓冲区")
        print(f"  {self.history_shortcut} - 显示对话历史")
        print(f"\n提示:")
        print("  - 可以输入多行需求，系统会综合分析")
        print("  - 尽量详细描述需求，包括平台类型、武器配置等")
        print("  - 系统会自动参考项目中的基础库和现有代码")
        print("  - 输入缓冲区限制为50行，避免内存溢出")
        print(f"{'='*50}\n")
        self.chat_system.input_state.is_processing = False
    
    def _show_system_info(self):
        """显示系统信息"""
        print(f"\n{'='*50}")
        print("系统信息:")
        print(f"{'='*50}")
        
        # 项目信息
        project_info = self.chat_system.get_project_info()
        print(f"项目统计:")
        print(f"  - 总文件数: {project_info['total_files']}")
        print(f"  - 基础库: {', '.join(project_info['base_libraries'])}")
        print(f"  - 代码模块: {len(project_info['code_modules'])} 个")
        print(f"  - 文件分类: {project_info['file_categories']}")
        
        # 数据库信息
        db_info = self.chat_system.get_vector_db_info()
        print(f"知识库: {db_info}")
        
        # 对话历史
        print(f"对话历史: {len(self.chat_system.conversation_history)} 条记录")
        
        # 输入缓冲区状态
        inputs = self.chat_system.input_state.get_all_inputs()
        print(f"当前输入缓冲区: {len(inputs)} 行")
        self.chat_system.input_state.clear()  # 清空临时获取的输入
        
        # 系统状态
        print(f"处理状态: {'处理中' if self.chat_system.input_state.is_processing else '等待输入'}")
        print(f"{'='*50}\n")
        self.chat_system.input_state.is_processing = False
    
    def _search_files(self, keyword: str):
        """搜索文件"""
        print(f"\n🔍 搜索文件: '{keyword}'")
        results = self.chat_system.search_project_files(keyword, 5)
        if results:
            print("找到以下相关文件:")
            for i, file_path in enumerate(results, 1):
                print(f"  {i}. {file_path}")
        else:
            print("未找到相关文件")
        print()
        self.chat_system.input_state.is_processing = False
    
    def _clear_input(self):
        """清空输入缓冲区"""
        self.chat_system.input_state.clear()
        print("✅ 输入缓冲区已清空")
        self.chat_system.input_state.is_processing = False
    
    def _show_conversation_history(self):
        """显示对话历史"""
        print(f"\n{'='*20}")
        print("对话历史:")
        print(f"{'='*20}")
        
        if not self.chat_system.conversation_history:
            print("暂无对话历史")
        else:
            for i, conv in enumerate(self.chat_system.conversation_history, 1):
                print(f"\n对话 {i} ({(time.time() - conv['timestamp'])/60:.1f}分钟前):")
                print(f"  问题: {conv['query'][:100]}...")
                print(f"  回答: {conv['response'][:100]}...")
                print(f"  参考: {conv['sources']} 个示例")
                if 'validation' in conv:
                    valid_status = "✅ 有效" if conv['validation']['is_valid'] else "❌ 有问题"
                    print(f"  验证: {valid_status}")
        
        print(f"{'='*20}\n")
        self.chat_system.input_state.is_processing = False
    
    def _process_complete_query(self):
        """处理完整的查询"""
        # 获取所有输入
        inputs = self.chat_system.input_state.get_all_inputs()
        
        if not inputs:
            print("❌ 输入为空，请重新输入需求")
            self.chat_system.input_state.is_processing = False
            return
        
        full_query = self._clean_and_optimize_query(inputs)
        
        # 尝试获取处理锁
        if not self.chat_system.input_state.acquire_processing_lock():
            print("⚠️  系统正在处理其他请求，请稍候...")
            return
        
        try:
            self.chat_system.input_state.is_processing = True
            
            print(f"\n⏳ 正在生成AFSIM代码...")
            start_time = time.time()
            
            result = self.chat_system.generate_enhanced_response(full_query)
            end_time = time.time()
            
            self._display_result(full_query, result, end_time - start_time)
            
        except Exception as e:
            self.logger.error(f"处理查询时出错: {e}")
            print(f"❌ 生成代码时出错: {e}")
        finally:
            # 确保状态恢复和锁释放
            self.chat_system.input_state.is_processing = False
            self.chat_system.input_state.release_processing_lock()
        
        print(f"\n 请输入新的AFSIM代码生成需求（输入 '{self.help_shortcut}' 查看帮助）:")
    
    def _display_result(self, query: str, result: Dict, duration: float):
        """显示生成结果"""
        print(f"\n{'✅'*40}")
        print(f"生成的AFSIM代码 (耗时: {duration:.2f}秒)")
        # print(f" 原始需求: {query}")
        # print(f"{'-'*80}")
        # print(f" 生成的代码:")
        # print(f"{'='*80}")
        print(result["result"])
        print(f"{'='*80}")
        
        # 显示验证结果
        # if result.get("validation"):
        #     validation = result["validation"]
        #     if validation["is_valid"]:
        #         print("✅ 代码验证: 通过")
        #     else:
        #         print("❌ 代码验证: 未通过")
        #         if validation.get("errors"):
        #             print("   错误:")
        #             for error in validation["errors"]:
        #                 print(f"     - {error}")
        #         if validation.get("warnings"):
        #             print("   警告:")
        #             for warning in validation["warnings"]:
        #                 print(f"     - {warning}")
        #         if validation.get("suggestions"):
        #             print("   建议:")
        #             for suggestion in validation["suggestions"]:
        #                 print(f"     - {suggestion}")
        
        # # 显示参考信息
        # if result.get("source_documents"):
        #     print(f"\n📚 参考了 {len(result['source_documents'])} 个代码示例")
        #     for i, doc in enumerate(result["source_documents"][:3], 1):
        #         source = doc.metadata.get('source', '未知') if hasattr(doc, 'metadata') else '未知'
        #         print(f"  {i}. {os.path.basename(source)}")
        
        # # 显示项目上下文信息
        # if result.get("project_context"):
        #     print(f"\n 项目上下文: 基于 {len(self.chat_system.project_learner.all_files)} 个文件分析")
        
        # print(f"{'✅'*40}\n")
    
    def _clean_and_optimize_query(self, inputs: List[str]) -> str:
        """清理和优化查询输入"""
        full_query = "\n".join(inputs)
        
        # 移除重复的要求
        lines = full_query.split('\n')
        unique_lines = []
        seen_lines = set()
        
        for line in lines:
            line_stripped = line.strip()
            # 跳过空行和过于重复的行
            if line_stripped and line_stripped not in seen_lines:
                # 简化重复的"必须包含"要求
                if "必须包含" in line_stripped and len(line_stripped) < 100:
                    unique_lines.append(line_stripped)
                    seen_lines.add(line_stripped)
                elif "必须包含" not in line_stripped:
                    unique_lines.append(line_stripped)
                    seen_lines.add(line_stripped)
        
        # 如果清理后内容太少，保留原始输入
        if len(unique_lines) < 3:
            return full_query
        
        optimized_query = "\n".join(unique_lines)
        self.logger.info(f"查询优化: {len(lines)} -> {len(unique_lines)} 行")
        
        return optimized_query
    