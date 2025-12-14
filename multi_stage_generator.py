import os
import json
import re
import time
import logging
from typing import Dict, List, Optional
from utils import ConfigManager


class AFSimProjectStructure:
    """AFSIM项目结构分析器"""
    
    def __init__(self):
        self.config = ConfigManager()
    
    def analyze_requirements(self, query: str, single_file_mode: bool = False) -> Dict:
        """分析需求，确定项目结构"""
        query_lower = query.lower()
                
        # 检测需要的组件
        components = self._detect_components(query_lower)
        
        # 构建项目结构
        structure = self._build_project_structure(components, single_file_mode)
        
        return {
            "components": components,
            "structure": structure,
            "stages": self._generate_stages(components, single_file_mode)
        }
    
    def _detect_components(self, query: str) -> Dict[str, bool]:
        """检测需要的组件"""
        return {
            "platforms": any(word in query for word in [
                "平台", "导弹", "炸弹", "车", "卫星", "船", "坦克", "飞行器", "飞机", "发射车"
            ]),
            "scenarios": any(word in query for word in [
                "红", "蓝", "队", "对抗"
            ]),
            "processors": any(word in query for word in [
                "处理器", "控制", "制导", "跟踪"
            ]),
            "weapons": any(word in query for word in [
                "武器平台", "武器", "导弹", "拦截弹","火箭", "炸弹", "火炮"
            ]),
            "sensors": any(word in query for word in [
                "传感器", "雷达", "探测", "跟踪", "红外", "光学"
            ]),
            "signatures": any(word in query for word in [
                "特征", "雷达反射", "红外特征", "光学特征", "雷达截面积", "隐身"
            ]),
        }
    
    def _build_project_structure(self, components: Dict, single_file_mode: bool) -> Dict:
        """构建项目结构"""
        if single_file_mode:
            return {
                "files": ["simulation_script.txt", "README.md"],
                "folders": []
            }

        structure = {
            "files": ["main.txt", "README.md", "project_structure.json"],
            "folders": []
        }
        
        # 文件夹映射
        folder_mapping = {
            "platforms": "platforms",
            "scenarios": "scenarios",
            "processors": "processors",
            "weapons": "weapons",
            "sensors": "sensors",
            "signatures": "signatures",
        }

        # 添加检测到的组件的文件夹
        for component, has_component in components.items():
            if has_component and component in folder_mapping:
                folder_name = folder_mapping[component]
                if folder_name not in structure["folders"]:
                    structure["folders"].append(folder_name)
        
        # 确保至少有平台文件夹
        if "platforms" not in structure["folders"] and components["platforms"]:
            structure["folders"].append("platforms")
        
        return structure
    
    def _generate_stages(self, components: Dict, single_file_mode: bool) -> List[Dict]:
        """生成阶段计划"""
        stages = []
        
        if single_file_mode:
            # 单文件模式：只有一个综合生成阶段
            stages.append({
                "name": "single_file_generation",
                "description": "生成完整的AFSIM仿真脚本（包含所有定义和场景）",
                "max_tokens": 3500,  # 保持较高的token限制
                "temperature": 0.2,
                "depends_on": [],
                "output_patterns": ["simulation_script.txt"]
            })
            return stages

        # 多文件模式：原有的分阶段逻辑
        # 首先添加项目结构阶段
        stages.append({
            "name": "project_structure",
            "description": "分析需求并规划项目结构",
            "max_tokens": 300,
            "temperature": 0.1,
            "depends_on": [],
            "output_patterns": ["project_structure.json"]
        })

        # 添加主程序阶段
        stages.append({
            "name": "main_program",
            "description": "生成主程序文件",
            "max_tokens": 800,
            "temperature": 0.2,
            "depends_on": ["project_structure"],
            "output_patterns": ["main.txt"]
        })
        
        # 根据检测到的组件添加相应阶段
        component_stage_mapping = {
            "platforms": {"name": "platforms", "description": "生成平台定义文件"},
            "scenarios": {"name": "scenarios", "description": "生成场景文件"},
            "processors": {"name": "processors", "description": "生成处理器文件"},
            "sensors": {"name": "sensors", "description": "生成传感器文件"},
            "weapons": {"name": "weapons", "description": "生成武器文件"},
            "signatures": {"name": "signatures", "description": "生成特征信号文件"},
        }

        # 添加检测到的组件的阶段
        for component, has_component in components.items():
            if has_component and component in component_stage_mapping:
                mapping = component_stage_mapping[component]
                
                # 设置依赖关系
                depends_on = ["project_structure"]
                if component == "scenarios":
                    depends_on = ["project_structure", "platforms"]
                elif component in ["processors", "sensors", "weapons"]:
                    depends_on = ["project_structure", "platforms"]
                
                # 创建阶段对象
                stage = {
                    "name": mapping["name"],
                    "description": mapping["description"],
                    "max_tokens": 1000,
                    "temperature": 0.15,
                    "depends_on": depends_on,
                    "output_patterns": [f"{mapping['name']}/*.txt"]
                }
                
                # 检查是否已存在同名阶段
                if not any(s["name"] == stage["name"] for s in stages):
                    stages.append(stage)
            
        return stages


class MultiStageGenerator:
    """简化的多阶段生成器"""
    
    def __init__(self, chat_system):
        self.chat_system = chat_system
        self.config = ConfigManager()
        self.logger = logging.getLogger(__name__)
        self.project_analyzer = AFSimProjectStructure()
        
        # 项目状态
        self.current_project = None
        self.generated_files = []
        self.project_context = {}
        self.stage_results = {}

    def generate_project(self, query: str, output_dir: str = None, single_file_mode: bool = False) -> Dict:
        """生成完整的AFSIM项目"""
        try:
            # 1. 分析需求
            self.logger.info(f"分析项目需求 (单文件模式: {single_file_mode})...")
            project_analysis = self.project_analyzer.analyze_requirements(query, single_file_mode)
            
            # 2. 准备输出目录
            if not output_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                suffix = "_single" if single_file_mode else ""
                output_dir = os.path.join(
                    self.config.get('generation.output.base_dir', 'generated_projects'),
                    f"afsim_project{suffix}_{timestamp}"
                )
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. 保存项目分析
            self.current_project = {
                "analysis": project_analysis,
                "output_dir": output_dir,
                "query": query,
                "start_time": time.time(),
                "stages": {},
                "single_file_mode": single_file_mode
            }
            
            # 创建项目结构
            self._create_project_structure(output_dir, project_analysis["structure"])
            
            # 4. 按阶段生成
            stages = project_analysis["stages"]
            
            for stage_info in stages:
                stage_name = stage_info["name"]
                stage_desc = stage_info["description"]
                
                self.logger.info(f"开始阶段: {stage_name} - {stage_desc}")
                
                # 检查依赖
                if not self._check_stage_dependencies(stage_info):
                    self.logger.warning(f"阶段 {stage_name} 的依赖未满足，跳过")
                    continue
                
                # 执行阶段生成
                stage_start = time.time()
                result = self._execute_stage(stage_info, query, output_dir, single_file_mode)
                stage_duration = time.time() - stage_start
                
                # 记录结果
                self.current_project["stages"][stage_name] = {
                    "status": "success" if result["success"] else "failed",
                    "output_files": result.get("output_files", []),
                    "error": result.get("error"),
                    "duration": stage_duration
                }
                
                if result["success"]:
                    self.generated_files.extend(result["output_files"])
                    self.project_context.update(result.get("context", {}))
                    self.stage_results[stage_name] = result
                    self.logger.info(f"阶段 {stage_name} 完成 ({stage_duration:.1f}秒)")
                else:
                    self.logger.error(f"阶段 {stage_name} 失败: {result.get('error')}")
            
            # 5. 生成项目报告
            report = self._generate_project_report()
            
            self.logger.info(f"项目生成完成: {output_dir}")
            
            return {
                "success": True,
                "project_dir": output_dir,
                "generated_files": self.generated_files,
                "report": report,
                "project_analysis": project_analysis
            }
            
        except Exception as e:
            self.logger.error(f"项目生成失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_project_structure(self, output_dir: str, structure: Dict):
        """创建项目文件夹结构"""
        self.logger.info(f"创建项目结构: {output_dir}")
        
        # 创建文件夹
        for folder in structure.get("folders", []):
            folder_path = os.path.join(output_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            self.logger.debug(f"创建文件夹: {folder_path}")
    
    def _check_stage_dependencies(self, stage_info: Dict) -> bool:
        """检查阶段依赖是否满足"""
        depends_on = stage_info.get("depends_on", [])
        if not depends_on:
            return True
        
        for dep in depends_on:
            if dep not in self.current_project["stages"]:
                return False
            if self.current_project["stages"][dep]["status"] != "success":
                return False
        
        return True
    
    def _execute_stage(self, stage_info: Dict, query: str, output_dir: str, single_file_mode: bool) -> Dict:
        """执行单个生成阶段"""
        stage_name = stage_info["name"]
        
        try:
            # 检查是否有阶段感知的RAG系统
            if hasattr(self.chat_system, 'generate_stage_response'):
                # 如果是单文件模式，使用极其严格的提示词策略
                if single_file_mode:
                    prompt = f"""
你是一个AFSIM仿真脚本代码生成器。请严格按照以下指令行动：

=== 🔴 核心指令 (必须遵守) ===
1. **严禁输出任何思考过程**：不要输出 "Thinking...", "Okay...", "Here is the code..." 等任何非代码文本。
2. **严禁输出解释性文字**：直接开始写代码。
3. **强制使用标记包裹**：你生成的代码必须包裹在 `<<<CODE_START>>>` 和 `<<<CODE_END>>>` 标记之间。系统只会提取这两个标记之间的内容。
4. **完整性**：生成的代码必须是一个完整的、可运行的脚本，包含所有定义，不依赖外部 `include`。

=== 📝 用户需求 ===
{query}

=== 💻 代码结构要求 ===
1. 头部：路径变量 (define_path_variable) 和日志配置 (log_file)。
2. 定义部分：平台类型 (platform_type)、武器、传感器等。
3. 场景部分：具体的平台实例 (platform)，包含位置 (position) 和阵营 (side)。
4. 尾部：仿真结束时间 (end_time)。

=== ✅ 预期输出格式示例 ===
<<<CODE_START>>>
define_path_variable NAME demo
log_file output/$(NAME).log

platform_type TANK WSF_PLATFORM
    icon tank
    mover WSF_GROUND_MOVER
    end_mover
end_platform_type

platform tank_1 TANK
    position 0 0 0
    side blue
end_platform

end_time 100 sec
<<<CODE_END>>>

请立即开始生成代码（从 <<<CODE_START>>> 开始）：
"""
                    rag_result = self.chat_system.generate_enhanced_response(prompt)
                else:
                    # 多文件模式
                    result = self.chat_system.generate_stage_response(
                        stage_name=stage_name,
                        query=query,
                        project_context=self.project_context
                    )
                    rag_result = result
                
                if not rag_result or "result" not in rag_result:
                    return {
                        "success": False,
                        "error": "生成结果为空"
                    }
                
                generated_content = rag_result["result"]
                
            else:
                # 回退方法
                prompt = f"生成{stage_info['description']}，需求:\n{query}"
                rag_result = self.chat_system.generate_enhanced_response(prompt)
                generated_content = rag_result["result"]
            
            # 提取文件内容
            files = self._extract_files_from_content(generated_content, stage_info, output_dir, single_file_mode)
            
            # 如果提取失败（比如模型没听话加标记），记录错误以便调试
            if not files:
                self.logger.warning(f"阶段 {stage_name} 未提取到有效文件，原始内容长度: {len(generated_content)}")
            
            # 保存文件
            output_files = self._save_generated_files(files, output_dir)
            
            # 更新上下文
            context = self._extract_context_from_content(generated_content)
            self.project_context.update(context)
            
            return {
                "success": True,
                "output_files": output_files,
                "context": context,
                "stage_name": stage_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_files_from_content(self, content: str, stage_info: Dict, output_dir: str, single_file_mode: bool) -> List[Dict]:
        """从生成的内容中提取文件"""
        files = []
        stage_name = stage_info["name"]
        
        if single_file_mode:
            clean_content = ""
            # 策略1：优先尝试提取 <<<CODE_START>>> 标记
            # 使用 re.DOTALL 让 . 匹配换行符
            marker_pattern = r'<<<CODE_START>>>\s*(.*?)\s*<<<CODE_END>>>'
            marker_match = re.search(marker_pattern, content, re.DOTALL)
            
            if marker_match:
                clean_content = marker_match.group(1)
                self.logger.info("成功提取 <<<CODE_START>>> 标记内容")
            else:
                # 策略2：尝试提取 Markdown 代码块
                code_block_pattern = r'```(?:afsim|txt|)\s*(.*?)\s*```'
                # 找最长的一个代码块，通常是主代码
                matches = re.findall(code_block_pattern, content, re.DOTALL)
                if matches:
                    # 假设最长的块是完整代码
                    clean_content = max(matches, key=len)
                    self.logger.info("提取到 Markdown 代码块")
                else:
                    # 策略3：最后的兜底 - 启发式清洗
                    # 尝试找到第一个像代码的行（以注释、define、platform 等开头）
                    self.logger.warning("未找到明确代码标记，尝试启发式提取")
                    lines = content.split('\n')
                    start_idx = -1
                    keywords = ['#', '//', 'define_path', 'log_file', 'platform', 'include', 'event_output']
                    
                    for i, line in enumerate(lines):
                        line_strip = line.strip()
                        if any(line_strip.startswith(kw) for kw in keywords):
                            start_idx = i
                            break
                    
                    if start_idx != -1:
                        clean_content = '\n'.join(lines[start_idx:])
                    else:
                        # 实在没办法，全部当做代码（虽然可能包含废话）
                        clean_content = content

            if clean_content.strip():
                files.append({
                    "path": "simulation_script.txt",
                    "content": clean_content.strip()
                })
            return files

        # 多文件模式的提取逻辑
        if stage_name == "project_structure":
            try:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    files.append({
                        "path": "project_structure.json",
                        "content": json.dumps(json.loads(json_str), indent=2, ensure_ascii=False)
                    })
            except json.JSONDecodeError:
                files.append({
                    "path": "project_structure.txt",
                    "content": content
                })
                
        elif stage_name == "main_program":
            files.append({
                "path": "main.txt",
                "content": content
            })
            
        elif stage_name in ["platforms", "scenarios", "processors", "sensors", "weapons", "signatures"]:
            folder_files = self._extract_multiple_files_simple(content, stage_name)
            files.extend(folder_files)
    
        return files
    
    def _extract_multiple_files_simple(self, content: str, folder_name: str) -> List[Dict]:
        """简单提取多个文件"""
        files = []
        
        # 查找文件分隔符
        file_patterns = [
            r'=== (.+?\.txt) ===\n(.*?)(?=\n=== |\Z)',
            r'// File: (.+?\.txt)\n(.*?)(?=\n// File: |\Z)',
            r'# File: (.+?\.txt)\n(.*?)(?=\n# File: |\Z)',
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                for filename, file_content in matches:
                    filename = filename.strip()
                    if not filename.endswith('.txt'):
                        filename += '.txt'
                    
                    files.append({
                        "path": f"{folder_name}/{filename}",
                        "content": file_content.strip()
                    })
                break
        
        # 如果没有找到明确的分隔符，创建单个文件
        if not files and content.strip():
            default_name = f"{folder_name}_main.txt"
            files.append({
                "path": f"{folder_name}/{default_name}",
                "content": content.strip()
            })
        
        return files
    
    def _save_generated_files(self, files: List[Dict], output_dir: str) -> List[str]:
        """保存生成的文件"""
        saved_files = []
        
        for file_info in files:
            try:
                file_path = os.path.join(output_dir, file_info["path"])
                
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file_info["content"])
                
                saved_files.append(file_info["path"])
                
            except Exception as e:
                self.logger.error(f"保存文件失败 {file_info['path']}: {e}")
        
        return saved_files
    
    def _extract_context_from_content(self, content: str) -> Dict:
        """从内容中提取上下文信息"""
        context = {}
        # 提取平台名称
        platform_matches = re.findall(r'platform_type\s+(\w+)', content, re.IGNORECASE)
        if platform_matches:
            context["platforms"] = list(set(platform_matches))
        return context
    
    def _generate_project_report(self) -> Dict:
        """生成项目报告"""
        if not self.current_project:
            return {}
        
        total_duration = time.time() - self.current_project["start_time"]
        
        return {
            "project_info": {
                "output_dir": self.current_project["output_dir"],
                "query": self.current_project["query"],
                "total_duration": total_duration,
                "generated_files_count": len(self.generated_files),
                "mode": "Single File" if self.current_project.get("single_file_mode") else "Multi-Stage"
            },
            "analysis": self.current_project["analysis"],
            "stage_results": self.current_project["stages"],
            "file_list": self.generated_files,
            "summary": {
                "total_stages": len(self.current_project["stages"]),
                "successful_stages": sum(1 for s in self.current_project["stages"].values() 
                                       if s["status"] == "success"),
                "total_files": len(self.generated_files),
                "avg_stage_duration": total_duration / max(len(self.current_project["stages"]), 1)
            }
        }


class MultiStageChatSystem:
    """支持多阶段生成的聊天系统"""
    
    def __init__(self, project_root: str, model_path: str = None):
        from rag_enhanced import EnhancedStageAwareRAGChatSystem
        from utils import setup_logging
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化基础RAG系统
        self.chat_system = EnhancedStageAwareRAGChatSystem(
            project_root=project_root,
            model_path=model_path
        )
        
        # 初始化多阶段生成器
        self.multi_stage_generator = MultiStageGenerator(self.chat_system)
    
    def generate_complete_project(self, query: str, output_dir: str = None, single_file_mode: bool = False) -> Dict:
        """生成完整的AFSIM项目"""
        self.logger.info(f"开始生成完整项目 (单文件模式: {single_file_mode}): {query[:100]}...")
        
        # 使用多阶段生成器
        result = self.multi_stage_generator.generate_project(query, output_dir, single_file_mode)
        
        # 记录到对话历史
        self.chat_system.conversation_history.append({
            'query': query,
            'type': 'project_generation',
            'result': result,
            'timestamp': time.time(),
            'single_file_mode': single_file_mode
        })
        
        return result
    
    def get_project_info(self):
        """获取项目信息"""
        return self.chat_system.get_project_info()
    
    def get_vector_db_info(self):
        """获取向量数据库信息"""
        return self.chat_system.get_vector_db_info()