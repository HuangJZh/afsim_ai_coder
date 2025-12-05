import os
import glob
import re
import logging
import json
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

from utils import FileReader, ConfigManager


@dataclass
class StageLearningResult:
    """阶段学习结果"""
    stage_name: str
    learned_patterns: Dict[str, List[str]]
    file_examples: Dict[str, List[str]]
    import_patterns: List[str]
    best_practices: List[str]

class AFSIMProjectLearner:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.config = ConfigManager()
        self.base_libraries = self.config.get('project.base_libraries', ["base_types", "base_types_nx"])
        self.code_folders = []
        self.all_files = {}
        self.import_dependencies = {}
        self.file_categories = {}
        self.logger = logging.getLogger(__name__)
        
        # 新增：阶段式学习存储
        self.stage_learning_results: Dict[str, StageLearningResult] = {}
        self.demo_projects: List[Dict[str, Any]] = []
        
    def analyze_project_structure(self):
        """分析项目结构"""
        self.logger.info("开始分析AFSIM项目结构...")
        
        if not os.path.exists(self.project_root):
            raise ValueError(f"项目目录不存在: {self.project_root}")
        
        # 获取所有文件夹
        all_folders = [f.name for f in os.scandir(self.project_root) if f.is_dir()]
        
        # 分离基础库和代码文件夹
        self.code_folders = [f for f in all_folders if f not in self.base_libraries]
        
        self.logger.info(f"发现基础库: {self.base_libraries}")
        self.logger.info(f"发现代码文件夹: {len(self.code_folders)} 个")
        
        # 收集所有文件
        self._collect_all_files()
        
        # 分析文件分类
        self._categorize_files()
        
        # 分析导入依赖
        self._analyze_imports()
        
        self.logger.info(f"项目分析完成: 总共 {len(self.all_files)} 个文件")
        
    def learn_from_demos(self, demos_folder: str = "demos"):
        """从demo项目中学习各阶段的模式"""
        self.logger.info(f"开始从{demos_folder}文件夹中学习项目模式...")
        
        demos_path = os.path.join(self.project_root, demos_folder)
        if not os.path.exists(demos_path):
            self.logger.warning(f"Demos文件夹不存在: {demos_path}")
            return
        
        # 发现所有demo项目
        demo_projects = []
        for item in os.listdir(demos_path):
            item_path = os.path.join(demos_path, item)
            if os.path.isdir(item_path):
                project_info = self._analyze_demo_project(item_path, item)
                demo_projects.append(project_info)
        
        self.demo_projects = demo_projects
        self.logger.info(f"发现 {len(demo_projects)} 个demo项目")
        
        # 按阶段学习
        self._learn_stage_patterns()
        
    def _analyze_demo_project(self, project_path: str, project_name: str) -> Dict[str, Any]:
        """分析单个demo项目"""
        project_info = {
            "name": project_name,
            "path": project_path,
            "files": {},
            "structure": {},
            "stages": {}
        }
        
        # 分析项目结构
        project_structure = self._analyze_project_structure(project_path)
        project_info["structure"] = project_structure
        
        # 收集项目文件
        txt_files = glob.glob(os.path.join(project_path, "**", "*.txt"), recursive=True)
        for file_path in txt_files:
            relative_path = os.path.relpath(file_path, project_path)
            try:
                content, encoding = FileReader.read_file_safely(file_path)
                if content.strip():
                    project_info["files"][relative_path] = {
                        'content': content,
                        'encoding': encoding,
                        'folder': os.path.dirname(relative_path)
                    }
            except Exception as e:
                self.logger.debug(f"跳过文件 {file_path}: {e}")
        
        # 识别项目中的阶段
        project_info["stages"] = self._identify_stages_in_project(project_info)
        
        return project_info
    
    def _analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """分析项目文件夹结构"""
        structure = {
            "folders": [],
            "files_by_type": {}
        }
        
        # 收集文件夹
        for item in os.listdir(project_path):
            item_path = os.path.join(project_path, item)
            if os.path.isdir(item_path):
                structure["folders"].append(item)
        
        # 收集文件类型统计
        file_patterns = [
            ("platforms", ["*platform*", "*acft*", "*mover*"]),
            ("weapons", ["*weapon*", "*missile*", "*gun*"]),
            ("sensors", ["*sensor*", "*radar*", "*detect*"]),
            ("processors", ["*processor*", "*controller*", "*tasker*"]),
            ("scenarios", ["*scenario*", "*mission*", "*setup*"]),
            ("signatures", ["*signature*", "*rcs*", "*emission*"]),
            ("main", ["main.txt", "Main.txt", "MAIN.txt"])
        ]
        
        for file_type, patterns in file_patterns:
            structure["files_by_type"][file_type] = []
            for pattern in patterns:
                matching_files = glob.glob(os.path.join(project_path, "**", pattern + ".txt"), recursive=True)
                for file_path in matching_files:
                    relative_path = os.path.relpath(file_path, project_path)
                    structure["files_by_type"][file_type].append(relative_path)
        
        return structure
    
    def _identify_stages_in_project(self, project_info: Dict) -> Dict[str, List[str]]:
        """识别项目中的生成阶段"""
        stages = {}
        
        # 阶段与文件类型的映射
        stage_mapping = {
            "project_structure": ["project_structure.json", "README.md"],
            "platforms": project_info["structure"]["files_by_type"].get("platforms", []),
            "weapons": project_info["structure"]["files_by_type"].get("weapons", []),
            "sensors": project_info["structure"]["files_by_type"].get("sensors", []),
            "processors": project_info["structure"]["files_by_type"].get("processors", []),
            "scenarios": project_info["structure"]["files_by_type"].get("scenarios", []),
            "signatures": project_info["structure"]["files_by_type"].get("signatures", []),
            "main_program": project_info["structure"]["files_by_type"].get("main", [])
        }
        
        for stage_name, file_patterns in stage_mapping.items():
            stage_files = []
            for file_path in file_patterns:
                if file_path in project_info["files"]:
                    stage_files.append(file_path)
            if stage_files:
                stages[stage_name] = stage_files
        
        return stages
    
    def _learn_stage_patterns(self):
        """学习各阶段的模式"""
        stages_to_learn = [
            "project_structure", "main_program", "platforms", 
            "weapons", "sensors", "processors", "scenarios", "signatures"
        ]
        
        for stage_name in stages_to_learn:
            self.logger.info(f"学习阶段: {stage_name}")
            result = StageLearningResult(
                stage_name=stage_name,
                learned_patterns={},
                file_examples={},
                import_patterns=[],
                best_practices=[]
            )
            
            # 收集该阶段在所有demo项目中的文件
            stage_files = []
            for project in self.demo_projects:
                if stage_name in project["stages"]:
                    for file_path in project["stages"][stage_name]:
                        if file_path in project["files"]:
                            stage_files.append({
                                "project": project["name"],
                                "path": file_path,
                                "content": project["files"][file_path]["content"]
                            })
            
            if not stage_files:
                self.logger.debug(f"阶段 {stage_name} 没有找到示例文件")
                continue
            
            # 学习模式
            result.learned_patterns = self._extract_patterns_from_stage_files(stage_files, stage_name)
            result.file_examples = self._extract_examples_from_stage_files(stage_files)
            result.import_patterns = self._extract_import_patterns(stage_files)
            result.best_practices = self._extract_best_practices(stage_files, stage_name)
            
            self.stage_learning_results[stage_name] = result
            
            self.logger.info(f"阶段 {stage_name} 学习完成: {len(result.file_examples)} 个示例")
    
    def _extract_patterns_from_stage_files(self, stage_files: List[Dict], stage_name: str) -> Dict[str, List[str]]:
        """从阶段文件中提取模式"""
        patterns = {}
        
        if stage_name == "project_structure":
            # 学习JSON结构模式
            json_patterns = []
            for file_info in stage_files:
                content = file_info["content"]
                try:
                    data = json.loads(content)
                    json_patterns.append(json.dumps(data, indent=2))
                except:
                    pass
            patterns["json_structure"] = json_patterns
            
        elif stage_name == "main_program":
            # 学习主程序模式
            patterns["import_sections"] = self._extract_import_sections([f["content"] for f in stage_files])
            patterns["initialization_blocks"] = self._extract_initialization_blocks([f["content"] for f in stage_files])
            patterns["event_loops"] = self._extract_event_loops([f["content"] for f in stage_files])
            
        elif stage_name in ["platforms", "weapons", "sensors", "processors"]:
            # 学习组件定义模式
            patterns["type_definitions"] = self._extract_type_definitions([f["content"] for f in stage_files])
            patterns["parameter_blocks"] = self._extract_parameter_blocks([f["content"] for f in stage_files])
            patterns["behavior_sections"] = self._extract_behavior_sections([f["content"] for f in stage_files])
            
        elif stage_name == "scenarios":
            # 学习场景模式
            patterns["platform_placements"] = self._extract_platform_placements([f["content"] for f in stage_files])
            patterns["event_sequences"] = self._extract_event_sequences([f["content"] for f in stage_files])
            patterns["environment_settings"] = self._extract_environment_settings([f["content"] for f in stage_files])
            
        elif stage_name == "signatures":
            # 学习特征模式
            patterns["signature_definitions"] = self._extract_signature_definitions([f["content"] for f in stage_files])
            patterns["rcs_patterns"] = self._extract_rcs_patterns([f["content"] for f in stage_files])
        
        return patterns
    
    def _extract_examples_from_stage_files(self, stage_files: List[Dict]) -> Dict[str, List[str]]:
        """提取文件示例"""
        examples = {}
        for file_info in stage_files:
            project_name = file_info["project"]
            if project_name not in examples:
                examples[project_name] = []
            
            # 截取文件内容的前500字符作为示例
            content_preview = file_info["content"][:500] + "..." if len(file_info["content"]) > 500 else file_info["content"]
            examples[project_name].append({
                "file": file_info["path"],
                "preview": content_preview
            })
        
        return examples
    
    def _extract_import_patterns(self, stage_files: List[Dict]) -> List[str]:
        """提取导入模式"""
        import_patterns = set()
        
        for file_info in stage_files:
            content = file_info["content"]
            imports = self._extract_imports(content)
            for imp in imports:
                # 标准化导入语句
                standardized = self._standardize_import(imp)
                if standardized:
                    import_patterns.add(standardized)
        
        return list(import_patterns)
    
    def _extract_best_practices(self, stage_files: List[Dict], stage_name: str) -> List[str]:
        """提取最佳实践"""
        best_practices = []
        
        # 基于文件内容提取最佳实践
        if stage_name == "main_program":
            best_practices.extend([
                "主程序应包含清晰的导入部分",
                "使用有意义的变量命名",
                "包含适当的错误处理",
                "添加必要的注释说明"
            ])
        elif stage_name == "platforms":
            best_practices.extend([
                "平台定义应包含完整的物理参数",
                "明确定义平台类型",
                "包含所有必要的组件引用",
                "设置合理的初始状态"
            ])
        
        return best_practices
    
    # 各种提取方法的实现...
    def _extract_import_sections(self, contents: List[str]) -> List[str]:
        """提取导入部分"""
        sections = []
        import_keywords = ["include", "import", "using", "require", "#include"]
        
        for content in contents:
            lines = content.split('\n')
            import_section = []
            in_import_section = False
            
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in import_keywords):
                    in_import_section = True
                    import_section.append(line.strip())
                elif in_import_section and line.strip() and not line.strip().startswith('#'):
                    # 遇到非注释行，结束导入部分
                    break
            
            if import_section:
                sections.append('\n'.join(import_section))
        
        return sections
    
    def _extract_type_definitions(self, contents: List[str]) -> List[str]:
        """提取类型定义"""
        definitions = []
        type_patterns = [
            r'platform_type\s+\w+\s*\{[^}]+\}',
            r'weapon_type\s+\w+\s*\{[^}]+\}',
            r'sensor_type\s+\w+\s*\{[^}]+\}',
            r'processor_type\s+\w+\s*\{[^}]+\}'
        ]
        
        for content in contents:
            for pattern in type_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                definitions.extend(matches)
        
        return definitions
    
    def _standardize_import(self, import_stmt: str) -> str:
        """标准化导入语句"""
        # 移除注释
        import_stmt = re.sub(r'#.*$', '', import_stmt)
        import_stmt = re.sub(r'//.*$', '', import_stmt)
        import_stmt = import_stmt.strip()
        
        # 提取关键部分
        if "include" in import_stmt.lower():
            # 提取文件名
            match = re.search(r'["<]([^">]+)[">]', import_stmt)
            if match:
                return f"include {match.group(1)}"
        
        return import_stmt
    
    def get_stage_learning_summary(self) -> Dict[str, Any]:
        """获取阶段学习摘要"""
        summary = {
            "total_demo_projects": len(self.demo_projects),
            "stages_learned": list(self.stage_learning_results.keys()),
            "stage_details": {}
        }
        
        for stage_name, result in self.stage_learning_results.items():
            summary["stage_details"][stage_name] = {
                "example_count": sum(len(examples) for examples in result.file_examples.values()),
                "pattern_count": sum(len(patterns) for patterns in result.learned_patterns.values()),
                "import_patterns": len(result.import_patterns),
                "best_practices": len(result.best_practices)
            }
        
        return summary
    
    def get_stage_context(self, stage_name: str, query: str = "") -> str:
        """获取阶段特定的上下文"""
        if stage_name not in self.stage_learning_results:
            return f"未找到阶段 {stage_name} 的学习结果"
        
        result = self.stage_learning_results[stage_name]
        
        context_parts = []
        
        # 1. 阶段概述
        context_parts.append(f"=== {stage_name.upper()} 阶段学习总结 ===")
        context_parts.append(f"从 {len(result.file_examples)} 个demo项目中学到的最佳实践:")
        
        # 2. 最佳实践
        if result.best_practices:
            context_parts.append("最佳实践:")
            for i, practice in enumerate(result.best_practices, 1):
                context_parts.append(f"  {i}. {practice}")
        
        # 3. 导入模式
        if result.import_patterns:
            context_parts.append("\n常用导入模式:")
            for pattern in result.import_patterns[:5]:  # 限制数量
                context_parts.append(f"  - {pattern}")
        
        # 4. 模式示例
        if result.learned_patterns:
            context_parts.append("\n常见模式:")
            for pattern_type, patterns in result.learned_patterns.items():
                if patterns:
                    context_parts.append(f"  {pattern_type}: {len(patterns)} 个示例")
                    if patterns and len(patterns[0]) < 200:
                        context_parts.append(f"    示例: {patterns[0][:150]}...")
        
        # 5. 文件示例（简略）
        if result.file_examples:
            context_parts.append("\n示例文件来源:")
            for project_name, examples in list(result.file_examples.items())[:3]:  # 限制项目数量
                context_parts.append(f"  {project_name}: {len(examples)} 个文件")
        
        # 6. 查询相关的特定建议
        if query:
            relevant_patterns = self._find_relevant_patterns_for_query(stage_name, query)
            if relevant_patterns:
                context_parts.append("\n查询相关建议:")
                for pattern in relevant_patterns[:3]:
                    context_parts.append(f"  - {pattern}")
        
        return "\n".join(context_parts)
    
    def _find_relevant_patterns_for_query(self, stage_name: str, query: str) -> List[str]:
        """为查询查找相关模式"""
        if stage_name not in self.stage_learning_results:
            return []
        
        result = self.stage_learning_results[stage_name]
        relevant = []
        query_lower = query.lower()
        
        # 检查最佳实践
        for practice in result.best_practices:
            if any(keyword in practice.lower() for keyword in query_lower.split()[:5]):
                relevant.append(f"最佳实践: {practice}")
        
        return relevant
        
    def _collect_all_files(self):
        """收集所有文件内容"""
        all_folders = self.base_libraries + self.code_folders
        skipped_files = 0
        processed_files = 0
        
        for folder in all_folders:
            folder_path = os.path.join(self.project_root, folder)
            if not os.path.exists(folder_path):
                self.logger.warning(f"文件夹不存在: {folder_path}")
                continue
                
            # 查找所有.txt文件
            txt_files = glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
            
            for file_path in txt_files:
                # 跳过有问题的文件
                if FileReader.should_skip_file(file_path):
                    skipped_files += 1
                    continue
                    
                relative_path = os.path.relpath(file_path, self.project_root)
                try:
                    content, encoding = FileReader.read_file_safely(file_path)
                    if content.strip():  # 只保存非空文件
                        self.all_files[relative_path] = {
                            'content': content,
                            'size': len(content),
                            'folder': folder,
                            'encoding': encoding
                        }
                        processed_files += 1
                    else:
                        skipped_files += 1
                        self.logger.debug(f"跳过空文件: {file_path}")
                except Exception as e:
                    self.logger.error(f"处理文件 {file_path} 时出错: {e}")
                    skipped_files += 1
        
        self.logger.info(f"成功加载 {processed_files} 个文件，跳过了 {skipped_files} 个文件")
    
    def _categorize_files(self):
        """对文件进行分类 - 修复版"""
        categories = {
            'platforms': [],
            'weapons': [],
            'sensors': [], 
            'processors': [],
            'scenarios': [],
            'behaviors': [],
            'signatures': [],
            'scripts': [],
            'other': []
        }
        
        for file_path, file_info in self.all_files.items():
            content = file_info['content'].lower()
            folder = file_info['folder']
            
            # 首先基于路径进行分类（优先级更高）
            # if self._categorize_by_path(file_path, categories):
            #     continue
            self._categorize_by_path(file_path, categories)
                
            # 如果路径分类失败，再基于内容分类
            # self._categorize_by_content(file_path, content, categories)
        
        self.file_categories = categories
        
        # 记录分类统计
        self.logger.info("文件分类统计:")
        for category, files in categories.items():
            self.logger.info(f"  {category}: {len(files)} 个文件")

    def _categorize_by_path(self, file_path: str, categories: dict) -> bool:
        """基于文件路径进行分类"""
        file_path_lower = file_path.lower()
        
        # 检查路径中的关键词
        if 'platform' in file_path_lower and 'processor' not in file_path_lower:
            categories['platforms'].append(file_path)
            return True
        elif 'weapon' in file_path_lower:
            categories['weapons'].append(file_path)
            return True
        elif 'sensor' in file_path_lower:
            categories['sensors'].append(file_path)
            return True
        elif 'processor' in file_path_lower:
            categories['processors'].append(file_path)
            return True
        elif 'scenario' in file_path_lower:
            categories['scenarios'].append(file_path)
            return True
        elif 'behavior' in file_path_lower:
            categories['behaviors'].append(file_path)
            return True
        elif 'signature' in file_path_lower:
            categories['signatures'].append(file_path)
            return True
        elif 'script' in file_path_lower:
            categories['scripts'].append(file_path)
            return True
        
        return False

    # def _categorize_by_content(self, file_path: str, content: str, categories: dict):
    #     """基于文件内容进行分类"""
    #     # 平台相关的关键词
    #     platform_keywords = [
    #         'platform_type', 'mover', 'acft', 'aircraft', 'vehicle', 
    #         'platform', 'flight_lead', 'fighter', 'bomber', 'awacs'
    #     ]
        
    #     # 处理器相关的关键词
    #     processor_keywords = [
    #         'processor', 'tasker', 'controller', 'assessment', 
    #         'quantum_agents', 'ep_operations', 'rcs_display'
    #     ]
        
    #     # 武器相关的关键词
    #     weapon_keywords = [
    #         'weapon_type', 'missile', 'gun', 'launch', 'warhead', 
    #         'munition', 'warhead', 'explosive'
    #     ]
        
    #     # 传感器相关的关键词
    #     sensor_keywords = [
    #         'sensor_type', 'radar', 'sensor', 'detect', 'track', 
    #         'esm', 'emitter', 'receiver'
    #     ]
        
    #     # 检查内容中的关键词
    #     if any(keyword in content for keyword in processor_keywords):
    #         categories['processors'].append(file_path)
    #     elif any(keyword in content for keyword in platform_keywords):
    #         categories['platforms'].append(file_path)
    #     elif any(keyword in content for keyword in weapon_keywords):
    #         categories['weapons'].append(file_path)
    #     elif any(keyword in content for keyword in sensor_keywords):
    #         categories['sensors'].append(file_path)
    #     elif 'scenario' in content or 'setup' in content:
    #         categories['scenarios'].append(file_path)
    #     elif 'behavior' in content:
    #         categories['behaviors'].append(file_path)
    #     elif 'signature' in content:
    #         categories['signatures'].append(file_path)
    #     elif 'script' in content:
    #         categories['scripts'].append(file_path)
    #     else:
    #         categories['other'].append(file_path)
    
    # def _is_platform_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为平台文件"""
    #     platform_keywords = ['platform_type', 'mover', 'acft', 'aircraft', 'vehicle', 'platform']
    #     return ('platform' in file_path.lower() or 
    #             any(keyword in content for keyword in platform_keywords))
    
    # def _is_weapon_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为武器文件"""
    #     weapon_keywords = ['missile', 'gun', 'launch', 'warhead', 'weapon_type', 'weapon']
    #     return ('weapon' in file_path.lower() or 
    #             any(keyword in content for keyword in weapon_keywords))
    
    # def _is_sensor_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为传感器文件"""
    #     sensor_keywords = ['radar', 'sensor', 'detect', 'track', 'sensor_type']
    #     return ('sensor' in file_path.lower() or 
    #             any(keyword in content for keyword in sensor_keywords))
    
    # def _is_processor_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为处理器文件"""
    #     processor_keywords = ['processor', 'behavior', 'tasker', 'controller']
    #     return ('processor' in file_path.lower() or 
    #             any(keyword in content for keyword in processor_keywords))
    
    # def _is_scenario_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为场景文件"""
    #     scenario_keywords = ['scenario', 'setup', 'mission', 'simulation']
    #     return ('scenario' in file_path.lower() or 
    #             any(keyword in content for keyword in scenario_keywords))
    
    # def _is_behavior_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为行为文件"""
    #     behavior_keywords = ['behavior', 'action', 'movement', 'maneuver']
    #     return ('behavior' in file_path.lower() or 
    #             any(keyword in content for keyword in behavior_keywords))
    
    # def _is_signature_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为特征文件"""
    #     signature_keywords = ['signature', 'radar_cross_section', 'rcs', 'emission']
    #     return ('signature' in file_path.lower() or 
    #             any(keyword in content for keyword in signature_keywords))
    
    # def _is_script_file(self, file_path: str, content: str) -> bool:
    #     """判断是否为脚本文件"""
    #     script_keywords = ['script', 'python', 'lua', 'javascript', 'function']
    #     return ('script' in file_path.lower() or 
    #             any(keyword in content for keyword in script_keywords))
    
    def _analyze_imports(self):
        """分析文件间的导入关系"""
        for file_path, file_info in self.all_files.items():
            content = file_info['content']
            imports = self._extract_imports(content)
            if imports:
                self.import_dependencies[file_path] = imports
    
    def _extract_imports(self, content: str) -> List[str]:
        """从内容中提取导入语句"""
        imports = []

        # 移除CUI头部信息
        content = self._remove_cui_header(content)

        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # 查找类似导入的语句
            if any(keyword in line.lower() for keyword in ['include', 'import', 'from', 'require', 'using']):
                # 清理注释
                line = re.sub(r'#.*$', '', line)  # 移除Python风格注释
                line = re.sub(r'//.*$', '', line)  # 移除C++风格注释
                line = re.sub(r'/\*.*?\*/', '', line)  # 移除C风格注释
                line = line.strip()
                if line:
                    imports.append(line)
        
        return imports
    
    def _remove_cui_header(self, content: str) -> str:
        """移除AFSIM CUI版权声明头部"""
        # 定义要移除的头部模式
        header_patterns = [
            r'# \*+\s*CUI\s*\*+.*?# \*+',  # 匹配整个CUI头部
            r'# \*+\s*The Advanced Framework for Simulation.*?and LICENSE for details.*?# \*+',  # 匹配AFSIM描述
        ]
        
        for pattern in header_patterns:
            # 使用re.DOTALL让.匹配换行符，从文件开头匹配
            content = re.sub(pattern, '', content, count=1, flags=re.DOTALL | re.IGNORECASE)
        
        # 更精确的匹配方式
        cui_header = re.compile(
            r'^\s*# \*+\s*\n\s*# CUI\s*\n\s*# \*+\s*\n.*?The Advanced Framework for Simulation.*?and LICENSE for details.*?# \*+\s*',
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )
        content = cui_header.sub('', content, count=1)
        
        return content.strip()
    
    def get_file_content(self, relative_path: str) -> str:
        """获取文件内容"""
        file_info = self.all_files.get(relative_path)
        return file_info['content'] if file_info else ""
    
    def find_related_files(self, query: str, max_results: int = 10) -> List[str]:
        """查找与查询相关的文件"""
        related_files = []
        query_lower = query.lower()
        
        for file_path, file_info in self.all_files.items():
            content = file_info['content'].lower()
            if query_lower in content:
                related_files.append(file_path)
        
        return related_files[:max_results]
    
    def get_library_examples(self, library_name: str, max_examples: int = 3) -> Dict[str, str]:
        """获取特定库的使用示例"""
        examples = {}
        
        for file_path, file_info in self.all_files.items():
            # 查找使用该库的文件（非基础库本身）
            if file_info['folder'] != library_name and any(lib in file_info['content'] for lib in [library_name]):
                examples[file_path] = file_info['content'][:800]  # 只取前800字符作为预览
                if len(examples) >= max_examples:
                    break
        
        return examples

    def get_category_examples(self, category: str, max_examples: int = 3) -> Dict[str, str]:
        """获取特定类别的文件示例"""
        examples = {}
        
        if category in self.file_categories:
            for file_path in self.file_categories[category][:max_examples]:
                file_info = self.all_files.get(file_path)
                if file_info:
                    examples[file_path] = file_info['content'][:800]
        
        return examples

    def generate_context_prompt(self, query: str) -> str:
        """生成包含项目上下文的提示词"""
        context_parts = []
        
        # 添加项目结构信息
        context_parts.append("=== AFSIM项目结构 ===")
        context_parts.append(f"基础库: {', '.join(self.base_libraries)}")
        context_parts.append(f"代码模块: {len(self.code_folders)} 个")
        context_parts.append("")
        
        # 添加文件分类信息
        context_parts.append("=== 文件分类 ===")
        for category, files in self.file_categories.items():
            if files:
                context_parts.append(f"{category}: {len(files)} 个文件")
        context_parts.append("")
        
        # 查找相关文件
        related_files = self.find_related_files(query, 5)
        if related_files:
            context_parts.append("=== 相关文件 ===")
            for file_path in related_files:
                context_parts.append(f"- {file_path}")
            context_parts.append("")
        
        # 添加基础库的关键内容示例
        context_parts.append("=== 基础库关键内容 ===")
        for base_lib in self.base_libraries:
            lib_files = [f for f in self.all_files.keys() if f.startswith(base_lib)]
            if lib_files:
                # 获取库中不同类型文件的示例
                platform_files = [f for f in lib_files if 'platform' in f.lower()]
                weapon_files = [f for f in lib_files if 'weapon' in f.lower()]
                sensor_files = [f for f in lib_files if 'sensor' in f.lower()]
                
                context_parts.append(f"{base_lib}:")
                if platform_files:
                    context_parts.append(f"  - 平台定义: {len(platform_files)} 个")
                if weapon_files:
                    context_parts.append(f"  - 武器定义: {len(weapon_files)} 个")
                if sensor_files:
                    context_parts.append(f"  - 传感器定义: {len(sensor_files)} 个")
        
        context_parts.append("")
        
        # 添加导入依赖示例
        if self.import_dependencies:
            context_parts.append("=== 导入模式示例 ===")
            import_examples = list(self.import_dependencies.items())[:3]
            for file_path, imports in import_examples:
                context_parts.append(f"{file_path}:")
                for imp in imports[:2]:  # 只显示前2个导入
                    context_parts.append(f"  - {imp}")
            context_parts.append("")
        
        return self._generate_enhanced_context_prompt(query)
    
    def _generate_enhanced_context_prompt(self, query: str) -> str:
        """生成增强的上下文提示词"""
        # 结合阶段学习结果
        context_parts = []
        
        # 原有的项目结构信息
        context_parts.append("=== AFSIM项目结构 ===")
        context_parts.append(f"基础库: {', '.join(self.base_libraries)}")
        context_parts.append(f"代码模块: {len(self.code_folders)} 个")
        context_parts.append("")
        
        # 添加阶段学习摘要
        if self.stage_learning_results:
            context_parts.append("=== 阶段学习总结 ===")
            for stage_name, result in self.stage_learning_results.items():
                example_count = sum(len(examples) for examples in result.file_examples.values())
                context_parts.append(f"{stage_name}: {example_count} 个示例")
            context_parts.append("")
        
        return "\n".join(context_parts)

    def get_project_summary(self) -> Dict:
        """获取项目摘要"""
        return {
            "total_files": len(self.all_files),
            "base_libraries": self.base_libraries,
            "code_modules": self.code_folders,
            "file_categories": {k: len(v) for k, v in self.file_categories.items()},
            "total_size": sum(info['size'] for info in self.all_files.values())
        }

    def save_analysis_report(self, output_path: str):
        """保存分析报告"""
        report = {
            "project_summary": self.get_project_summary(),
            "file_categories_details": self.file_categories,
            "import_dependencies": self.import_dependencies
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"分析报告已保存到: {output_path}")