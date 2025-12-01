# project_learner.py (修复版)
import os
import glob
import re
import logging
from typing import Dict, List, Set, Tuple
from pathlib import Path
import json
from utils import FileReader, ConfigManager

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