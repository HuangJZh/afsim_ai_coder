# multi_stage_generator.py (整合版)
import os
import json
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class GenerationStage:
    """生成阶段定义"""
    name: str
    description: str
    max_tokens: int = 500
    temperature: float = 0.2
    depends_on: List[str] = field(default_factory=list)
    output_patterns: List[str] = field(default_factory=list)

class AFSimProjectStructure:
    """AFSIM项目结构分析器"""
    
    def __init__(self):
        self.structure_templates = {
            "simple": {
                "files": ["main.txt"],
                "folders": ["platforms", "scenarios"]
            },
            "standard": {
                "files": ["main.txt", "README.md", "input_variables.txt"],
                "folders": ["platforms", "scenarios", "processors", "weapons", "sensors"]
            },
            "complex": {
                "files": ["main.txt", "README.md", "input_variables.txt", 
                         "event_output.txt", "run_me_first.txt"],
                "folders": ["platforms", "scenarios", "processors", 
                           "weapons", "sensors", "patterns", "avionics", "doc"]
            },
            "satellite": {
                "files": ["main.txt", "README.md", "satellite.json"],
                "folders": ["platforms", "scenarios", "TLE", "satellites", 
                           "ground_networks", "avionics"]
            }
        }
    
    def analyze_requirements(self, query: str) -> Dict:
        """分析需求，确定项目结构"""
        query_lower = query.lower()
        
        # 检测项目类型
        project_type = "standard"
        if any(word in query_lower for word in ["卫星", "satellite", "轨道", "orbit"]):
            project_type = "satellite"
        elif any(word in query_lower for word in ["通信", "communication", "comm", "网络"]):
            project_type = "complex"
        elif any(word in query_lower for word in ["简单", "simple", "基础", "basic"]):
            project_type = "simple"
        
        # 检测需要的组件
        components = self._detect_components(query_lower)
        
        # 构建项目结构
        structure = self._build_project_structure(project_type, components)
        
        return {
            "project_type": project_type,
            "components": components,
            "structure": structure,
            "stages": self._generate_stages(project_type, components)
        }
    
    def _detect_components(self, query: str) -> Dict[str, bool]:
        """检测需要的组件"""
        return {
            "platforms": any(word in query for word in ["平台", "platform", "飞机", "aircraft", "飞行器"]),
            "weapons": any(word in query for word in ["武器", "weapon", "导弹", "missile", "火炮"]),
            "sensors": any(word in query for word in ["传感器", "sensor", "雷达", "radar", "探测"]),
            "processors": any(word in query for word in ["处理器", "processor", "控制", "control", "算法"]),
            "scenarios": any(word in query for word in ["场景", "scenario", "任务", "mission", "想定"]),
            "communications": any(word in query for word in ["通信", "communication", "comm", "网络", "network"]),
            "navigation": any(word in query for word in ["导航", "navigation", "gps", "定位"]),
            "patterns": any(word in query for word in ["模式", "pattern", "阵型", "formation"]),
        }
    
    def _build_project_structure(self, project_type: str, components: Dict) -> Dict:
        """构建项目结构"""
        base_structure = self.structure_templates.get(project_type, self.structure_templates["standard"])
        
        # 根据组件调整结构
        adjusted_structure = {
            "files": base_structure["files"].copy(),
            "folders": base_structure["folders"].copy()
        }
        
        # 添加必要的文件夹
        if components["communications"]:
            if "avionics" not in adjusted_structure["folders"]:
                adjusted_structure["folders"].append("avionics")
            if "ground_networks" not in adjusted_structure["folders"]:
                adjusted_structure["folders"].append("ground_networks")
        
        if components["navigation"]:
            if "TLE" not in adjusted_structure["folders"]:
                adjusted_structure["folders"].append("TLE")
            if "satellites" not in adjusted_structure["folders"]:
                adjusted_structure["folders"].append("satellites")
        
        return adjusted_structure
    
    def _generate_stages(self, project_type: str, components: Dict) -> List[Dict]:
        """生成阶段计划"""
        stages = [
            {
                "name": "project_structure",
                "description": "分析需求并规划项目结构",
                "depends_on": [],
                "output_patterns": ["project_structure.json"]
            },
            {
                "name": "main_program",
                "description": "生成主程序文件",
                "depends_on": ["project_structure"],
                "output_patterns": ["main.txt"]
            }
        ]
        
        # 根据组件添加阶段
        if components["platforms"]:
            stages.append({
                "name": "platforms",
                "description": "生成平台定义文件",
                "depends_on": ["project_structure"],
                "output_patterns": ["platforms/*.txt"]
            })
        
        if components["scenarios"]:
            stages.append({
                "name": "scenarios",
                "description": "生成场景文件",
                "depends_on": ["platforms", "main_program"],
                "output_patterns": ["scenarios/*.txt"]
            })
        
        if components["processors"]:
            stages.append({
                "name": "processors",
                "description": "生成处理器文件",
                "depends_on": ["platforms"],
                "output_patterns": ["processors/*.txt"]
            })
        
        if components["sensors"]:
            stages.append({
                "name": "sensors",
                "description": "生成传感器文件",
                "depends_on": ["platforms"],
                "output_patterns": ["sensors/*.txt"]
            })
        
        if components["weapons"]:
            stages.append({
                "name": "weapons",
                "description": "生成武器文件",
                "depends_on": ["platforms"],
                "output_patterns": ["weapons/*.txt"]
            })
        
        # 卫星项目特殊阶段
        if project_type == "satellite":
            stages.extend([
                {
                    "name": "satellites",
                    "description": "生成卫星轨道文件",
                    "depends_on": ["platforms"],
                    "output_patterns": ["satellites/**/*.txt", "TLE/*.txt"]
                },
                {
                    "name": "ground_networks",
                    "description": "生成地面站网络文件",
                    "depends_on": ["platforms"],
                    "output_patterns": ["ground_networks/*.txt"]
                },
                {
                    "name": "avionics",
                    "description": "生成航电系统文件",
                    "depends_on": ["platforms"],
                    "output_patterns": ["avionics/**/*.txt"]
                }
            ])
        
        # 集成阶段
        stages.append({
            "name": "integration",
            "description": "整合和验证所有文件",
            "depends_on": [stage["name"] for stage in stages[1:] if stage["name"] != "integration"],
            "output_patterns": ["README.md", "run_instructions.txt"]
        })
        
        return stages

class MultiStageGenerator:
    """多阶段生成器"""
    
    def __init__(self, chat_system, config):
        self.chat_system = chat_system
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.project_analyzer = AFSimProjectStructure()
        
        # 项目状态
        self.current_project = None
        self.generated_files = []
        self.current_stage = None
        self.project_context = {}
        self.stage_results = {}
        
    def generate_project(self, query: str, output_dir: str = None) -> Dict:
        """生成完整的AFSIM项目"""
        try:
            # 1. 分析需求
            self.logger.info("分析项目需求...")
            print("🔍 分析项目需求...")
            project_analysis = self.project_analyzer.analyze_requirements(query)
            
            # 2. 准备输出目录
            if not output_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.join(
                    self.config.get('generation.output.base_dir', 'generated_projects'),
                    f"afsim_project_{timestamp}"
                )
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 3. 保存项目分析
            self.current_project = {
                "analysis": project_analysis,
                "output_dir": output_dir,
                "query": query,
                "start_time": time.time(),
                "stages": {}
            }
            
            # 创建项目结构
            self._create_project_structure(output_dir, project_analysis["structure"])
            
            # 4. 按阶段生成
            stages = project_analysis["stages"]
            total_stages = len(stages)
            
            for idx, stage_info in enumerate(stages, 1):
                stage_name = stage_info["name"]
                stage_desc = stage_info["description"]
                
                self.current_stage = stage_name
                self.logger.info(f"开始阶段 {idx}/{total_stages}: {stage_name} - {stage_desc}")
                print(f"\n📋 阶段 {idx}/{total_stages}: {stage_desc}")
                
                # 检查依赖
                if not self._check_stage_dependencies(stage_info):
                    self.logger.warning(f"阶段 {stage_name} 的依赖未满足，跳过")
                    print(f"⚠️  跳过阶段 {stage_name}（依赖未满足）")
                    continue
                
                # 执行阶段生成
                stage_start = time.time()
                result = self._execute_stage(stage_info, query, output_dir)
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
                    print(f"✅ 阶段 {stage_name} 完成 ({stage_duration:.1f}秒)")
                else:
                    self.logger.error(f"阶段 {stage_name} 失败: {result.get('error')}")
                    print(f"❌ 阶段 {stage_name} 失败: {result.get('error')}")
                
                # 显示生成的文件
                if result.get("output_files"):
                    print(f"   生成文件: {', '.join(result['output_files'])}")
            
            # 5. 生成项目报告
            report = self._generate_project_report()
            
            self.logger.info(f"项目生成完成: {output_dir}")
            print(f"\n🎉 项目生成完成！位置: {output_dir}")
            
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
    
    def _execute_stage(self, stage_info: Dict, query: str, output_dir: str) -> Dict:
        """执行单个生成阶段"""
        stage_name = stage_info["name"]
        
        try:
            # 构建阶段特定的提示词
            prompt = self._build_stage_prompt(stage_info, query)
            
            # 生成内容
            result = self.chat_system.generate_enhanced_response(prompt)
            
            if not result or "result" not in result:
                return {
                    "success": False,
                    "error": "生成结果为空"
                }
            
            # 解析生成的内容
            generated_content = result["result"]
            
            # 提取文件内容
            files = self._extract_files_from_content(generated_content, stage_info, output_dir)
            
            # 保存文件
            output_files = self._save_generated_files(files, output_dir)
            
            # 更新上下文
            context = self._extract_context_from_content(generated_content)
            
            return {
                "success": True,
                "output_files": output_files,
                "context": context,
                "raw_content": generated_content,
                "stage_name": stage_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_stage_prompt(self, stage_info: Dict, query: str) -> str:
        """构建阶段特定的提示词"""
        stage_name = stage_info["name"]
        stage_desc = stage_info["description"]
        
        # 阶段特定的指令
        stage_instructions = {
            "project_structure": f"""请分析AFSIM项目需求并规划项目结构。

原始需求：{query}

请输出一个JSON格式的项目结构规划，包含以下信息：
1. 项目类型（simple/standard/complex/satellite）
2. 需要的组件列表
3. 建议的文件结构
4. 主要平台名称
5. 主要场景描述

请只输出JSON格式，不要其他内容。""",
            
            "main_program": f"""请生成AFSIM主程序文件（main.txt）。

基于以下项目需求：{query}

项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}

已生成的文件：{chr(10).join(f"- {f}" for f in self.generated_files)}

主程序必须包含：
1. 必要的导入语句（使用include语句）
2. 全局变量和常量定义
3. 场景初始化和设置
4. 主事件循环
5. 输出配置
6. 仿真控制参数

请生成完整的main.txt文件内容：""",
            
            "platforms": f"""请生成AFSIM平台定义文件。

基于项目需求：{query}

项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}

需要生成以下平台的定义：
{self._get_platform_requirements()}

每个平台文件应该包含：
1. 平台类型定义（platform_type）
2. 物理参数（尺寸、重量、动力等）
3. 初始状态（位置、速度、方向）
4. 组件配置（传感器、武器、处理器等）
5. 行为定义

请为每个平台生成单独的.txt文件。
使用以下格式分隔不同文件：
=== 平台名称.txt ===
[平台文件内容]
=== 另一个平台.txt ===
[另一个平台文件内容]

请开始生成：""",
            
            "scenarios": f"""请生成AFSIM场景文件。

基于项目需求：{query}
已生成的平台：{json.dumps(self.project_context.get('platforms', []), ensure_ascii=False)}

需要创建以下场景：
1. 主测试场景
2. 训练场景
3. 验证场景

每个场景文件应该包含：
1. 场景名称和描述
2. 参与平台及其初始配置
3. 环境设置（地形、天气、时间）
4. 任务目标和约束
5. 事件序列和触发器

请为每个场景生成单独的.txt文件。
使用以下格式分隔不同文件：
=== 场景名称.txt ===
[场景文件内容]

请开始生成：""",
            
            "processors": f"""请生成AFSIM处理器文件。

基于项目需求：{query}
已生成的平台：{json.dumps(self.project_context.get('platforms', []), ensure_ascii=False)}

需要生成以下处理器：
1. 战术决策处理器
2. 传感器数据处理处理器
3. 武器控制处理器
4. 通信处理器

每个处理器文件应该包含：
1. 处理器类型定义
2. 输入输出接口
3. 处理算法和逻辑
4. 配置参数
5. 性能指标

请为每个处理器生成单独的.txt文件。
使用以下格式分隔不同文件：
=== 处理器名称.txt ===
[处理器文件内容]

请开始生成：""",
            
            "sensors": f"""请生成AFSIM传感器文件。

基于项目需求：{query}
已生成的平台：{json.dumps(self.project_context.get('platforms', []), ensure_ascii=False)}

需要生成以下传感器：
1. 雷达传感器
2. 光电传感器
3. 电子支援措施（ESM）
4. 通信传感器

每个传感器文件应该包含：
1. 传感器类型定义
2. 探测参数（范围、精度、分辨率）
3. 工作模式
4. 数据输出格式
5. 功耗和性能

请为每个传感器生成单独的.txt文件。
使用以下格式分隔不同文件：
=== 传感器名称.txt ===
[传感器文件内容]

请开始生成：""",
            
            "weapons": f"""请生成AFSIM武器文件。

基于项目需求：{query}
已生成的平台：{json.dumps(self.project_context.get('platforms', []), ensure_ascii=False)}

需要生成以下武器：
1. 空对空导弹
2. 空对地导弹
3. 机炮系统
4. 炸弹

每个武器文件应该包含：
1. 武器类型定义
2. 性能参数（射程、速度、精度）
3. 制导系统
4. 战斗部配置
5. 发射控制

请为每个武器生成单独的.txt文件。
使用以下格式分隔不同文件：
=== 武器名称.txt ===
[武器文件内容]

请开始生成：""",
            
            "integration": f"""请整合所有生成的AFSIM文件，创建项目文档。

项目需求：{query}

已生成的文件列表：
{chr(10).join(f"- {f}" for f in self.generated_files)}

需要生成：
1. README.md - 项目说明文档
2. run_instructions.txt - 运行说明

README.md应该包含：
1. 项目概述
2. 文件结构说明
3. 平台和组件介绍
4. 运行方法
5. 配置说明

run_instructions.txt应该包含：
1. 运行步骤
2. 参数配置
3. 预期输出
4. 故障排除

请生成完整的文档内容。"""
        }
        
        instruction = stage_instructions.get(stage_name, f"请根据需求生成{stage_desc}。")
        return instruction
    
    def _get_platform_requirements(self) -> str:
        """获取平台需求描述"""
        if "platforms" in self.project_context:
            platforms = self.project_context["platforms"]
            return "\n".join([f"- {p}" for p in platforms])
        return "根据项目需求生成合适的平台"
    
    def _extract_files_from_content(self, content: str, stage_info: Dict, output_dir: str) -> List[Dict]:
        """从生成的内容中提取文件"""
        files = []
        stage_name = stage_info["name"]
        
        if stage_name == "project_structure":
            # 尝试解析JSON
            try:
                # 提取JSON部分
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    # 保存JSON文件
                    files.append({
                        "path": "project_structure.json",
                        "content": json.dumps(json.loads(json_str), indent=2, ensure_ascii=False)
                    })
                    
                    # 解析并更新上下文
                    structure_data = json.loads(json_str)
                    if "platforms" in structure_data:
                        self.project_context["platforms"] = structure_data.get("platforms", [])
            except json.JSONDecodeError:
                # 如果不是JSON，保存为文本
                files.append({
                    "path": "project_structure.txt",
                    "content": content
                })
                
        elif stage_name == "main_program":
            files.append({
                "path": "main.txt",
                "content": content
            })
            
        elif stage_name in ["platforms", "scenarios", "processors", "sensors", "weapons", 
                           "satellites", "ground_networks", "avionics"]:
            # 使用智能文件分割
            files.extend(self._extract_multiple_files_smart(content, stage_name))
            
        elif stage_name == "integration":
            # 处理集成文档
            files.extend(self._extract_integration_files(content))
        
        return files
    
    def _extract_multiple_files_smart(self, content: str, folder_name: str) -> List[Dict]:
        """智能提取多个文件"""
        files = []
        
        # 多种文件分隔模式
        patterns = [
            (r'=== (.+?\.txt) ===\n(.*?)(?=\n=== |\Z)', re.DOTALL),  # === 文件名.txt ===
            (r'// File: (.+?\.txt)\n(.*?)(?=\n// File: |\Z)', re.DOTALL),  # // File: 文件名.txt
            (r'# File: (.+?\.txt)\n(.*?)(?=\n# File: |\Z)', re.DOTALL),  # # File: 文件名.txt
            (r'文件：(.+?\.txt)\n(.*?)(?=\n文件：|\Z)', re.DOTALL),  # 文件：文件名.txt
            (r'(\w+)_platform\.txt:\n(.*?)(?=\n\w+_platform\.txt:|\Z)', re.DOTALL),  # 平台名_platform.txt:
        ]
        
        for pattern, flags in patterns:
            matches = re.findall(pattern, content, flags)
            if matches:
                for filename, file_content in matches:
                    # 清理文件名
                    filename = filename.strip()
                    if not filename.endswith('.txt'):
                        filename += '.txt'
                    
                    # 清理文件内容
                    file_content = file_content.strip()
                    
                    files.append({
                        "path": f"{folder_name}/{filename}",
                        "content": file_content
                    })
                break
        
        # 如果没有明确的分割，尝试其他方法
        if not files:
            files = self._extract_files_by_platform_pattern(content, folder_name)
        
        # 如果还是没有找到文件，创建一个默认文件
        if not files and content.strip():
            default_name = f"{folder_name}_main.txt"
            files.append({
                "path": f"{folder_name}/{default_name}",
                "content": content.strip()
            })
        
        return files
    
    def _extract_files_by_platform_pattern(self, content: str, folder_name: str) -> List[Dict]:
        """根据平台模式提取文件"""
        files = []
        
        # 查找平台定义
        platform_patterns = [
            r'platform_type\s+(\w+)',
            r'class\s+(\w+)\s*\{',
            r'(\w+)_platform\s*\{'
        ]
        
        all_platforms = []
        for pattern in platform_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            all_platforms.extend(matches)
        
        # 为每个平台提取相关内容
        for platform in set(all_platforms):
            # 查找该平台的相关内容
            platform_content = self._extract_platform_content(content, platform)
            if platform_content:
                filename = f"{platform}.txt"
                files.append({
                    "path": f"{folder_name}/{filename}",
                    "content": platform_content
                })
        
        return files
    
    def _extract_platform_content(self, content: str, platform: str) -> str:
        """提取特定平台的内容"""
        # 查找以平台名开始的部分
        patterns = [
            fr'platform_type\s+{platform}.*?\n}}(?=\n|$)',
            fr'class\s+{platform}.*?\n}}(?=\n|$)',
            fr'{platform}_platform.*?\n}}(?=\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group().strip()
        
        return ""
    
    def _extract_integration_files(self, content: str) -> List[Dict]:
        """提取集成文件"""
        files = []
        
        # 尝试分离README和运行说明
        readme_patterns = [
            r'# README\.md.*?\n(.*?)(?=# run_instructions\.txt|# 运行说明|\Z)',
            r'README.*?\n(.*?)(?=运行说明|\Z)',
        ]
        
        run_instructions_patterns = [
            r'# run_instructions\.txt.*?\n(.*?)(?=# |\Z)',
            r'运行说明.*?\n(.*?)(?=\Z)',
        ]
        
        readme_content = ""
        run_instructions_content = ""
        
        for pattern in readme_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                readme_content = match.group(1).strip()
                break
        
        for pattern in run_instructions_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                run_instructions_content = match.group(1).strip()
                break
        
        if not readme_content and not run_instructions_content:
            # 如果无法分离，整个内容作为README
            files.append({
                "path": "README.md",
                "content": content
            })
        else:
            if readme_content:
                files.append({
                    "path": "README.md",
                    "content": readme_content
                })
            if run_instructions_content:
                files.append({
                    "path": "run_instructions.txt",
                    "content": run_instructions_content
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
                self.logger.debug(f"保存文件: {file_info['path']}")
                
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
        
        # 提取武器名称
        weapon_matches = re.findall(r'weapon_type\s+(\w+)', content, re.IGNORECASE)
        if weapon_matches:
            context["weapons"] = list(set(weapon_matches))
        
        # 提取传感器名称
        sensor_matches = re.findall(r'sensor_type\s+(\w+)', content, re.IGNORECASE)
        if sensor_matches:
            context["sensors"] = list(set(sensor_matches))
        
        # 提取场景名称
        scenario_matches = re.findall(r'scenario\s+(\w+)', content, re.IGNORECASE)
        if scenario_matches:
            context["scenarios"] = list(set(scenario_matches))
        
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
                "generated_files_count": len(self.generated_files)
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
        from rag_enhanced import EnhancedRAGChatSystem
        from utils import setup_logging, ConfigManager
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化基础RAG系统
        self.chat_system = EnhancedRAGChatSystem(
            project_root=project_root,
            model_path=model_path
        )
        
        # 加载配置
        self.config = ConfigManager()
        
        # 初始化多阶段生成器
        self.project_analyzer = AFSimProjectStructure()
        self.multi_stage_generator = MultiStageGenerator(self.chat_system, self.config)
    
    def generate_complete_project(self, query: str, output_dir: str = None) -> Dict:
        """生成完整的AFSIM项目"""
        self.logger.info(f"开始生成完整项目: {query[:100]}...")
        
        # 使用多阶段生成器
        result = self.multi_stage_generator.generate_project(query, output_dir)
        
        # 记录到对话历史
        self.chat_system.conversation_history.append({
            'query': query,
            'type': 'project_generation',
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def get_project_info(self):
        """获取项目信息"""
        return self.chat_system.get_project_info()
    
    def get_vector_db_info(self):
        """获取向量数据库信息"""
        return self.chat_system.get_vector_db_info()