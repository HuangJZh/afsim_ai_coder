# multi_stage_generator.py
import os
import json
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from utils import ConfigManager  # 导入ConfigManager

@dataclass
class GenerationStage:
    """生成阶段定义"""
    name: str
    description: str
    max_tokens: int = 2048
    temperature: float = 0.2
    depends_on: List[str] = field(default_factory=list)
    output_patterns: List[str] = field(default_factory=list)

class AFSimProjectStructure:
    """AFSIM项目结构分析器"""
    
    def __init__(self):
        # 获取配置管理器
        self.config = ConfigManager()
        # 基础文件
        self.base_files = [
            "main.txt",
            "README.md",
            "project_structure.json"
        ]
    
    def analyze_requirements(self, query: str) -> Dict:
        """分析需求，确定项目结构"""
        query_lower = query.lower()
                
        # 检测需要的组件
        components = self._detect_components(query_lower)
        
        # 构建项目结构
        structure = self._build_project_structure(components)
        
        return {
            "components": components,
            "structure": structure,
            "stages": self._generate_stages(components)
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
    
    def _build_project_structure(self, components: Dict) -> Dict:
        """构建项目结构"""
        structure = {
            "files": self.base_files.copy(),
            "folders": []
        }
        
        # 根据检测到的组件添加相应的文件夹
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
        
        # 确保至少有平台和场景文件夹（大部分项目都需要）
        if "platforms" not in structure["folders"] and components["platforms"]:
            structure["folders"].append("platforms")
        if "scenarios" not in structure["folders"] and components["scenarios"]:
            structure["folders"].append("scenarios")
        
        # 排序文件夹，让常用文件夹在前面
        preferred_order = ["platforms", "scenarios", "weapons", "sensors", "processors"]
        structure["folders"] = sorted(
            structure["folders"],
            key=lambda x: (preferred_order.index(x) if x in preferred_order else len(preferred_order), x)
        )
        
        return structure
    
    def _generate_stages(self, components: Dict) -> List[Dict]:
        """生成阶段计划，从config.yaml读取参数"""
        # 从配置获取阶段定义
        config_stages = self.config.get('generation.stages', [])
        
        # 创建阶段列表
        stages = []
        
        # 首先添加项目结构阶段
        project_structure_stage = next(
            (stage for stage in config_stages if stage['name'] == 'project_structure'),
            {
                "name": "project_structure",
                "description": "分析需求并规划项目结构",
                "max_tokens": 300,
                "temperature": 0.1
            }
        )
        stages.append({
            "name": project_structure_stage["name"],
            "description": project_structure_stage["description"],
            "max_tokens": project_structure_stage.get("max_tokens", 300),
            "temperature": project_structure_stage.get("temperature", 0.1),
            "depends_on": [],
            "output_patterns": ["project_structure.json"]
        })

        # 添加主程序阶段
        main_program_stage = next(
            (stage for stage in config_stages if stage['name'] == 'main_program'),
            {
                "name": "main_program",
                "description": "生成主程序文件",
                "max_tokens": 800,
                "temperature": 0.2
            }
        )
        stages.append({
            "name": main_program_stage["name"],
            "description": main_program_stage["description"],
            "max_tokens": main_program_stage.get("max_tokens", 800),
            "temperature": main_program_stage.get("temperature", 0.2),
            "depends_on": ["project_structure"],
            "output_patterns": ["main.txt"]
        })
        
        # 根据检测到的组件添加相应阶段
        component_stage_mapping = {
            "platforms": {
                "config_name": "platforms",
                "default": {
                    "name": "platforms",
                    "description": "生成平台定义文件",
                    "max_tokens": 1200,
                    "temperature": 0.15
                }
            },
            "scenarios": {
                "config_name": "scenarios",
                "default": {
                    "name": "scenarios",
                    "description": "生成场景文件",
                    "max_tokens": 1000,
                    "temperature": 0.15
                }
            },
            "processors": {
                "config_name": "processors",
                "default": {
                    "name": "processors",
                    "description": "生成处理器文件",
                    "max_tokens": 900,
                    "temperature": 0.15
                }
            },
            "sensors": {
                "config_name": "sensors",
                "default": {
                    "name": "sensors",
                    "description": "生成传感器文件",
                    "max_tokens": 700,
                    "temperature": 0.15
                }
            },
            "weapons": {
                "config_name": "weapons",
                "default": {
                    "name": "weapons",
                    "description": "生成武器文件",
                    "max_tokens": 700,
                    "temperature": 0.15
                }
            },
            "signatures": {
                "config_name": None,  # 配置中可能没有signatures阶段
                "default": {
                    "name": "signatures",
                    "description": "生成特征信号文件",
                    "max_tokens": 600,
                    "temperature": 0.1
                }
            }
        }

        # 添加检测到的组件的阶段
        for component, has_component in components.items():
            if has_component and component in component_stage_mapping:
                mapping = component_stage_mapping[component]
                
                # 从配置获取阶段参数或使用默认值
                if mapping["config_name"]:
                    stage_config = next(
                        (stage for stage in config_stages if stage['name'] == mapping["config_name"]),
                        mapping["default"]
                    )
                else:
                    stage_config = mapping["default"]
                
                # 设置依赖关系
                depends_on = ["project_structure"]
                if component == "scenarios":
                    depends_on = ["project_structure", "platforms"]
                elif component in ["processors", "sensors", "weapons"]:
                    depends_on = ["project_structure", "platforms"]
                
                # 创建阶段对象
                stage = {
                    "name": stage_config["name"],
                    "description": stage_config["description"],
                    "max_tokens": stage_config.get("max_tokens", mapping["default"]["max_tokens"]),
                    "temperature": stage_config.get("temperature", mapping["default"]["temperature"]),
                    "depends_on": depends_on,
                    "output_patterns": [f"{stage_config['name']}/*.txt"]
                }
                
                # 检查是否已存在同名阶段
                if not any(s["name"] == stage["name"] for s in stages):
                    stages.append(stage)
            
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

    def _execute_stage(self, stage_info: Dict, query: str, output_dir: str) -> Dict:
        """执行单个生成阶段 - 使用阶段感知RAG"""
        stage_name = stage_info["name"]
        stage_max_tokens = stage_info.get("max_tokens", 1024)
        stage_temperature = stage_info.get("temperature", 0.3)
        
        try:
            self.logger.info(f"执行阶段 {stage_name}，使用阶段感知RAG...")
            
            # 构建阶段特定的查询
            stage_query = self._build_stage_query(stage_info, query)
            
            # 检查是否有阶段感知的RAG系统
            if hasattr(self.chat_system, 'generate_stage_response'):
                # 使用阶段感知RAG生成
                result = self.chat_system.generate_stage_response(
                    stage_name=stage_name,
                    query=stage_query,
                    project_context=self.project_context
                )
                
                if not result or "result" not in result:
                    return {
                        "success": False,
                        "error": "生成结果为空"
                    }
                
                generated_content = result["result"]
                
                # 从结果中提取额外信息
                if "stage_context" in result:
                    self.logger.debug(f"阶段 {stage_name} 上下文: {result['stage_context'][:200]}...")
                
            else:
                # 回退到原来的方法
                prompt = self._build_stage_prompt(stage_info, query)
                generation_params = {
                    "max_tokens": stage_max_tokens,
                    "temperature": stage_temperature
                }
                
                if hasattr(self.chat_system, 'generate_enhanced_response_with_params'):
                    rag_result = self.chat_system.generate_enhanced_response_with_params(
                        prompt, 
                        **generation_params
                    )
                else:
                    rag_result = self.chat_system.generate_enhanced_response(prompt)
                
                if not rag_result or "result" not in rag_result:
                    return {
                        "success": False,
                        "error": "生成结果为空"
                    }
                
                generated_content = rag_result["result"]
            
            # 提取文件内容
            files = self._extract_files_from_content(generated_content, stage_info, output_dir)
            
            # 保存文件
            output_files = self._save_generated_files(files, output_dir)
            
            # 更新上下文
            context = self._extract_context_from_content(generated_content)
            self.project_context.update(context)
            
            # 记录阶段特定的学习
            if hasattr(self.chat_system, 'project_learner') and stage_name in self.chat_system.project_learner.stage_learning_results:
                stage_learning = self.chat_system.project_learner.stage_learning_results[stage_name]
                self.logger.info(f"阶段 {stage_name} 使用了 {len(stage_learning.file_examples)} 个学习示例")
            
            return {
                "success": True,
                "output_files": output_files,
                "context": context,
                "raw_content": generated_content[:500] + "..." if len(generated_content) > 500 else generated_content,
                "stage_name": stage_name,
                "method": "stage_aware_rag"
            }
            
        except Exception as e:
            self.logger.error(f"执行阶段 {stage_name} 失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        
    def _build_stage_query(self, stage_info: Dict, query: str) -> str:
        """构建阶段特定的查询"""
        stage_name = stage_info["name"]
        
        # 阶段特定的查询增强
        stage_queries = {
            "project_structure": f"分析以下AFSIM项目需求并生成项目结构规划:\n{query}",
            "main_program": f"根据项目需求生成主程序文件，需求:\n{query}",
            "platforms": f"生成平台定义，基于项目需求:\n{query}\n已确定平台: {self.project_context.get('platforms', [])}",
            "scenarios": f"生成场景文件，基于项目需求:\n{query}\n可用平台: {self.project_context.get('platforms', [])}",
            "processors": f"生成处理器文件，基于项目需求:\n{query}\n平台上下文: {self.project_context.get('platforms', [])}",
            "sensors": f"生成传感器文件，基于项目需求:\n{query}\n平台上下文: {self.project_context.get('platforms', [])}",
            "weapons": f"生成武器文件，基于项目需求:\n{query}\n平台上下文: {self.project_context.get('platforms', [])}",
            "signatures": f"生成特征信号文件，基于项目需求:\n{query}\n平台类型: {self.project_context.get('platforms', [])}"
        }
        
        return stage_queries.get(stage_name, f"生成{stage_info['description']}，需求:\n{query}")
        
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
                stage_max_tokens = stage_info.get("max_tokens", 1024)
                stage_temperature = stage_info.get("temperature", 0.3)
                
                self.current_stage = stage_name
                self.logger.info(f"开始阶段 {idx}/{total_stages}: {stage_name} - {stage_desc} (tokens: {stage_max_tokens}, temp: {stage_temperature})")
                print(f"\n📋 阶段 {idx}/{total_stages}: {stage_desc}")
                print(f"   参数: max_tokens={stage_max_tokens}, temperature={stage_temperature}")
                
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
                    "duration": stage_duration,
                    "max_tokens": stage_max_tokens,
                    "temperature": stage_temperature
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
        stage_max_tokens = stage_info.get("max_tokens", 1024)
        stage_temperature = stage_info.get("temperature", 0.3)
        
        try:
            # 构建阶段特定的提示词
            prompt = self._build_stage_prompt(stage_info, query)
            
            # 设置生成参数
            generation_params = {
                "max_tokens": stage_max_tokens,
                "temperature": stage_temperature
            }
            
            # 生成内容（假设chat_system支持传入参数）
            if hasattr(self.chat_system, 'generate_enhanced_response_with_params'):
                result = self.chat_system.generate_enhanced_response_with_params(
                    prompt, 
                    **generation_params
                )
            else:
                # 回退到默认方法
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
                "stage_name": stage_name,
                "generation_params": generation_params
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
        
        # 使用配置中的参数构建提示词
        stage_instructions = {
            "project_structure": f"""分析AFSIM项目需求并规划项目结构。

原始需求：{query}

输出一个JSON格式的项目结构规划，包含以下信息：
1. 需要的组件列表
2. 建议的文件结构
3. 主要平台名称
4. 主要场景描述

**只输出AFSIM代码格式**。""",
            
            "signatures": f"""生成AFSIM特征信号文件。

基于项目需求：{query}
项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}
已生成的文件：{chr(10).join(f"- {f}" for f in self.generated_files)}

需要生成以下特征信号定义：
1. 雷达特征（radar_signature）
2. 红外特征（infrared_signature）  
3. 光学特征（optical_signature）

每个特征信号文件应该包含：
1. 特征类型定义
2. RCS值或辐射强度
3. 角度依赖性
4. 双基地特征（如适用）

为每种特征信号生成单独的.txt文件。
使用以下格式分隔不同文件：
=== Feature_Name_signature.txt ===
[特征信号文件内容]

开始生成：""",
            
            "main_program": f"""生成AFSIM主程序文件（main.txt）。

基于以下项目需求：{query}

项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}

已生成的文件：{chr(10).join(f"- {f}" for f in self.generated_files)}

主程序必须包含：
1. 必要的include导入语句
2. 全局变量和常量定义
3. 场景初始化和设置
4. 主事件循环
5. 输出配置
6. 仿真控制参数

生成完整的main.txt内容：""",
            
            "platforms": f"""生成AFSIM平台定义文件。

基于项目需求：{query}

项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}
已生成的文件：{chr(10).join(f"- {f}" for f in self.generated_files)}

需要生成以下平台的定义：
{self._get_platform_requirements()}

每个平台文件应该包含：
1. 平台类型定义（platform_type）
2. 物理参数（尺寸、重量、动力等）
3. 初始状态（位置、速度、方向）
4. 组件配置（传感器、武器、处理器等）
5. 行为定义
6. 特征信号引用（如适用）

为每个平台生成单独的.txt文件。
使用以下格式分隔不同文件：
=== Platform_Name.txt ===
[平台文件内容]
=== 另一个平台.txt ===
[另一个平台文件内容]

开始生成：""",
            
            "scenarios": f"""生成AFSIM场景文件。

基于项目需求：{query}
项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}
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

为每个场景生成单独的.txt文件。
使用以下格式分隔不同文件：
=== Scene_Name.txt ===
[场景文件内容]

开始生成：""",
            
            "processors": f"""生成AFSIM处理器文件。

基于项目需求：{query}
项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}
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

为每个处理器生成单独的.txt文件。
使用以下格式分隔不同文件：
=== Processor_Name.txt ===
[处理器文件内容]

开始生成：""",
            
            "sensors": f"""生成AFSIM传感器文件。

基于项目需求：{query}
项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}
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

为每个传感器生成单独的.txt文件。
使用以下格式分隔不同文件：
=== Sensor_Name.txt ===
[传感器文件内容]

开始生成：""",
            
            "weapons": f"""生成AFSIM武器文件。

基于项目需求：{query}
项目上下文：{json.dumps(self.project_context, ensure_ascii=False)}
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

为每个武器生成单独的.txt文件。
使用以下格式分隔不同文件：
=== Weapon_Name.txt ===
[武器文件内容]

开始生成："""
        }
        
        instruction = stage_instructions.get(stage_name, f"根据需求生成{stage_desc}。")
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
                           "signatures"]:
            # 使用智能文件分割
            files.extend(self._extract_multiple_files_smart(content, stage_name))
    
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
            fr'platform_type\s+{platform}.*?\n}}(?=\n|$)' if '}' in content else fr'platform_type\s+{platform}.*?(?=\nplatform_type|\Z)',
            fr'class\s+{platform}.*?\n}}(?=\n|$)' if '}' in content else fr'class\s+{platform}.*?(?=\nclass|\Z)',
            fr'{platform}_platform.*?\n}}(?=\n|$)' if '}' in content else fr'{platform}_platform.*?(?=\n\w+_platform|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group().strip()
        
        return ""
    
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
                "avg_stage_duration": total_duration / max(len(self.current_project["stages"]), 1),
                "stage_params": {
                    stage_name: {
                        "max_tokens": stage_info.get("max_tokens"),
                        "temperature": stage_info.get("temperature")
                    }
                    for stage_name, stage_info in self.current_project["stages"].items()
                }
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