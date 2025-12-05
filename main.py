import os
import sys
import warnings
from typing import Dict, Any

# 解决OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 过滤警告
warnings.filterwarnings("ignore")

import time
import json
from pathlib import Path

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_enhanced import EnhancedStageAwareRAGChatSystem
from multi_stage_generator import MultiStageChatSystem  
from utils import setup_logging, ConfigManager

def check_environment():
    """检查环境配置"""
    print("检查环境配置...")
    
    # 检查必要的库
    try:
        import torch
        import transformers
        import langchain
        import yaml
        import tqdm
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Transformers: {transformers.__version__}")
        print(f"✅ LangChain: {langchain.__version__}")
        print(f"✅ PyYAML: {yaml.__version__}")
        print(f"✅ TQDM: {tqdm.__version__}")
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e}")
        return False
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA不可用，将使用CPU")
    
    return True

def collect_project_requirements():
    """收集项目需求"""
    print("\n" + "=" * 80)
    print("AFSIM 多阶段项目生成系统")
    print("=" * 80)
    
    print("\n请输入AFSIM项目需求（支持多行输入）：")
    print("示例：创建一个战斗机空战模拟项目，包含F-22和SU-35平台")
    print("     配置雷达、导弹系统，创建空战场景")
    print("     实现飞行控制和战术处理器")
    print("提示：输入 //end 结束输入")
    print("-" * 80)
    
    requirements = []
    print("开始输入：")
    
    line_count = 0
    while True:
        try:
            line = input().strip()
            # 检查是否输入了结束命令
            if line == "//end":
                if line_count == 0:
                    print("⚠️  尚未输入任何内容，请至少输入一行需求")
                    continue
                break
            if line:
                requirements.append(line)
                line_count += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  输入中断，返回主菜单")
            return "CANCEL"  # 返回特殊标记表示用户取消
        except EOFError:
            print("\n\n输入结束")
            if line_count > 0:
                break
            return None
    
    if not requirements:
        print("❌ 需求不能为空")
        return None
    
    return "\n".join(requirements)

def display_generated_project(result, total_time):
    """显示生成的项目结果"""
    print("\n" + "=" * 80)
    print("生成完成！")
    print("=" * 80)
    
    if result.get("success"):
        print(f"✅ 项目生成成功！")
        print(f"📁 项目位置: {result['project_dir']}")
        print(f"📄 生成文件数: {len(result.get('generated_files', []))}")
        print(f"⏱️  总耗时: {total_time:.1f}秒")
        
        # 显示文件列表
        print("\n📋 生成的文件:")
        print("-" * 40)
        for i, file_path in enumerate(result.get('generated_files', []), 1):
            print(f"  {i:2d}. {file_path}")
        
        # 显示阶段统计
        if result.get("report", {}).get("summary"):
            summary = result["report"]["summary"]
            print(f"\n📊 阶段统计:")
            print(f"   成功阶段: {summary['successful_stages']}/{summary['total_stages']}")
            print(f"   平均阶段耗时: {summary['avg_stage_duration']:.1f}秒")
        
        # 显示阶段详情
        if result.get("report", {}).get("stage_results"):
            print(f"\n📈 阶段详情:")
            for stage_name, stage_info in result["report"]["stage_results"].items():
                status_icon = "✅" if stage_info.get("status") == "success" else "❌"
                files_count = len(stage_info.get("output_files", []))
                print(f"   {status_icon} {stage_name}: {stage_info.get('duration', 0):.1f}秒, {files_count}个文件")
        
        print(f"\n💡 提示: 请查看 {result['project_dir']} 文件夹获取完整项目。")
        
        return True
    else:
        print(f"❌ 项目生成失败")
        if result.get("error"):
            print(f"错误: {result['error']}")
        return False

def show_main_menu():
    """显示主菜单"""
    print("\n" + "=" * 80)
    print("AFSIM 智能代码生成系统")
    print("=" * 80)
    print("1. 🚀 生成完整项目 (多阶段)")
    print("2. 💬 单轮对话模式")
    print("3. 📊 查看系统信息")
    print("4. 🔍 测试检索功能")
    print("5. 📚 查看学习结果")
    print("6. ⚙️  系统设置")
    print("0. 🚪 退出系统")
    print("-" * 80)
    
    choice = input("请选择操作 (0-6): ").strip()
    return choice

def show_system_info(chat_system):
    """显示系统信息"""
    print("\n" + "=" * 80)
    print("系统信息")
    print("=" * 80)
    
    try:
        # 获取项目信息
        project_info = chat_system.get_project_info()
        if project_info:
            print(f"📁 项目信息:")
            print(f"   文件总数: {project_info.get('total_files', 0)}")
            print(f"   基础库: {', '.join(project_info.get('base_libraries', []))}")
            print(f"   代码模块: {len(project_info.get('code_modules', []))}")
        
        # 获取向量数据库信息
        vector_db_info = chat_system.get_vector_db_info()
        print(f"\n📊 向量数据库: {vector_db_info}")
        
        # 如果有阶段学习功能，显示学习结果
        if hasattr(chat_system, 'project_learner'):
            print(f"\n🎓 学习模块:")
            if hasattr(chat_system.project_learner, 'get_stage_learning_summary'):
                try:
                    learning_summary = chat_system.project_learner.get_stage_learning_summary()
                    print(f"   从 {learning_summary.get('total_demo_projects', 0)} 个demo项目中学习")
                    print(f"   已学习阶段: {len(learning_summary.get('stages_learned', []))}")
                    
                    # 显示阶段详情
                    if learning_summary.get('stage_details'):
                        print(f"\n   阶段详情:")
                        for stage_name, details in learning_summary['stage_details'].items():
                            print(f"     {stage_name}: {details.get('example_count', 0)} 个示例")
                except Exception as e:
                    print(f"   学习模块暂时不可用: {e}")
        
        # 显示模型信息
        print(f"\n🤖 模型信息:")
        config = ConfigManager()
        model_path = config.get('model.path', '未知')
        print(f"   模型路径: {model_path}")
        
    except Exception as e:
        print(f"获取系统信息失败: {e}")

def test_retrieval(chat_system):
    """测试检索功能"""
    print("\n" + "=" * 80)
    print("测试检索功能")
    print("=" * 80)
    
    query = input("请输入测试查询: ").strip()
    if not query:
        print("❌ 查询不能为空")
        return
    
    print(f"\n🔍 测试查询: {query}")
    print("正在检索相关文档...")
    
    try:
        # 使用基础检索
        if hasattr(chat_system, 'search_similar_documents_cached'):
            start_time = time.time()
            docs = chat_system.search_similar_documents_cached(query, top_k=3)
            end_time = time.time()
            
            print(f"✅ 检索完成 (耗时: {end_time - start_time:.3f}秒)")
            print(f"📄 找到 {len(docs)} 个相关文档:")
            
            for i, doc in enumerate(docs, 1):
                print(f"\n  [{i}] 文档片段:")
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                print(f"     {content_preview}")
                
                if hasattr(doc, 'metadata'):
                    source = doc.metadata.get('source', '未知')
                    print(f"     来源: {source}")
        
        # 测试阶段感知检索
        print(f"\n🎯 阶段感知检索测试:")
        stages = ["platforms", "weapons", "sensors", "scenarios"]
        for stage in stages:
            if hasattr(chat_system, 'get_stage_aware_qa_chain'):
                try:
                    qa_chain = chat_system.get_stage_aware_qa_chain(stage)
                    docs = qa_chain.retriever.get_relevant_documents(query)
                    print(f"   {stage}: {len(docs)} 个文档")
                except:
                    print(f"   {stage}: 不可用")
        
    except Exception as e:
        print(f"❌ 检索测试失败: {e}")

def show_learning_results(chat_system):
    """显示学习结果"""
    print("\n" + "=" * 80)
    print("阶段学习结果")
    print("=" * 80)
    
    if not hasattr(chat_system, 'project_learner'):
        print("❌ 系统未启用学习功能")
        return
    
    try:
        if hasattr(chat_system.project_learner, 'get_stage_learning_summary'):
            learning_summary = chat_system.project_learner.get_stage_learning_summary()
            
            print(f"📊 学习摘要:")
            print(f"   Demo项目数: {learning_summary.get('total_demo_projects', 0)}")
            print(f"   已学习阶段: {len(learning_summary.get('stages_learned', []))}")
            
            if learning_summary.get('stage_details'):
                print(f"\n📈 阶段详情:")
                for stage_name, details in learning_summary['stage_details'].items():
                    example_count = details.get('example_count', 0)
                    pattern_count = details.get('pattern_count', 0)
                    print(f"   {stage_name}:")
                    print(f"     示例文件: {example_count}")
                    print(f"     学习模式: {pattern_count}")
                    print(f"     导入模式: {details.get('import_patterns', 0)}")
                    print(f"     最佳实践: {len(details.get('best_practices', []))}")
            
            # 显示具体阶段的上下文示例
            print(f"\n🔍 阶段上下文示例:")
            stages_to_show = ["platforms", "main_program", "scenarios"]
            for stage in stages_to_show:
                if stage in learning_summary.get('stages_learned', []):
                    context = chat_system.project_learner.get_stage_context(stage, "测试查询")
                    if context:
                        print(f"\n   [{stage}] 上下文摘要:")
                        lines = context.split('\n')[:5]
                        for line in lines:
                            print(f"      {line}")
        
        # 如果有demo项目信息
        if hasattr(chat_system.project_learner, 'demo_projects'):
            demo_projects = chat_system.project_learner.demo_projects
            if demo_projects:
                print(f"\n📁 Demo项目列表:")
                for project in demo_projects[:5]:  # 只显示前5个
                    print(f"   {project.get('name')}: {len(project.get('files', {}))} 个文件")
                if len(demo_projects) > 5:
                    print(f"   ... 还有 {len(demo_projects) - 5} 个项目")
        
    except Exception as e:
        print(f"❌ 获取学习结果失败: {e}")

def system_settings():
    """系统设置"""
    print("\n" + "=" * 80)
    print("系统设置")
    print("=" * 80)
    
    config = ConfigManager()
    
    while True:
        print("\n当前配置:")
        print(f"1. 模型路径: {config.get('model.path')}")
        print(f"2. 项目根目录: {config.get('project.root')}")
        print(f"3. 生成策略: {config.get('generation.strategy')}")
        print(f"4. 向量数据库目录: {config.get('vector_db.persist_dir')}")
        print(f"5. 日志级别: {config.get('logging.level')}")
        print("6. 查看完整配置")
        print("0. 返回主菜单")
        
        choice = input("\n请选择要修改的配置 (0-6): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            new_path = input("请输入新的模型路径: ").strip()
            if new_path and os.path.exists(new_path):
                config.set('model.path', new_path)
                print("✅ 模型路径已更新")
            else:
                print("❌ 路径无效")
        elif choice == "2":
            new_root = input("请输入新的项目根目录: ").strip()
            if new_root and os.path.exists(new_root):
                config.set('project.root', new_root)
                print("✅ 项目根目录已更新")
            else:
                print("❌ 路径无效")
        elif choice == "6":
            print("\n完整配置:")
            config_data = config.get_all()
            for key, value in config_data.items():
                print(f"{key}: {value}")

def single_chat_mode(chat_system):
    """单轮对话模式"""
    print("\n" + "=" * 80)
    print("单轮对话模式")
    print("=" * 80)
    print("输入 'quit' 或 'exit' 返回主菜单")
    print("-" * 80)
    
    while True:
        print("\n💭 请输入您的问题:")
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q', '0']:
            print("返回主菜单...")
            break
        
        if not user_input:
            continue
        
        print("🤔 正在思考...")
        start_time = time.time()
        
        try:
            # 使用增强的响应生成
            response = chat_system.generate_enhanced_response(user_input)
            
            end_time = time.time()
            
            print(f"\n💡 回答 (耗时: {end_time - start_time:.2f}秒):")
            print("-" * 50)
            print(response.get("result", "未生成回答"))
            print("-" * 50)
            
            # 显示来源信息
            if response.get("source_documents"):
                print(f"\n📚 参考了 {len(response['source_documents'])} 个文档片段")
                if len(response['source_documents']) > 0:
                    print("   来源示例:")
                    for i, doc in enumerate(response['source_documents'][:2], 1):
                        if hasattr(doc, 'metadata'):
                            source = doc.metadata.get('source', '未知')
                            print(f"     {i}. {os.path.basename(source)}")
            
            # 显示验证结果
            if response.get("validation"):
                validation = response["validation"]
                if validation.get("is_valid"):
                    print("✅ 代码验证通过")
                else:
                    print(f"⚠️  代码验证警告:")
                    for error in validation.get('errors', [])[:3]:
                        print(f"     - {error}")
            
        except Exception as e:
            print(f"❌ 生成回答时出错: {e}")

def main():
    """主函数"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请安装必要的依赖")
        return
    
    # 设置日志
    setup_logging()
    
    # 加载配置
    config = ConfigManager()
    
    try:
        # 检查项目路径
        project_root = config.get('project.root')
        if not os.path.exists(project_root):
            print(f"❌ 项目路径不存在: {project_root}")
            print("请检查config.yaml中的project.root配置")
            return
        
        # 获取模型路径
        model_path = config.get('model.path')
        if not os.path.exists(model_path):
            print(f"⚠️  模型路径不存在: {model_path}")
            print("请检查config.yaml中的model.path配置")
            use_default = input("是否使用默认路径? (y/n): ").strip().lower()
            if use_default != 'y':
                return
            # 尝试使用相对路径
            model_path = None
        
        print(f"\n✅ 环境检查通过")
        print(f"📁 项目路径: {project_root}")
        print(f"🤖 模型路径: {model_path or '使用配置默认值'}")
        
        # 初始化聊天系统
        print("\n🔧 正在初始化系统...")
        start_init = time.time()
        
        # 根据配置选择系统
        strategy = config.get('generation.strategy', 'multi_stage')
        enable_stage_aware = config.get('generation.enable_stage_aware_rag', True)
        
        if strategy == 'multi_stage' and enable_stage_aware:
            print("🚀 初始化阶段感知多阶段生成系统...")
            chat_system = MultiStageChatSystem(
                project_root=project_root,
                model_path=model_path
            )
        else:
            print("💬 初始化单轮对话系统...")
            chat_system = EnhancedStageAwareRAGChatSystem(
                project_root=project_root,
                model_path=model_path,
                embedding_model=config.get('embedding.model')
            )
        
        end_init = time.time()
        print(f"✅ 系统初始化完成 (耗时: {end_init - start_init:.1f}秒)")
        
        # 显示学习结果
        if hasattr(chat_system, 'project_learner'):
            try:
                learning_summary = chat_system.project_learner.get_stage_learning_summary()
                print(f"🎓 从 {learning_summary.get('total_demo_projects', 0)} 个demo项目中学习了 {len(learning_summary.get('stages_learned', []))} 个阶段")
            except:
                print("ℹ️  未启用阶段学习功能")
        
        # 主菜单循环
        while True:
            choice = show_main_menu()
            
            if choice == "0":
                print("\n👋 感谢使用AFSIM智能代码生成系统，再见！")
                break
            
            elif choice == "1":
                # 生成完整项目
                requirements = collect_project_requirements()
                if requirements == "CANCEL":
                    continue
                elif not requirements:
                    print("❌ 未输入有效需求")
                    continue
                
                print(f"\n🔍 分析需求: {requirements[:100]}...")
                print("🚀 开始多阶段项目生成...")
                
                start_time = time.time()
                
                try:
                    # 使用多阶段生成
                    if isinstance(chat_system, MultiStageChatSystem):
                        result = chat_system.generate_complete_project(requirements)
                    else:
                        # 如果当前不是多阶段系统，创建临时实例
                        multi_stage_system = MultiStageChatSystem(
                            project_root=project_root,
                            model_path=model_path
                        )
                        result = multi_stage_system.generate_complete_project(requirements)
                    
                    end_time = time.time()
                    
                    display_generated_project(result, end_time - start_time)
                    
                except Exception as e:
                    print(f"❌ 项目生成失败: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif choice == "2":
                # 单轮对话模式
                single_chat_mode(chat_system)
            
            elif choice == "3":
                # 查看系统信息
                show_system_info(chat_system)
            
            elif choice == "4":
                # 测试检索功能
                test_retrieval(chat_system)
            
            elif choice == "5":
                # 查看学习结果
                show_learning_results(chat_system)
            
            elif choice == "6":
                # 系统设置
                system_settings()
            
            else:
                print("❌ 无效选择，请重新输入")
            
            # 等待用户确认继续
            if choice != "0":
                input("\n按Enter键继续...")
    
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出系统")
    except Exception as e:
        print(f"\n❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()