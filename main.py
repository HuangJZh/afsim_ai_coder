import os
import sys
import warnings
from typing import Dict

# 解决OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 过滤警告
warnings.filterwarnings("ignore")

import time
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
    print("-" * 80)
    
    requirements = []
    print("开始输入（输入空行结束）：")
    
    line_count = 0
    while True:
        try:
            line = input().strip()
            if line == "" and line_count > 0:
                break
            if line:
                requirements.append(line)
                line_count += 1
        except KeyboardInterrupt:
            print("\n\n输入中断")
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
        
        print(f"\n💡 提示: 请查看 {result['project_dir']} 文件夹获取完整项目。")
        
        return True
    else:
        print(f"❌ 项目生成失败")
        if result.get("error"):
            print(f"错误: {result['error']}")
        return False

def view_file_content(project_dir, files):
    """查看特定文件的内容"""
    print("\n可查看的文件:")
    for i, file_path in enumerate(files[:10], 1):
        print(f"  {i:2d}. {file_path}")
    
    try:
        choice = input("\n请输入文件编号（0跳过）: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            file_idx = int(choice) - 1
            file_path = files[file_idx]
            full_path = os.path.join(project_dir, file_path)
            
            if os.path.exists(full_path):
                print(f"\n{'='*60}")
                print(f"文件内容: {file_path}")
                print('='*60)
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content[:1000])  # 只显示前1000字符
                    if len(content) > 1000:
                        print(f"\n... (文件过长，只显示前1000字符)")
                print('='*60)
    except:
        pass

def main():
    """主函数"""
    print("=" * 80)
    print("AFSIM 智能代码生成系统 (多阶段生成版)")
    print("=" * 80)
    
    # 设置日志
    setup_logging()
    
    # 检查环境
    if not check_environment():
        print("环境检查失败，请安装必要的依赖库")
        return
    
    # 加载配置
    config = ConfigManager()
    
    # 配置路径
    project_root = config.get('project.root')
    model_path = config.get('model.path')
    
    if not os.path.exists(project_root):
        print(f"❌ 错误: 项目目录不存在: {project_root}")
        print("请修改配置文件中的 project.root")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型目录不存在: {model_path}")
        print("请修改配置文件中的 model.path")
        return
    
    try:
        # 初始化多阶段生成系统 - 现在只需要导入一个类
        print("\n正在初始化多阶段生成系统...")
        chat_system = MultiStageChatSystem(
            project_root=project_root,
            model_path=model_path
        )
        
        print("✅ 多阶段生成系统初始化完成！")
        
        # 显示系统信息
        project_info = chat_system.get_project_info()
        print(f"\n📊 项目知识库统计:")
        print(f"   总文件数: {project_info['total_files']}")
        print(f"   基础库: {', '.join(project_info['base_libraries'])}")
        print(f"   知识库: {chat_system.get_vector_db_info()}")
        
        # 主循环 (保持不变)
        while True:
            # 收集项目需求
            query = collect_project_requirements()
            if query is None:
                continue
            
            # 确认生成
            print(f"\n需求摘要: {query[:100]}...")
            confirm = input("\n是否开始生成项目？(y/n): ").strip().lower()
            if confirm != 'y' and confirm != 'yes':
                print("生成取消")
                continue
            
            # 生成项目
            print("\n" + "=" * 80)
            print("开始生成AFSIM项目...")
            print("=" * 80)
            
            start_time = time.time()
            result = chat_system.generate_complete_project(query)
            total_time = time.time() - start_time
            
            # 显示结果
            success = display_generated_project(result, total_time)
            
            # 如果生成成功，询问是否查看文件内容
            if success:
                view_files = input("\n是否查看某个文件的具体内容？(y/n): ").strip().lower()
                if view_files == 'y' or view_files == 'yes':
                    view_file_content(result['project_dir'], result.get('generated_files', []))
            
            print("\n" + "=" * 80)
            
            # 询问是否继续
            continue_gen = input("\n是否生成另一个项目？(y/n): ").strip().lower()
            if continue_gen != 'y' and continue_gen != 'yes':
                print("\n感谢使用 AFSIM 智能代码生成系统！再见！")
                break
            
    except KeyboardInterrupt:
        print(f"\n\n程序被用户中断")
        print("感谢使用 AFSIM 智能代码生成系统！再见！")
    except Exception as e:
        print(f"❌ 系统运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保资源清理
        if 'chat_system' in locals():
            del chat_system
        # 清理GPU内存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()