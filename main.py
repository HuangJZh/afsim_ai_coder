# main.py (优化版)
import os
import sys
import warnings

# 解决OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 过滤警告
warnings.filterwarnings("ignore")

# 添加当前目录到Python路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from rag_enhanced import EnhancedRAGChatSystem, EnhancedInputHandler
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

def main():
    """主函数"""
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
    project_root = config.get('project.root')  # 请根据实际情况修改
    model_path = config.get('model.path')
    
    if not os.path.exists(project_root):
        print(f"❌ 错误: 项目目录不存在: {project_root}")
        print("请修改 main.py 中的 project_root 变量为正确的路径")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型目录不存在: {model_path}")
        print("请修改配置文件中的 model.path 或确保模型路径正确")
        return
    
    try:
        # 初始化增强系统
        print("\n正在初始化系统...")
        chat_system = EnhancedRAGChatSystem(
            project_root=project_root,
            model_path=model_path
        )
        
        print("✅ 系统初始化完成！")
        
        # 显示项目摘要
        project_info = chat_system.get_project_info()
        print(f"\n 项目摘要:")
        print(f"  总文件数: {project_info['total_files']}")
        print(f"  基础库: {', '.join(project_info['base_libraries'])}")
        print(f"  代码模块: {len(project_info['code_modules'])} 个")
        print(f"  数据总量: {project_info['total_size'] // 1024} KB")
        
        # 显示数据库信息
        db_info = chat_system.get_vector_db_info()
        print(f" {db_info}")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 启动输入处理器
    input_handler = EnhancedInputHandler(chat_system)
    input_handler.start_input_listener()
    
    # 保存分析报告（可选）
    try:
        report_path = "project_analysis_report.json"
        chat_system.project_learner.save_analysis_report(report_path)
        print(f" 项目分析报告已保存: {report_path}")
    except Exception as e:
        print(f"⚠️  保存分析报告失败: {e}")
    
    print(f"\n 系统已就绪，可以开始输入AFSIM代码生成需求！")
    print(f"{'='*80}\n")
    
    try:
        # 主循环
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n\n{'='*20}")
        print("感谢使用 AFSIM 智能代码生成系统！再见！")
        print(f"{'='*20}")
    finally:
        # 确保资源清理
        if 'chat_system' in locals():
            del chat_system

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()