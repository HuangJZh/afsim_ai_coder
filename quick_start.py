# quick_start.py
"""
AFSIM RAG系统快速启动脚本
直接运行此文件即可启动系统
"""
import os
import sys

def check_dependencies():
    """检查依赖"""
    required = ['torch', 'transformers', 'sentence_transformers', 'chromadb']
    missing = []
    
    for lib in required:
        try:
            __import__(lib.replace('-', '_'))
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        response = input("是否安装? (y/n): ").strip().lower()
        if response == 'y':
            import subprocess
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["accelerate", "bitsandbytes"])
                print("安装完成！")
            except Exception as e:
                print(f"安装失败: {e}")
                return False
        else:
            return False
    
    return True

def main():
    """主函数"""
    print("="*60)
    print("AFSIM RAG 代码生成系统")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        print("请手动安装依赖后重试")
        input("按回车键退出...")
        return
    
    # 检查教程文件夹
    tutorials_dir = "tutorials"
    if not os.path.exists(tutorials_dir):
        print(f"❌ 找不到教程文件夹: {tutorials_dir}")
        print("请确保教程文件夹存在")
        input("按回车键退出...")
        return
    
    # 列出教程文件
    md_files = []
    for root, dirs, files in os.walk(tutorials_dir):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    
    if not md_files:
        print(f"❌ 教程文件夹中没有找到.md文件")
        input("按回车键退出...")
        return
    
    print(f"✅ 找到 {len(md_files)} 个教程文件:")
    for file in md_files[:10]:
        print(f"  • {os.path.basename(file)}")
    if len(md_files) > 10:
        print(f"  ... 还有 {len(md_files)-10} 个文件")
    
    print("\n选择运行模式:")
    print("1. 扫描并索引文档 (首次运行)")
    print("2. 命令行交互模式")
    print("3. 退出")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == "1":
        print("正在初始化系统...")
        try:
            from rag_afsim_system import AFSIMRAGSystem
            system = AFSIMRAGSystem()
            print("正在扫描教程文件夹...")
            success = system.load_documents_from_folder(tutorials_dir)
            if success:
                print(f"✅ 成功加载 {system.collection.count()} 个文档块")
                print("现在可以运行交互模式了")
                input("按回车键继续...")
                system.interactive_chat()
            else:
                print("❌ 加载文档失败")
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "2":
        print("启动交互模式...")
        try:
            from rag_afsim_system import AFSIMRAGSystem
            system = AFSIMRAGSystem()
            # 检查文档是否已加载
            if system.collection.count() == 0:
                print("数据库为空，建议先扫描文档")
                response = input("是否现在扫描? (y/n): ").strip().lower()
                if response == 'y':
                    system.load_documents_from_folder(tutorials_dir)
            system.interactive_chat()
        except Exception as e:
            print(f"❌ 启动失败: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "3":
        print("退出系统")
        return
    
    else:
        print("无效选择")
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()