# main.py
#!/usr/bin/env python3
"""
AFSIM RAG系统主启动脚本（兼容Gradio 3.x）
"""
import argparse
import sys
import os
from utils import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="AFSIM RAG代码生成系统")
    parser.add_argument("--mode", choices=["cli", "web", "index", "test"], 
                       default="web", help="运行模式")
    parser.add_argument("--query", type=str, help="CLI模式下的查询")
    parser.add_argument("--docs", type=str, default=None,
                       help="文档列表文件路径")
    parser.add_argument("--db", type=str, default=None,
                       help="Chroma数据库路径")
    parser.add_argument("--port", type=int, default=None,
                       help="Web服务器端口")
    parser.add_argument("--share", action="store_true",
                       help="是否生成公网分享链接")
    
    args = parser.parse_args()
    
    # 获取配置
    config = ConfigManager()
    
    # 使用配置或命令行参数
    docs_path = args.docs or config.get('paths.documents_list', 'tree_of_tutorials.txt')
    db_path = args.db or config.get('database.chroma_path', './chroma_db')
    port = args.port or config.get('web.port', 7860)
    
    # 导入必要的模块
    try:
        from rag_afsim_system import AFSIMRAGSystem
    except ImportError as e:
        print(f"导入模块失败: {e}")
        print("请确保rag_afsim_system.py在同一个目录下")
        sys.exit(1)
    
    if args.mode == "index":
        print("开始索引文档...")
        system = AFSIMRAGSystem(chroma_db_path=db_path)
        system.load_documents_from_folder(docs_path)
        print("文档索引完成！")
    
    elif args.mode == "cli":
        system = AFSIMRAGSystem(chroma_db_path=db_path)
        
        if args.query:
            print(f"查询: {args.query}")
            result = system.generate_response(args.query)
            print("\n" + "="*60)
            print("回答：")
            print(result["response"])
            print("\n来源：")
            for source in result["sources"]:
                print(f"  - {source}")
            print("="*60)
        else:
            print("交互式CLI模式（输入'exit'退出）")
            print("注意：首次运行需要先加载文档，输入 'load' 加载文档")
            
            # 检查文档是否已加载
            doc_count = system.collection.count()
            if doc_count == 0:
                print(f"⚠ 数据库中有 {doc_count} 个文档，建议先加载文档")
            
            while True:
                try:
                    query = input("\n请输入问题: ").strip()
                    if query.lower() in ["exit", "quit", "q"]:
                        print("再见！")
                        break
                    elif query.lower() == "load":
                        print("正在加载文档...")
                        system.load_documents_from_folder(docs_path)
                        print("文档加载完成！")
                        continue
                    elif not query:
                        continue
                    
                    result = system.generate_response(query)
                    print("\n" + "="*60)
                    print(result["response"])
                    print("-"*40)
                    print("参考来源:")
                    for source in result["sources"]:
                        print(f"  • {source}")
                    print("="*60)
                    
                except KeyboardInterrupt:
                    print("\n程序已中断")
                    break
                except Exception as e:
                    print(f"错误: {e}")
    
    elif args.mode == "web":
        print(f"启动Web界面... 访问 http://localhost:{port}")
        try:
            from app import launch_app
            launch_app(share=args.share, port=port)
        except ImportError:
            print("错误: 找不到app.py")
            print("请确保app.py在同一个目录下")
            sys.exit(1)
    
    elif args.mode == "test":
        print("运行系统测试...")
        test_system()

def test_system():
    """测试系统功能"""
    from rag_afsim_system import AFSIMRAGSystem
    from utils import ConfigManager
    
    config = ConfigManager()
    
    print("1. 初始化系统...")
    system = AFSIMRAGSystem(
        chroma_db_path="./chroma_db_test"
    )
    
    print("2. 加载文档...")
    docs_path = config.get('paths.tutorials_folder', 'tutorials')
    system.load_documents_from_folder(docs_path)
    
    print("3. 测试查询...")
    test_queries = [
        "什么是AFSIM?",
        "如何创建移动平台?"
    ]
    
    for query in test_queries:
        print(f"\n测试查询: {query}")
        result = system.generate_response(query)
        print(f"响应长度: {len(result['response'])} 字符")
        print(f"参考来源: {result['sources']}")
        print("-"*50)
    
    print("测试完成！")

if __name__ == "__main__":
    main()