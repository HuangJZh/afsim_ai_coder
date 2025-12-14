import gradio as gr
from rag_afsim_system import AFSIMRAGSystem
import json
import os
from loguru import logger

# 初始化系统
system = None

def initialize_system():
    """初始化RAG系统"""
    global system
    try:
        system = AFSIMRAGSystem(
            model_path="D:/Qwen/Qwen/Qwen3-4B",
            embedding_model="BAAI/bge-small-zh-v1.5",
            chroma_db_path="./chroma_db"
        )
        return "✅ 系统初始化成功！"
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return f"❌ 初始化失败: {str(e)}"

def load_documents(file_list_path):
    """加载文档"""
    if system is None:
        return "请先初始化系统"
    
    try:
        system.load_documents(file_list_path, base_dir=".")
        return "✅ 文档加载完成！"
    except Exception as e:
        return f"❌ 文档加载失败: {str(e)}"

def query_afsim(query, history=None):
    """处理查询"""
    if system is None:
        return "请先初始化系统", []
    
    try:
        result = system.generate_response(query)
        
        # 格式化显示
        response = f"{result['response']}\n\n"
        
        if result['sources']:
            response += "**参考来源:**\n"
            for source in result['sources']:
                response += f"- {source}\n"
        
        # 更新历史记录
        if history is None:
            history = []
        
        history.append((query, response))
        
        return response, history
        
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        logger.error(error_msg)
        return error_msg, history

# 创建Gradio界面
with gr.Blocks(title="AFSIM RAG代码生成系统") as demo:
    gr.Markdown("# 🚀 AFSIM RAG增强代码生成系统")
    gr.Markdown("基于Qwen3-4B + BGE嵌入 + Chroma构建的AFSIM智能助手")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 系统控制")
            
            init_btn = gr.Button("初始化系统", variant="primary")
            init_status = gr.Markdown("等待初始化...")
            
            file_input = gr.Textbox(
                label="文档列表文件路径",
                value="tree_of_tutorials.txt",
                placeholder="输入文档列表文件路径"
            )
            load_btn = gr.Button("加载文档", variant="secondary")
            load_status = gr.Markdown("")
            
            gr.Markdown("### 示例查询")
            examples = [
                "如何创建AFSIM移动平台？",
                "如何配置传感器参数？",
                "生成一个武器系统控制的示例代码",
                "如何可视化仿真结果？",
                "解释AFSIM中的路由配置"
            ]
            
            example_selector = gr.Examples(
                examples=examples,
                inputs=[gr.Textbox(visible=False)]  # 这将绑定到聊天输入
            )
            
            # 系统信息显示
            gr.Markdown("### 系统信息")
            info_text = gr.Textbox(
                label="状态",
                value="等待初始化...",
                interactive=False
            )
            
        with gr.Column(scale=3):
            gr.Markdown("## AFSIM助手")
            
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="输入你的AFSIM相关问题",
                    placeholder="例如：如何创建AFSIM移动平台？",
                    scale=4,
                    lines=2
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("清空对话", variant="secondary", scale=1)
                export_btn = gr.Button("导出历史", variant="secondary", scale=1)
    
    # 事件绑定
    def on_init():
        status = initialize_system()
        return status, status
    
    init_btn.click(
        fn=on_init,
        outputs=[init_status, info_text]
    )
    
    def on_load(file_path):
        if system is None:
            return "请先初始化系统", "系统未初始化"
        status = load_documents(file_path)
        return status, status
    
    load_btn.click(
        fn=on_load,
        inputs=file_input,
        outputs=[load_status, info_text]
    )
    
    def clear_chat():
        return [], "", "对话已清空"
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, msg, info_text]
    )
    
    def export_chat(history):
        """导出对话历史为JSON"""
        if not history:
            return "没有对话历史可导出"
        
        export_data = []
        for q, a in history:
            export_data.append({
                "question": q,
                "answer": a[:500] + "..." if len(a) > 500 else a
            })
        
        # 保存到文件
        import time
        filename = f"afsim_chat_history_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return f"✅ 对话历史已导出到: {filename}"
    
    export_btn.click(
        fn=export_chat,
        inputs=chatbot,
        outputs=info_text
    )
    
    # 提交查询
    def process_query(message, history):
        if not message.strip():
            return "", history, "请输入问题"
        
        response, new_history = query_afsim(message, history)
        return "", new_history, f"已回答: {message[:30]}..."
    
    submit_btn.click(
        fn=process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, info_text]
    )
    
    # 回车提交
    msg.submit(
        fn=process_query,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, info_text]
    )

# 启动函数
def launch_app(share=False, port=7860):
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        debug=True
    )

if __name__ == "__main__":
    launch_app()