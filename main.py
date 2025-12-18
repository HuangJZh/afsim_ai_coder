# AFSIM 代码生成助手 (本地 Python 版)
# 已修改为使用 OpenAI 兼容 API

import os
import sys
import time
import requests  


API_BASE_URL = "https://api.deepseek.com"  
API_KEY = "sk-75ed608c4f8342c6934bb5c6f9cb4888"  
MODEL_NAME = "deepseek-chat"  



KNOWLEDGE_DIR = "./tutorials"  

def load_knowledge_base(directory):
    """加载指定目录下的所有文本文件作为知识库"""
    knowledge = ""
    if not os.path.exists(directory):
        print(f"警告: 目录 '{directory}' 不存在。请创建该目录并放入 AFSIM 规则文件。")
        return knowledge

    print(f"正在加载 '{directory}' 中的规则文件...")
    file_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 支持的文件扩展名
            if file.lower().endswith(('.txt', '.md', '.cpp', '.h', '.json', '.xml')):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        knowledge += f"\n--- 文档: {file} ---\n{content}\n----------------\n"
                        file_count += 1
                except Exception as e:
                    print(f"跳过文件 {file}: {e}")
    
    print(f"共加载 {file_count} 个文件。")
    return knowledge

def chat_completion(messages, system_instruction="", max_tokens=4000):
    """使用 OpenAI 兼容 API 进行对话"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 如果有系统指令，添加到消息列表开头
    if system_instruction:
        messages = [{"role": "system", "content": system_instruction}] + messages
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False  # 简化版本，先不使用流式
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            error_msg = f"API 错误: {response.status_code} - {response.text}"
            print(f"\n[错误] {error_msg}")
            return f"错误: {error_msg}"
            
    except requests.exceptions.Timeout:
        return "错误: 请求超时，请稍后重试"
    except Exception as e:
        return f"错误: {str(e)}"

def main():
    if not API_KEY or API_KEY.startswith("sk-xxxxxxxx"):
        print("="*50)
        print("请先配置 API Key!")
        print("="*50)
        print("\n请选择以下方式之一获取免费 API Key:")
        print("1. DeepSeek (推荐):")
        print("   - 访问: https://platform.deepseek.com/api_keys")
        print("   - 注册后获取免费 API Key")
        print("   - 将 API_KEY 替换为您的密钥")
        print("\n2. OpenRouter:")
        print("   - 访问: https://openrouter.ai/settings/keys")
        print("   - 注册后获取免费 API Key")
        print("   - 取消注释 OpenRouter 配置")
        print("\n3. 本地 Ollama (需要安装):")
        print("   - 安装: https://ollama.com/")
        print("   - 运行: ollama run qwen2.5:7b")
        print("   - 设置 API_BASE_URL = 'http://localhost:11434/v1'")
        print("   - 设置 API_KEY = 'ollama' (任意值)")
        print("   - 设置 MODEL_NAME = 'qwen2.5:7b'")
        print("\n按 Enter 退出...")
        input()
        return
    
    knowledge_base = load_knowledge_base(KNOWLEDGE_DIR)

    system_instruction = """你是一个精通 AFSIM 代码生成专家。
你的任务是根据用户的需求，参考提供的知识库规则，生成正确、规范的 AFSIM 代码脚本。
请遵循以下规则：
1. 在每个文件的开头用注释说明文件名。
1. 优先使用知识库中提供的语法和模式。
2. 保持代码简洁，逻辑清晰。
3. 不需要解释代码，只需提供代码本身。
"""

    if knowledge_base:
        system_instruction += f"\n=== 本地知识库/参考文档 开始 ===\n{knowledge_base}\n=== 本地知识库/参考文档 结束 ===\n"

    print("\n" + "="*50)
    print("AFSIM 本地代码助手启动")
    print(f"API: {API_BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print("使用说明: 输入多行内容，最后另起一行输入 //end 提交")
    print("输入 'quit' 或 'exit' (单独一行) 退出程序")
    print("="*50 + "\n")

    # 对话历史
    conversation_history = []

    while True:
        try:
            print("需求 > (输入完成后，另起一行输入 //end 结束)")
            lines = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break # 处理 Ctrl+D/Z 的情况
                
                # 检查结束标记
                if line.strip() == "//end":
                    break
                
                lines.append(line)
            
            user_input = "\n".join(lines)
            # --- 修改结束 ---

            # 检查退出命令
            if user_input.strip().lower() in ['quit', 'exit']:
                break
            
            if not user_input.strip():
                print("输入为空，请重新输入。\n")
                continue

            print("\n正在生成代码...", flush=True)
            
            # 添加到对话历史
            conversation_history.append({"role": "user", "content": user_input})
            
            # 发送请求
            response = chat_completion(conversation_history, system_instruction)
            
            print("\n" + "-"*50)
            print("生成的代码:")
            print("-"*50)
            print(response)
            
            # 将助手回复添加到历史
            conversation_history.append({"role": "assistant", "content": response})
            
            print("\n" + "-"*50 + "\n")
            
            # 避免频繁请求导致限流
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n程序已停止。")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main()