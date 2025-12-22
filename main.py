# AFSIM 代码生成助手 (分步引导版)
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
    """加载指定目录下的所有文本文件作为知识库 """
    knowledge = ""
    if not os.path.exists(directory):
        print(f"警告: 目录 '{directory}' 不存在。")
        return knowledge

    print(f"正在加载 '{directory}' 中的规则文件...")
    file_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
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

def get_multiline_input(prompt_label):
    """分步获取多行输入的辅助函数"""
    print(f"\n>>> 请输入【{prompt_label}】 (完成后另起一行输入 //end 提交，输入 'exit' 退出):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().lower() == 'exit':
                return 'exit'
            if line.strip() == "//end":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

def chat_completion(messages, system_instruction="", max_tokens=4000):
    """使用 OpenAI 兼容 API 进行对话 """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    if system_instruction:
        messages = [{"role": "system", "content": system_instruction}] + messages
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False
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
            return f"API 错误: {response.status_code} - {response.text}"
    except Exception as e:
        return f"错误: {str(e)}"

def main():
    if not API_KEY or API_KEY.startswith("sk-xxxxxxxx"):
        print("请先配置 API Key!")
        return
    
    knowledge_base = load_knowledge_base(KNOWLEDGE_DIR)

    system_instruction = """你是一个精通 AFSIM 代码生成与调试的专家。
你的任务是根据用户的需求，参考提供的知识库规则，生成正确、规范的 AFSIM 代码脚本。
请遵循以下规则：
1. 在每个文件的开头用注释说明文件名。
2. 优先使用知识库中提供的语法和模式（如 mover 类型、通信网络名等）。
3. 保持代码简洁，逻辑清晰，不需要多余解释。
4. **代码修复模式**：如果用户提供了任务目标、错误代码和报错信息，请分析原因并输出修复后的完整代码。
"""

    if knowledge_base:
        system_instruction += f"\n=== 本地知识库/参考文档 开始 ===\n{knowledge_base}\n=== 本地知识库/参考文档 结束 ===\n"

    print("\n" + "="*50)
    print("AFSIM 本地代码助手启动")
    print("-" * 50)
    
    conversation_history = []

    while True:
        print("\n请选择操作模式：")
        print("1. [生成] 直接输入需求生成代码")
        print("2. [修复] 分步提供任务、代码和报错进行修复")
        print("输入 'clear' 清空历史, 'exit' 退出")
        
        choice = input("选择 (1/2) > ").strip().lower()

        if choice == 'exit':
            break
        elif choice == 'clear':
            conversation_history = []
            print("\n[系统] 对话历史已清空。")
            continue
        elif choice not in ['1', '2']:
            print("无效选择，请重新输入。")
            continue

        if choice == '1':
            user_input = get_multiline_input("代码生成需求")
            if user_input == 'exit': break
        else:
            goal = get_multiline_input("任务目标")
            if goal == 'exit': break
            
            old_code = get_multiline_input("之前的错误代码")
            if old_code == 'exit': break
            
            error_msg = get_multiline_input("编译器报错信息")
            if error_msg == 'exit': break
            
            user_input = f"【任务目标】\n{goal}\n\n【之前的错误代码】\n{old_code}\n\n【编译器报错信息】\n{error_msg}"

        if not user_input.strip():
            continue

        print("\n正在处理...", flush=True)
        conversation_history.append({"role": "user", "content": user_input})
        
        response = chat_completion(conversation_history, system_instruction)
        
        print("\n" + "-"*50)
        print("输出结果:")
        print("-" * 50)
        print(response)
        
        conversation_history.append({"role": "assistant", "content": response})
        print("-" * 50)
        time.sleep(1)

if __name__ == "__main__":
    main()