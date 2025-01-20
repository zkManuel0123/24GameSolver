import re
import torch
from tqdm import tqdm
from modelscope import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import time
import json

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def extract_answer(self, response: str) -> str:
        """从模型输出中提取答案表达式"""
        print("\n原始回答:", response)
        
        # 匹配"答案是："后面的表达式
        pattern = r"答案是[：:]\s*([0-9\s\+\-\*\/\(\)]+?)(?:\s*=\s*\d+)?[\n\r]?"
        match = re.search(pattern, response)
        
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'\s+', '', answer)  # 移除空格
            return answer
        
        print("未能提取到答案")
        return ""
    
    def validate_answer(self, expression: str, expected_numbers: List[int]) -> bool:
        """验证答案是否正确"""
        if not expression:
            return False
            
        # 提取表达式中的所有数字
        used_numbers = [int(n) for n in re.findall(r'\d+', expression)]
        
        # 检查数字使用是否正确
        if sorted(used_numbers) != sorted(expected_numbers):
            print(f"数字使用不正确: 期望 {expected_numbers}, 实际 {used_numbers}")
            return False
        
        try:
            # 计算表达式的值
            result = eval(expression)
            return abs(result - 24) < 1e-10
        except Exception as e:
            print(f"表达式计算出错: {e}")
            return False
    
    def generate_response(self, question: str) -> str:
        """生成模型回答"""
        prompt = f"""请解决这个24点游戏问题：{question}

规则：
1. 必须恰好使用给定的四个数字，每个数字使用且仅使用一次
2. 只能使用基本的算术运算（+、-、*、÷）和括号
3. 请展示你的详细思考过程，一步一步地思考
4. 最终表达式必须等于24
5. 请以"答案是："开头，直接给出表达式，不要包含等号或结果

"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.5,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # 加载JSON测试数据
    with open("24_game_test.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        if isinstance(test_cases, dict):  # 如果是单个样本
            test_cases = [test_cases]
    
    evaluator = ModelEvaluator("./final_model")
    correct = 0
    total = len(test_cases)
    
    # 记录开始时间
    start_time = time.time()
    
    for sample in tqdm(test_cases):
        question = sample['question']
        numbers = [int(n) for n in re.findall(r'\d+', question)][:4]
        
        try:
            # 生成答案
            response = evaluator.generate_response(question)
            
            # 提取答案
            answer = evaluator.extract_answer(response)
            
            # 验证答案
            is_correct = evaluator.validate_answer(answer, numbers)
            if is_correct:
                correct += 1
                
            print(f"\n问题: {question}")
            print(f"提取的数字: {numbers}")
            print(f"参考答案: {sample.get('answer', '无')}")
            print(f"模型回答: {response}")
            print(f"是否正确: {is_correct}")
            
        except Exception as e:
            print(f"\n处理问题时出错: {question}")
            print(f"错误信息: {e}")
            continue
    
    # 计算总用时
    total_time = time.time() - start_time
    
    # 输出统计信息
    accuracy = correct / total
    print(f"\n评测统计:")
    print(f"总样本数: {total}")
    print(f"正确数量: {correct}")
    print(f"准确率: {accuracy:.2%}")
    print(f"总用时: {total_time:.2f}秒")
    print(f"平均每题用时: {total_time/total:.2f}秒")

if __name__ == "__main__":
    main()