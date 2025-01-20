import re
from typing import List, Optional

def evaluate_expression(expression: str) -> float:
    """计算数学表达式的值"""
    try:
        # 移除所有空格
        expression = expression.replace(' ', '')
        # 安全地评估表达式
        return eval(expression)
    except:
        raise ValueError("Invalid expression")

class Game24Solver:
    def __init__(self):
        self.operators = ['+', '-', '*', '/']
        self.epsilon = 1e-10  # 用于浮点数比较
    
    def is_valid_24(self, expression: str, numbers: List[int]) -> bool:
        """验证表达式是否是有效的24点解"""
        try:
            # 提取表达式中的数字
            used_nums = sorted([int(n) for n in re.findall(r'\d+', expression)])
            if sorted(numbers) != used_nums:
                return False
            
            # 计算表达式的值
            result = evaluate_expression(expression)
            return abs(result - 24) < self.epsilon
        except:
            return False
    
    def solve(self, numbers: List[int]) -> Optional[str]:
        """求解24点问题"""
        if len(numbers) != 4:
            return None
        
        def generate_expressions(nums: List[float], expressions: List[str]) -> List[tuple]:
            """生成表达式及其结果
            Args:
                nums: 当前可用的数字列表
                expressions: 对应的表达式列表
            Returns:
                List[tuple]: (结果, 表达式)的列表
            """
            if len(nums) == 1:
                return [(nums[0], expressions[0])]
            
            results = []
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    num1, num2 = nums[i], nums[j]
                    expr1, expr2 = expressions[i], expressions[j]
                    
                    # 创建剩余数字和表达式列表
                    remaining_nums = nums[:i] + nums[i+1:j] + nums[j+1:]
                    remaining_exprs = expressions[:i] + expressions[i+1:j] + expressions[j+1:]
                    
                    for op in self.operators:
                        # 跳过除数为0的情况
                        if op == '/' and abs(num2) < self.epsilon:
                            continue
                        
                        try:
                            if op == '+':
                                new_num = num1 + num2
                                new_expr = f"({expr1}+{expr2})"
                            elif op == '-':
                                new_num = num1 - num2
                                new_expr = f"({expr1}-{expr2})"
                            elif op == '*':
                                new_num = num1 * num2
                                new_expr = f"({expr1}*{expr2})"
                            else:  # op == '/'
                                new_num = num1 / num2
                                new_expr = f"({expr1}/{expr2})"
                            
                            # 如果只剩一个数字，检查是否等于24
                            if len(remaining_nums) == 0:
                                if abs(new_num - 24) < self.epsilon:
                                    return [(new_num, new_expr)]
                            
                            # 递归生成子表达式
                            sub_results = generate_expressions(
                                remaining_nums + [new_num],
                                remaining_exprs + [new_expr]
                            )
                            results.extend(sub_results)
                            
                        except Exception as e:
                            continue
            
            return results
        
        # 将数字转换为字符串表达式
        initial_expressions = [str(n) for n in numbers]
        
        # 尝试生成所有可能的表达式
        print(f"\n开始求解: {numbers}")
        all_results = generate_expressions(numbers, initial_expressions)
        
        # 找到第一个等于24的有效表达式
        for result, expr in all_results:
            if abs(result - 24) < self.epsilon:
                print(f"找到解: {expr} = {result}")
                return expr
        
        print("未找到解")
        return None

# 测试代码
if __name__ == "__main__":
    solver = Game24Solver()
    numbers = [7, 5, 0, 2]
    
    result = solver.solve(numbers)
    print(f"Numbers: {numbers}")
    print(f"Solution: {result}")
