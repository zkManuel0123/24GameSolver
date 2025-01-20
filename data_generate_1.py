import random
import math
import json
from itertools import combinations, permutations
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass

@dataclass
class Expression:
    value: float
    expr_str: str
    used_numbers: Set[int]

class TwentyFourGame:
    def __init__(self, eps: float = 1e-10):
        self.eps = eps
        self.operators = ['+', '-', '*', '/']
        self.steps_log = []
        
    def reset_log(self):
        """Reset the steps log for a new problem"""
        self.steps_log = []
        
    def log_step(self, message: str):
        """Add a step to the reasoning process"""
        self.steps_log.append(f"Step{len(self.steps_log) + 1}: {message}")
        
    def evaluate(self, a: float, b: float, op: str) -> Optional[float]:
        """Evaluate a simple binary operation"""
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/' and abs(b) > self.eps:
            return a / b
        return None
        
    def is_close_to_24(self, value: float) -> bool:
        """Check if a value is close enough to 24"""
        return abs(value - 24) < self.eps
        
    def solve(self, numbers: List[int]) -> Tuple[bool, List[str], Optional[str]]:
        """
        Solve the 24 game for given numbers using DFS
        Returns: (has_solution, steps_log, solution_expr)
        """
        self.reset_log()
        
        # Start with converting numbers to expressions
        expressions = [Expression(float(n), str(n), {n}) for n in numbers]
        
        # Try to find solution
        found_solution = self._dfs_solve(expressions)
        
        if found_solution:
            solution = self.steps_log[-1].split("Found valid solution: ")[1]
            return True, self.steps_log, solution
        else:
            self.log_step("After exhaustive search, no valid solution exists.")
            return False, self.steps_log, None
            
    def _dfs_solve(self, expressions: List[Expression]) -> bool:
        """Helper function for DFS search"""
        if len(expressions) == 1:
            if self.is_close_to_24(expressions[0].value):
                self.log_step(f"Found valid solution: {expressions[0].expr_str} = 24")
                return True
            return False
            
        # Try all pairs of expressions
        for i, j in combinations(range(len(expressions)), 2):
            expr1, expr2 = expressions[i], expressions[j]
            
            # Skip if numbers would be reused
            if not expr1.used_numbers.isdisjoint(expr2.used_numbers):
                continue
                
            # Try all operators
            for op in self.operators:
                # Try both orders for non-commutative operations
                for a, b in [(expr1, expr2), (expr2, expr1)]:
                    if op in ['+', '*'] and b.value > a.value:
                        # Skip redundant commutative operations
                        continue
                        
                    result = self.evaluate(a.value, b.value, op)
                    if result is None:
                        self.log_step(f"Trying {a.expr_str} {op} {b.expr_str}: Invalid operation")
                        continue
                        
                    new_expr = Expression(
                        value=result,
                        expr_str=f"({a.expr_str} {op} {b.expr_str})",
                        used_numbers=a.used_numbers | b.used_numbers
                    )
                    
                    self.log_step(f"Trying {new_expr.expr_str} = {result:.2f}")
                    
                    # Create new expression list with this result
                    new_expressions = [expr for k, expr in enumerate(expressions) 
                                     if k != i and k != j]
                    new_expressions.append(new_expr)
                    
                    # Recurse
                    if self._dfs_solve(new_expressions):
                        return True
                        
                    self.log_step(f"Backtracking from {new_expr.expr_str}, trying different approach")
                    
        return False

def generate_diverse_numbers(min_num: int = 1, max_num: int = 13, size: int = 4) -> List[int]:
    """Generate a list of numbers for the 24 game"""
    return [random.randint(min_num, max_num) for _ in range(size)]

from multiprocessing import Pool, cpu_count

def generate_single_example(args):
    min_num, max_num = args
    game = TwentyFourGame()
    numbers = generate_diverse_numbers(min_num, max_num)
    has_solution, steps, solution = game.solve(numbers)
    return {
        "question": {
            "numbers": numbers,
            "goal": 24
        },
        "long_cot": "\n".join(steps),
        "answer": solution if has_solution else "No valid solution"
    }

def generate_dataset(num_train: int = 1000, num_test: int = 100, 
                    min_num: int = 1, max_num: int = 9) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training and test datasets for the 24 game
    Returns: (training_data, test_data)
    """
    # Use multiprocessing for faster generation
    num_processes = cpu_count()
    with Pool(num_processes) as pool:
        # Generate training data in parallel
        args = [(min_num, max_num) for _ in range(num_train * 2)]  # Generate extra to account for duplicates
        training_data = pool.map(generate_single_example, args)
        
        # Generate test data in parallel
        args = [(min_num, max_num) for _ in range(num_test * 2)]
        test_data = pool.map(generate_single_example, args)
    
    # Remove duplicates
    seen_numbers = set()
    
    # Generate training data
    while len(training_data) < num_train:
        numbers = generate_diverse_numbers(min_num, max_num)
        numbers_key = tuple(sorted(numbers))
        
        # Skip if we've seen this combination
        if numbers_key in seen_numbers:
            continue
            
        seen_numbers.add(numbers_key)
        
        # Solve and generate CoT
        has_solution, steps, solution = game.solve(numbers)
        
        # Create data entry
        entry = {
            "question": {
                "numbers": numbers,
                "goal": 24
            },
            "long_cot": "\n".join(steps),
            "answer": solution if has_solution else "No valid solution"
        }
        
        training_data.append(entry)
        
    # Generate test data (similar process but without CoT)
    test_seen = set()
    while len(test_data) < num_test:
        numbers = generate_diverse_numbers(min_num, max_num)
        numbers_key = tuple(sorted(numbers))
        
        # Skip if we've seen this in training or test
        if numbers_key in seen_numbers or numbers_key in test_seen:
            continue
            
        test_seen.add(numbers_key)
        
        # For test data, we only need the question
        entry = {
            "question": {
                "numbers": numbers,
                "goal": 24
            }
        }
        
        test_data.append(entry)
        
    return training_data, test_data

def save_datasets(train_data: List[Dict], test_data: List[Dict], 
                 train_file: str = "24game_train.json", 
                 test_file: str = "24game_test.json"):
    """Save the generated datasets to JSON files"""
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
        
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

def main():
    import time
    start_time = time.time()
    print(f"Generating datasets using {cpu_count()} CPU cores...")
    print("This may take a few minutes depending on the dataset size...")
    train_data, test_data = generate_dataset(num_train=5000, num_test=500)
    
    # Save to files
    print("Saving datasets...")
    save_datasets(train_data, test_data)
    
    # Print some statistics and timing
    total_time = time.time() - start_time
    print(f"\nGenerated {len(train_data)} training examples")
    print(f"Generated {len(test_data)} test examples")
    
    # Print a sample
    print(f"\nTotal generation time: {total_time:.2f} seconds")
    print(f"Average time per example: {total_time/(len(train_data) + len(test_data)):.3f} seconds")
    
    print("\nSample training example:")
    print(json.dumps(train_data[0], indent=2))

if __name__ == "__main__":
    main()