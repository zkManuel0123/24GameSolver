# 24GameSolver
A research project exploring the capabilities of Large Language Models in solving the 24 Game through fine-tuning. This project investigates whether LLMs can learn mathematical reasoning and problem-solving skills specific to the 24 Game challenge. This project investigates whether LLMs can learn mathematical reasoning and problem-solving skills specific to the 24 Game challenge.
🎯 Project Overview
The 24 Game is a mathematical card game where players need to find a way to manipulate four numbers using basic arithmetic operations to arrive at 24. This project attempts to teach this capability to language models through fine-tuning with Chain-of-Thought (CoT) reasoning.
Experiments
We conducted experiments with two different model scales:

Qwen2.5-0.5B (500M parameters)
Qwen2.5-14B (14B parameters)

🛠 Technical Implementation
Data Preparation

Generated training (1000 samples) and test (100 samples) datasets using Depth-First Search
Each sample includes:

Input numbers
Detailed step-by-step solution (Chain-of-Thought)
Final arithmetic expression



Model Architecture

Base models: Qwen2.5-0.5B and Qwen2.5-14B
Fine-tuning method: LoRA (Low-Rank Adaptation)
Training framework: Hugging Face Transformers

📊 Results and Findings
Performance Metrics

Training converged successfully with loss decreasing from 0.6745 to ~0.13
However, the model failed to generate correct solutions on the test set
Average response time: 6.04 seconds
Accuracy: 0.00%

Key Insights

Model size isn't everything

Despite the 30x difference in parameter count, both models struggled with the task
Training strategy and data quality might be more crucial than model scale


Loss metrics can be misleading

Despite good convergence in loss values, models failed to learn the actual problem-solving skills
Traditional loss metrics might not be sufficient for evaluating mathematical reasoning capabilities


Data considerations

1000 training samples might be insufficient for complex mathematical reasoning
Future work should focus on data augmentation and more detailed reasoning steps



🚀 Future Improvements

Data Enhancement

Increase training dataset size
Improve quality and granularity of reasoning steps
Implement data augmentation techniques


Training Strategy

Design better curriculum learning approaches
Explore alternative fine-tuning methods
Investigate specialized architectures for mathematical reasoning


Evaluation Metrics

Develop better metrics for assessing mathematical reasoning capabilities
Implement step-by-step evaluation of solution process


⚙️ Requirements
Python 3.8+
PyTorch
Transformers
PEFT
ModelScope
Wandb

📝 Installation
```
# Clone the repository
git clone https://github.com/yourusername/24-game-solver-llm.git
cd 24-game-solver-llm

# Install dependencies
pip install -r requirements.txt
```
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Thanks to the Qwen team for providing the base models
Thanks to Hugging Face for their excellent Transformers library
