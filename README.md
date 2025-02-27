# Machine Unlearning

SemEval 2025 [Homepage](https://semeval.github.io/SemEval2025/tasks) <br>
[Task 4](https://llmunlearningsemeval2025.github.io/) Machine Unlearning <br>
[Course page](https://github.com/cicl-iscl/cicl2024)

### Team member: Chi Kuan Lai, Yifei Chen <br>


### Methods
1. Perform Gradient Ascent on forgetting set
Gradient Ascent: to maximize a function <br>
- We negate the loss, changing the model's update direction to "increase loss".
- This effect makes the model "less proficient" at remembering the answers for these positions, as the increased loss indicates poorer performance in this area.
By continuously performing gradient ascent on this data, we can gradually reduce the model's reliance on this information, achieving an "unlearning" effect.

2. Perform Kullback-Leibler Divergence 
The Kullback-Leibler Divergence score: quantifies how much one probability distribution differs from another probability distribution.
-  used to measure the prediction differences between the current model and the pre-trained model on normal samples, thereby ensuring that the model does not deviate from learning normal samples during the "unlearning" process.

### Quick Acess
1. [Data sets](https://github.com/devychen/SemEval2025_Task4_NEKO/tree/main/Data%20sets)
2. [Final code (.py)](https://github.com/devychen/SemEval2025_Task4_NEKO/blob/main/unlearning_final.py)
3. [Pseudo code](https://github.com/devychen/SemEval2025_Task4_NEKO/blob/main/pseudo_codes.yaml)


