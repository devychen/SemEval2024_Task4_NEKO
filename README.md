# Machine Unlearning

SemEval 2025 [Homepage](https://semeval.github.io/SemEval2025/tasks) <br>
[Course page](https://github.com/cicl-iscl/cicl2024)

### Team member: Chi Kuan Lai, Yifei Chen <br>

### Task chosen
[Task 4](https://llmunlearningsemeval2025.github.io/) Machine Unlearning <br>

### Paper Writing
[Submission requirements](https://semeval.github.io/paper-requirements.html)  
[Writing guidelines](https://semeval.github.io/system-paper-template.html)

### Initial idea
### 1. Perform Gradient Ascent on forgetting set
Gradient Ascent: to maximize a function
- We negate the loss, changing the model's update direction to "increase loss".
- This effect makes the model "less proficient" at remembering the answers for these positions, as the increased loss indicates poorer performance in this area.
By continuously performing gradient ascent on this data, we can gradually reduce the model's reliance on this information, achieving an "unlearning" effect.

### 2. Perform Kullback-Leibler Divergence 
The Kullback-Leibler Divergence score: quantifies how much one probability distribution differs from another probability distribution.
-  used to measure the prediction differences between the current model and the pre-trained model on normal samples, thereby ensuring that the model does not deviate from learning normal samples during the "unlearning" process.
  
### 3. Perform Gradient Descent on retaining set or RAG（Retrieval Augmented Generation)
For Gradient Descent:
- Train the model again with the retain set using Gradient Descent and minimize the loss in order to predict the correctly retained answer.
  
For RAG (Retrieval Augmented Generation):
- It is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM's internal representation of information.
- The model retrieves the retained data set when the input is related to it.


### Instructions

#### Suggested Timetable
25.10.2024: team formation, task choices <br>
01.11.2024: final team formation, task choices for SemEval <br>
08.11.2024: informal task presentations, initial ideas, baselines (SemEval, at least one member from each team should join) <br>
30.11.2024: deadline for interim report (for SemEval, no meeting only a brief report) <br>
02.12.2024: informal presentation (meeting with all participants) <br>
10.01.2025: informal presentation (meeting with all participants) <br>
07.02.2025: final presentation (meeting with all participants) <br>



