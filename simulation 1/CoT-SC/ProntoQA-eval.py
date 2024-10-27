df = pd.read_csv('/content/prontoqa-falseontology-1hop.csv')

prompt = f"""
A logical approach will help us quickly arrive at the solution to this problem. Your output should be of the following format:

Reasoning:
Your reasoning here.

Answer:
Your final answer (either true or false).

Context:
[DOCS]
"""
modified_prompts = []
for i in range(len(df)):
    # Convert each document to a string and join them with newline characters
    new_prompt = prompt.replace('[DOCS]', df['question'].iloc[i])

    modified_prompts.append(new_prompt)

acc_1_hop = []
df_pred = cot_sc(modified_prompts)
df_test = pd.read_csv('/content/prontoqa-falseontology-1hop.csv')

pred = df_pred['majority_answer'].apply(lambda x: convert_logits(x))
df_test['answer_label'] = df_test['ground_truth'].apply(lambda x: convert_logits(x))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(df_test['answer_label'].to_numpy(), pred.to_numpy())
acc_1_hop.append(accuracy)