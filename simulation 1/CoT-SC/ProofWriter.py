df = pd.read_csv('/content/proofwriter-depth1-2000sample.csv')

prompt = f"""
A logical approach will help us quickly arrive at the solution to this problem. Your output should be of the following format:

Reasoning:
Your reasoning here.

Answer:
Your final answer (either true or false).

Premise:
[PREMISE]

Hypothesis:
Is it true or false? [HYPOTHESIS]
"""

modified_prompts = []
for i in range(len(df)):
    # Convert each document to a string and join them with newline characters

    new_prompt = prompt.replace('[PREMISE]', df['premise'].iloc[i])
    new_prompt = new_prompt.replace('[HYPOTHESIS]', df['hypothesis'].iloc[i])

    modified_prompts.append(new_prompt)

acc_depth_1 = []
df_test = pd.read_csv('/content/proofwriter-depth1-2000sample.csv')
df_pred = cot_sc(modified_prompts)

pred = df_pred['majority_answer'].apply(lambda x: convert_logits(x))
df_test['answer_label'] = df_test['answer'].apply(lambda x: convert_logits(x))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(df_test['answer_label'].to_numpy(), pred.to_numpy())
acc_depth_1.append(accuracy)