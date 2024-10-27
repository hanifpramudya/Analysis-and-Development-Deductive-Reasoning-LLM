df = pd.read_csv('/content/prontoqa-falseontology-1hop.csv')

# prompt template
prompt = f"""
Context:
[DOCS]

"""

modified_prompts = []
for i in range(len(df)):
    # Convert each document to a string and join them with newline characters
    new_prompt = prompt.replace('[DOCS]', df['question'].iloc[i])

    modified_prompts.append(new_prompt)

modified_prompts = []
for i in range(len(df)):
    # Convert each document to a string and join them with newline characters
    new_prompt = prompt.replace('[DOCS]', df['question'].iloc[i])

    modified_prompts.append(new_prompt)

acc_1_hop = []
df_pred = io_prompting(modified_prompts)
df_test = pd.read_csv('/content/prontoqa-falseontology-1hop.csv')

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(df_test['answer_label'].to_numpy(), pred.to_numpy())
acc_1_hop.append(accuracy)