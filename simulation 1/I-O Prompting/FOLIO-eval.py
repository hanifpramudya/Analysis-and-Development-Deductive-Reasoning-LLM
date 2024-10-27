df = pd.read_csv('/content/FOLIO-training.csv')

prompt = f"""
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

acc_fol = []

df_pred = io_prompting(modified_prompts)
df = pd.read_csv('/content/FOLIO-training.csv')

# Calculate accuracy using sklearn
accuracy = accuracy_score(df['answer_label'].to_numpy(), pred.to_numpy())

# Append the accuracy to the acc_fol list
acc_fol.append(accuracy)