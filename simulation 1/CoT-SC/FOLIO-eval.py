from COT-SC import cot-sc

df = pd.read_csv('/content/FOLIO-training.csv')

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

# List to accumulate accuracy results
acc_fol = []

# Load the generated CSV for evaluation
df_pred = cot_sc(modified_prompts)
df = pd.read_csv('/content/FOLIO-training.csv')

# Apply conversion function to predictions and labels
pred = df_pred['majority_answer'].apply(lambda x: convert_logits(x))
df['answer_label'] = df['label'].apply(lambda x: convert_logits(x))

# Calculate accuracy using sklearn
accuracy = accuracy_score(df['answer_label'].to_numpy(), pred.to_numpy())

# Append the accuracy to the acc_fol list
acc_fol.append(accuracy)