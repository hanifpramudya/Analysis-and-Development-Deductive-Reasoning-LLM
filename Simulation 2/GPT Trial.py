acc_list = []
content_list = []
token_list = []
majority_answer = []
id_list = []

for id in range(27):

  from tqdm import tqdm

  output_list = []

  for prompt in tqdm(modified_prompts):
      start = time.time()
      gpt_res = generate_gpt(prompt, temp = 1, num_return_seq = 5, top_p = 0.95)
      output_list.append(gpt_res)
      end = time.time()
      elapsed_time = end - start
      if elapsed_time < 1.5:
        time.sleep(1.5 - elapsed_time)

  import pandas as pd
  sample_1 = []
  sample_2 = []
  sample_3 = []
  sample_4 = []
  sample_5 = []
  problem_id = []

  # Assuming output_list contains multiple outputs
  for i,output in enumerate(output_list):
      id_list.append(f'id-{id+1}')
      problem_id.append(f'problem-{i+1}')
      content_list.append(f"sample-1: {output.choices[0].message.content}\n"
                          f"sample-2: {output.choices[1].message.content}\n"
                          f"sample-3: {output.choices[2].message.content}\n"
                          f"sample-4: {output.choices[3].message.content}\n"
                          f"sample-5: {output.choices[4].message.content}")
      sample_1.append(answer(output.choices[0].message.content))
      sample_2.append(answer(output.choices[1].message.content))
      sample_3.append(answer(output.choices[2].message.content))
      sample_4.append(answer(output.choices[3].message.content))
      sample_5.append(answer(output.choices[4].message.content))
      token_list.append(output.usage.completion_tokens)

      cot_sc_answer = pd.DataFrame({'sample_1': sample_1, 'sample_2': sample_2, 'sample_3': sample_3, 'sample_4': sample_4, 'sample_5': sample_5}).mode(axis=1)
      cot_sc_answer = cot_sc_answer.iloc[:, 0].map({ 'true':1, 'false':0, 'unknown':0})

  df = pd.read_csv('/content/proofwriter-depth2-samplesurvey.csv', delimiter = ";")
  df['answer_label'] = df['label'].apply(lambda x: convert_logits(x))

  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(df['answer_label'].to_numpy(), cot_sc_answer.to_numpy())
  acc_depth_2.append(accuracy)
  majority_answer.append(cot_sc_answer.to_list())

  # Zip the lists together
majority_answer_flatten = [item for sublist in majority_answer for item in sublist]
data = {'id':id_list,'problem':id_list,'response': content_list, 'gpt_4o': majority_answer_flatten, 'generated_tokens': token_list}

  # Create DataFrame
df = pd.DataFrame(data)