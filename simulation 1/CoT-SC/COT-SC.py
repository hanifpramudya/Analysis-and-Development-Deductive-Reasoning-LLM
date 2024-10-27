def cot_sc(modified_prompts):
    output_list = []

    # Collect GPT results for each modified prompt
    for prompt in tqdm(modified_prompts):
        gpt_res = generate_gpt(prompt, temp=1, num_return_seq=5, top_p=0.95)
        output_list.append(gpt_res)

    # Initialize lists to store data
    content_list = []
    token_list = []
    id_list = []
    sample_1 = []
    sample_2 = []
    sample_3 = []
    sample_4 = []
    sample_5 = []

    # Process GPT outputs and append to corresponding lists
    for i, output in enumerate(output_list):
        id_list.append(f'id-{i+1}')
        content_list.append(f"sample-1: {output.choices[0].text}\n"
                            f"sample-2: {output.choices[1].text}\n"
                            f"sample-3: {output.choices[2].text}\n"
                            f"sample-4: {output.choices[3].text}\n"
                            f"sample-5: {output.choices[4].text}")
        
        # Append the processed answer results to corresponding sample lists
        sample_1.append(answer(output.choices[0].text))
        sample_2.append(answer(output.choices[1].text))
        sample_3.append(answer(output.choices[2].text))
        sample_4.append(answer(output.choices[3].text))
        sample_5.append(answer(output.choices[4].text))
        
        # Append the token count for each response
        token_list.append(output.usage.completion_tokens)

    # Zip the lists together into a dictionary
    data = {
        'id': id_list,
        'response': content_list,
        'sample-1': sample_1,
        'sample-2': sample_2,
        'sample-3': sample_3,
        'sample-4': sample_4,
        'sample-5': sample_5,
        'generated_tokens': token_list
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Replace values in the DataFrame (true, false, unknown)
    df[['sample-1', 'sample-2', 'sample-3', 'sample-4', 'sample-5']] = df[
        ['sample-1', 'sample-2', 'sample-3', 'sample-4', 'sample-5']].replace(
        {'true': 1, 'false': -1, 'unknown': 0, 'NaN': 0})

    # Get the majority answer for each row using mode
    majority_answer = df[['sample-1', 'sample-2', 'sample-3', 'sample-4', 'sample-5']].mode(axis=1)

    # Map the index labels to the corresponding answer
    majority_answer = majority_answer.iloc[:, 0].map({1: 'true', -1: 'false', 0: 'unknown'})

    # Add the new column for the majority answer
    df['majority_answer'] = majority_answer

    return df