def io_prompting(modified_prompts):
    output_list = []

    for prompt in tqdm(modified_prompts):

        gpt_res = generate_gpt(prompt, max_tokens=5)
        output_list.append(gpt_res)

    import pandas as pd

    response_list = []
    token_list = []
    id_list = []

    answer = []

    # Assuming output_list contains multiple outputs
    for i,output in enumerate(output_list):
        id_list.append(f'id-{i+1}')
        response_list.append(f"{output.choices[0].text}")
        answer.append(convert_logits(output.choices[0].text))
        token_list.append(output.usage.completion_tokens)

    # Zip the lists together
    data = {'id':id_list,'response': response_list, 'answer':answer, 'generated_tokens': token_list}

    df = pd.DataFrame(data)

    return df