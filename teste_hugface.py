import requests
from transformers import AutoTokenizer
from huggingface_hub import login


# Utilizando a API do HuggingFace
modelo = "mistralai/Mixtral-8x7B-Instruct-v0.1"
url = f'https://api-inference.huggingface.co/models/{modelo}'
access_token_write = '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'
# login(token=access_token_write, add_to_git_credential=True)
tokenizer_mixtral = AutoTokenizer.from_pretrained(modelo)
pergunta = ''
chat = []
headers = {'Authorization': f'Bearer {access_token_write}'}


while pergunta != 'sair':
    pergunta = input('Digite sua pergunta(digite \'sair\' para encerrar): ')
    if pergunta == 'sair':
        break
    chat.append({'role': 'user', 'content': pergunta})
    template_mix = tokenizer_mixtral.apply_chat_template(chat, tokenize=False,
                                                         add_generation_prompt=True)

    json = {
        'inputs': template_mix,
        'options': {'use_cache': False, 'wait_for_model': True},
        'parameters': {'max_new_tokens': 1000}
    }

    response = requests.post(url, json=json, headers=headers).json()
    print(response)
    resposta = response[0]['generated_text'].split('[/INST]')[-1]    # -1 é o último
    chat.append({'role': 'assistant', 'content': resposta})

    print(f'Resposta: {resposta}')

    # ---------------------------------------------------
    # Usar auto tokenizador para não precisar formatar os inputs para cada modelo

