import pandas as pd
from unsloth import FastLanguageModel
from tqdm import tqdm

df = pd.read_pickle('df.pkl').reset_index(drop=True)

dict_system_prompts = {
    "name": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *Compound name*. It can be more than one compound name. 
Your output should be a python list with the names. Return only the python list, without any additional information.
""",
    "bioActivity": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *Biological Activity*.  It can be more than one biological activity. 
Your output should be a python list with the biological activities. Return only the python list, without any additional information.
""",
    "collectionSpecie": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *Collection Specie*, i.e., Species from which natural products were extracted. Provide the scientific name, binomial form. Family name can be provided. For example Tithonia diversifolia, Styrax camporum (Styracaceae), or Colletotrichum gloeosporioides (Phyllachoraceae).
Your output should be a python list with the collection species. Return only the python list, without any additional information.
""",
    "collectionType": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the Collection Type*, i.e., Collection type of the species. 
Your output should be a python list with the collection type. Return only the python list, without any additional information.
""",
    "collectionSite": """You are a scientist trained in chemistry. 
You must extract information from scientific papers identifying relevant properties associated with each natural product discussed in the academic publication.
For each paper, you have to analyze the content (text) to identify the *collection Site *, i.e., the place of the collection. 
Your output should be a python list with the place or places. Return only the python list, without any additional information.
"""
}

MODEL_CONFIGS = {
    'qwen14b': {
        'chat_template': "qwen2.5",
        'instruction_part': "<|im_start|>user\n",
        'response_part': "<|im_start|>assistant\n",
        'string_initial': "<|im_start|>assistant\n",
        'string_final': "<|im_end|>",
        'pretrained_model': "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    },
    'phi14b': {
        'chat_template': "phi-4",
        'instruction_part': "<|im_start|>user<|im_sep|>",
        'response_part': "<|im_start|>assistant<|im_sep|>",
        'string_initial': "<|im_start|>assistant<|im_sep|>",
        'string_final': "<|im_end|>",
        'pretrained_model': "unsloth/phi-4-unsloth-bnb-4bit"
    },
    'llama8b': {
        'chat_template': "llama3",
        'instruction_part': "<|start_header_id|>user<|end_header_id|>\n\n",
        'response_part': "<|start_header_id|>assistant<|end_header_id|>\n\n",
        'string_initial': "<|start_header_id|>assistant<|end_header_id|>",
        'string_final': "<|eot_id|>",
        'pretrained_model': "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    }
}

def llm_call(dict_system_prompts, tarefa, input, model, tokenizer, temperature, model_key):
    messages = [
        {"role": "system", "content": dict_system_prompts[tarefa]},
        {"role": "user", "content": input}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=2048,
        use_cache=True,
        temperature=temperature,
        min_p=0.1
    )

    decoded = tokenizer.batch_decode(outputs)[0]
    
    result = decoded[decoded.find(MODEL_CONFIGS[model_key]['string_initial']):].replace(MODEL_CONFIGS[model_key]['string_initial'], "").replace(MODEL_CONFIGS[model_key]['string_final'], "")
    
    result = result.replace("```python", '').replace("```", "")

    return result.strip()

max_seq_length = 16384  
load_in_4bit = True  

model_key = 'llama8b'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_CONFIGS[model_key]['pretrained_model'],
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit
)

FastLanguageModel.for_inference(model)

tarefas = ['bioActivity', 'collectionSite', 'collectionSpecie', 'collectionType', 'name']

temperature = 0.00001

for index, row in tqdm(df.iterrows()):
    for tarefa in tarefas:
        input = row['texto']
        output = llm_call(dict_system_prompts, tarefa, input, model, tokenizer, temperature, model_key)
        try:
            l_out = eval(output)
            df.at[index,tarefa + '_llm'] = l_out
        except:
            df.at[index,tarefa + '_llm'] = output
            print(row['doi'], tarefa)

df.to_pickle(model_key +'-zero-shot.pkl')