from PIL import Image 
import requests
from transformers import AutoModelForCausalLM, AutoProcessor,AutoConfig  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_moe = "lamm-mit/Cephalo-Phi-3-MoE-vision-128k-3x4b-beta"

processor = AutoProcessor.from_pretrained(model_name_moe, trust_remote_code=True) 
moe_model = AutoModelForCausalLM.from_pretrained(
    model_name_moe,
    trust_remote_code=True,  torch_dtype=torch.bfloat16,    
).to(device)

question = "What is shown in this image, and what is the relevance for materials design? Include a discussion of multi-agent AI."

messages = [ 
    {"role": "user", "content": f"<|image_1|>\n{question}"}, 
    ] 

url = "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg" 

image = Image.open(requests.get(url, stream=True).raw) 

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

generation_args = { 
                    "max_new_tokens": 256, 
                    "temperature": 0.1, 
                    "do_sample": True, 
                    "stop_strings": ['<|end|>',
                                     '<|endoftext|>'],
                    "tokenizer": processor.tokenizer,
                  } 

generate_ids = moe_model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

print(response)
