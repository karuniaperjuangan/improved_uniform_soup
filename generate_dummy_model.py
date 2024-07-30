from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch
model_name = "Qwen/Qwen2-0.5B"

config = AutoConfig.from_pretrained(model_name)

for i in range(3):
    model = AutoModel.from_config(config,torch_dtype=torch.float16)
    model.save_pretrained(f'./dummy_models/{model_name.replace("/","--")}_{i}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f'./dummy_models/{model_name.replace("/","--")}_{i}')
    print(f"Model {i} saved!")