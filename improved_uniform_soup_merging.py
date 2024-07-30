"""
This code is an implementation of a model merging algorithms. The step of merging algorithms is given below:
1. Load the models that will be merged.
2. Initially, create a initial merged model by averaging each parameter of the models.
3. Do a forward pass of the initial merged model to get the loss.
4. Calculate the gradient of the loss with respect to the parameters of the initial merged model. Get the opposite sign of the gradient (sign(-gradient)) and save it as a direction tensor.
For each parameter:
 If the direction sign is positive:
  - Average the parameter from the initial merged model parameter plus any source models with the parameter that is larger than initial merged model parameter.
 If the direction sign is negative:
  - Average the parameter from the initial merged model parameter plus any source models with the parameter that is smaller than initial merged model parameter.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List
import torch, os
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
input_texts = ["Hello, world!", "My name is Bob", "I work at Google"]

#model_name = "Qwen/Qwen2-0.5B"
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


list_model_name_to_merge = ["./dummy_models/Qwen--Qwen2-0.5B_0", "./dummy_models/Qwen--Qwen2-0.5B_1", "./dummy_models/Qwen--Qwen2-0.5B_1"]
#models = [AutoModelForCausalLM.from_pretrained(model_name).to(device) for model_name in list_model_name_to_merge]


def calculate_direction(model, input_texts, device,tokenizer):
    model.train()
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    outputs.loss.backward()
    os.makedirs("./temp/sign_tensors", exist_ok=True)
    for name, param in (pbar :=tqdm(model.named_parameters())):
        pbar.set_description(f"calculating direction for {name}")
        #sign tensor is reverse of gradient, so negative gradient is positive sign while positive gradient is negative sign
        sign_tensor = (torch.sign(param.grad.data) < 0)
        #save sign tensor
        torch.save(sign_tensor, f"./temp/sign_tensors/{name}.pt")
    model.eval()
    print("Direction tensors saved to",f"./temp/sign_tensors/{name}.pt")
    return f"./temp/sign_tensors/{name}.pt"

def store_params_temp(model:AutoModelForCausalLM, model_name):
    os.makedirs(f"./temp/{model_name}", exist_ok=True)
    for name, param in (pbar:=tqdm(model.named_parameters())):
        pbar.set_description(f"saving {name} from {model_name} into temporary directory")
        torch.save(param.data.to(dtype), f"./temp/{model_name}/{name}.pt")
    #return directory path
    return f"./temp/{model_name}"

def average_weights(temp_model_dirs):
    #get name of parameters by looking at file of temp_model_dirs ending with .pt
    params_to_merge = [item.replace('.pt','') for item in os.listdir(temp_model_dirs[0])]
    #print("params_to_merge:", params_to_merge)
    os.makedirs("./temp/merged_model", exist_ok=True)
    for param_name in (pbar:=tqdm(params_to_merge)):
        #print(f"merging parameter {param_name}")
        pbar.set_description(f"merging parameter {param_name}")
        param_tensors = [torch.load(f"{temp_model_dir}/{param_name}.pt") for temp_model_dir in temp_model_dirs]
        avg_param = torch.mean(torch.stack(param_tensors), dim=0)
        torch.save(avg_param, f"./temp/merged_model/{param_name}.pt")
    return "./temp/merged_model"

def compile_folder_of_tensor_to_model(folder_path, output_dir):
    model:AutoModelForCausalLM = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(list_model_name_to_merge[0]), torch_dtype=dtype).to(device)
    #tokenizer = AutoTokenizer.from_pretrained(list_model_name_to_merge[0])
    for name, param in model.named_parameters():
        param.data = torch.load(f"{folder_path}/{name}.pt").to(device)
    #model.save_pretrained(output_dir)
    #tokenizer.save_pretrained(output_dir)
    print(f"Model {folder_path} has been compiled to {output_dir}")
    return model

def average_weight_with_direction(sign_tensors_dir, temp_model_dirs, initial_merged_model_dir):
    """
        For each parameter:
    If the direction sign is positive:
    - Average the parameter from the initial merged model parameter plus any source models with the parameter that is larger than initial merged model parameter.
    If the direction sign is negative:
    - Average the parameter from the initial merged model parameter plus any source models with the parameter that is smaller than initial merged model parameter.
    """
    #get name of parameters by looking at file of temp_model_dirs ending with .pt
    params_to_merge = [item.replace('.pt','') for item in os.listdir(temp_model_dirs[0])]
    os.makedirs("./temp/final_merged_model", exist_ok=True)
    for param_name in (pbar:=tqdm(params_to_merge)):
        pbar.set_description(f"merging parameter {param_name}")
        param_tensors:List[torch.Tensor] = [torch.load(f"{temp_model_dir}/{param_name}.pt") for temp_model_dir in temp_model_dirs]
        param_initial_merged:torch.Tensor = torch.load(f"{initial_merged_model_dir}/{param_name}.pt")
        
        param_tensors_delta = [(param > param_initial_merged) for param in param_tensors]
        # get sign tensor for this parameter
        sign_tensor = torch.load(f"{sign_tensors_dir}/{param_name}.pt")
        # for every parameter in param_tensors, replace element in param_tensors to 0 if sign_tensor != param_tensors_delta
        param_tensors = [param * (sign_tensor == param_delta) # if sign_tensor == param_delta, keep the value, else set to 0
                         for param, param_delta in zip(param_tensors, param_tensors_delta)]
        # append param_initial_merged to param_tensors
        param_tensors.append(param_initial_merged)

        # average the parameter, excluding parameter with 0 value
        avg_param = torch.sum(torch.stack(param_tensors), dim=0) / torch.sum(torch.stack(param_tensors) != 0, dim=0)

        torch.save(avg_param, f"./temp/final_merged_model/{param_name}.pt")
    return "./temp/final_merged_model"


if __name__ == '__main__':
    # for each model in list_model_name_to_merge, save to temp directory
    temp_model_dirs = [store_params_temp(AutoModelForCausalLM.from_pretrained(model_name).to(device), model_name.split('/')[-1]) for model_name in list_model_name_to_merge]
    print("Model parameters saved to temp directory")
    #average weights
    average_weight_path = average_weights(temp_model_dirs)

    model = compile_folder_of_tensor_to_model(average_weight_path, "./temp/merged_model_compiled")
    
    #calculate direction
    tokenizer = AutoTokenizer.from_pretrained(list_model_name_to_merge[0])
    sign_tensor_dir = calculate_direction(model, input_texts, device, tokenizer)

    #average weight with direction
    final_merged_model_path = average_weight_with_direction(sign_tensor_dir, temp_model_dirs, average_weight_path)