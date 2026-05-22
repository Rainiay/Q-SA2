import argparse
import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import TaskType, get_peft_model
from qsa2_utils import qsa2_init
from peft import LoraConfig
import torch.nn as nn
from safetensors.torch import safe_open


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model with Q-SA2.")
    parser.add_argument("--model_name_or_path", type=str, default=None, required=True, help="The name or path of the fp32/16 model.", )
    parser.add_argument("--token", type=str, default=None, help="The access token to download model from HuggingFace Hub.")
    parser.add_argument("--bits", type=int, default=2, help="The quantized bits")
    parser.add_argument("--iter", type=int, default=1, help="The alternating steps in Q-SA2")
    parser.add_argument("--de_iter", type=int, default=1, help="The decomposition steps in Q-SA2")
    parser.add_argument("--rank", type=int, default=16, help="The rank of the LoRA adapter")
    parser.add_argument("--save_dir", type=str, default="./model_zoo/qsa2/", help="The rank of the LoRA adapter")
    args = parser.parse_args()
    return args


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)


def quantize_and_save():
    args = arg_parse()

    # Download weights and configure LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        token=args.token,
        trust_remote_code=True,
        device_map="auto",
    )
    task_type = TaskType.CAUSAL_LM
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    lora_model = get_peft_model(model, lora_config)

    for name, module in lora_model.named_modules():
        if any(target_key in name for target_key in target_modules):
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                weight = module.weight.data
                print(f"Quantizing {name} | shape: {weight.shape}")
                qweight, lora_A, lora_B = qsa2_init(weight, num_bits=args.bits, reduced_rank=args.rank, num_iter=args.iter, de_iter=args.de_iter)
                if hasattr(module, 'lora_A'):
                    module.lora_A['default'].weight = torch.nn.Parameter(lora_A)
                if hasattr(module, 'lora_B'):
                    module.lora_B['default'].weight = torch.nn.Parameter(lora_B)
                module.weight.data = qweight

    # Save Q-SA2 model
    model_name = args.model_name_or_path.split("/")[-1] + f"-{args.bits}bit" + f"-{args.rank}rank"
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_model_dir = os.path.join(args.save_dir, model_name, "qsa2_init")

    lora_model.save_pretrained(lora_model_dir)
    print_model(lora_model, "lora_model")

    # remove lora adapters and save the backbone
    base_model = lora_model.get_base_model()
    unwrap_model(base_model)
    print_model(base_model, "base_model")
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)

    # convert safetensor to bin
    tensors = {}
    with safe_open(os.path.join(lora_model_dir, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(lora_model_dir, "adapter_model.bin"))

    # change adapter_config.json
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = base_model_dir
        adapter_config['init_lora_weights'] = True
        fp.close()
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)

    return base_model_dir, lora_model_dir


if __name__ == "__main__":
    base_model_dir, lora_model_dir = quantize_and_save()

