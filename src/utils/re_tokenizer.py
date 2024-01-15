import argparse
from transformers import AutoTokenizer
from constants import FUNCTION_MAP, ARG_TOKENS

def get_tokenzier(model_name_or_path):
    return AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

def add_functional_tokens(tokenizer, function_map, arg_tokens):
    special_tokens_dict = {"additional_special_tokens": []}
    for func_name in function_map:
        special_tokens_dict["additional_special_tokens"].append(function_map[func_name]["start"])
        special_tokens_dict["additional_special_tokens"].append(function_map[func_name]["end"])
    for arg_token in arg_tokens:
        special_tokens_dict["additional_special_tokens"].append(arg_token)
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

def save_tokenizer(tokenizer, output_dir):
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    tokenizer = get_tokenzier(args.model_name_or_path)
    tokenizer = add_functional_tokens(tokenizer, FUNCTION_MAP, ARG_TOKENS)
    save_tokenizer(tokenizer, args.output_dir)