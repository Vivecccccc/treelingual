import torch
from typing import Iterator
from transformers import PreTrainedTokenizerBase

def _get_qr_pairs(input_ids: torch.Tensor, output_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> str:
    queries, responses = [], []
    assert input_ids.size(0) == output_ids.size(0)
    batch_size = input_ids.size(0)
    actual_query = input_ids.detach().cpu()
    actual_output = output_ids[:, input_ids.size(-1):].detach().cpu()
    for i in range(batch_size):
        query_start_at = (actual_query[i] != tokenizer.pad_token_id).nonzero()[0].item()
        response_end_at = (actual_output[i] == tokenizer.eos_token_id).nonzero()
        if len(response_end_at) == 0:
            response_length = 1 # allow empty response
        else:
            response_length = response_end_at[-1].item() + 1
        queries.append(actual_query[i, query_start_at:]) # remove padding from left
        responses.append(actual_output[i, :response_length]) # remove padding from right
    return queries, responses

class TreeIterator(Iterator):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ids: torch.Tensor):
        self.tokenizer = tokenizer
        self.ids = ids
        self.index = 0

    def has_next(self):
        return self.index < len(self)

    def __len__(self):
        return self.ids.size(0)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        elem = self.ids[self.index]
        if not isinstance(elem.item(), int):
            raise ValueError(f"element at index {self.index} is not an integer")
        self.index += 1
        return elem.item(), self.tokenizer.decode(elem)