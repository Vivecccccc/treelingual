import re
import math
import torch
from zss import Node, simple_distance
from xml.etree import ElementTree as ET
from transformers import PreTrainedTokenizerBase
from typing import List, Tuple, Optional

from ..postprocessing import PostprocessingTree

INITIALS = "Output:\n"

def _decode_label_and_resp(l_tensors: List[torch.Tensor], 
                           r_tensors: List[torch.Tensor],
                           tokenizer: PreTrainedTokenizerBase) -> Tuple[List[str], List[str]]:
    assert len(l_tensors) == len(r_tensors)
    labels = tokenizer.batch_decode(l_tensors)
    responses = tokenizer.batch_decode(r_tensors)
    return labels, responses

def _is_wellformed(s: str) -> Tuple[bool, Optional[ET.Element]]:
    if not s.startswith(INITIALS):
        return False, None
    s = PostProcessingTreeForRewards.preprocess_xml(s.replace(INITIALS, ''))
    try:
        root = ET.fromstring(s)
    except:
        return False, None
    return True, root

def _build_zss_tree_full(node: ET.Element) -> Node:
    label = f"<{node.tag}>"
    if label == "<arg>":
        label = node.text
        label = label.replace("&", "&amp;") \
                        .replace('"', "&quot;") \
                        .replace("'", "&apos;") \
                        .replace("<", "&lt;") \
                        .replace(">", "&gt;")
        return Node(label)
    children = [_build_zss_tree_full(child) for child in node]
    return Node(label, children)

def _build_zss_tree_partial(s: str) -> Node:
    try:
        tag_start_at, tag_end_at, tags = zip(*[(m.start(0), m.end(0), s[m.start(0):m.end(0)]) 
                                                for m in re.finditer(r'<[^>]+>', s)])
    except ValueError:
        return Node("")
    root = Node(tags[0])
    tag_stack = [root]
    arg_pool = []
    tag_start_at, tag_end_at, tags = tag_start_at[1:], tag_end_at[1:], tags[1:]
    for i, tag in enumerate(tags):
        if tag.endswith("/>"):
            tag = (tag[:-2]).strip() + ">"
            new_node = Node(tag)
            tag_stack[-1].addkid(new_node)
            continue
        # Check if it is an opening tag
        if not tag.startswith("</"):
            # Create a new node for the tag
            if tag == "<arg>":
                arg_pool.append(i)
                continue
            new_node = Node(tag)
            # The new node is a child of the last node in the stack
            tag_stack[-1].addkid(new_node)
            # Push the new node onto the stack
            tag_stack.append(new_node)
        elif tag.startswith("</"):
            if tag == "</arg>":
                arg_start_at = arg_pool.pop()
                assert arg_start_at == i - 1 and not arg_pool
                # get the text content between <arg> and </arg>
                arg_text = s[tag_end_at[arg_start_at]:tag_start_at[i]]
                arg_node = Node(arg_text)
                tag_stack[-1].addkid(arg_node)
                continue
            # It's a closing tag, pop from the stack
            tag_stack.pop()
    return root

class PostProcessingTreeForRewards(PostprocessingTree):
    def __init__(self, kb_file_path: str):
        self.is_simple = False
        self.init_engine(kb_file_path)

    @classmethod
    def preprocess_xml(cls, s: str) -> str:
        s = s.replace("&", "&amp;") \
             .replace('"', "&quot;") \
             .replace("'", "&apos;")
        def _escape_angle_brackets_within_arg(s):
        # This regular expression finds all occurrences of <arg>...</arg>
            arg_content_pattern = re.compile(r'(<arg>)(.*?)(</arg>)', re.DOTALL)
            def __escape_match(m):
                # Escape < and > inside the <arg> element
                return m.group(1) + m.group(2).replace('<', '&lt;').replace('>', '&gt;') + m.group(3)
            # Replace < and > inside <arg>...</arg> without affecting the tags themselves
            return arg_content_pattern.sub(__escape_match, s)
        return _escape_angle_brackets_within_arg(s)
    
    @classmethod
    def comparing_ans(cls, gt, pred):
        def _equal(gt, pred):
            assert gt is not None, "ground truth should not be None"
            if not pred:
                return False
            if isinstance(gt, list):
                if isinstance(pred, list):
                    if pred[0] == gt[0]:
                        return 1.0
                    return len(set(gt) & set(pred)) / len(set(gt))
                if pred == gt[0]:
                    return 1.0
                return len(set(gt) & set([pred])) / len(set(gt))
            else:
                if isinstance(pred, list):
                    if pred[0] == gt:
                        return 1.0
                    return len(set([gt]) & set(pred)) / len(set([gt]))
                return (gt == pred) * 1.0
        return _equal(gt, pred)
    
    def load(self):
        raise NotImplementedError

class TreeRewards:
    alpha = 5
    beta = 1 / 20
    initials = INITIALS
    def __init__(self, tokenizer: PreTrainedTokenizerBase, kb_file_path: str):
        self.tokenizer = tokenizer
        self.pp = PostProcessingTreeForRewards(kb_file_path)

    @classmethod
    def award_format(cls, s) -> Tuple[float, Optional[ET.Element]]:
        is_wellformed, root = _is_wellformed(s)
        if not is_wellformed:
            return 0.0, None
        return 1.0, root
    
    @classmethod
    def award_distance(cls,
                       gt_xml: ET.Element,
                       pred_tensor: torch.Tensor,
                       tokenizer: PreTrainedTokenizerBase) -> List[float]:
        gt_tree_full = _build_zss_tree_full(gt_xml)
        pred_s = ""
        dist_list = []
        initial_token_ids = tokenizer.encode(cls.initials)
        # remove initial_token_ids from pred_tensor
        pred_tensor = pred_tensor[len(initial_token_ids):]
        tensor_iter = iter(pred_tensor)
        while True:
            dist_list.append(simple_distance(gt_tree_full, _build_zss_tree_partial(pred_s)))
            try:
                token_id = next(tensor_iter).item()
            except StopIteration:
                break
            pred_s += tokenizer.decode(token_id)
            pred_s = PostProcessingTreeForRewards.preprocess_xml(pred_s)
        assert len(dist_list) == pred_tensor.size(0) + 1
        score_func = lambda step_1, step_0: cls.alpha * (math.exp(-cls.beta * step_1) - math.exp(-cls.beta * step_0))
        score_list = [score_func(step_1, step_0) for step_1, step_0 in zip(dist_list[1:], dist_list)]
        return score_list
    
    @classmethod
    def award_induction(cls,
                        gt_xml: ET.Element,
                        pred_xml: ET.Element,
                        processor: PostProcessingTreeForRewards) -> float:
        gt_seq = processor._convert_to_raw_label(processor._parse_tree(gt_xml))
        pred_seq = processor._convert_to_raw_label(processor._parse_tree(pred_xml))
        gt_ans = processor._infer(processor.prepare_args_for_infer(gt_seq))
        pred_ans = processor._infer(processor.prepare_args_for_infer(pred_seq))
        return PostProcessingTreeForRewards.comparing_ans(gt_ans, pred_ans)
    
    def batch_award(self,
                    label_tensors: List[torch.Tensor],
                    response_tensors: List[torch.Tensor]) -> List[float]:
        assert len(label_tensors) == len(response_tensors)
        labels, responses = _decode_label_and_resp(label_tensors, response_tensors, self.tokenizer)
        rewards = []
        for i, (label, response) in enumerate(zip(labels, responses)):
            reward_format, reward_distance, reward_induction = 0.0, [], 0.0
            _, gt_xml = TreeRewards.award_format(label)
            assert gt_xml is not None, "ground truth should not be None"
            reward_format, pred_xml = TreeRewards.award_format(response)
            if pred_xml is None:
                rewards.append((reward_format, [0.0 for _ in range(response_tensors[i])], reward_induction))
                continue
            reward_distance = TreeRewards.award_distance(gt_xml, response_tensors[i], self.tokenizer)
            reward_induction = TreeRewards.award_induction(gt_xml, pred_xml, self.pp)
            rewards.append((reward_format, reward_distance, reward_induction))
        return rewards
    
    def combine_rewards(cls, rewards: Tuple[float, List[float], float]) -> float:
        reward_format, reward_distance, reward_induction = rewards
        return [reward_format * d * reward_induction for d in reward_distance]