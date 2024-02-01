import re
import torch
from collections import Counter
from itertools import accumulate, product
from typing import List, Tuple, Optional
from xml.etree import ElementTree as ET

from zss import Node, simple_distance
from transformers import PreTrainedTokenizerBase
from nltk.metrics.distance import edit_distance

from ..postprocessing import PostprocessingTree
from .trainer_utils import TreeIterator

INITIALS = "Output:\n"

def _decode_label_and_resp(l_tensors: List[torch.Tensor], 
                           r_tensors: List[torch.Tensor],
                           tokenizer: PreTrainedTokenizerBase) -> Tuple[List[str], List[str]]:
    assert len(l_tensors) == len(r_tensors)
    labels = tokenizer.batch_decode(l_tensors)
    responses = tokenizer.batch_decode(r_tensors)
    return labels, responses

def _all_is_well(s: str) -> Tuple[bool, Optional[ET.Element]]:
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

class RewardNode(Node):
    def __init__(self, label, children=None):
        super().__init__(label, children)
        self.parent: Optional[RewardNode] = None
        self.enter_reward = None
        self.exit_reward = None
    
    def get_parent(self):
        return self.parent
    
    def addkid(self, node, before=False):
        node.parent = self
        super().addkid(node, before)

    def removekid(self, node):
        if node in self.children:
            node.parent = None
            self.children.remove(node)
        return self

    def get_enter_reward(self):
        return self.enter_reward
    
    def get_exit_reward(self):
        return self.exit_reward

    def award_enter(self, ref: Node):
        full_dist = simple_distance(Node(""), ref)
        if self.parent is None:
            self.enter_reward = (full_dist - simple_distance(self, ref)) / full_dist
        else:
            cloned_parent = RewardNode(self.parent.label, [ch for ch in self.parent.children])
            cloned_self = RewardNode(self.label, [ch for ch in self.children])
            cloned_parent.removekid(cloned_self)
            self.enter_reward = (simple_distance(cloned_parent, ref) - simple_distance(self.parent, ref)) / full_dist
        return self.enter_reward
    
    def award_exit(self, ref: Node):
        full_dist = simple_distance(Node(""), ref)
        self.exit_reward = (full_dist - simple_distance(self, ref)) / full_dist
        return self.exit_reward
    
    def get_combine_reward(self):
        assert self.enter_reward is not None and self.exit_reward is not None
        num_children = len(self.children)
        # recursively get the combined reward of children
        if num_children == 0:
            return self.enter_reward + self.exit_reward
        return self.enter_reward + self.exit_reward + \
            sum([child.get_combine_reward() for child in self.children]) / num_children

class PostProcessingTreeForRewards(PostprocessingTree):
    def __init__(self, 
                #  kb_file_path: str
                 ):
        self.is_simple = False
        # self.init_engine(kb_file_path)

    @classmethod
    def preprocess_xml(cls, s: str, ignore_angle_brackets: bool = False) -> str:
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
        if ignore_angle_brackets:
            return s.replace("<", "&lt;").replace(">", "&gt;")
        return _escape_angle_brackets_within_arg(s)
    
    @classmethod
    def comparing_ans(cls, gt, pred):
        raise NotImplementedError
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
    initials = INITIALS
    malform_penalty = -5.0
    def __init__(self, tokenizer: PreTrainedTokenizerBase, kb_file_path: str):
        self.tokenizer = tokenizer
        self.pp = PostProcessingTreeForRewards(kb_file_path)

    @classmethod
    def award_format(cls, 
                     pred_ids: torch.Tensor, 
                     tokenizer: PreTrainedTokenizerBase
                     ) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
        leading_ids = tokenizer.encode(cls.initials)
        leading_tokens = [tokenizer.decode(x) for x in leading_ids]
        remains = pred_ids[len(leading_ids):]
        if pred_ids[:len(leading_ids)].tolist() == leading_ids:
            return [(x, 1.0) for x in leading_tokens], remains
        return [(x, cls.malform_penalty) for x in leading_tokens], remains

    @classmethod
    def award_distance(cls,
                       gt_xml: ET.Element,
                       pred_ids: torch.Tensor,
                       tokenizer: PreTrainedTokenizerBase) -> List[Tuple[str, float]]:
        gt_tree = _build_zss_tree_full(gt_xml)
        valid_tag_ids = set(tokenizer.additional_special_tokens_ids)
        it = TreeIterator(tokenizer, pred_ids)

        tag_stack: List[RewardNode] = []
        rewards: List[Tuple[str, float]] = []
        
        arg_pool = []

        def __get_args_in_tree(tree: RewardNode) -> List[Tuple[str, str]]:
            arg_list = []
            for child in tree.children:
                if child.label == "<args>":
                    arg_list.extend([(tree.label, x.label) for x in child.children])
                else:
                    arg_list.extend(__get_args_in_tree(child))
            return arg_list

        def __award_tag(tag: str):
            assert not tag.endswith("/>"), "self-closed tag is not allowed"
            if tag.startswith("</"):
                latest_tag = tag_stack.pop()
                latest_tag.award_exit(gt_tree)
                if latest_tag.label != f"<{tag[2:-1]}>":
                    latest_tag.exit_reward += cls.malform_penalty
                rewards.append((tag, latest_tag.get_combine_reward()))
            else:
                new_node = RewardNode(tag)
                if tag_stack:
                    tag_stack[-1].addkid(new_node)
                new_node.award_enter(gt_tree)
                tag_stack.append(new_node)
                rewards.append((tag, new_node.get_enter_reward()))

        def __award_arg(s: str):
            is_malformed = False
            if s not in ("<arg>", "</arg>"):
                s = PostProcessingTreeForRewards.preprocess_xml(s, ignore_angle_brackets=True)
            # hack: put it outside of the `if` to handle the case where 
            # the first or last tag is <arg> or </arg>
            arg_pool.append(s)
            if it.has_next() and s != "</arg>":
                pid, ps = next(it)
                if pid in valid_tag_ids and ps != "</arg>":
                    # hack: if the next tag is not </arg>, then the current arg is malformed,
                    # return to one recursion level up to handle the elements in `arg_pool`
                    rewards.append((ps, cls.malform_penalty))
                    is_malformed = True
                    return is_malformed
                is_malformed = __award_arg(ps)
            if not arg_pool:
                # hack: arg_pool would be cleared at the deepest recursion level
                return is_malformed
            
            sub_rewards = []
            arg_enter_reward = 0.0

            latest_tag = tag_stack[-1]
            if latest_tag.label == "<args>":
                latest_tag = latest_tag.get_parent()
            
            arg_content = [x for x in arg_pool if x not in ("<arg>", "</arg>")]
            arg_str = "".join(arg_pool)
            if arg_pool[-1] != "</arg>":
                arg_str += "</arg>"
            arg_text = ET.fromstring(arg_str).text
            arg_text = arg_text if arg_text else ""
            
            gt_arg_roots, gt_args = zip(*__get_args_in_tree(gt_tree))
            root2args = {root: [] for root in gt_arg_roots}
            for root, arg in zip(gt_arg_roots, gt_args):
                root2args[root].append(arg)

            def __args_edit_distance():
                _arg_content = [""] + arg_content
                nearest_arg = sorted([(arg, edit_distance("".join(_arg_content), arg)) for arg in gt_args], 
                                     key=lambda x: x[1])[0][0]
                dists = [edit_distance(acc_tok, arg) for acc_tok, arg in product(accumulate(_arg_content), [nearest_arg])]
                dist_diffs = [st_0 - st_1 for st_0, st_1 in zip(dists[:-1], dists[1:])]
                assert len(dist_diffs) == len(_arg_content) - 1
                return [(tok, d / len(dist_diffs)) for tok, d in zip(_arg_content[1:], dist_diffs)]

            if latest_tag is not None and latest_tag.label in gt_arg_roots:
                arg_enter_reward += 2 / Counter(gt_arg_roots)[latest_tag.label]
                if arg_text in root2args[latest_tag.label]:
                    arg_enter_reward += 2 / len(root2args[latest_tag.label])
            sub_rewards.insert(0, ("<arg>", arg_enter_reward))
            sub_rewards.extend(__args_edit_distance())
            if is_malformed:
                # get the appended malformed penalty
                penalty = rewards.pop()
                sub_rewards.append(penalty)
                is_malformed = False
            else:
                sub_rewards.append(("</arg>", arg_enter_reward))
            
            rewards.extend(sub_rewards)

            new_node = RewardNode("".join(arg_content))
            tag_stack[-1].addkid(new_node)
            new_node.award_enter(gt_tree)
            new_node.award_exit(gt_tree)

            arg_pool.clear()
            return is_malformed

        while it.has_next():
            pid, ps = next(it)
            if pid not in valid_tag_ids:
                rewards.append((ps, cls.malform_penalty))
            else:
                if ps != "<arg>":
                    __award_tag(ps)
                else:
                    __award_arg(ps)

        return rewards
                    
    @classmethod
    def award_induction(cls,
                        gt_xml: ET.Element,
                        pred_xml: ET.Element,
                        processor: PostProcessingTreeForRewards) -> float:
        raise NotImplementedError
        gt_seq = processor._convert_to_raw_label(processor._parse_tree(gt_xml))
        pred_seq = processor._convert_to_raw_label(processor._parse_tree(pred_xml))
        gt_ans = processor._infer(processor.prepare_args_for_infer(gt_seq))
        pred_ans = processor._infer(processor.prepare_args_for_infer(pred_seq))
        return PostProcessingTreeForRewards.comparing_ans(gt_ans, pred_ans)

    def batch_award(self,
                    label_tensors: List[torch.Tensor],
                    response_tensors: List[torch.Tensor]
                    ) -> Tuple[List[float], List[torch.Tensor], List[torch.Tensor]]:
        assert len(label_tensors) == len(response_tensors)
        labels, responses = _decode_label_and_resp(label_tensors, response_tensors, self.tokenizer)
        rewards = []
        label_ids_list = [x for x in label_tensors]
        pred_ids_list = [x for x in response_tensors]
        for i, (label, _) in enumerate(zip(labels, responses)):
            reward: List[Tuple[str, float]] = []
            _, gt_xml = _all_is_well(label)
            try:
                reward_format, remains = TreeRewards.award_format(response_tensors[i], self.tokenizer)
                reward_distance = TreeRewards.award_distance(gt_xml, remains, self.tokenizer)
                reward.extend(reward_format)
                reward.extend(reward_distance)
                rewards.append(reward)
            except:
                # remove this item from the batch
                label_ids_list.pop(i)
                pred_ids_list.pop(i)
        return rewards, label_ids_list, pred_ids_list