from collections import deque
import os
import re
import json
from xml.etree import ElementTree as ET
from typing import List, Dict, Tuple, Optional

class Postprocessing:
    def __init__(self, generated_file_path: str, output_file_name: Optional[str] = None):
        assert os.path.exists(generated_file_path), f"{generated_file_path} does not exist"
        self.generated_file_path = generated_file_path
        self.output_file_name = output_file_name
        self._gen_list = []
        self._ans_dict = {}
        self.engine = None

    def init_engine(self, kb_file_path: str):
        from kopl.kopl import KoPLEngine
        self.engine = KoPLEngine(json.load(open(kb_file_path)))

    def get_gen(self) -> List:
        return self._gen_list
    
    def get_ans(self) -> Dict:
        return self._ans_dict

    def load(self):
        with open(self.generated_file_path, "r", encoding="utf-8") as f:
            jsons_list = list(f)
        for json_str in jsons_list:
            self._gen_list.append(json.loads(json_str))

    def prepare_args_for_infer(self, seq: str) -> Tuple[List, List]:
        func_list = []
        inputs_list = []
        chunks = seq.split('<func>')
        for chunk in chunks:
            chunk = chunk.strip()
            res = chunk.split('<arg>')
            res = [_.strip() for _ in res]
            if len(res) > 0:
                func = res[0]
                inputs = []
                if len(res) > 1:
                    for x in res[1:]:
                        inputs.append(x)
                else:
                    inputs = []
                func_list.append(func)
                inputs_list.append(inputs)
        return func_list, inputs_list

    def _infer(self, program: Tuple[List, List]):
        assert self.engine is not None, "inference engine is not initialized"
        ans = self.engine.forward(*program, ignore_error=True)
        if ans is None:
            ans = 'no'
        elif isinstance(ans, list):
            ans = ans[0] if len(ans) > 0 else 'None'
        return ans
    
    def infer(self, programs: List[Tuple[List, List]], key: str = "default"):
        self._ans_dict.update({key: [self._infer(program) for program in programs]})
        return self._ans_dict[key]
    
    def comparing_ans(self):
        assert self._check_ans(), "answer dict is empty"
        length = len(next(iter(self._ans_dict.values())))
        assert all(len(val) == length for val in self._ans_dict.values()), "answer lists should have the same length"
        def _equal(gt, pred):
            assert gt is not None, "ground truth should not be None"
            if not pred:
                return False
            if isinstance(gt, list):
                if isinstance(pred, list):
                    return gt[0] == pred[0]
                return gt[0] == pred
            else:
                if isinstance(pred, list):
                    return gt == pred[0]
                return gt == pred
        ans_zip = zip(*self._ans_dict.values())
        comp_list = []
        for ans in ans_zip:
            if not all(_equal(ans[0], x) for x in ans):
                comp_list.append(False)
                continue
            comp_list.append(True)    
        return comp_list
    
    def dump(self):
        assert self.output_file_name is not None, "output file name is not specified"
        assert self._check_ans(), "answer dict is empty"
        assert all([isinstance(x, str) for lst in self._ans_dict.values() for x in lst]), "any answer lists should be a list of strings"
        for k, v in self._ans_dict.items():
            with open(f"{self.output_file_name}-{k}.txt", "w", encoding="utf-8") as f:
                for ans in v:
                    f.write(f"{ans}\n")
    
    def _check_ans(self):
        return self._ans_dict
    
class PostprocessingSeq(Postprocessing):
    def __init__(self, generated_file_path: str, output_file_path: Optional[str] = None):
        super().__init__(generated_file_path, output_file_path)

class PostprocessingTree(Postprocessing):
    def __init__(self, generated_file_path: str, output_file_name: Optional[str] = None, simple: bool = False):
        super().__init__(generated_file_path, output_file_name)
        self.is_simple = simple
    
    def _preprocess_xml(self, s: str) -> str:
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
        def _escape_xml_attribute_value(s):
            s = s.replace("<", "&lt;") \
                 .replace(">", "&gt;")
            return s
        if self.is_simple:
            s = re.sub(r'args="(.*?)"', lambda match: f'args="{_escape_xml_attribute_value(match.group(1))}"', s)
        else:
            s = _escape_angle_brackets_within_arg(s)
        return s
    
    def _remove_leftmost_leaf(self, node: ET.Element):
        # 删除最左侧的叶子节点并返回其值
        is_root = False
        if len(node) == 0:
            root_tag = f"<{node.tag}>"
            is_root = True
            return root_tag, False, is_root
        
        if len(node[0]) == 0:  # 如果第一个子节点是叶子节点
            if node[0].tag == "args": # 如果args所有的arg被remove，args成为叶子节点，舍弃之
                is_arg = False
                node.remove(node[0])
                return None, is_arg, is_root 
            if node.tag != "args":
                leaf_value = f"<{node[0].tag}>"  # 获取叶子节点的值
                is_arg = False
            else:
                leaf_value = node[0].text
                is_arg = True
            node.remove(node[0])  # 从树中删除叶子节点
            return leaf_value, is_arg, is_root
        else:
            return self._remove_leftmost_leaf(node[0])

    def _remove_leftmost_leaf_simple(self, node: ET.Element):
        is_root = False
        args = []
        def _check_args(node):
            if "args" in node.attrib:
                args.extend(node.attrib["args"].split(";;"))
                
        if len(node) == 0:
            root_tag = f"<{node.tag}>"
            is_root = True
            _check_args(node)
            return [root_tag, *args], is_root
        if len(node[0]) == 0:
            leaf_value = f"<{node[0].tag}>"
            _check_args(node[0])
            node.remove(node[0])
            return [leaf_value, *args], is_root
        else:
            return self._remove_leftmost_leaf_simple(node[0])
        
    def _convert_to_raw_label(self, lst):
        from utils.constants import FUNCTION_LIST
        func_pattern = set([f"<{x}>" for x in FUNCTION_LIST])
        remove_angle_brac = lambda x: x.replace("<", "").replace(">", "")
        s_lst = []
        if lst[0] in func_pattern:
            s_lst.append(remove_angle_brac(lst[0]))
        else:
            raise
        for i in range(1, len(lst)):
            curr_elem = lst[i]
            if curr_elem in func_pattern:
                s_lst.append("<func>")
                s_lst.append(remove_angle_brac(curr_elem))
            else:
                s_lst.append("<arg>")
                s_lst.append(curr_elem)
        return " ".join(s_lst)
        
    def load(self):
        with open(self.generated_file_path, "r", encoding="utf-8") as f:
            jsons_list = list(f)
        for json_str in jsons_list:
            self._gen_list.append(json.loads(json_str))
        for i, elem in enumerate(self._gen_list):
            for k, v in elem.items():
                v = self._preprocess_xml(v.replace("Output:\n", ""))
                try:
                    root = ET.fromstring(v)
                except Exception as e:
                    self._gen_list[i][k] = ""
                    continue
                result_list = self._parse_tree(root)
                self._gen_list[i][k] = self._convert_to_raw_label(result_list)

    def _parse_tree(self, root: ET.Element):
        is_root = False
        result_list = []
        if not self.is_simple:
            args = deque()
                    # 循环处理树直到所有叶子节点都被移除
            while not is_root:
                leaf_value, is_arg, is_root = self._remove_leftmost_leaf(root)
                if leaf_value is not None:
                    if not is_arg:
                        result_list.append(leaf_value.strip())
                        while len(args) != 0:
                            result_list.extend([x.strip() for x in args])
                            args.clear()
                    else:
                        args.append(leaf_value)
        else:
            while not is_root:
                content, is_root = self._remove_leftmost_leaf_simple(root)
                result_list.extend([x.strip() for x in content])
        return result_list
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_file_path", type=str, required=True)
    parser.add_argument("--output_file_name", type=str, required=True)
    parser.add_argument("--kb_file_path", type=str, required=True)
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--seq", action="store_true")
    args = parser.parse_args()
    if args.seq:
        postprocessor = PostprocessingSeq(args.generated_file_path, args.output_file_name)
    else:
        postprocessor = PostprocessingTree(args.generated_file_path, args.output_file_name, args.simple)
    postprocessor.load()
    from pprint import pprint
    pprint(postprocessor.get_gen()[0])
    postprocessor.init_engine(args.kb_file_path)
    from collections import defaultdict
    programs_dict = defaultdict(list)
    for elem in postprocessor.get_gen():
        for k, v in elem.items():
            func_list, inputs_list = postprocessor.prepare_args_for_infer(v)
            programs_dict[k].append((func_list, inputs_list))
    for k, v in programs_dict.items():
        postprocessor.infer(v, k)
    from collections import Counter
    ans_matched_list = postprocessor.comparing_ans()
    print(f"answer acc: {Counter(ans_matched_list)[True] / len(ans_matched_list)}")
    postprocessor.dump()

