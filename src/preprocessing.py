import os
import json
import argparse
from typing import List, Dict, Optional

class Preprocessing:
    def __init__(self, src_paths: List, dst_paths: List, create_manifest_at: Optional[str] = None):
        assert isinstance(src_paths, list) and isinstance(dst_paths, list) and len(src_paths) == len(dst_paths)
        self._src_paths = src_paths
        self._dst_paths = dst_paths
        self._map = {p_src: p_dst for p_src, p_dst in zip(src_paths, dst_paths)}
        self._src_dict = {p: [] for p in src_paths}
        self._dst_dict = {p: [] for p in dst_paths}
        self.manifest_path = create_manifest_at
        self.INST = None

    def get_src(self) -> Dict:
        return self._src_dict
    
    def get_dst(self) -> Dict:
        return self._dst_dict

    def load(self) -> List:
        for src_path in self._src_paths:
            assert os.path.exists(src_path)
            src = self._load_src(src_path)
            self._src_dict.update({src_path: src})

    def dump(self):
        assert self._check_not_empty(), "please load and process data first"
        if self.INST is None:
            print("INSTRUCTION would fall back to blank due to not specified")
        for p_dst, dst in self._dst_dict.items():
            data = [
                {
                    "instruction": "" if self.INST is None else self.INST,
                    "input": f"Input:\n{elem['question']}\n",
                    "output": f"Output:\n{elem['program']}",
                } 
                for elem in dst
            ]
            with open(p_dst, "w", encoding="utf-8") as f:
                print(f"Dump to {p_dst}")
                json.dump(data, f, ensure_ascii=False, indent=4)
            if self.manifest_path is not None:
                self._dump_manifest(p_dst)
    
    def _get_inst(self):
        raise NotImplementedError
    
    def _dump_manifest(self, p_dst):
        assert self.manifest_path is not None
        added_dataset = {
            os.path.basename(p_dst): {
                "file_name": os.path.basename(p_dst),
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                }
            }
        }
        data = {}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data.update(added_dataset)
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def _load_src(self, src_path: str):
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                src = json.load(f)
            assert isinstance(src, list)
        except Exception as e:
            print(f"Load src file error: {e}")
            src = []
        return src
    
    def _check_not_empty(self):
        return all([len(self._src_dict[p]) > 0 for p in self._src_paths]) and all([len(self._dst_dict[p]) > 0 for p in self._dst_paths])
    
    def _check_len(self):
        return all([len(self._src_dict[p]) == len(self._dst_dict[self._map[p]]) for p in self._src_paths])
    
class PreprocessingSeq(Preprocessing):
    def __init__(self, src_paths: List, dst_paths: List, create_manifest_at: Optional[str] = None):
        super().__init__(src_paths, dst_paths, create_manifest_at)
        self.INST = self._get_inst()

    def _get_inst(self):
        from utils.constants import INST_TEMPLATE_SEQ
        return INST_TEMPLATE_SEQ
    
    def process(self):
        def _get_program_seq(program: List) -> str:
            seq = []
            for item in program:
                func = item['function']
                inputs = item['inputs']
                args = ''
                for input in inputs:
                    args += ' <arg> ' + input
                seq.append(func + args)
            seq = ' <func> '.join(seq)
            return seq
        for p_src, src in self._src_dict.items():
            if not src:
                raise ValueError(f"please load data first")
            p_dst = self._map[p_src]
            for i, item in enumerate(src):
                question = item["question"]
                program = _get_program_seq(item["program"])
                self._dst_dict[p_dst].append({"id": i, "question": question, "program": program})
        assert self._check_len(), "length of src and dst not equal"

class PreprocessingTree(Preprocessing):
    def __init__(self, src_paths: List, dst_paths: List, create_manifest_at: Optional[str] = None, simple: bool = False):
        super().__init__(src_paths, dst_paths, create_manifest_at)
        self.is_simple = simple
        self.INST = self._get_inst()
        
        from utils.constants import FUNCTION_MAP, ARG_TOKENS
        self.FUNCTION_MAP = FUNCTION_MAP
        self.ARG_TOKENS = ARG_TOKENS

    def _get_inst(self):
        if self.is_simple:
            from utils.constants import INST_TEMPLATE_TREE_SIMPLE
            return INST_TEMPLATE_TREE_SIMPLE
        else:
            from utils.constants import INST_TEMPLATE_TREE_COMPLEX
            return INST_TEMPLATE_TREE_COMPLEX
    
    def _find_root_indices(self, program):
        # All indices are initially assumed to be roots
        possible_roots = set(range(len(program)))
        # Remove any index that is a dependency of another function
        for func in program:
            for dep in func['dependencies']:
                possible_roots.discard(dep)
        return possible_roots
    
    def _build_dependency_tree(self, program, func_index, level=0):
        func = program[func_index]
        func_name = func['function']
        func_name = "QueryName" if func_name == "What" else func_name
        func_start = self.FUNCTION_MAP[func_name]["start"]
        dependencies = func['dependencies']
        inputs = func['inputs']
        children = []
        for dep_index in dependencies:
            children.append(self._build_dependency_tree(program, dep_index, level + 1))
        if self.is_simple:
            if inputs:
                assert all([";;" not in inp for inp in inputs])
                input_str = ";;".join(inputs)
                func_start = func_start[:-1]
                func_start = f"{func_start} args=\"{input_str}\">"
            if not children:
                func_start = func_start[:-1]
                func_start = f"{func_start} />"
                children.insert(0, func_start)
            else:
                children.insert(0, func_start)
                children.append(self.FUNCTION_MAP[func_name]["end"])
        else:
            if inputs:
                input_list = [self.ARG_TOKENS[0]]
                for inp in inputs:
                    input_list.append(self.ARG_TOKENS[2])
                    input_list.append(inp)
                    input_list.append(self.ARG_TOKENS[3])
                input_list.append(self.ARG_TOKENS[1])
                children.append("".join(input_list))
            children.insert(0, func_start)
            children.append(self.FUNCTION_MAP[func_name]["end"])
        return "".join(children)
    
    def process(self):
        def _get_program_tree(program: List, root_index: int) -> str:
            tree = self._build_dependency_tree(program, root_index)
            return tree
        for p_src, src in self._src_dict.items():
            if not src:
                raise ValueError(f"please load data first")
            p_dst = self._map[p_src]
            for i, item in enumerate(src):
                question = item["question"]
                program = item["program"]
                root_indices = self._find_root_indices(program)
                tree = None
                try:
                    assert len(root_indices) == 1, "programmes have multiple root"
                    root_index = root_indices.pop()
                    if self.is_simple:
                        tree = _get_program_tree(program, root_index)
                    else:
                        tree = _get_program_tree(program, root_index)
                except Exception as e:
                    print(f"exception when parsing {i}: {e}")
                if tree is not None:
                    self._dst_dict[p_dst].append({"id": i, "question": question, "program": tree})
        assert self._check_len(), "length of src and dst not equal"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_paths", type=str, nargs="+", required=True)
    parser.add_argument("--dst_paths", type=str, nargs="+", required=True)
    parser.add_argument("--create_manifest_at", type=str, default=None)
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--seq", action="store_true")
    args = parser.parse_args()
    if args.seq:
        preprocessor = PreprocessingSeq(args.src_paths, args.dst_paths, args.create_manifest_at)
    else:
        preprocessor = PreprocessingTree(args.src_paths, args.dst_paths, args.create_manifest_at, args.simple)
    preprocessor.load()
    preprocessor.process()
    preprocessor.dump()