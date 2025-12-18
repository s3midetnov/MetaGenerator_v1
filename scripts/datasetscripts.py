import json
import random
from typing import List, Any, Optional

def parse_concatenated_json(path) -> List[Any]:
    objects = []
    buffer = []
    brace_level = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Count opening/closing braces not inside strings
            brace_level += line.count("{")
            brace_level -= line.count("}")

            buffer.append(line)

            # When brace level drops to zero we have a complete JSON object
            if brace_level == 0 and buffer:
                block = "".join(buffer).strip()
                if block:
                    objects.append(json.loads(block))
                buffer = []

    return objects


def format_augmented_goal(
    s: str, premises: List[Any], max_len: Optional[int] = None, p_drop: float = 0.0
    ) -> str:
    """Format a state with retrieved premises and drop some of them with probability ``p_drop``."""
    aug_s = ""
    length = 0
    if max_len is None:
        max_len = 9999999999999999999999
    max_premises_len = max_len - len(bytes(s.encode("utf-8")))

    for p in premises:
        if random.random() < p_drop:
            continue
        p_str = f"{p}\n\n"
        l = len(bytes(p_str.encode("utf-8")))
        if length + l > max_premises_len:
            continue
        length += l
        aug_s = p_str + aug_s

    aug_s += s
    # print('xxxxxxxxxxx state xxxxxxxxxxxxxx = \n', aug_s)
    return aug_s

# Code that transforms Arend lib samples into trainable dataOld, LLM version 0
def jsonl_to_prompt_refactor0(entry : dict) -> dict:
    two_field_dict = dict()
    two_field_dict['prompt'] = str(entry['Context']) + " " + str(entry['Premises']) + " " + entry['Expected type']
    two_field_dict['completion'] = entry['Expression']
    return two_field_dict


# Code that transforms Arend lib samples into trainable dataOld, LLM version 1
def jsonl_to_prompt_refactor1(entry : dict) -> dict:
    two_field_dict = dict()
    two_field_dict['prompt'] = "<<<Context:>>> "+ str(entry['Context']) + "<<<Premises:>>> " + str(entry['Premises']) + "<<<Expected type:>>> " + entry['Expected type']
    two_field_dict['completion'] = entry['Expression']
    return two_field_dict

# Code that transforms Arend lib samples into trainable dataOld, LLM version 2

def jsonl_to_prompt_refactor2(entry : dict) -> dict:
        processed_prompt = {
            "instruction": f"[GOAL]\n{entry['Context']} {entry['Premises']} {entry['Expected type']}\n[PROOFSTEP]\n",
            "input": "",
            "output": entry["Expression"],
        }
        return processed_prompt
