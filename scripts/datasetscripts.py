import json
import random
from typing import List, Any, Optional, Tuple, Dict

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

def dump_data(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")



def proportional_split(
        A: List,
        B: List,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        random_seed: int = 42
) -> Dict[str, Dict[str, List]]:
    """
    Split two lists A and B proportionally into train, validation, and test sets.
    Lists can have different lengths - each will be split according to the same ratios.

    Args:
        A: First list
        B: Second list
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        test_ratio: Proportion for test set (default: 0.1)
        shuffle: Whether to shuffle before splitting (default: True)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary containing train, val, test splits for both A and B
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(random_seed)

    def split_list(lst, shuffle_data):
        n = len(lst)
        indices = list(range(n))

        # Shuffle indices if requested
        if shuffle_data:
            random.shuffle(indices)

        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return {
            'train': [lst[i] for i in train_idx],
            'val': [lst[i] for i in val_idx],
            'test': [lst[i] for i in test_idx]
        }

    # Split both lists independently
    A_splits = split_list(A, shuffle)
    B_splits = split_list(B, shuffle)

    # Organize results
    result = {
        'train': {
            'A': A_splits['train'],
            'B': B_splits['train']
        },
        'val': {
            'A': A_splits['val'],
            'B': B_splits['val']
        },
        'test': {
            'A': A_splits['test'],
            'B': B_splits['test']
        }
    }

    return result


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
