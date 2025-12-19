import torch

from scripts.datasetscripts import format_augmented_goal, parse_concatenated_json


def ask_model_prompt(prompt : dict, checkpoint_path : str, model, device) -> str :
    prompt_ = format_augmented_goal(
            prompt["Expected type"],
            prompt["Premises"],
            512,
            0.0,
        )
    prompts = [prompt_]
    enc = model.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=getattr(model.hparams, "max_inp_seq_len", 512),
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model.generator.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=4, # getattr(model.hparams, "num_beams", 4),
            length_penalty=getattr(model.hparams, "length_penalty", 0.0),
            max_length=getattr(model.hparams, "max_oup_seq_len", 128),
        )
    preds = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for p in preds:
        return p


def ask_model_index(index : int, prompts, checkpoint_path : str, model, device) -> str :
    if type(prompts) == str:
        parsed_prompts = parse_concatenated_json(prompts)
        myprompt = parsed_prompts[index]
    if type(prompts) == list:
        myprompt = prompts[index]
    return ask_model_prompt(myprompt, checkpoint_path, model, device)