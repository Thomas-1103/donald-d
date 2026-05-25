import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    logging,
)

from .plotting import plot_2d_matrix_with_ellipses

logging.set_verbosity_error()


def sentence_embeddings_to_matrix_from_hf_hidden_states(
    hidden_states,
    tokenizer=None,
    input_ids=None,
    skip_special_tokens=True,
):
    hs_list = []

    for h in hidden_states:
        if hasattr(h, "detach"):
            h = h.detach().cpu().numpy()

        arr = np.asarray(h)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)

        if arr.ndim != 2:
            raise ValueError(f"Hidden state layer must be 2D after squeeze. Got shape: {arr.shape}")

        hs_list.append(arr)

    if len(hs_list) > 1:
        stack = np.stack(hs_list[1:], axis=0)
    else:
        stack = np.stack(hs_list, axis=0)

    tokens_clean = None

    if tokenizer is not None and input_ids is not None:
        ids = np.asarray(input_ids).ravel().tolist()
        toks = tokenizer.convert_ids_to_tokens(ids)

        if skip_special_tokens and hasattr(tokenizer, "all_special_tokens"):
            mask_keep = np.array([t not in tokenizer.all_special_tokens for t in toks], dtype=bool)
        else:
            mask_keep = np.ones(len(toks), dtype=bool)

        keep_idx = np.where(mask_keep)[0]

        if keep_idx.size > 0:
            stack = stack[:, keep_idx, :]
            tokens_clean = [toks[i] for i in keep_idx]
        else:
            tokens_clean = toks

    M = stack.mean(axis=2)

    return M, tokens_clean


def DONALD_D(
    sentence,
    model_name="bert-base-uncased",
    model_kind="encoder",
    output_file="DONALD-D_visualisation.svg",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if model_kind == "encoder":
        model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(device).eval()

    tokens = tokenizer(sentence, return_tensors="pt", truncation=True)

    tokens_cpu = {k: v.cpu() for k, v in tokens.items()}
    tokens_device = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        out = model(
            **tokens_device,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = out.hidden_states

    M, token_list = sentence_embeddings_to_matrix_from_hf_hidden_states(
        hidden_states,
        tokenizer=tokenizer,
        input_ids=tokens_cpu["input_ids"],
    )

    print(f"Matrix shape: {M.shape}")

    return plot_2d_matrix_with_ellipses(
        M,
        tokens=token_list,
        output_file=output_file,
    )