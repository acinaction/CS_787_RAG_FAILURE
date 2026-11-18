PROMPT_VERSION = "prompt_v1"
SYSTEM = "You are a helpful assistant."
STD = "Answer the question using the passages below. Give a short answer (a few words only)."
ABSTAIN_EXTRA = " If the passages do not contain the answer, reply exactly: I don't know."
MAX_CHARS = 12000

def build_user(row, template="std"):
    instr = STD + (ABSTAIN_EXTRA if template == "abstain" else "")
    tail = f"\n\nQuestion: {row['question']}"
    block = "\n\n".join(f"Passage {i+1}: {p['text']}" for i, p in enumerate(row["passages"]))
    room = MAX_CHARS - len(instr) - len(tail) - 2
    return f"{instr}\n\n{block[:room]}{tail}"   # truncate passages, never the question

def build_messages(row, template="std"):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(row, template)}]

def encode_prompt(tok, row, template="std"):
    """Token ids [1, P] of the chat prompt, exactly as used for generation AND for feature extraction.
    Works whether apply_chat_template returns a tensor (transformers 4.x) or a BatchEncoding (5.x)."""
    out = tok.apply_chat_template(build_messages(row, template), add_generation_prompt=True, return_tensors="pt")
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return out
