"""The canonical prompt, mirroring web/prompt.js exactly.

If these two drift, the fine-tune optimises a prompt the browser never sends and
accuracy quietly collapses. check_parity.py proves they agree.
"""

LETTERS = "ABCDEFGHIJKLMNOPQRST"

INSTRUCTION = (
    "Apply the criterion to the evidence and choose exactly one listed option. "
    "Answer with only the uppercase letter of that option."
)


def head_text(state: str) -> str:
    """Everything up to and including the state. Identical across every criterion."""
    return f"{INSTRUCTION}\n\nEvidence:\n{state}\n"


def tail_text(criterion: dict) -> str:
    """One criterion's own question and options. Never shared."""
    lines = [f"\nCriterion: {criterion['question']}", "Options:"]
    for letter, option in zip(LETTERS, criterion["options"]):
        lines.append(f"{letter}. {option['description']}")
    lines.append("Answer:")
    return "\n".join(lines)


def full_prompt(state: str, criterion: dict) -> str:
    return head_text(state) + tail_text(criterion)


def answer_slots(tokenizer, prompt: str, count: int):
    """Token ids of the answer letters at this exact prompt boundary.

    These are not contiguous in any vocabulary we have checked, and whether the
    natural continuation is " A" or "A" depends on the preceding character, so both
    are probed and we keep whichever appends exactly one token without disturbing
    the prompt's own tokenization.
    """
    base = tokenizer.encode(prompt, add_special_tokens=False)
    if not base:
        raise ValueError("Prompt encoded to zero tokens")
    for separator in (" ", ""):
        slots = []
        for letter in LETTERS[:count]:
            encoded = tokenizer.encode(prompt + separator + letter, add_special_tokens=False)
            if len(encoded) != len(base) + 1 or encoded[: len(base)] != base:
                slots = []
                break
            slots.append(encoded[-1])
        if slots and len(slots) == len(set(slots)):
            return slots, separator
    raise ValueError("No answer-letter continuation keeps the prompt boundary stable.")
