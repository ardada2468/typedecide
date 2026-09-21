# PromptError "no answer-letter continuation" on a new tokenizer

## Symptoms

Training stops during preflight, within seconds of starting:

```text
error: Decision 'T-0-queue' (criterion 'queue'): No answer-letter continuation keeps the prompt boundary stable for this tokenizer, so a readout would score tokens that are not the answer. The prompt tail needs adjusting for this model.
```

During evaluation or export the same text arrives as a `PromptError` without the
decision prefix. It typically appears the first time you point `base_model` at a
model family you have not used before.

## Diagnosis

[`answer_slots`](../concepts/answer-slots.md) requires that, for one separator out of
`" "` and `""`, **every** letter appends exactly one token to the encoded prompt,
leaves the prompt's own tokens unchanged, and yields distinct ids. Find out which
condition fails:

```python
from transformers import AutoTokenizer

from typedecide import Criterion, Option, full_prompt
from typedecide.schema import LETTERS

MODEL = "your/model-id"
N_OPTIONS = 4

tokenizer = AutoTokenizer.from_pretrained(MODEL)
criterion = Criterion("probe", "Which option applies?", tuple(
    Option(f"opt{i}", f"Option {LETTERS[i]}") for i in range(N_OPTIONS)
))
prompt = full_prompt("Some ordinary evidence.", criterion)
base = tokenizer.encode(prompt, add_special_tokens=False)
print("prompt tail tokens:", tokenizer.convert_ids_to_tokens(base[-4:]))

for separator in (" ", ""):
    for letter in LETTERS[:N_OPTIONS]:
        ids = tokenizer.encode(prompt + separator + letter, add_special_tokens=False)
        print(repr(separator), letter,
              "appended", len(ids) - len(base),
              "prefix kept", ids[:len(base)] == base,
              "tokens", tokenizer.convert_ids_to_tokens(ids[len(base) - 1:]))
```

Read the output against these causes:

| Observation | Cause |
|---|---|
| `appended 2` for both separators | The tokenizer splits the letter, or emits a separate whitespace token. Common with byte-level or character-level vocabularies |
| `prefix kept False` | Adding the letter re-tokenizes the end of the prompt: `:` merges with what follows |
| `appended 1`, prefix kept, for only **some** letters | Later letters (often beyond `J` or so) are not single tokens. You have more options than this vocabulary supports |
| `encode` adds tokens even with `add_special_tokens=False` | The tokenizer ignores the flag. The library has no workaround for that |
| Two letters map to the same id | The vocabulary normalises case or strips the character |

## Fix

- **Too many options for this vocabulary.** Reduce the number of options on the
  affected criteria, or split one criterion into two staged ones. This is the only
  fix that needs no template change.

- **The boundary re-tokenizes, or the letter never lands as one token.** The prompt
  tail has to change for this model, as the message says. That is a change to
  `typedecide/prompt.py` **and** `web/prompt.js` in the same commit, followed by
  retraining every adapter built on the old prompt. Read
  [Prompt parity](../concepts/prompt-parity.md) first. The template is not
  configurable at run time in this release.

- **Choose another base model.** If you are evaluating model families, a tokenizer
  that cannot give one clean token per answer letter is a legitimate reason to
  prefer a different one. This project's measurements all use Qwen3-0.6B.

!!! danger "Do not catch the error and carry on"
    The check exists because the alternative runs to completion and produces numbers
    that mean nothing: the loss lands on, or the readout reads, a token that is not the
    answer.

## Verify

Run the probe through the library, which applies the full rule:

```python
from transformers import AutoTokenizer

from typedecide.export import answer_letter_slots

tokenizer = AutoTokenizer.from_pretrained("your/model-id")
slots, separator = answer_letter_slots(tokenizer, 4)
print(slots, repr(separator))
```

- It returns 4 distinct ids and a separator without raising.
- `typedecide train` then gets past preflight, and the resulting `manifest.json`
  contains an `answer_slots` entry for every criterion key.
- If you changed the template, the [parity check](prompt-parity-ci.md) passes.
