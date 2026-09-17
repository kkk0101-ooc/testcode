
import torch
from torch import nn

V, D = 10, 4
embedding = nn.Embedding(
    num_embeddings=V,
    embedding_dim=D,
    padding_idx=0,
)

input_ids = torch.tensor([
    [2, 5, 7],
    [4, 3, 0],
], dtype=torch.long)                    # [B=2, T=3]

hidden = embedding(input_ids)           # [2, 3, 4]

print(embedding.weight.shape)           # torch.Size([10, 4])
print(input_ids.shape, input_ids.dtype) # torch.Size([2, 3]) torch.int64
print(hidden.shape)                     # torch.Size([2, 3, 4])
torch.testing.assert_close(hidden[0, 1], embedding.weight[5])


def add(x, y):
    return x + y


from transformers import AutoTokenizer

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

samples = {
    "natural_language": "Tokenization changes the effective context length.",
    "chinese": "今天天气很好，我们学习分词。",
    "numbers": "The values are 123456, 3.14159, and 2026-09-11.",
    "code": "def add(x, y):\n    return x + y\n",
}

print("name_or_path:", tokenizer.name_or_path)
print("class:", type(tokenizer).__name__)
print("vocab_size:", tokenizer.vocab_size)
print("special_tokens_map:", tokenizer.special_tokens_map)

for category, text in samples.items():
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)

    print(f"\n[{category}]")
    print("repr(text):", repr(text))
    print("tokens:", tokens)
    print("ids:", ids)
    print("offsets:", encoded["offset_mapping"])
    print("decoded:", repr(tokenizer.decode(ids)))
    print("unk_count:", ids.count(tokenizer.unk_token_id))


def measure(text: str, tokenizer) -> dict:
    ids = tokenizer.encode(text, add_special_tokens=False)
    codepoints = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    token_count = len(ids)

    return {
        "codepoints": codepoints,
        "utf8_bytes": utf8_bytes,
        "tokens": token_count,
        "chars_per_token": (
            codepoints / token_count if token_count else float("nan")
        ),
        "bytes_per_token": (
            utf8_bytes / token_count if token_count else float("nan")
        ),
        "tokens_per_char": (
            token_count / codepoints if codepoints else float("nan")
        ),
    }

for category, text in samples.items():
    result = measure(text, tokenizer)
    print(category, result)


totals = {"codepoints": 0, "utf8_bytes": 0, "tokens": 0}

for text in samples.values():
    result = measure(text, tokenizer)
    for key in totals:
        totals[key] += result[key]

print({
    **totals,
    "chars_per_token": totals["codepoints"] / totals["tokens"],
    "bytes_per_token": totals["utf8_bytes"] / totals["tokens"],
    "tokens_per_char": totals["tokens"] / totals["codepoints"],
})


from collections import Counter

sequences = [list("low"), list("low"), list("lower")]

def pair_counts(sequences):
    return Counter(
        pair
        for sequence in sequences
        for pair in zip(sequence, sequence[1:])
    )

def merge_pair(sequence, pair):
    merged = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and (sequence[index], sequence[index + 1]) == pair
        ):
            merged.append(sequence[index] + sequence[index + 1])
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged

for round_index in range(1, 5):
    counts = pair_counts(sequences)
    best_pair = max(counts, key=counts.get)
    sequences = [merge_pair(sequence, best_pair) for sequence in sequences]
    print(round_index, best_pair, counts[best_pair], sequences)


import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

texts = ["hello world", "你好，世界"]
batch = tokenizer(texts, padding=True, return_tensors="pt")
embedding = model.get_input_embeddings()

print("input_ids:", batch["input_ids"].shape, batch["input_ids"].dtype)
print("attention_mask:", batch["attention_mask"].shape)
print("embedding weight:", embedding.weight.shape)

token_vectors = embedding(batch["input_ids"])
print("lookup output:", token_vectors.shape)

with torch.inference_mode():
    outputs = model(**batch)
print("last_hidden_state:", outputs.last_hidden_state.shape)

B, T = batch["input_ids"].shape
V, D = embedding.weight.shape
assert token_vectors.shape == (B, T, D)
assert outputs.last_hidden_state.shape == (B, T, D)

token_id = batch["input_ids"][0, 1]
torch.testing.assert_close(
    token_vectors[0, 1],
    embedding.weight[token_id],
)


output_embedding = model.get_output_embeddings()
if output_embedding is None:
    print("当前 AutoModel 不包含语言模型输出头")
else:
    tied = embedding.weight.data_ptr() == output_embedding.weight.data_ptr()
    print("output embedding:", output_embedding.weight.shape)
    print("weights tied:", tied)

