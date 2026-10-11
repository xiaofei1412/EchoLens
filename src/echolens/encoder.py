"""One Chinese encoder with 18 independent four-class aspect heads."""
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel


class AspectEncoder(nn.Module):
    def __init__(self, path, aspect_count, pretrained=True):
        super().__init__()
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        self.encoder = (AutoModel.from_pretrained(path, local_files_only=True, attn_implementation="sdpa")
                        if pretrained else AutoModel.from_config(config, attn_implementation="sdpa"))
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(config.hidden_size, aspect_count * 4)
        self.aspect_count = aspect_count

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state[:, 0]
        return self.head(self.dropout(hidden)).reshape(-1, self.aspect_count, 4)


def tokenized_loader(tokenizer, texts, labels, config, shuffle=False, generator=None):
    encoded = tokenizer(texts, truncation=True, max_length=config["max_length"], padding=False)
    records = [{k: encoded[k][i] for k in encoded} for i in range(len(texts))]

    def collate(indices):
        inputs = tokenizer.pad([records[i] for i in indices], padding=True, pad_to_multiple_of=8, return_tensors="pt")
        return inputs, torch.as_tensor(labels[indices] + 2, dtype=torch.long)

    return DataLoader(list(range(len(texts))), batch_size=config["batch_size"], shuffle=shuffle,
                      collate_fn=collate, num_workers=0, generator=generator)


def predict(model, loader):
    model.eval()
    outputs, times = [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for inputs, _ in loader:
            torch.cuda.synchronize()
            started = time.perf_counter()
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**inputs)
            outputs.append(output.argmax(dim=-1).cpu().numpy() - 2)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
    return np.concatenate(outputs), {"forward_seconds": sum(times), "batch_p50_seconds": float(np.median(times)),
                                     "batch_p95_seconds": float(np.percentile(times, 95)),
                                     "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                     "timing_scope": "Per batch GPU transfer + forward + prediction transfer, excludes tokenization and model loading"}
