import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, default_data_collator
from modeling_llama import LlamaForCausalLM
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp8'], help='precision')
args = parser.parse_args()


checkpoint = "deepseek-ai/deepseek-llm-7b-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
config = AutoConfig.from_pretrained(checkpoint)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16


model = LlamaForCausalLM(config, precision=args.precision).to(device=device, dtype=dtype)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

from datasets import load_dataset
dataset = load_dataset("PeterBrendan/Ads_Creative_Text_Programmatic")
train_dataset = dataset["train"]

small_dataset = train_dataset.select(range(100))

split_dataset = small_dataset.train_test_split(
    test_size=0.1, 
    seed=42,
    shuffle=True 
)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=default_data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=default_data_collator)

model.train()

for epoch in range(1, 11):
    print('epoch', epoch)
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs[1]
        loss.backward()
        optimizer.step()

print('evluation')
eval_loss = 0
with torch.no_grad():
    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs[1]
        print('loss', loss)
        eval_loss += loss.item()
avg_eval_loss = eval_loss / len(eval_dataloader)
print('avg_eval_loss', avg_eval_loss)
model.train()

# torch.cuda.synchronize()
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# starter.record()

# torch.cuda.synchronize()
# ender.record()
# torch.cuda.synchronize()
# curr_time = starter.elapsed_time(ender)

# mean_syn = curr_time / 1000

# print(mean_syn)
# peak_memory = torch.cuda.max_memory_allocated() / (1024**2) 
# print(f"{dtype} - Peak memory: {peak_memory:.2f} MB")
# print("Batch size:", inputs["input_ids"].shape)
