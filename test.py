from transformers import BertConfig, BertModel, BertTokenizerFast
import torch
import psutil
import time
import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

device = "cpu"

config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
model = BertModel.from_pretrained('bert-base-uncased', config=config).to(device)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

model.eval()

# print(model)

seq = "I ate a hotdog."
input = tokenizer(seq, return_tensors="pt", padding='max_length', max_length=12)

# compare
# 1. output of BERT model
# 2. manually call every layer
# 3. compre the output of the first hidden layer (first encoder block)
# 4. if same, then manual process correct, can use manual to extract output of attention layer


### BERT model
with torch.no_grad():
    out = model(**input)
    print(out.keys())

    # print(len(out.hidden_states))       # 13  (12 encoder blocks + 1 embedding layer)
    # print(out.hidden_states[-1].shape)      # (batch size, sequence length, embedding dim)
    # print(((out.last_hidden_state - out.hidden_states[-1]) ** 2).sum())     # 0, last element of out.hidden_states is the last_hidden_state
    # print(((embeddings - out.hidden_states[0]) ** 2).sum())         # 0, first element of out.hidden_states is the output of the embeddings layer
    # out.hidden_states: (output of embedding layer, output of 1st encoder, ..., output of the last encoder (12th for BERT-BASE) )


    # out.attentions: 12-tuple (for BERT BASE), one for each layer
    # out.attentions[0]: attention of first layer (batch size, # heads, seq length, seq length), # heads for BERT BASE is 12


### Manual first encoder output
input_ids = input['input_ids']
token_type_ids = input['token_type_ids']

# compute the actual attention mask passed to each attention layer
attention_mask = input['attention_mask']
attention_mask = attention_mask[:, None, None, :]
attention_mask = (1.0 - attention_mask) * torch.finfo(model.dtype).min

with torch.no_grad():
    # get embeddings of input tokens
    embeddings = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)   # output of embedding, input for first attention layer

    # get output of attention layer
    attention_out = model.encoder.layer[0].attention.self(embeddings, attention_mask=attention_mask) # 1-tuple of (batch size, sequence length, embedding dim)
    # print(attention_out)
    # print(len(attention_out))
    # print(attention_out[0].shape)

    attention_linear_out = model.encoder.layer[0].attention.output(attention_out[0], embeddings)    # refer to Attention is All You Need: Add & Norm
    feedforward_out1 = model.encoder.layer[0].intermediate(attention_linear_out)
    encoder_out = model.encoder.layer[0].output(feedforward_out1, attention_linear_out)     # Add & Norm here as well

print( ((out.hidden_states[1] - encoder_out) ** 2).sum() )      # 9.962e-11, pretty much the same

# further verify visually
# print(out.hidden_states[1][0, 0])
# print(encoder_out[0,0])
# indeed, same

### Manual (n + 1)-th encoder output
n = 6
embeddings = out.hidden_states[n]

with torch.no_grad():
    attention_out = model.encoder.layer[n].attention.self(embeddings, attention_mask=attention_mask)
    attention_linear_out = model.encoder.layer[n].attention.output(attention_out[0], embeddings)
    feedforward_out1 = model.encoder.layer[n].intermediate(attention_linear_out)
    encoder_out = model.encoder.layer[n].output(feedforward_out1, attention_linear_out)

print( ((out.hidden_states[n + 1] - encoder_out) ** 2).sum() )     
# indeed, manual procedure is correct.


# """
# Dataset test
# """
# from datasets import load_dataset, interleave_datasets
# from torch.utils.data import DataLoader

# bookcorpus = load_dataset("bookcorpus", split='train[:500]')
# wiki = load_dataset("wikipedia", "20220301.en", split='train[:500]')
# combined_dataset = interleave_datasets([bookcorpus, wiki])
# # print(bookcorpus)
# # print(wiki)
# # print(combined_dataset)     # union of columns, with None filled for those not present


# def tokenization(example):
#     # return_overflowing_tokens: break a long sequence into chunks of max_length
#     # can set stride=n to perform sliding window.
#     return tokenizer(example['text'], truncation=True, padding='max_length', max_length=12, return_overflowing_tokens=True)

# # print(tokenization(bookcorpus[0:3]))

# start_time = time.time()

# combined_dataset = combined_dataset.map(tokenization, batched=True, remove_columns=combined_dataset.column_names)
# combined_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

# # print(time.time() - start_time, "seconds.")
# ### 100K examples
# # batched = True: 11.5 seconds
# # batched = False: 46.17 seconds

# dataloader = DataLoader(combined_dataset, shuffle=True, batch_size=32)

# for batch in dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     print(outputs.last_hidden_state.shape)      # (batch_size, sequence length, embedding dim)
#     break

# # out = model(**combined_dataset[:3].to(device))

# # print(combined_dataset)
# # print(combined_dataset[:3])


# ### size of dataset:
# # RAM used:
# # print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# # Size of dataset:
# # size_gb = bookcorpus.dataset_size / (1024 ** 3)
# # print(f"Size of dataset is {size_gb}GB.")





# # First start with BERT TINY
# # tiny model, short sequences. visualize weight for short sequences (6 tokens?)