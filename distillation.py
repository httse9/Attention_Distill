from transformers import BertConfig, BertModel, BertTokenizerFast
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from linear import LinearLayer
from dataset import create_dataset, preprocess_for_distillation
import numpy as np
import pickle
import time
from os.path import join
import json
import os
from plot import plot_distillation_linear_layers_loss
from utils import set_random_seed, seed_worker

VERBOSE = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""
Steps:
1. Create dataset (done)
    a. Load dataset
    b. Preprocess
2. Load BERT pretrained model (done)
3. Create MLP (done)
4. Training loop (done)
    a. get inputs/outputs of pretrained BERT
    b. get outputs of MLP
    c. minimize loss
5. Record/Visualize training loss
6. Record/Visualize parameters of MLP
7. hyperparameter search?
"""

"""
Todo:
save checkpoint regularly
"""

def distillation(dataset, bert_model, mlp_models, args):
    
    # prepare batches
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

    g = torch.Generator()
    g.manual_seed(args.seed)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=2,
                            worker_init_fn=seed_worker, generator=g)

    n_bert_layers = len(mlp_models)

    # optimizer
    optimizers = []
    for mlp_model in mlp_models:
        optimizers.append(torch.optim.AdamW(mlp_model.parameters(), lr=args.lr))
    loss_fn = nn.MSELoss()

    loss_list = []

    for _ in range(args.epoch):

        losses = [[] for _ in range(n_bert_layers)]   

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            ### compute actual attention mask passed to each attention layer
            attention_mask = batch['attention_mask']        # (batch size, sequence length)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(bert_model.dtype).min

            ### pass input through bert model
            with torch.no_grad():
                outputs = bert_model(**batch)
            
            # print(len(outputs.hidden_states))     # number of layers + 1
                
            for i, (mlp_model, optimizer) in enumerate(zip(mlp_models, optimizers)):

                ### Get BERT attention input/output pair
                atten_in = outputs.hidden_states[i]     # input
                with torch.no_grad():
                    atten_out = bert_model.encoder.layer[i].attention.self(atten_in, attention_mask=attention_mask)[0]  # output

                    # ### for testing only, output should be 0 if implementation is correct.
                    # atten_linear_out = bert_model.encoder.layer[i].attention.output(atten_out, outputs.hidden_states[i])
                    # feedforward_out1 = bert_model.encoder.layer[i].intermediate(atten_linear_out)
                    # encoder_out = bert_model.encoder.layer[i].output(feedforward_out1, atten_linear_out)
                    # print( ((encoder_out - outputs.hidden_states[i + 1]) ** 2).sum())     # 0

                # print(atten_in.shape)     # (batch size, sequence length, embedding dim)
                # print(atten_out.shape)
                    
                ### Get output of MLP
                mlp_out = mlp_model(atten_in)
                # print(mlp_out.shape)      # (batch size, sequence length, embedding dim)

                ### compute loss
                loss = loss_fn(mlp_out, atten_out)
                losses[i].append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        losses = [np.mean(l) for l in losses]
        print(losses)
        loss_list.append(losses)

    return np.array(loss_list).T    # (# layers, # epochs)
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # training
    parser.add_argument("max_seq_length", type=int, help="Max sequence length to use. Affects the \
                        input/output dimension of the MLP.")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument('--epoch', default=5, type=int, help="Number of pass through the dataset")

    # BERT
    parser.add_argument("--bert", type=str, default="prajjwal1/bert-tiny", help="BERT model to use")

    # datasets
    parser.add_argument("--wiki", action='store_true', help="Whether to use wiki dataset")
    parser.add_argument("--book", action="store_true", help="Whether to use bookcorpus dataset")
    parser.add_argument("--n_examples", type=int, default=None, help="Number of examples from dataset used. \
                        Mainly for specifying a small number for code testing. Default is None, and uses \
                        the entire dataset.")
    
    # experiment name for saving checkpoints, trained model, config, etc.
    parser.add_argument('--exp_name', default='test', type=str)

    # random seed
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # saving
    parser.add_argument("--save_every_epoch", type=int, default=None, help="Save model checkpoint every [.] epochs.")

    ### parse args
    args = parser.parse_args()
    set_random_seed(args.seed)

    ### experiment dir
    exp_dir = join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=False)
    # save args
    with open(join(exp_dir, "config.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)


    ### create/preprocess datasets
    start_time = time.time()
    dataset = create_dataset(args.wiki, args.book, args.n_examples, VERBOSE)
    dataset = preprocess_for_distillation(dataset, args.bert, args.max_seq_length, VERBOSE)
    print("Process dataset time:", time.time() - start_time, "seconds.")

    ### create models
    # BERT
    config = BertConfig.from_pretrained(args.bert, output_hidden_states=True, output_attentions=True)
    bert_model = BertModel.from_pretrained(args.bert, config=config).to(device)
    bert_model.eval()

    n_bert_layers = bert_model.config.num_hidden_layers

    # MLP for each attention layer to distill
    mlp_models = []
    for i in range(n_bert_layers):
        mlp_model = LinearLayer(args.max_seq_length, config.hidden_size, join(exp_dir, f"layer{i+1}")).to(device)
        mlp_models.append(mlp_model)

    

    ### distillation
    start_time = time.time()
    losses = distillation(dataset, bert_model, mlp_models, args)
    print("Distillation time:", time.time() - start_time, "seconds.")

    # save loss
    with open(f"experiments/{args.exp_name}/loss.pkl", "wb") as f:
        pickle.dump(losses, f)

    # plot loss
    plot_distillation_linear_layers_loss(losses, exp_dir)

    

    ### visualize weight matrix, save model
    for mlp_model in mlp_models:
        mlp_model.visualize()
        mlp_model.save_model()




#####
# tests on GPU utilization:
# setting; time

# max sequence length: 12 (book)
# batch size 256 
# -c 4, num workers = 10, batch size 256; 71.44 seconds, gpu util ~65%
# -c 4, num workers = 6, batch size 256; 65.12 seconds, gpu util ~65%
# -c 4, num workers = 4, batch size 256; 59.26 seconds, gpu util ~70%
# -c 4, num workers = 3, batch size 256; 58.15s seconds, gpu util ~70%
# -c 4, num workers = 2, batch size 256; 55.30 seconds, gpu util ~ 70%
# -c 4, num workers = 1, batch size 256; 74.08 seconds, gpu util ~ 53%

# batch size 1024
# -c 4, num workers = 8, batch size 1024; 43.48 seconds, gpu util ~90%
# -c 4, num workers = 4, batch size 1024; 37.51 seconds, gpu util ~90%
# -c 4, num workers = 2, batch size 1024; 34.99 seconds, gpu util ~90%

# batch size 2048
# -c 4, num workers = 6, batch size 2048; 39.99 seconds, gpu util ~95%
# -c 4, num workers = 4, batch size 2048; 36.76 seconds, gpu util ~90%
# -c 4, num workers = 2, batch size 2048; 39.69 seconds, gpu util ~85%
        
# batch size 4096
# -c 4, num workers = 2, batch size 4096; 41.67 seconds, gpu util ~75%
# -c 4, num workers = 2, batch size 4096; 42.20 seconds, gpu util ~70%
        
### Best Config: batch size 1024, 2 workers
### !!! Note: for different sequence length (and thus different data size), figure out optimal batch size and num workers again!!!
    

# max sequence length: 36 (wiki 1K), mem ~ 1GB
# batch size 1024, num workers 1; 35.45s gpu util 99%
# batch size 1024, num workers 2; 35.66s, gpu util 99%
# batch size 1024, num workers 4; 35.88s, gpu util 99%
# batch size 1024, num workers 8; 36.83s gpu util 99%
# batch size 512, num workers 1; 40.74s gpu util 98%
# batch size 512, num workers 2; 40.50s gpu util 98%
# batch size 512, num workers 4; 40.99s gpu util 98%
# batch size 2048, num workers 1; 34.04s gpu util 99%
# batch size 2048, num workers 2; 34.14s gpu util 99%
# batch size 2048, num workers 4; 34.50s gpu util 99%
# batch size 1500, num workers 2; 34.75s gpu util 99%
# batch size 4096, num workers 2; 34.70s gpu util 99.5%
### FINAL: batch size 2048, num workers 2

### tokenization speed test (wiki, 20K):
# num_proc = 1; ~70-75 examples/s
# num_proc = 2; ~120 examples/s; 264s
# num_proc = 4; ~320 examples/s; 165s
# num_proc = 6; ~320 examples/s; 165s
# num_proc = 8; ~300 examples/s; 165s
# num_proc = 12; ~250 examples/s; 226s
    