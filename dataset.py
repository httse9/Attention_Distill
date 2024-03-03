from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizerFast

NUM_PROC = 4

def create_dataset(wiki, book, n_examples=None, verbose=False):
    """
    wiki: (bool) whether to use wiki dataset
    book: (bool) whether to use bookcorpus dataset
    n_examples: (int) number of examples from dataset to use
    """
    assert wiki or book       # at least use one dataset
    if verbose:
        print(f"Using dataset(s): {'wiki ' if wiki else ''}{'bookcorpus ' if book else ''}")

    n_datasets = int(wiki) + int(book)
    if n_examples is not None:
        n_examples_per_dataset = n_examples // n_datasets
    else:
        n_examples_per_dataset = ""

    # add_special_tokens=False, can use this to avoid adding CLS or SEP
    combined_dataset = []
    if wiki:
        wiki = load_dataset("wikipedia", "20220301.en", split=f'train[:{n_examples_per_dataset}]', trust_remote_code=True, num_proc = NUM_PROC)
        combined_dataset.append(wiki)
    if book:
        book = load_dataset("bookcorpus", split=f'train[:{n_examples_per_dataset}]', num_proc = NUM_PROC)
        combined_dataset.append(book)

    combined_dataset = concatenate_datasets(combined_dataset)
    if verbose:
        print(combined_dataset)

    # test:
    # Command; n_rows
    # 1. --wiki; 6458670
    # 2. --book; 74004228
    # 3. --wiki --book; 80462898
    # 4. --wiki --n_examples 10000; 10000
    # 5. --book --n_examples 10000; 10000
    # 6. --wiki --book --n_examples 10000; 10000

    return combined_dataset

def preprocess_for_distillation(dataset, bert_model, max_length, verbose=False):
    """
    Preprocess dataset for distillation: 
    1. truncate long sequence to fixed length
    2. pad short sequence to fixed length 

    bert_model: the BERT model to use
    max_length: the fixed sequence length
    """

    tokenizer = BertTokenizerFast.from_pretrained(bert_model)

    def tokenization(example):
        # return_overflowing_tokens: break a long sequence into chunks of max_length
        # can set stride=n to perform sliding window.
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=max_length, return_overflowing_tokens=True)

    dataset = dataset.map(tokenization, batched=True, remove_columns=dataset.column_names, num_proc = NUM_PROC)

    if verbose:
        print("Preprocessed dataset:", dataset)

    return dataset

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()

#     # BERT
#     parser.add_argument("max_seq_length", type=int, help="Max sequence length to use. Affects the \
#                         input/output dimension of the MLP.")
#     parser.add_argument("--bert", type=str, default="prajjwal1/bert-tiny", help="BERT model to use")

#     # datasets
#     parser.add_argument("--wiki", action='store_true', help="Whether to use wiki dataset")
#     parser.add_argument("--book", action="store_true", help="Whether to use bookcorpus dataset")
#     parser.add_argument("--n_examples", type=int, default=None, help="Number of examples from dataset used. \
#                         Mainly for specifying a small number for code testing. Default is None, and uses \
#                         the entire dataset.")

#     args = parser.parse_args()


#     ### create/preprocess datasets
#     VERBOSE = True
#     dataset = create_dataset(args.wiki, args.book, args.n_examples, VERBOSE)
#     dataset = preprocess_for_distillation(dataset, args.bert, args.max_seq_length, VERBOSE)