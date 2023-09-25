import pandas as pd
from datasets import Dataset
from get_model import *


_ ,tokenizer = get_model_tokenizer()

def make_dataset(data_path = "NvidiaDocumentationQandApairs.csv" ,split_rate = None, print_info = None):
    data = pd.read_csv("NvidiaDocumentationQandApairs.csv")[["question", "answer"]]
    if print_info != None:
        print("Data Shape: ",data.shape)
        print("="*30)
        print(data.head(5))
        
    if split_rate == None:
        data = Dataset.from_pandas(data)
        tokenized_datasets = data.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['question', 'answer',])
        
    if print_info != None:
        print("="*30)
        print("Training_Data Details: ", tokenized_datasets)
    
    return tokenized_datasets
        
def tokenize_function(example):
    start_prompt = '\n\n'
    end_prompt = '\n\nAnswer: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["question"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt", max_length=200).input_ids
    example['labels'] = tokenizer(example["answer"], padding="max_length", truncation=True, return_tensors="pt",max_length=200).input_ids
    
    return example