import json
from datasets import load_dataset
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatoForSeq2Seq
        )

def get_dataset(dataset_name: str,
                name: str = "japanese") -> datasets.DatasetDict:
    ds = load_dataset(dataset_name, name=name)
    return ds

def preprocess(ds: datasets.DatasetDict,
               tokenizer: AutoTokenizer) -> datasets.DatasetDict:
    def preprocess_function(example):
        # Max token size is 14536 and 215 for inputs and labels, espectively.
        # Here I restrict these token size
        input_feature = tokenizer(example["text"], truncation=True, max_length=1024)
        label = tokenizer(example["summary"], truncation=True, max_length=128)
        return {
                "input_ids": input_feature["input_ids"],
                "attention_mask": input_feature["attention_mask"],
                "labels": label["input_ids"],
                }
    tokenized_dataset: datasets.DatasetDict
    return tokenized_dataset

def get_collator(tokenizer: AutoTokenizer,
                 model: AutoMode):
    data_collator = DataCollatoForSeq2Seq(
        tokenizer,
        model=model,
        return_tensors="pt"
    )
    return data_collator
