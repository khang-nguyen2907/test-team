import torch
from transformers import (AutoConfig,
                          AutoModelForSeq2SeqLM,
                          AutoTokenizer)

def get_config(model_path_or_name: str,
               max_length: int,
               length_pentalty: float,
               no_repeat_ngram_size: float,
               num_beams:int) -> AutoConfig:
    config: AutoConfig
    return config

def get_tokenizer(model_path_or_name: str
                  ):
    tokenizer: AutoTokenizer
    return tokenizer

def get_model(model_path_or_name: str,
              config: AutoConfig):
    model: AutoModelForSeq2SeqLM
    return model


