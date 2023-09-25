from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch




def get_model_tokenizer(model_name = "google/flan-t5-base"):
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return original_model, tokenizer