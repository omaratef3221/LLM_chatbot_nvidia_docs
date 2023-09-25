from transformers import GenerationConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch
from prepare_datasets import *
from get_model import *
import time


## Hyper Parameters and other parameters
EPOCHS = 25
PRINT_INFO = True
LR = 1e-3
BATCH_SIZE = 8
LORA_RANK = 32
#########


original_model, _ = get_model_tokenizer()
original_model.to("mps")

# Get the dataset
training_data = make_dataset(print_info=PRINT_INFO)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


lora_config = LoraConfig(
    r=LORA_RANK, # Rank
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config)

if PRINT_INFO:
    print("="*30)
    print(print_number_of_trainable_model_parameters(peft_model))


training_output_dir = f'./Nvidia-chatbot-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=training_output_dir,
    auto_find_batch_size=True,
    learning_rate=LR, 
    num_train_epochs=25,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=1,
    logging_strategy = 'epoch',
    max_steps=-1, 
    use_mps_device= True,
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=training_data,
)

peft_trainer.train()

peft_model_path="./peft-Nvidia-chatbot-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

if PRINT_INFO:
    print("="*30)
    print("Training Done and Model saved at: ", peft_model_path)
    print("="*30)