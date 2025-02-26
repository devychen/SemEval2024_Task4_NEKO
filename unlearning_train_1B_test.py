import argparse
import logging
import random
import time
import os

from datasets import Dataset
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, get_scheduler

# Set random seeds for reproducibility
SEED = 8888
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def create_dataloader_from_parquet(tokenizer, parquet_file, batch_size, shuffle=True, max_length=150):
    """
    Create a DataLoader from a Parquet file containing retain or forget data.

    Args:
        tokenizer: Tokenizer.
        parquet_file: Path to a Parquet file with 'input' and 'output' columns.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the DataLoader.
        max_length: Maximum sequence length for tokenization.

    Returns:
        DataLoader of retain or forget set.
    """
    def preprocess(examples):
        """
        Preprocess examples by determining type based on the presence of '?' in input.
        """
        full_texts = []
        start_locs = []

        for inp, outp in zip(examples['input'], examples['output']):
            if "?" in inp:  # Classify as QA
                full_text = f"### Question: {inp}\n ### Answer: {outp}"
                start_loc = len(tokenizer(f"### Question: {inp}\n ### Answer: ", truncation=True, max_length=max_length)["input_ids"])
            else:  # Classify as text generation
                full_text = f"### Text: {inp} {outp}"
                start_loc = len(tokenizer(f"### Text: {inp} ", truncation=True, max_length=max_length)["input_ids"])
            
            full_texts.append(full_text)
            start_locs.append(start_loc)
           
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "start_locs": start_locs,
            "labels": tokenized["input_ids"], 
        }
    
    dataset = Dataset.from_parquet(parquet_file)
    
    processed_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_locs", "labels"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    dataloader = torch.utils.data.DataLoader(
        processed_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle
    )
    
    return dataloader

def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute forward Kullback Leibler divergence as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    
    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
    retain_probs = torch.nn.functional.log_softmax(pretrained_outputs.logits, dim=-1)
    retain_probs = retain_probs.view(-1, pretrained_outputs.logits.shape[-1])

    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    current_probs = torch.nn.functional.log_softmax(normal_outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, normal_outputs.logits.shape[-1])

    retain_loss = torch.nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
    
    return retain_loss

def ga_loss(batch, model, device):
    """
    Compute the (gradient ascent) loss on the answer (i.e. y) part.

    Args:
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """

  
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    

    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        position_loss = -position_loss

        # Define position weights: 1 for answer part, 0 for other parts
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part(input)
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss

# Configuration constants
MAX_UNLEARN_STEPS = 500
BAD_WEIGHT = 0.2
NORMAL_WEIGHT = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.00005
SAVE_EVERY = 100
LOG_FILE = "logs/default1B4.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w+",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d-%H-%M",
    level=logging.INFO,
)

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 4  # number of validation checks without improvement

def train(input_model_path, retain_set, forget_set, retain_val_set, output_model_path):
    
    accelerator = Accelerator() 
    accelerator.even_batches = False 
    device = accelerator.device

  
    model = AutoModelForCausalLM.from_pretrained(input_model_path)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")

    pretrained_model = AutoModelForCausalLM.from_pretrained(input_model_path)
       
    train_bad_loader = create_dataloader_from_parquet(tokenizer, forget_set, batch_size=BATCH_SIZE)
    train_normal_loader = create_dataloader_from_parquet(tokenizer, retain_set, batch_size=BATCH_SIZE)
    #val_bad_loader = create_dataloader_from_parquet(tokenizer, forget_val_set, batch_size=BATCH_SIZE)
    val_normal_loader = create_dataloader_from_parquet(tokenizer, retain_val_set, batch_size=BATCH_SIZE)


    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = MAX_UNLEARN_STEPS
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    # Prepare with Accelerator
    model, pretrained_model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler = accelerator.prepare(
        model, pretrained_model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )

    best_val_loss = float("inf")
    no_improve_steps = 0
    model.train()

    step = 0
    start_time = time.time() 

    while step < MAX_UNLEARN_STEPS:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            if step >= MAX_UNLEARN_STEPS:
                break

            
            bad_loss = ga_loss(bad_batch, model, device=device)
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device=device)
           
            loss = BAD_WEIGHT * bad_loss + NORMAL_WEIGHT * normal_loss

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            logging.info(f"Step: {step}, Bad Loss: {bad_loss:.2f}, KL Loss: {normal_loss:.2f}")
            step += 1

            # Validation 
            if step % SAVE_EVERY == 0:
                model.eval()
                val_normal_loss = 0

                with torch.no_grad():
                    for val_normal_batch in val_normal_loader:
                        val_normal_loss += compute_kl(pretrained_model, model, val_normal_batch, device=device).item()

                val_normal_loss /= len(val_normal_loader)

                logging.info(f"Validation - Step: {step}, Val Normal Loss: {val_normal_loss:.2f}")

                if val_normal_loss < best_val_loss:
                    best_val_loss = val_normal_loss
                    no_improve_steps = 0
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_model_path, save_function=accelerator.save)
                else:
                    no_improve_steps += 1
                    logging.info(f"No improvement for {no_improve_steps} validation steps.")

                if no_improve_steps >= EARLY_STOPPING_PATIENCE:
                    logging.info("Early stopping triggered. Stopping training.")
                    break

                model.train()

        if no_improve_steps >= EARLY_STOPPING_PATIENCE:
            break

    end_time = time.time()
    logging.info(f"Total training time: {int(end_time - start_time)} seconds")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_model_path)
    logging.info("Training complete.")
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLURM-compatible training script for unlearning model")
    parser.add_argument("--input_model_path", type=str, required=True, help="Path to the input model")
    parser.add_argument("--retain_set", type=str, required=True, help="Parquet file containing retain set")
    parser.add_argument("--forget_set", type=str, required=True, help="Parquet file containing forget set")
    parser.add_argument("--retain_val_set", type=str, required=True, help="Parquet file containing retain validation set")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the output model")
    args = parser.parse_args()

    # Configure logging
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Run training
    train(args.input_model_path, args.retain_set, args.forget_set, args.retain_val_set, args.output_model_path)