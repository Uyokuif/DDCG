#!/usr/bin/env python3
"""
DGAP-v2 Dual-Critic Training with DeepSpeed
Implements hierarchical training strategy with DeepSpeed optimization:
1. CriticE: Executability Critic (Binary Classification) 
2. CriticQ: Quality Critic Ensemble (Regression)

Based on paper specifications:
- AdamW optimizer, lr=1e-5, warmup=0.1, batch_size=32
- RoBERTa initialization with L2 loss for score prediction
"""

import os
import sys
import json
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import numpy as np
import deepspeed
from tqdm import tqdm
from datetime import datetime

import transformers
from transformers import (
    RobertaTokenizer, 
    RobertaModel,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed
)
from torch.utils.data import Dataset, DataLoader

# Setup wandb
try:
    import wandb
    WANDB_AVAILABLE = True
    os.environ["WANDB_API_KEY"] = ''
    os.environ["WANDB_MODE"] = "online"
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="roberta-base",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    cache_dir: str = field(
        default="./cache",
        metadata={"help": "Cache directory for models"}
    )
    dropout: float = field(
        default=0.3,
        metadata={"help": "Dropout rate for critics"}
    )

@dataclass 
class DataArguments:
    """Arguments for data configuration"""
    positive_file: str = field(
        default="/home/msj/planning/d2c/VirtualHome/virtualhome/dataset/flant5_training_data/critic_e_positive_samples.jsonl",
        metadata={"help": "Path to positive samples file"}
    )
    negative_file: str = field(
        default="/home/msj/planning/d2c/VirtualHome/virtualhome/dataset/flant5_training_data/critic_e_negative_samples.jsonl", 
        metadata={"help": "Path to negative samples file"}
    )
    quality_file: str = field(
        default="/home/msj/planning/d2c/VirtualHome/virtualhome/dataset/flant5_training_data/critic_q_training_data.jsonl",
        metadata={"help": "Path to quality training data file"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    balance_ratio: float = field(
        default=1.0,
        metadata={"help": "Negative to positive sample ratio"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Extended training arguments"""
    critic_type: str = field(
        default="executability",
        metadata={"help": "Type of critic to train: executability or quality"}
    )
    ensemble_size: int = field(
        default=3,
        metadata={"help": "Number of critics in quality ensemble"}
    )
    noise_factor: float = field(
        default=0.01,
        metadata={"help": "Noise factor for ensemble diversity"}
    )
    data_sampling_ratio: float = field(
        default=1.0,
        metadata={"help": "Data sampling ratio for ensemble diversity"}
    )

class DGAPDataset(Dataset):
    """DGAP Dataset for DeepSpeed training"""
    
    def __init__(self, positive_file: str, negative_file: str = None, quality_file: str = None, 
                 tokenizer=None, max_length: int = 512, task_type: str = 'executability', 
                 balance_ratio: float = 1.0, data_sampling_ratio: float = 1.0, 
                 random_seed: int = 42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.data = []
        
        if task_type == 'executability':
            # Load positive and negative samples separately for executability
        logger.info(f"Loading positive samples from: {positive_file}")
        pos_count = 0
        with open(positive_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)
                pos_count += 1
        
        logger.info(f"Loading negative samples from: {negative_file}")
        target_neg_count = int(pos_count * balance_ratio)
        
        with open(negative_file, 'r', encoding='utf-8') as f:
            neg_samples = []
            for line in f:
                sample = json.loads(line.strip())
                neg_samples.append(sample)
            
            if len(neg_samples) > target_neg_count:
                neg_samples = random.sample(neg_samples, target_neg_count)
            
            self.data.extend(neg_samples)
            neg_count = len(neg_samples)
        
            logger.info(f"Dataset loaded: {pos_count} positive, {neg_count} negative samples")
            
        else:  # quality task
            # Load quality data from single file
            logger.info(f"Loading quality samples from: {quality_file}")
            with open(quality_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    self.data.append(sample)
            
            logger.info(f"Quality dataset loaded: {len(self.data)} samples")
        
        # Apply data sampling for ensemble diversity
        if data_sampling_ratio < 1.0:
            random.seed(random_seed)
            original_size = len(self.data)
            sample_size = int(original_size * data_sampling_ratio)
            self.data = random.sample(self.data, sample_size)
            logger.info(f"Data sampling applied: {original_size} → {len(self.data)} samples (ratio: {data_sampling_ratio})")
        
        random.seed(random_seed)
        random.shuffle(self.data)
        logger.info(f"Final dataset size: {len(self.data)} samples for {task_type} task")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        input_text = sample['input']
        score = int(sample['Score'])
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if self.task_type == 'executability':
            label = 1.0 if score >= 2 else 0.0
        else:  # quality
            label = score / 10.0
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(), 
            'labels': torch.tensor(label, dtype=torch.float32)
        }

class CriticE(nn.Module):
    """Executability Critic with RoBERTa backbone"""
    
    def __init__(self, model_name='roberta-base', dropout=0.3):
        super(CriticE, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        # Freeze embeddings for efficiency
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        hidden_size = self.roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output).squeeze()
        
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            # Ensure dtype consistency for bf16
            logits = logits.to(labels.dtype)
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}

class CriticQ(nn.Module):
    """Quality Critic with RoBERTa backbone - Ensemble Version"""
    
    def __init__(self, model_name='roberta-base', dropout=0.3, noise_factor=0.0):
        super(CriticQ, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.noise_factor = noise_factor
        
        # Freeze embeddings for efficiency  
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        hidden_size = self.roberta.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Add ensemble diversity noise during training
        if self.training and self.noise_factor > 0:
            noise = torch.randn_like(pooled_output) * self.noise_factor
            pooled_output = pooled_output + noise
        
        scores = self.regressor(pooled_output).squeeze()
        
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()  # L2 loss as per paper
            # Ensure dtype consistency for bf16
            scores = scores.to(labels.dtype)
            loss = loss_fn(scores, labels)
            
            # Additional diversity loss for ensemble training
            if self.training and self.noise_factor > 0:
                diversity_loss = torch.var(scores) * 0.01  # Encourage prediction diversity
                loss = loss - diversity_loss  # Subtract to encourage diversity
        
        return {"loss": loss, "logits": scores}

def compute_metrics_executability(eval_pred):
    """Compute metrics for executability task"""
    predictions, labels = eval_pred
    predictions = (predictions > 0.5).astype(int)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def compute_metrics_quality(eval_pred):
    """Compute metrics for quality task"""
    predictions, labels = eval_pred
    mse = ((predictions - labels) ** 2).mean()
    mae = abs(predictions - labels).mean()
    return {"mse": mse, "mae": mae}

def train_critic(
    model_args: ModelArguments,
    data_args: DataArguments, 
    training_args: TrainingArguments
):
    """Train single critic with DeepSpeed"""
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    
    # Set seed
    set_seed(training_args.seed)
    
    # Initialize tokenizer and dataset
    tokenizer = RobertaTokenizer.from_pretrained(model_args.model_name_or_path)
    
    if training_args.critic_type == 'executability':
    train_dataset = DGAPDataset(
            positive_file=data_args.positive_file,
            negative_file=data_args.negative_file,
            tokenizer=tokenizer,
        max_length=data_args.max_length,
        task_type=training_args.critic_type,
        balance_ratio=data_args.balance_ratio
    )
    else:  # quality
        train_dataset = DGAPDataset(
            positive_file=None,
            quality_file=data_args.quality_file,
            tokenizer=tokenizer,
            max_length=data_args.max_length,
            task_type=training_args.critic_type,
            data_sampling_ratio=training_args.data_sampling_ratio,
            random_seed=training_args.seed
        )
    
    # Initialize model
    if training_args.critic_type == 'executability':
        model = CriticE(model_args.model_name_or_path, model_args.dropout)
        compute_metrics = compute_metrics_executability
    else:
        model = CriticQ(
            model_name=model_args.model_name_or_path, 
            dropout=model_args.dropout,
            noise_factor=training_args.noise_factor
        )
        compute_metrics = compute_metrics_quality
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Start training
    logger.info(f"Starting {training_args.critic_type} critic training")
    trainer.train()
    
    # Save model
    output_dir = f"./models/dgap_v2_{training_args.critic_type}"
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    return trainer

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize wandb
    if WANDB_AVAILABLE and not training_args.disable_tqdm:
        wandb.init(
            project="dgap-v2-deepspeed",
            name=f"dgap_v2_{training_args.critic_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "critic_type": training_args.critic_type,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs
            }
        )
    
    # Train critic
    trainer = train_critic(model_args, data_args, training_args)
    
    if WANDB_AVAILABLE and not training_args.disable_tqdm:
        wandb.finish()

if __name__ == "__main__":
    main() 
