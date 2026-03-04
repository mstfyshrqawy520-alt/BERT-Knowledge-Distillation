# DistillBERT-Lite

A practical implementation of **Knowledge Distillation** for compressing a large Transformer model into a lightweight student model while preserving performance.

This project demonstrates how a large **BERT teacher model** can transfer its knowledge to a smaller **student transformer**, reducing model size and improving inference speed with minimal accuracy loss.

---

## Project Idea

Large transformer models like BERT achieve high accuracy but are computationally expensive.

Knowledge Distillation solves this by training a smaller model to mimic the behavior of a larger pretrained model.

Teacher Model → BERT-base  
Student Model → Compact Transformer (4 layers)

The student learns from:

1. **Hard labels** (ground truth)
2. **Soft targets** produced by the teacher model

---

## Architecture

Teacher Model (BERT-base)
        ↓
Soft Predictions (Temperature Scaling)
        ↓
Student Model (Tiny Transformer)
        ↓
Efficient Sentiment Classifier

---

## Dataset

IMDB Movie Reviews Dataset

Binary classification:

Positive Review  
Negative Review

---

## Knowledge Distillation Loss

The training objective combines two losses:

Hard Loss:
Cross Entropy with ground truth labels

Soft Loss:
KL Divergence between teacher and student predictions.

Final loss:

Loss = α * HardLoss + (1 - α) * SoftLoss

Where:

α controls the balance between hard labels and teacher knowledge.

Temperature scaling is used to soften probability distributions.

---

## Tech Stack

Python  
PyTorch  
HuggingFace Transformers  
HuggingFace Datasets  
Evaluate

---

## Project Structure
