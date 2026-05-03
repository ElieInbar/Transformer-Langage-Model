from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)

    # Inputs are everything except the last token in the sequence
    inputs = batch[:, :-1]
    # Labels are everything except the first token (shifted by one step into the future)
    labels = batch[:, 1:]
    return (inputs, labels)

def compute_loss(logits, gold_labels):
    # logits size is (batch, seq_len, vocab_size)
    # gold_labels size is (batch, seq_len)

    # Flatten the logits to (batch * seq_len, vocab_size)
    # view(-1, logits.size(-1)) tells PyTorch to figure out the first dimension automatically
    # and keep the last dimension (vocab_size) intact.
    logits_flat = logits.view(-1, logits.size(-1))

    # Flatten the gold labels to (batch * seq_len)
    # .contiguous() ensures the tensor is laid out nicely in memory before flattening
    labels_flat = gold_labels.contiguous().view(-1)

    # Calculate Cross Entropy Loss
    # In CharTokenizer, <PAD> is the very first symbol added, so its ID is 0.
    # ignore_index=0 ensures the model isn't trained to predict padding.
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)

    return loss
