from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math

ATTENTION_CACHE = {'active': False, 'matrices': []}

def create_kqv_matrix(input_vector_dim, n_heads=1):
    # Each head processes a fraction of the full embedding dimension
    head_dim = input_vector_dim // n_heads
    # The output dimension is 3 * head_dim to account for K, Q, and V
    return nn.Linear(input_vector_dim, 3 * head_dim)

def kqv(x, linear):
    B, N, D = x.size()
    # compute k, q, v
    # compute the linear operation
    out = linear(x)

    # Split the output into 3 equal chunks along the last dimension
    k, q, v = torch.chunk(out, 3, dim=2)


    return k, q, v

def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    # b is Q (B, N2, D2), a is K (B, N1, D1).
    # We want Q @ K^T, so we transpose K to (B, D1, N1) and do b @ a.transpose
    A = (b @ a.transpose(1,2)) / math.sqrt(D1)
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):

    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant.

    # a matrix max*max of ones
    ones = torch.ones(max_context_len, max_context_len)
    # make it lower triangular and add a dimension in the front for the batch size
    mask = torch.tril(ones).unsqueeze(0)
    return mask

def self_attention(v, A, mask = None):
    # compute sa (corresponding to y in the assignemnt text).
    # Get the current sequence length (N)
    # v has shape (B, N, D), A has shape (B, N, N)
    B, N, D = v.size()

    if mask is not None:
        # Slice the mask to match the current sequence length N
        # The mask is (1, MaxLen, MaxLen), we want (1, N, N)
        mask_N = mask[:, :N, :N]

        # we replace all the zeros with -inf
        A = A.masked_fill(mask_N == 0, float("-inf"))

    # we apply softmax to the last dimension of A, the columns
    A = F.softmax(A, dim=2)

    global ATTENTION_CACHE
    if ATTENTION_CACHE['active']:
        ATTENTION_CACHE['matrices'].append(A.detach().cpu())

    # Compute the weighted sum of the values: sa = A @ V
    sa = torch.matmul(A, v)

    return sa

#this function maps from x to sa
def self_attention_layer(x, kqv_matrix, attention_mask):
    # we compute k, q and v using the kqv function
    # which applies the linear transformation given by kqv_matrix to x.
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    # sa is self attention, which is the output of the self attention layer.
    # it's A @ v, where A is the attention scores after applying the mask and softmax.
    # It has the same dimensions as x (b x n x d)
    # where b is the batch size, n is the sequence length, and d is the dimension of each vector.
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    # This is most easily done using calls to self_attention_layer, each with a different
    # entry in kqv_matrices, and combining the results.

    # a list to store the results of each head
    heads_outputs = []

    # for each head, we compute the self attention layer and store the result in heads_outputs
    for kqv_matrix in kqv_matrices:

        # compute the self attention layer for this head, using the corresponding kqv_matrix and the mask.
        head_output = self_attention_layer(x, kqv_matrix, mask)

        # store the result in heads_outputs
        heads_outputs.append(head_output)

    # we concatenate the results of each head in the last dimension (the dimension of the vectors)
    sa = torch.cat(heads_outputs, dim=2)

    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        # Final linear projection to "mix" the concatenated attention heads (d to d)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        return self.proj(sa)
