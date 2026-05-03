import torch
import math
from attention import attention_scores, self_attention, create_causal_mask

def test_attention_scores():
    # A simple test with batch=1, seq_len=2, dim=2
    # We choose K and Q so their dot product is very easy to calculate manually
    q = torch.tensor([[[1.0, 0.0],
                       [0.0, 2.0]]])

    k = torch.tensor([[[1.0, 0.0],
                       [0.0, 1.0]]])

    # Q @ K^T should be:
    # [[(1*1 + 0*0), (1*0 + 0*1)],  ==  [[1.0, 0.0],
    #  [(0*1 + 2*0), (0*0 + 2*1)]]       [0.0, 2.0]]
    #
    # Scaled by sqrt(d) = sqrt(2)
    expected_scores = torch.tensor([[[1.0 / math.sqrt(2), 0.0],
                                     [0.0, 2.0 / math.sqrt(2)]]])

    scores = attention_scores(k, q)

    assert torch.allclose(scores, expected_scores), "attention_scores failed!"
    print("test_attention_scores passed!")

def test_self_attention():
    # We craft an 'A' matrix (pre-softmax) where the softmax output is predictable:
    # 100 vs -100 turns into [1.0, 0.0] after softmax.
    # 0.0 vs 0.0 turns into [0.5, 0.5] after softmax.
    A = torch.tensor([[[100.0, -100.0],
                       [0.0, 0.0]]])

    # Simple values for V
    v = torch.tensor([[[10.0, 20.0],
                       [30.0, 40.0]]])

    # Softmax(A) will be roughly: [[1.0, 0.0], [0.5, 0.5]]
    # So the weighted sum Softmax(A) @ V will be:
    # row 1: 1.0*[10, 20] + 0.0*[30, 40] = [10.0, 20.0]
    # row 2: 0.5*[10, 20] + 0.5*[30, 40] = [20.0, 30.0]
    expected_sa = torch.tensor([[[10.0, 20.0],
                                 [20.0, 30.0]]])

    sa = self_attention(v, A, mask=None)

    assert torch.allclose(sa, expected_sa), "self_attention failed!"
    print("test_self_attention passed!")

def test_causal_mask():
    # Request a mask for a small max_context_len = 3
    mask = create_causal_mask(embed_dim=8, n_heads=2, max_context_len=3)

    # We expect a simple 3x3 lower triangular matrix
    expected_mask = torch.tensor([[[1.0, 0.0, 0.0],
                                   [1.0, 1.0, 0.0],
                                   [1.0, 1.0, 1.0]]])

    assert torch.allclose(mask, expected_mask), "create_causal_mask failed!"
    print("test_causal_mask passed!")

if __name__ == "__main__":
    test_attention_scores()
    test_self_attention()
    test_causal_mask()
