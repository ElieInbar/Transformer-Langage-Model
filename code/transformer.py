from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len: int,
                 with_residuals: bool = False, use_pre_norm: bool = True):
        super().__init__()
        # Causal self-attention layer to attend to previous tokens only
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        # Feed-forward MLP layer
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        # Layer normalization before attention
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        # Layer normalization before MLP
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        # Flag to enable/disable residual connections
        self.with_residuals = with_residuals
        # Flag to choose between pre-norm and post-norm architecture
        self.use_pre_norm = use_pre_norm

    def forward(self, inputs):
        x = inputs
        if self.with_residuals:
            if self.use_pre_norm:
                residual = x
                x = self.layer_norm_1(x)
                x = self.causal_attention(x)
                x = x + residual
                residual = x
                x = self.layer_norm_2(x)
                x = self.mlp(x)
                x = x + residual
            else:
                x = x + self.causal_attention(x)
                x = self.layer_norm_1(x)
                x = x + self.mlp(x)
                x = self.layer_norm_2(x)

        else:
            if self.use_pre_norm:
                x = self.layer_norm_1(x)
                x = self.causal_attention(x)
                x = self.layer_norm_2(x)
                x = self.mlp(x)

            else:
                x = self.causal_attention(x)
                x = self.layer_norm_1(x)
                x = self.mlp(x)
                x = self.layer_norm_2(x)
        return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        b, n = x.size()

        # Get the token embeddings: shape (b, n, d)
        tok_embeddings = self.token_embeddings(x)

        # Generate position indices [0, 1, ..., n-1]
        # We use device=x.device to ensure the positions tensor is on the same device as the input x (CPU or GPU).
        positions = torch.arange(n, device=x.device)
        # Get the position embeddings: shape (n, d)
        pos_embeddings = self.position_embeddings(positions)

        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            ):
        super().__init__()
        self.config = {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'embed_size': embed_size,
            'max_context_len': max_context_len,
            'vocab_size': vocab_size,
            'mlp_hidden_size': mlp_hidden_size,
            'with_residuals': with_residuals,
        }
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        # initialize weights
        # TODO implement initialization logic for embeddings and linear layers.
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal(module.weight, mean=0.0, std=0.02)

    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def save(self, path: str):
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'TransformerLM':
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint['config']
        model = cls(
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            embed_size=config['embed_size'],
            max_context_len=config['max_context_len'],
            vocab_size=config['vocab_size'],
            mlp_hidden_size=config['mlp_hidden_size'],
            with_residuals=config['with_residuals'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32))
                logits_for_last_token = logits[0][-1]
                
                # Apply temperature
                if temperature > 0:
                    logits_for_last_token = logits_for_last_token / temperature
                
                # Apply Top-K filtering
                if topK > 0 and topK < logits_for_last_token.size(-1):
                    # Find top K values
                    top_k_logits, top_k_indices = torch.topk(logits_for_last_token, topK)
                    # Create a tensor of -inf
                    filtered_logits = torch.full_like(logits_for_last_token, float('-inf'))
                    # Scatter the top-k values back into the -inf tensor
                    filtered_logits.scatter_(0, top_k_indices, top_k_logits)
                    logits_for_last_token = filtered_logits
                
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1).item()
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated
