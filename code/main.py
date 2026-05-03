from __future__ import annotations

import torch
import json
import os

if __name__ == "__main__":
    import lm
    import torch
    from torch import nn, optim
    from transformer import TransformerLM

    import data

    config_path = "config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    mode = config.get("mode", "train")
    data_path = config.get("data_path", "../data/en/")
    save_model_path = config.get("save_model_path", "model.pt")
    load_model_path = config.get("load_model_path", "model.pt")
    save_tokenizer_path = config.get("save_tokenizer_path", "tokenizer.json")
    load_tokenizer_path = config.get("load_tokenizer_path", "tokenizer.json")

    seq_len = config.get("seq_len", 128)
    batch_size = config.get("batch_size", 64)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 6)
    embed_size = config.get("embed_size", 192)
    mlp_hidden_size = embed_size * 4

    learning_rate = config.get("learning_rate", 5e-4)
    gradient_clipping = config.get("gradient_clipping", 1.0)
    temperature = config.get("temperature", 1.0)
    topK = config.get("topK", 5)

    num_batches_to_train = config.get("num_batches", 50000)

    if mode == "train":
        print(f"Loading data from {data_path}...")
        tokenizer, tokenized_data = data.load_data(data_path)
        
        print(f"Saving tokenizer to {save_tokenizer_path}...")
        tokenizer.save(save_tokenizer_path)

        data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

        model: torch.nn.Module = TransformerLM(
            n_layers,
            n_heads,
            embed_size,
            seq_len,
            tokenizer.vocab_size(),
            mlp_hidden_size,
            with_residuals=True,
        )

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

        model.train()

        num_batches = 0
        while True:
            for batch in data.batch_items(data_iter, batch_size):
                if num_batches >= num_batches_to_train:
                    break

                batch_x, batch_y = lm.batch_to_labeled_samples(batch)

                logits = model(batch_x)

                loss = lm.compute_loss(logits, batch_y)

                # parameters update
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                num_batches += 1
                if num_batches % 10 == 0:
                    print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                    if num_batches % 100 == 0:
                        for _ in range(1):
                            model.eval()
                            sampled = tokenizer.detokenize(
                                model.better_sample_continuation(tokenizer.tokenize("Hello"), 500, temperature, topK)
                            )
                            model.train()
                            print(f"Model sample: '''{sampled}'''")
                        print(f"Saving model to {save_model_path}...")
                        model.save(save_model_path)
                        print("")
            
            if num_batches >= num_batches_to_train:
                break
        
        print(f"Training complete. Final model saved to {save_model_path}.")
        model.save(save_model_path)

    elif mode == "eval":
        if not os.path.exists(load_tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {load_tokenizer_path}")
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Model not found at {load_model_path}")
            
        print(f"Loading tokenizer from {load_tokenizer_path}...")
        tokenizer = data.CharTokenizer.load(load_tokenizer_path)
        
        print(f"Loading model from {load_model_path}...")
        model = TransformerLM.load(load_model_path)
        model.eval()
        
        prompt = "Hello"
        print(f"Generating continuation for prompt: '{prompt}' (temp: {temperature}, topK: {topK})")
        sampled = tokenizer.detokenize(
            model.better_sample_continuation(tokenizer.tokenize(prompt), 500, temperature, topK)
        )
        print(f"Model generated sample: '''{sampled}'''\n")
