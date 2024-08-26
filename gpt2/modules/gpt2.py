import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import Block


class GPT2(nn.Module):
    """GPT Language Model"""

    def __init__(self, vocab_size, max_len, d_model, n_head, n_layer, dropout):
        super().__init__()
        self.max_len = max_len
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, d_model),
                wpe=nn.Embedding(max_len, d_model),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            d_model=d_model,
                            n_head=n_head,
                            max_len=max_len,
                            dropout=dropout,
                        )
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=nn.LayerNorm(d_model),
            )
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        support_models = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        import os

        model_name = model_name_or_path.removeprefix("openai-community/")
        if os.path.isdir(model_name_or_path):
            basename = os.path.basename(model_name_or_path)
            for model in sorted(support_models, key=len, reverse=True):
                if model in basename:
                    model_name = model
                    break
            else:
                raise ValueError(
                    f"To specify the custom GPT model structure, the dir name should be one of {support_models} "
                )
        elif model_name not in support_models:
            raise ValueError(f"Unsupport model {model_name} or dir not found")

        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_name)

        # n_layer, n_head and d_model are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, d_model=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, d_model=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, d_model=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, d_model=1600),  # 1558M params
        }[model_name]
        print("forcing vocab_size=50257, max_len=1024")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["max_len"] = 1024  # always 1024 for GPT model checkpoints
        config_args["dropout"] = 0.1

        model = GPT2(**config_args)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self, lr, weight_decay, betas=(0.9, 0.999), device_type=None
    ):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def forward(self, input, input_mask=None):
        device = input.device
        b, t = input.size()
        assert (
            t <= self.max_len
        ), f"Cannot forward sequence of length {t}, block size is only {self.max_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        if input_mask is not None and input_mask.dim() == 2:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            input_mask = input_mask.unsqueeze(1).unsqueeze(2)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(
            input
        )  # token embeddings of shape (b, t, d_model)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, d_model)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, input_mask)
        x = self.transformer.ln_f(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_len
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len :]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
