import os
from typing import Union

import torch
import torch.nn as nn

from .layers import ViTEmbeddings, ViTEncoder


class ViTModel(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_channels=3,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.embeddings = ViTEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.encoder = ViTEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )
        self.layernorm = nn.LayerNorm(hidden_size)

        self.apply(self._init_weights)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        encoded_output = self.encoder(hidden_states)
        return self.layernorm(encoded_output)

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=0.02
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32), mean=0.0, std=0.02
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32), mean=0.0, std=0.02
            ).to(module.cls_token.dtype)


class ViTForImageClassification(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        hidden_size=768,
        num_channels=3,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=10,
        dropout=0.1,
    ):
        super().__init__()
        self.vit = ViTModel(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_channels=num_channels,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.classifier.weight.data = nn.init.trunc_normal_(
            self.classifier.weight.data.to(torch.float32), mean=0.0, std=0.02
        ).to(self.classifier.weight.dtype)
        self.classifier.bias.data.zero_()

    def forward(self, pixel_values):
        hidden_states = self.vit(pixel_values)
        cls_output = hidden_states[:, 0]  # CLS token
        logits = self.classifier(cls_output)
        return logits

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, num_labels=10):
        support_models = {
            "vit-base-patch16-224",
            "vit-large-patch16-224",
            "vit-huge-patch14-224",
        }

        model_name = model_name_or_path
        if os.path.isdir(model_name_or_path):
            basename = os.path.basename(model_name_or_path)
            for model in sorted(support_models, key=len, reverse=True):
                if model in basename:
                    model_name = model
                    break
            else:
                raise ValueError(
                    f"To specify the custom ViT model structure, the dir name should be one of {support_models} "
                )
        else:
            for model in support_models:
                if model_name_or_path.endswith(model):
                    model_name = model
                    break
            else:
                raise ValueError(f"Unsupported model {model_name} or dir not found")

        from transformers import (
            ViTForImageClassification as HFViTForImageClassification,
        )

        print("Loading weights from pretrained ViT: %s" % model_name)

        # Define configurations based on the model type
        config_args = {
            "vit-base-patch16-224": dict(
                num_hidden_layers=12,
                num_attention_heads=12,
                hidden_size=768,
                intermediate_size=3072,
            ),
            "vit-large-patch16-224": dict(
                num_hidden_layers=24,
                num_attention_heads=16,
                hidden_size=1024,
                intermediate_size=4096,
            ),
            "vit-huge-patch14-224": dict(
                num_hidden_layers=32,
                num_attention_heads=16,
                hidden_size=1280,
                intermediate_size=5120,
            ),
        }[model_name]

        model = cls(num_labels=num_labels, **config_args)
        sd = model.state_dict()

        # Load from HuggingFace pretrained weights
        model_hf = HFViTForImageClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        sd_hf = model_hf.state_dict()

        # Ensure all the parameters align between HuggingFace model and our model
        sd_keys = sd.keys()
        sd_keys_hf = sd_hf.keys()

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"Mismatch in number of keys: {len(sd_keys_hf)} vs {len(sd_keys)}"

        for k in sd_keys:
            assert (
                sd_hf[k].shape == sd[k].shape
            ), f"Shape mismatch for key {k}: {sd_hf[k].shape} vs {sd[k].shape}"
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(
        self, lr, weight_decay, betas=(0.9, 0.999), device_type=None
    ):
        # Get all parameters and filter those that require gradient
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Separate decay and no-decay parameter groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Print parameter counts
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Use fused AdamW if available
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer
