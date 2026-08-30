"""Semantic channel: TAG → descriptor → sentence encoder → MLP → classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class GraniteDescriptorGenerator(nn.Module):
    """Convert a raw TAG into a natural-language descriptor with Granite."""

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-4.2-3b",
        max_new_tokens: int = 256,
        freeze: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens
        self.freeze = freeze
        if freeze:
            for param in self.llm.parameters():
                param.requires_grad = False
        self.llm.eval()

    def forward(self, tags: list[str]) -> list[str]:
        descriptors = []
        for tag in tags:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Convert the following TAG into a concise "
                        "natural-language descriptor suitable for "
                        "semantic classification.\n\n"
                        f"TAG: {tag}\n\n"
                        "Return only the descriptor."
                    ),
                }
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=1.0,
                    top_p=0.95,
                    do_sample=True,
                )
            generated_tokens = outputs[0, inputs.input_ids.shape[-1] :]
            descriptor = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            descriptors.append(descriptor)
        return descriptors


class HuggingFaceSentenceEncoder(nn.Module):
    """Mean-pool a pretrained transformer into a sentence embedding."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        max_length: int = 256,
        freeze: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.max_length = max_length
        self.freeze = freeze
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self.backbone.config.hidden_size

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        if self.freeze:
            with torch.no_grad():
                outputs = self.backbone(**encoded)
        else:
            outputs = self.backbone(**encoded)

        token_embeddings = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        if self.normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class SemanticChannel(nn.Module):
    """TAG → descriptor → sentence embedding → MLP → class logits."""

    def __init__(
        self,
        descriptor_generator: GraniteDescriptorGenerator,
        sentence_encoder: HuggingFaceSentenceEncoder,
        hidden_dim: int,
        semantic_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.descriptor_generator = descriptor_generator
        self.sentence_encoder = sentence_encoder
        sentence_dim = sentence_encoder.embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(sentence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
        )
        self.classifier = nn.Linear(semantic_dim, num_classes)

    def forward(self, tags: list[str]):
        descriptors = self.descriptor_generator(tags)
        x_semantic = self.sentence_encoder(descriptors)
        h_semantic = self.mlp(x_semantic)
        z_semantic = self.classifier(h_semantic)
        return descriptors, x_semantic, h_semantic, z_semantic
