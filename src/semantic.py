"""Semantic channel: TAG → descriptor → sentence encoder → MLP → classifier."""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: `transformers` is imported lazily inside the two backbone classes, so a
# run with --semantic-embeddings (precomputed view) needs no transformers install.


def strip_label_declarations(text: str, class_names: list[str]) -> str:
    """Remove label declarations from a generated explanation before encoding.

    Encoding an explanation WITH its predicted-label line reports mostly label
    leakage, not semantic signal — and prefix-matching "Answer:" alone is not
    enough (measured on the LLMNodeBed gpt-4o-mini dumps: only ~60-70% of
    explanations start that way; the rest open with "The categories are:",
    a bare class list, or bolded per-class headers). So strip structurally:
    leading declaration lines, lines that are only class-name tokens, and
    per-item class-name headers.
    """
    if not class_names:
        return text
    decl = re.compile(
        r"^(answer|prediction|label|category|categories|the (most )?likely"
        r" (category|categories|class(es)?)|the categor(y|ies))\b.{0,120}$",
        re.I,
    )
    names = "|".join(re.escape(c) for c in class_names)
    only_names = re.compile(rf"^({names})([\s,;:/&·-]+({names}))*[\s.,;:]*$", re.I)
    item_prefix = re.compile(r"^\s*(\d+[.)]|[-*•])\s*|\*+|_{1,2}|:\s*$")

    out, leading = [], True
    for line in text.splitlines():
        bare = item_prefix.sub("", line.strip()).strip()
        if leading and (not bare or decl.match(bare) or only_names.match(bare)):
            continue  # drop the leading declaration block
        leading = False
        if only_names.match(bare):
            continue  # drop per-item label headers inside the body
        out.append(line)
    cleaned = "\n".join(out).strip()
    return cleaned or text  # never hand the encoder an empty string


# TAPE's prompt shape (He et al., ICLR 2024, arXiv:2305.19523): node text, then a
# closed category list, then a ranked prediction plus a free-text explanation.
# Deliberately NO chain-of-thought (Chen et al. 2024, Obs. 13: unreliable) and NO
# neighbor summaries (Obs. 15-16: help only under homophily, hurt PubMed).
TAPE_PROMPT = (
    "{text}\n\n"
    "Question: Which of the following categories does this paper belong to: {class_names}?\n"
    "Give your best guesses as a comma-separated list ordered from most to least likely, "
    "then explain your reasoning based on the text.\n"
    "Answer:"
)


class GraniteDescriptorGenerator(nn.Module):
    """TAPE-style prediction+explanation for each node's text (LLM is swappable;
    TAPE Table 4 shows robustness to the backbone, so it is fixed, not ablated)."""

    def __init__(
        self,
        model_name: str = "ibm-granite/granite-4.2-3b",
        max_new_tokens: int = 256,
        class_names: list[str] | None = None,
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.class_names = class_names or []
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
        class_names = ", ".join(self.class_names) if self.class_names else "the dataset's classes"
        descriptors = []
        for tag in tags:
            messages = [
                {
                    "role": "user",
                    "content": TAPE_PROMPT.format(text=tag, class_names=class_names),
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
                # greedy decoding: the explanation is a FEATURE, so it must be the
                # same text every time the run is repeated with the same seed
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
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
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        # E5 models are trained with an input prefix and underperform without it
        # (Wang et al. 2022, arXiv:2212.03533)
        self.input_prefix = "query: " if "e5" in model_name.lower() else ""
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
        if self.input_prefix:
            texts = [self.input_prefix + t for t in texts]
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
    """Text → [optional TAPE-style descriptor] → sentence embedding → MLP → logits.

    With `descriptor_generator=None` the node's OWN text is encoded directly —
    the primary arm: a frozen strong encoder on raw text (Chen et al. 2024,
    Obs. 3/6). With a generator, the LLM's prediction+explanation is encoded
    instead (the TAPE arm), with label declarations stripped first unless
    `strip_labels=False` (the un-stripped variant IS the leak control).

    The descriptor LLM and the sentence encoder are frozen, so their output is
    computed ONCE per run and cached: without the cache every epoch (and every
    validation pass) would re-run LLM inference over all nodes and the semantic
    view would change under the model while it trains.
    """

    requires_texts = True

    def __init__(
        self,
        descriptor_generator: GraniteDescriptorGenerator | None,
        sentence_encoder: HuggingFaceSentenceEncoder,
        hidden_dim: int,
        semantic_dim: int,
        num_classes: int,
        class_names: list[str] | None = None,
        strip_labels: bool = True,
    ):
        super().__init__()
        self.descriptor_generator = descriptor_generator
        self.sentence_encoder = sentence_encoder
        self.class_names = class_names or []
        self.strip_labels = strip_labels
        sentence_dim = sentence_encoder.embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(sentence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
        )
        self.classifier = nn.Linear(semantic_dim, num_classes)
        self._cached_descriptors: list[str] | None = None
        self._cached_x: torch.Tensor | None = None

    def forward(self, tags: list[str]):
        if self._cached_x is None:
            if self.descriptor_generator is None:
                texts = tags  # primary arm: the node's own text
            else:
                self._cached_descriptors = self.descriptor_generator(tags)
                texts = self._cached_descriptors
                if self.strip_labels:
                    texts = [strip_label_declarations(t, self.class_names) for t in texts]
            self._cached_x = self.sentence_encoder(texts).detach()
        descriptors, x_semantic = self._cached_descriptors, self._cached_x
        h_semantic = self.mlp(x_semantic)
        z_semantic = self.classifier(h_semantic)
        return descriptors, x_semantic, h_semantic, z_semantic


class PrecomputedSemanticChannel(nn.Module):
    """Precomputed sentence embeddings → MLP → class logits.

    For encoder views that cannot run locally (API-only models such as
    text-embedding-3-large) and for freezing one view across seeds and arms.
    """

    requires_texts = False

    def __init__(self, embeddings: torch.Tensor, hidden_dim: int, semantic_dim: int, num_classes: int):
        super().__init__()
        self.register_buffer("embeddings", embeddings)
        self.mlp = nn.Sequential(
            nn.Linear(embeddings.size(1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, semantic_dim),
        )
        self.classifier = nn.Linear(semantic_dim, num_classes)

    def forward(self, tags: list[str] | None = None):
        x_semantic = self.embeddings
        h_semantic = self.mlp(x_semantic)
        z_semantic = self.classifier(h_semantic)
        return None, x_semantic, h_semantic, z_semantic
