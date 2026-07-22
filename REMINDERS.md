# Reminders — discussing LLMs

Loose notes for how we talk about LLMs, and for anything that builds on that talk later (docs, evals, harnesses). Keep this light: prefer stable framing over strong claims.

## Scope of depth

- Default: do **not** go deep on how a model is built (architecture, training stack, internals).
- Exception: open-source or open-weights models — deeper discussion is fair when weights/code are actually inspectable.
- Closed / proprietary models: stay at the interface — behavior, APIs, reported capabilities, evaluation surface.

## Assumptions that are clear enough to rely on

These are the shared ground we can treat as settled for discussion and for future harness design:

- An LLM maps text (and often multimodal) input to token predictions / generated output.
- Behavior is shaped by training data, objectives, and post-training (alignment, instruction tuning, etc.) — without needing a full recipe.
- Outputs are stochastic unless decoding is fixed; repeats are not guaranteed identical.
- Prompting, context, tools, and system instructions change what you get; the “model alone” is rarely the whole system.
- Evaluation needs a defined task, inputs, and scoring rule — “vibes” are not a harness.
- Open weights ≠ open everything: weights can be public while data, code, or training details stay opaque.

## Assumptions that are *not* fully clear

Treat these as open or model-/setup-dependent. Do not hard-code them into harness assumptions:

- Exact training data mix, data quality, or contamination for a given model.
- Whether a capability is “understood,” memorized, or scaffolded by the surrounding system.
- How much of observed behavior is the base model vs. the product wrapper (safety layers, retrieval, tools, routing).
- Fair comparisons across labs when recipes, compute, and eval protocol differ.
- Long-horizon reliability, agency, or “reasoning” claims without a concrete operational definition.
- What will stay true as models and serving stacks change.

## For future harnesses

Motivate harnesses from the clear list above; leave the unclear list as knobs, caveats, or out-of-scope — not as load-bearing premises.

- Prefer black-box, interface-level tests unless the target is open weights / open source.
- Make decoding, context, tools, and version/identity of the model explicit inputs.
- Separate “model” from “system under test” when wrappers or tools are involved.
- Keep scoring rules and task definitions independent of unverified training-story claims.
- When comparing models, document what you *cannot* control as clearly as what you can.

## CGU-3 vs Augment (SimCLR parallel)

When applying a test in this project, **state the motivation for running it on CGU-3** (`cg3` in code) rather than on an Augment-style framework.

- Augment is the natural SimCLR parallel: contrastive learning via augmented views of the input.
- That Augment line is what GeoAug, LLM-based augmentation, and similar view-generation methods sit next to — so it is the easy baseline family for “augmentation as the contrastive signal.”
- CGU-3 is a different contrastive setup (graph-to-graph / hierarchy, not just aug-views). Choosing it changes what the test is probing.
- Always write down *why* CGU-3 (not Augment) so the choice is explicit and results stay comparable to Augment variants (GeoAug, LLM, etc.) without readers having to reverse-engineer the design decision.

Update this file when an assumption moves from unclear → clear (or the reverse), rather than baking temporary stories into harness code.
