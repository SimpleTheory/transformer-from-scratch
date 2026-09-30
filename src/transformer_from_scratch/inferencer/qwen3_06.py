from tokenizers import Tokenizer
from transformer_from_scratch import project_root
from safetensors.torch import load_file
from transformer_from_scratch.base_objects.blocks_and_models import SmallQwen3Model
import torch
from transformer_from_scratch.trainer.utility import to_device

base_model_file = project_root('/data/qwen3_06/qwen3-0.6B-base.pth')
control_reasoning_model_file = project_root('/data/qwen3_06/qwen3-0.6B-reasoning.pth')
tokenizer_base = Tokenizer.from_file(str(project_root("data/qwen3_06/tokenizer-base.json")))
tokenizer_reasoning = Tokenizer.from_file(str(project_root("data/qwen3_06/tokenizer-reasoning.json")))

# TODO FOR RIGHT NOW THESE FUNCTIONS ONLY SUPPORT SINGLE SEQUENCE
# BUT I SHOULD ADD MULTI-LENGTH BATCH GENERATION
# For that though I need to refactor the model to allow for padded inputs such that they don't affect:
    # Embedding
    # Attention Results
    # Rope
# And create helpers that actually pad the sequences initially
def tokenize(text: str, tokenizer: Tokenizer = tokenizer_base, device: torch.device | str | None = None,) -> torch.Tensor:
    """
    Convert text into token IDs with a batch dimension:

        "Hello" -> (sequence_length,)
                -> (1, sequence_length)

    The model expects:
        (batch, sequence_length)
    """
    token_ids = tokenizer.encode(text).ids
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    if device is not None:
        token_ids = token_ids.to(device)
    return token_ids


def detokenize(token_ids: torch.Tensor | list[int], tokenizer: Tokenizer = tokenizer_base, skip_special_tokens: bool = False,) -> str:
    """
    Convert token IDs back into text.

    Accepts either:
        (sequence_length,)
        (1, sequence_length)
        list[int]
    """
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim == 2:
            if token_ids.shape[0] != 1:
                raise ValueError("detokenize() currently expects a single sequence.")
            token_ids = token_ids[0]
        token_ids = token_ids.detach().cpu().tolist()

    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens,)


def get_eos_token_id(tokenizer: Tokenizer) -> int | None:
    """
    Find the Qwen EOS token directly from the tokenizer instead of
    hard-coding its integer ID.
    """
    for token in ("<|endoftext|>", "<|im_end|>",):
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return token_id
    return None

@torch.no_grad()
def generate_text(
        model: SmallQwen3Model,
        prompt: str,
        tokenizer: Tokenizer = tokenizer_base,
        max_new_tokens: int = 1000,
        start_with_eos: bool = True,
        use_cache: bool = True,
        return_only_new_text: bool = False,
) -> str:
    """
    Tokenize -> generate -> detokenize.
    """
    model.eval()
    device = next(model.parameters()).device
    eos_token_id = get_eos_token_id(tokenizer)
    # (batch size, sequence length)
    input_ids = tokenize(prompt, tokenizer=tokenizer, device=device,)
    if start_with_eos:
        # since batch size is always one this gives (1, sequence length) as well
        shaped_eos_token = torch.tensor([eos_token_id], device=device, dtype=input_ids.dtype).unsqueeze(0)
        input_ids = torch.cat((shaped_eos_token, input_ids), dim=-1)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        eos_token_id=eos_token_id,
    )

    if return_only_new_text:
        generated_ids = generated_ids[:, input_ids.shape[1]:]

    return detokenize(generated_ids, tokenizer=tokenizer,)


@torch.no_grad()
def load(target, source_name, weights):
    if source_name not in weights:
        raise KeyError(f"Weight {source_name!r} not found in checkpoint")
    source = weights[source_name]
    if target.shape != source.shape:
        raise ValueError(f"Shape mismatch for {source_name}: model={tuple(target.shape)}, checkpoint={tuple(source.shape)}")
    target.copy_(source.to(device=target.device, dtype=target.dtype))

@torch.no_grad()
def load_qwen3_weights_from_official_safetensors(model, path):
    weights = load_file(path, device="cpu")

    # Embedding
    load(model.embedding_layer.embedding_matrix,"model.embed_tokens.weight", weights)

    # Transformer blocks
    for index, block in enumerate(model.attention_blocks):
        prefix = f"model.layers.{index}"

        # Attention
        load(block.attention.query_weights, f"{prefix}.self_attn.q_proj.weight", weights)
        load(block.attention.key_weights, f"{prefix}.self_attn.k_proj.weight", weights)
        load(block.attention.value_weights, f"{prefix}.self_attn.v_proj.weight", weights)
        load(block.attention.final_linear_layer, f"{prefix}.self_attn.o_proj.weight", weights)

        # Q/K normalization
        load(block.attention.query_norm.weights, f"{prefix}.self_attn.q_norm.weight", weights)
        load(block.attention.key_norm.weights, f"{prefix}.self_attn.k_norm.weight", weights)

        # Block normalization
        load(block.norm1.weights, f"{prefix}.input_layernorm.weight", weights)
        load(block.norm2.weights, f"{prefix}.post_attention_layernorm.weight", weights)

        # SwiGLU
        load(block.ffn.gate_weights, f"{prefix}.mlp.gate_proj.weight", weights)
        load(block.ffn.up_weights, f"{prefix}.mlp.up_proj.weight", weights)
        load(block.ffn.down_weights, f"{prefix}.mlp.down_proj.weight", weights)

    # Final RMSNorm
    load(model.final_norm.weights, "model.norm.weight", weights)

    return model

@torch.no_grad()
def load_qwen3_weights_from_raschka_pt(model, checkpoint_path):
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Token embeddings
    load(model.embedding_layer.embedding_matrix, "tok_emb.weight", weights)

    # Transformer blocks
    for index, block in enumerate(model.attention_blocks):
        prefix = f"trf_blocks.{index}"
        att = block.attention

        # QKV projections
        load(att.create_query.weights, f"{prefix}.att.W_query.weight", weights)
        load(att.create_key.weights, f"{prefix}.att.W_key.weight", weights)
        load(att.create_value.weights, f"{prefix}.att.W_value.weight", weights)

        # Attention output projection
        load(att.final_linear_layer.weights, f"{prefix}.att.out_proj.weight", weights)

        # Q/K RMSNorm
        load(att.query_norm.weights, f"{prefix}.att.q_norm.scale", weights)
        load(att.key_norm.weights, f"{prefix}.att.k_norm.scale", weights)

        # Transformer-block RMSNorms
        load(block.norm1.weights, f"{prefix}.norm1.scale", weights)
        load(block.norm2.weights, f"{prefix}.norm2.scale", weights)

        # SwiGLU FFN
        load(block.ffn.gate.weights, f"{prefix}.ff.fc1.weight", weights)  # This is confusing naming wise
        load(block.ffn.up.weights, f"{prefix}.ff.fc2.weight", weights)
        load(block.ffn.down.weights, f"{prefix}.ff.fc3.weight", weights)

    # Final RMSNorm
    load(model.final_norm.weights, "final_norm.scale", weights)

    return model


config = SmallQwen3Model.config_factory(True)
model = SmallQwen3Model(**config)
to_device(model)
load_qwen3_weights_from_raschka_pt(model, control_reasoning_model_file)

if __name__ == '__main__':
    prompt = "Who was the first person to land on the moon"
    generated_text = generate_text(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer_reasoning,
        max_new_tokens=2000,
        use_cache=True,
    )
    print(generated_text)
