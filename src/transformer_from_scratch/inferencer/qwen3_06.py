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

# TODO implement tokenizer and generator and detokenizer (maybe make the tokenizers global?)

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
load_qwen3_weights_from_raschka_pt(model, base_model_file)

if __name__ == '__main__':
    # TODO Generate basic text with the model loaded to CUDA
    for name, param in model.named_parameters():
        print(name, param.shape)
