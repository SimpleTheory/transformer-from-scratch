from tokenizers import Tokenizer
from transformer_from_scratch import project_root

tokenizer_base = Tokenizer.from_file(project_root("data/qwen3_06/tokenizer-base.json"))
tokenizer_reasoning = Tokenizer.from_file(project_root("data/qwen3_06/tokenizer-reasoning.json"))

# TODO implement

