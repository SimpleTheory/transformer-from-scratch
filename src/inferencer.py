import torch
import nn_modules
import trainer
import tiktoken
from utility import to_device

encoder = tiktoken.get_encoding('gpt2')
n_vocab = encoder.n_vocab
config = trainer.Config('./whatever',
                        load_path=r'C:\Users\arigf\Downloads\model_results_from_kaggle\transformer_from_scratch.pt')


def encode(text, start_with_boundary=True, convert_to_tensor=True):
    result = []
    if start_with_boundary:
        result.append(encoder.eot_token)
    result += encoder.encode_ordinary(text)
    if convert_to_tensor:
        return to_device(torch.tensor(result))
    return result + encoder.encode_ordinary(text)


def decode(token_ids) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().flatten().tolist()
    else:
        token_ids = list(token_ids)
    return encoder.decode(token_ids)


def convenient_generate(text_or_texts: str | list[str], max_tokens: int = None):
    if isinstance(text_or_texts, str):
        # This adds a batch size of 1 at position 0: so (3,) -> (1,3)
        m_input = encode(text_or_texts).unsqueeze(0)
        if max_tokens is None:
            max_tokens = config.context_length - len(text_or_texts)
    else:
        m_input = torch.stack([encode(text_or_texts) for _ in text_or_texts])
        if max_tokens is None:
            max_tokens = config.context_length - len(text_or_texts[0])
    return decode(model.generate(m_input, max_tokens))


# model.generate(torch.tensor(encode("There was a king that lived in"), dtype=torch.long), 100)

if __name__ == '__main__':
    model = nn_modules.GPTModel(
        vocab_size=n_vocab,
        embedding_dimension=config.embedding_dim,
        max_sequence_length=config.context_length,
        total_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout_probability=config.dropout,
    )
    to_device(model)

    model.load_state_dict(torch.load(config.load_path))
    out = convenient_generate('The princess was poisoned by an apple and fell into a deep sleep')
    print(out)
