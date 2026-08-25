from pathlib import Path
import torch
from transformer_from_scratch.base_objects.blocks_and_models import GPTModel
import transformer_from_scratch.trainer.trainer as trainer
import tiktoken
from transformer_from_scratch.trainer.utility import to_device

encoder = tiktoken.get_encoding('gpt2')
n_vocab = encoder.n_vocab
config = trainer.Config('./whatever', load_path=r'C:\Users\arigf\Downloads\model_result_from_brandon\checkpoint.pt')
eot_token = '<|endoftext|>'

def isolate_prompted_story(text: str):
    if text.startswith(eot_token):
        text = text[len(eot_token):]
    return text.split(eot_token)[0]


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


def convenient_generate(
        text_or_texts: str | list[str],
        max_tokens: int = None,
        start_with_eot: bool = False,
        get_only_prompted_story: bool = True
):
    if isinstance(text_or_texts, str):
        # This adds a batch size of 1 at position 0: so (3,) -> (1,3)
        m_input = encode(text_or_texts, start_with_eot).unsqueeze(0)
        if max_tokens is None:
            max_tokens = config.context_length - len(text_or_texts)
        result = decode(model.generate(m_input, max_tokens))
        if get_only_prompted_story:
            return isolate_prompted_story(result)
        return result
    else:
        m_input = torch.stack([encode(text, start_with_eot) for text in text_or_texts])
        if max_tokens is None:
            max_tokens = config.context_length - len(text_or_texts[0])
    if get_only_prompted_story:
        return [isolate_prompted_story(decode(out)) for out in model.generate(m_input, max_tokens).tolist()]
    return [decode(out) for out in model.generate(m_input, max_tokens).tolist()]
    # return decode(model.generate(m_input, max_tokens))

def save_to_file(path, text_or_texts, preface=''):
    path = Path(path)
    if isinstance(text_or_texts, list):
        text_or_texts = '\n-----------------------------------------\n'.join(text_or_texts)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(preface + text_or_texts, encoding='utf-8')


# model.generate(torch.tensor(encode("There was a king that lived in"), dtype=torch.long), 100)

if __name__ == '__main__':
    model = GPTModel(
        vocab_size=n_vocab,
        embedding_dimension=config.embedding_dim,
        max_sequence_length=config.context_length,
        total_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout_probability=config.dropout,
    )
    to_device(model)

    model.load_state_dict(torch.load(config.load_path))
    prompt = "If she comes in time, we"
    repeat_times = 5
    start_with_eot = True
    out = convenient_generate([prompt]*repeat_times, start_with_eot=start_with_eot)
    # out = convenient_generate('The princess was poisoned by an apple and fell into a deep sleep')
    save_to_file(
        trainer.data_dir / 'inference_outputs/she_comes_in_time_3.txt',
        out,
f'PROMPT: {prompt}\nTIMES REPEATED: {repeat_times}\nSTART W/ EOT: {start_with_eot}\n--------------------\n')
