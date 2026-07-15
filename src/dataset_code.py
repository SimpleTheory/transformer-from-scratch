from collections.abc import Sequence
from pathlib import Path
import tiktoken
import torch
from torch.utils.data import Dataset


class TransformerTextDataset(Dataset):
    """
    Creates fixed-length input/target sequences for next-token prediction.

    inputs:  [token_0, token_1, ..., token_n]
    targets: [token_1, token_2, ..., token_n+1]
    """

    def __init__(
            self,
            texts: str | Sequence[str],
            context_length: int,
            encoding_name: str = "gpt2",
            stride: int | None = None,
            add_end_of_text: bool = True,
    ):
        if context_length < 1:
            raise ValueError("context_length must be at least 1.")

        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.context_length = context_length
        self.stride = context_length if stride is None else stride

        if self.stride < 1:
            raise ValueError("stride must be at least 1.")

        if isinstance(texts, str):
            texts = [texts]

        token_ids: list[int] = []

        for text in texts:
            token_ids.extend(self.encode(text))

            if add_end_of_text:
                token_ids.append(self.end_of_text_token)

        self.tokens = torch.tensor(token_ids, dtype=torch.long)

        if len(self.tokens) <= context_length:
            raise ValueError(
                f"Dataset contains {len(self.tokens)} tokens, but needs more than "
                f"context_length={context_length}."
            )

    @classmethod
    def from_file(
            cls,
            path: str | Path,
            context_length: int,
            file_encoding: str = "utf-8",
            **kwargs,
    ) -> "TransformerTextDataset":
        text = Path(path).read_text(encoding=file_encoding)
        return cls(text, context_length=context_length, **kwargs)

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.max_token_value + 1

    @property
    def end_of_text_token(self) -> int:
        return self.tokenizer.eot_token

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids: torch.Tensor | Sequence[int]) -> str:
        """Convert token IDs back into text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().flatten().tolist()

        return self.tokenizer.decode(token_ids)

    untokenize = decode

    def __len__(self) -> int:
        available_starts = len(self.tokens) - self.context_length - 1
        return available_starts // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        sequence = self.tokens[start: start + self.context_length + 1]

        inputs = sequence[:-1]
        targets = sequence[1:]

        return inputs, targets
