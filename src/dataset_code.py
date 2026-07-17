from collections.abc import Iterable, Sequence
from itertools import islice
from pathlib import Path
import tiktoken
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
from torch.utils.data import Dataset


class TransformerTextDataset(Dataset):
    """
    Fixed-length dataset for causal next-token prediction.

    inputs:  [token_0, token_1, ..., token_n]
    targets: [token_1, token_2, ..., token_n+1]
    """

    def __init__(
        self,
        data: str | Iterable[str] | HuggingFaceDataset | DatasetDict,
        context_length: int,
        *,
        split: str = "train",
        text_column: str | None = None,
        encoding_name: str = "gpt2",
        stride: int | None = None,
        max_documents: int | None = None,
        selection_seed: int | None = None,
        add_document_boundaries: bool = True,
    ):
        if context_length < 1:
            raise ValueError("context_length must be at least 1.")

        self.tokenizer = tiktoken.get_encoding(encoding_name)
        self.context_length = context_length
        self.stride = context_length if stride is None else stride

        if self.stride < 1:
            raise ValueError("stride must be at least 1.")

        texts = self._get_texts(
            data=data,
            split=split,
            text_column=text_column,
            max_documents=max_documents,
            selection_seed=selection_seed,
        )

        token_ids: list[int] = []

        # One shared boundary token acts as the start of the first
        # document and the end/start boundary between documents.
        if add_document_boundaries:
            token_ids.append(self.document_boundary_token)

        document_count = 0

        for text in texts:
            if not isinstance(text, str):
                raise TypeError(
                    "Every document must be a string. "
                    "Check that text_column references the correct column."
                )

            token_ids.extend(self.encode(text))

            if add_document_boundaries:
                token_ids.append(self.document_boundary_token)

            document_count += 1

        if document_count == 0:
            raise ValueError("The dataset contains no documents.")

        self.tokens = torch.tensor(token_ids, dtype=torch.long)

        if len(self.tokens) <= context_length:
            raise ValueError(
                f"The dataset contains {len(self.tokens)} tokens but needs "
                f"more than context_length={context_length}."
            )

    @staticmethod
    def _infer_text_column(dataset: HuggingFaceDataset) -> str:
        """Infer the dataset's text column."""

        if "text" in dataset.column_names:
            return "text"

        string_columns = [
            name
            for name, feature in dataset.features.items()
            if getattr(feature, "dtype", None) in {"string", "large_string"}
        ]

        if len(string_columns) == 1:
            return string_columns[0]

        raise ValueError(
            "Could not infer the text column. "
            f"Available columns: {dataset.column_names}. "
            "Pass text_column explicitly."
        )

    @classmethod
    def _get_texts(
        cls,
        data,
        split: str,
        text_column: str | None,
        max_documents: int | None,
        selection_seed: int | None,
    ) -> Iterable[str]:
        """Extract an iterable of documents from supported data sources."""

        if isinstance(data, DatasetDict):
            if split not in data:
                raise KeyError(
                    f"Split {split!r} does not exist. "
                    f"Available splits: {list(data.keys())}"
                )

            data = data[split]

        if isinstance(data, HuggingFaceDataset):
            text_column = text_column or cls._infer_text_column(data)

            if text_column not in data.column_names:
                raise KeyError(
                    f"Column {text_column!r} does not exist. "
                    f"Available columns: {data.column_names}"
                )

            if max_documents is not None:
                if max_documents < 1:
                    raise ValueError("max_documents must be at least 1.")

                if selection_seed is not None:
                    data = data.shuffle(seed=selection_seed)

                data = data.select(
                    range(min(max_documents, len(data)))
                )

            return data[text_column]

        if isinstance(data, str):
            return [data]

        if max_documents is not None:
            return islice(data, max_documents)

        return data

    @classmethod
    def from_files(
        cls,
        paths: Iterable[str | Path],
        context_length: int,
        *,
        file_encoding: str = "utf-8",
        **kwargs,
    ) -> "TransformerTextDataset":
        texts = (
            Path(path).read_text(encoding=file_encoding)
            for path in paths
        )

        return cls(
            texts,
            context_length=context_length,
            **kwargs,
        )

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def document_boundary_token(self) -> int:
        return self.tokenizer.eot_token

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode_ordinary(text)

    def decode(
        self,
        token_ids: torch.Tensor | Sequence[int],
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().flatten().tolist()
        else:
            token_ids = list(token_ids)

        return self.tokenizer.decode(token_ids)

    untokenize = decode

    def __len__(self) -> int:
        return (
            len(self.tokens) - self.context_length - 1
        ) // self.stride + 1

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)

        if not 0 <= index < len(self):
            raise IndexError("Dataset index out of range.")

        start = index * self.stride
        end = start + self.context_length + 1

        sequence = self.tokens[start:end]

        inputs = sequence[:-1]
        targets = sequence[1:]

        return inputs, targets