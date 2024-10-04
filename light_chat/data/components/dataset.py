import torch
from torch.utils.data import Dataset

from light_chat.utils.dataset import (
    character_to_index_dictionary,
    index_to_character_dictionary,
)


class NgramDataset(Dataset):
    def __init__(self, file_path: str, transform: bool = True) -> None:
        self.lines = open(file_path, "r").read().splitlines()
        self.ngrams, self.next_chars, self.stoi, self.itos = self.build_dataset()
        self.transform = transform

    def build_dataset(self, n=3):
        ngrams = []
        next_chars = []
        characters = "." + "".join(self.lines)
        stoi = character_to_index_dictionary(characters)
        itos = index_to_character_dictionary(characters)
        for line in self.lines:
            ngram = [0] * n
            for c in line + ".":
                ix = stoi[c]
                ngrams.append(ngram)
                next_chars.append(ix)
                ngram = ngram[1:] + [ix]
        return ngrams, next_chars, stoi, itos

    def __len__(self) -> int:
        return len(self.ngrams)

    def __getitem__(self, index: int):
        ngram = torch.tensor(self.ngrams[index])
        next_char = torch.tensor(self.next_chars[index])
        return {"ngram": ngram, "label": next_char}

    def __repr__(self) -> str:
        return f"NgramDataset(n_instances={len(self.ngrams)})"
