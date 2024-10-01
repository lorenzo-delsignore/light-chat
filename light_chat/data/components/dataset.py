import torch
from torch.utils.data import Dataset

from light_chat.utils.dataset import (
    character_to_index_dictionary,
    index_to_character_dictionary,
)


class TrigramDataset(Dataset):
    def __init__(self, file_path: str, transform: bool = True) -> None:
        self.lines = open(file_path, "r").read().splitlines()
        (
            self.trigram_matrix,
            self.bigrams,
            self.next_chars,
            self.bigram_to_idx,
            self.char_to_idx,
            self.idx_to_bigram,
            self.idx_to_char,
        ) = self.build_dataset()
        self.transform = transform

    def build_dataset(self):
        trigram_count = {}
        bigrams = []
        next_chars = []
        count_matrix = torch.ones((601, 27), dtype=torch.float32)  # count smoothing
        for line in self.lines:
            line = f".{line}."
            for i in range(len(line) - 2):
                bigram = line[i : i + 2]
                next_char = line[i + 2]
                trigram = (bigram, next_char)
                trigram_count[trigram] = trigram_count.get(trigram, 0) + 1
                bigrams.append(bigram)
                next_chars.append(next_char)
        bigram_to_idx, char_to_idx = (
            character_to_index_dictionary(bigrams),
            character_to_index_dictionary(next_chars),
        )
        idx_to_bigram, idx_to_char = (
            index_to_character_dictionary(bigrams),
            index_to_character_dictionary(next_chars),
        )

        for trigram, count in trigram_count.items():
            bigram_idx, char_idx = bigram_to_idx[trigram[0]], char_to_idx[trigram[1]]
            count_matrix[bigram_idx, char_idx] = count
        count_matrix = count_matrix / count_matrix.sum(dim=1, keepdim=True)
        return (
            count_matrix,
            bigrams,
            next_chars,
            bigram_to_idx,
            char_to_idx,
            idx_to_bigram,
            idx_to_char,
        )

    def __len__(self) -> int:
        return len(self.trigram_matrix)

    def __getitem__(self, index: int):
        bigram = self.bigrams[index]
        next_char = self.next_chars[index]
        bigram_idx = self.bigram_to_idx[bigram]
        next_char_idx = self.char_to_idx[next_char]
        probability_matrix = self.trigram_matrix[bigram_idx]
        return {
            "bigram": bigram,
            "bigram_idx": bigram_idx,
            "label": next_char,
            "label_idx": next_char_idx,
            "probability_logits": probability_matrix,
        }

    def __repr__(self) -> str:
        return f"TrigramDataset(n_instances={len(self.trigram_matrix)})"
