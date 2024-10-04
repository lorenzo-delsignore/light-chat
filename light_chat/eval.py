import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from light_chat.models.trigram_module import NgramModuleVanilla
from light_chat.utils import (
    RankedLogger,
    extras,
)

log = RankedLogger(__name__, rank_zero_only=True)


def evaluate(cfg: DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.
    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    idx_to_char = datamodule.itos
    model = NgramModuleVanilla()
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    for i in range(5):
        out = []
        ix = [0, 0, 0]
        while True:
            ngram = {"ngram": torch.tensor([ix])}
            logits = model(ngram)
            probs = logits.exp()
            probs = probs / probs.sum(dim=1, keepdim=True)
            next_char = torch.multinomial(probs, num_samples=1, replacement=True).item()
            ix = ix[1:] + [next_char]
            out.append(idx_to_char[next_char])
            if next_char == 0:
                break
        print("".join(out))


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
