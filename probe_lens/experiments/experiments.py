from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from sae_lens import SAE, HookedSAETransformer
from tqdm.autonotebook import tqdm
from transformer_lens import HookedTransformer

"""
Probe experiments are used to generate data for probing tasks.
"""


class ProbeExperiment(ABC):
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.data = None

    def __repr__(self) -> str:
        return self.task_name

    @abstractmethod
    def get_data(self) -> list[tuple[str, int]]:
        pass

    @abstractmethod
    def get_classes(self) -> list[str]:
        pass

    def generate_probe_data(
        self,
        hooked_model: HookedSAETransformer | HookedTransformer,
        sae: SAE | None = None,
        device: str = "cpu",  # consistent with sae_lens and transformer_lens
    ) -> torch.utils.data.TensorDataset:
        sae_acts = []
        answer_classes = []
        for i, (prompt, answer_class) in enumerate(
            tqdm(self.get_data(), desc="Generating probe data")
        ):
            if sae is not None:
                _, cache = hooked_model.run_with_cache_with_saes(
                    prompt, saes=[sae], stop_at_layer=sae.cfg.hook_layer + 1
                )
                sae_acts.append(
                    cache[sae.cfg.hook_name + ".hook_sae_acts_post"][0, -1, :]
                )
            else:
                raise ValueError("Not implemented for non-SAE models.")

            answer_classes.append(answer_class)

        one_hot_answer_classes = F.one_hot(
            torch.tensor(answer_classes, device=device),
            num_classes=len(self.get_classes()),
        ).float()
        dataset = torch.utils.data.TensorDataset(
            torch.stack(sae_acts), one_hot_answer_classes
        )
        return dataset
