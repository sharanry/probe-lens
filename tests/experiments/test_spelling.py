import torch
from sae_lens import SAE, HookedSAETransformer
from torch.utils.data import DataLoader

from probe_lens.experiments.spelling import LETTERS, FirstLetterSpelling
from probe_lens.probes import LinearProbe

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def test_first_letter_spelling():
    words = ["example", "words", "to", "spell"]
    spelling_task = FirstLetterSpelling(words)
    data = spelling_task.data
    classes = [c for _, c in data]
    assert classes == [LETTERS.index(word.lower()[0]) for word in words]


def test_first_letter_spelling_default_dataset():
    spelling_task = FirstLetterSpelling()
    assert len(spelling_task.data) == 10000


def test_first_letter_spelling_probe_data():
    words = ["example", "words", "to", "spell"]
    spelling_task = FirstLetterSpelling(words)
    model = HookedSAETransformer.from_pretrained_no_processing(
        "gpt2-small", device=DEVICE
    )
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
        device=DEVICE,
    )
    dataset = spelling_task.generate_probe_data(model, sae, device=DEVICE)
    assert len(dataset) == len(words)


def test_first_letter_spelling_probe_training():
    words = ["example", "words", "to", "spell"]
    spelling_task = FirstLetterSpelling(words)
    model = HookedSAETransformer.from_pretrained_no_processing(
        "gpt2-small", device=DEVICE
    )
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
        sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
        device=DEVICE,
    )
    dataset = spelling_task.generate_probe_data(model, sae, device=DEVICE)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    X, y = next(iter(dataloader))
    probe = LinearProbe(X.shape[1], y.shape[1], device=DEVICE, class_names=LETTERS)
    probe.train_probe(
        dataloader, torch.optim.SGD(probe.parameters(), lr=0.01), val_dataloader=None
    )
