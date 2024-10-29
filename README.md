# ProbeLens
### A framework to enable probing of language models.
![CI](https://github.com/sharanry/probe-lens/actions/workflows/ci.yaml/badge.svg)

## API
### Example Usage

Here is an example of how to use the ProbeLens framework to generate probe data and train a linear probe on a spelling task:

```python
from probe_lens.experiments.spelling import FirstLetterSpelling

words = ["example", "words", "to", "spell"]
spelling_task = FirstLetterSpelling(words)
```
```python
from sae_lens import HookedSAETransformer, SAE
DEVICE = "mps"
model = HookedSAETransformer.from_pretrained_no_processing("gpt2-small", device=DEVICE)
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",  # see other options in sae_lens/pretrained_saes.yaml
    sae_id="blocks.8.hook_resid_pre",  # won't always be a hook point
    device=DEVICE,
)
```

```python
from torch.utils.data import DataLoader
dataset = spelling_task.generate_probe_data(model, sae, device=DEVICE)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

```python
from probe_lens.probes import LinearProbe
X, y = next(iter(dataloader))
probe = LinearProbe(X.shape[1], y.shape[1], class_names=spelling_task.get_classes())
```

```python
import torch.optim as optim
probe.train_probe(dataloader, optim.SGD(probe.parameters(), lr=0.01), val_dataloader=None, epochs=1000)
plot = probe.visualize_performance(dataloader)
```


## Roadmap
### Functionalities
- [x] Add basic linear probe with tests
- [ ] Add regularization
- [ ] Add sparsity
- [ ] Add other kind of probes
    - [ ] Non-linear probes
    - [ ] ... ?

### Applications
- [ ] Reproduce results from SAE-Spelling
    - [ ] First letter data generation
- [ ] Add more probing experiments
    - [ ] ... ?

### Visualization
- [x] Add Confusion Matrix 
- [x] Add f2 score and accuracy
- [ ] Add more visualization experiments
    - [ ] ... ?

