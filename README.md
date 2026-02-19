# Opti Hub
Opti Hub is a lightweight registry and installer for state‑of‑the‑art optimizers used in modern ML and LLM research.

## Why This Exists
Since the advent of Muon, we have seen a resurgence in the creation of new optimizers for machine learning, especially for large language models. Many of these methods appear first in papers or GitHub repositories and are not yet integrated into mainstream frameworks like PyTorch.

Opti Hub is built to help researchers:
- quickly install and try new or SOTA optimizers
- benchmark optimizers that are not yet in the mainline ML stacks
- use `registry.toml` as a learning map of what is available and where it lives

## How It Works
Opti Hub reads `registry.toml` to discover optimizer packages, install sources, and import paths. The file is intended to be both a registry and a learning map you can expand.

## Install
If you are on Python 3.11+, `tomllib` is built in. For Python < 3.11, install `tomli`.

```bash
pip install tomli
```

## Example Usage
Basic installation of optimizers from the registry:

```python
from opti_hub import OptiHub

tool = OptiHub()
tool.install("Muon", "Swan")
```

Instantiate an optimizer by name:

```python
from opti_hub import OptiHub
import torch

tool = OptiHub()
model = torch.nn.Linear(128, 128)

optimizer = tool.get_optimizer("Muon", model.parameters(), lr=1e-3, weight_decay=0.01)
```

## Registry
The `registry.toml` file is the source of truth. It declares:
- the package or repo to install
- the import module path
- the optimizer class name

Feel free to add new entries and submit improvements.

## Notes
- Installation is done through `pip` in the active Python environment.
- Some optimizers may require specific hyperparameters; consult their upstream docs.

