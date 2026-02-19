# Opti Hub
Opti Hub is a lightweight registry and installer for state‑of‑the‑art optimizers used in modern ML and LLM research.

## Why This Exists
Since the advent of Muon, we have seen a resurgence in the utilization/creation of new optimizers for machine learning, especially for large language models. Many of these methods appear first in papers or GitHub repositories and are not yet integrated into mainstream frameworks like PyTorch.

Opti Hub is built to help researchers:
- quickly install and try new or SOTA optimizers
- benchmark optimizers that are not yet in the mainline ML stacks
- use `registry.toml` as a learning map of what is available and where it lives

## How It Works
Opti Hub reads `registry.toml` to discover optimizer packages, install sources, and import paths. The file is intended to be both a registry and a learning map you can expand.
This registry is actively updated. If you have a newly developed optimizer you would like added, please open an issue.

## Install

Install directly from GitHub:

```bash
pip install git+https://github.com/chen-dylan-liang/opti-hub.git
```

Install from a specific branch:

```bash
pip install "git+https://github.com/chen-dylan-liang/opti-hub.git@main"
```

Install locally for development:

```bash
pip install -e .
```

Dependency note:
- Python 3.11+: uses built-in `tomllib`
- Python < 3.11: installs `tomli` automatically via package dependency

## Example Usage
Basic installation of optimizers from the registry:

```python
from opti_hub import OptiHub

tool = OptiHub()
tool.install("Muon", "Shampoo")
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
- the paper or blog where it lives
- the import module path
- the optimizer class name

Feel free to add new entries and submit improvements.

## Notes
- Installation is done through `pip` in the active Python environment.
- Some optimizers may require specific hyperparameters; consult their upstream docs.
