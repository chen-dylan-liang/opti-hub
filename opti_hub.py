import sys
import subprocess
import importlib

# Handle TOML parsing depending on the Python version
try:
    import tomllib  # Built-in for Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for Python < 3.11
    except ImportError:
        print("❌ 'tomli' is required for Python < 3.11. Please run: pip install tomli")
        sys.exit(1)
        
class OptiHub:
    """The core tool for managing and installing SOTA optimizers."""
    
    def __init__(self, registry_path="registry.toml"):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _load_registry(self):
        """Loads the optimizer registry from the TOML file."""
        try:
            with open(self.registry_path, "rb") as f:
                data = tomllib.load(f)
                return data.get("optimizers", {})
        except FileNotFoundError:
            print(f"❌ Registry file '{self.registry_path}' not found.")
            sys.exit(1)

    def install(self, *optimizer_names):
        """Installs the requested optimizers using pip."""
        packages_to_install = []
        
        for name in optimizer_names:
            if name not in self.registry:
                print(f"⚠️ Unknown optimizer: '{name}'. Supported: {list(self.registry.keys())}")
                continue
                
            opt_info = self.registry[name]
            if opt_info.get("installable", True) is False:
                print(f"⚠️ Skipping '{name}': not pip-installable yet (see registry.toml).")
                continue
                
            source = opt_info["source"]
            packages_to_install.append(source)
            print(f"⏳ Queued '{name}' for installation from: {source}")

        if not packages_to_install:
            print("No valid optimizers found to install.")
            return

        print("\n🚀 Starting pip install process...")
        # Use sys.executable to ensure we install into the current active Python environment
        cmd = [sys.executable, "-m", "pip", "install"] + packages_to_install
        
        try:
            # check_call will raise an exception if the pip command fails
            subprocess.check_call(cmd)
            print(f"\n✅ Successfully installed: {', '.join(optimizer_names)}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Installation failed with error: {e}\n")
            
    def get_optimizer(self, name, params, **kwargs):
        """
        Dynamically loads and instantiates the requested optimizer.
        
        Args:
            name (str): The name of the optimizer in the registry (e.g., 'Muon').
            params (iterable): Model parameters to optimize.
            **kwargs: Hyperparameters like lr, weight_decay, etc.
        """
        if name not in self.registry:
            raise ValueError(f"⚠️ Unknown optimizer '{name}'. Supported: {list(self.registry.keys())}")
        
        if name == "Muon":
            # Special handling based on the official Muon usage pattern.
            # Prefer explicit param_groups with `use_muon` flags, otherwise build a heuristic split.
            import torch

            param_groups = kwargs.pop("param_groups", None)
            if param_groups is None and isinstance(params, list) and params:
                if isinstance(params[0], dict) and "use_muon" in params[0]:
                    param_groups = params

            if param_groups is None:
                params_list = list(params)
                muon_params = []
                adam_params = []

                if params_list and isinstance(params_list[0], tuple) and len(params_list[0]) == 2:
                    # named_parameters() style: (name, param)
                    for param_name, p in params_list:
                        if p.ndim >= 2 and "embed" not in param_name and "head" not in param_name:
                            muon_params.append(p)
                        else:
                            adam_params.append(p)
                else:
                    # Best-effort split based on dimensionality
                    muon_params = [p for p in params_list if p.ndim >= 2]
                    adam_params = [p for p in params_list if p.ndim < 2]
                    print(
                        "⚠️ Muon: using a heuristic split (ndim>=2 vs <2). "
                        "For best results, pass explicit param_groups with `use_muon`, "
                        "or use named_parameters() to exclude embeddings/heads."
                    )

                if not muon_params:
                    raise ValueError(
                        "❌ Muon requires >=2D parameters for the Muon group. "
                        "Pass explicit param_groups with `use_muon` for full control."
                    )

                muon_group = dict(
                    params=muon_params,
                    use_muon=True,
                    lr=kwargs.pop("lr", 0.02),
                    weight_decay=kwargs.pop("weight_decay", 0),
                    momentum=kwargs.pop("momentum", 0.95),
                )
                adam_group = dict(
                    params=adam_params,
                    use_muon=False,
                    lr=kwargs.pop("adam_lr", 3e-4),
                    betas=kwargs.pop("adam_betas", (0.9, 0.95)),
                    weight_decay=kwargs.pop("adam_weight_decay", 0),
                    eps=kwargs.pop("adam_eps", 1e-10),
                )
                param_groups = [muon_group, adam_group]

            if kwargs:
                raise ValueError(f"❌ Unused keyword arguments for Muon: {sorted(kwargs.keys())}")

            module = importlib.import_module("muon")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                optimizer_class = getattr(module, "MuonWithAuxAdam")
            else:
                optimizer_class = getattr(module, "SingleDeviceMuonWithAuxAdam")

            print("✅ Successfully loaded Muon (with aux Adam groups)")
            return optimizer_class(param_groups)

        opt_info = self.registry[name]
        module_path = opt_info.get("module_path")
        class_name = opt_info.get("class_name")
        
        if not module_path or not class_name:
            raise ValueError(f"❌ Registry entry for '{name}' is missing 'module_path' or 'class_name'.")
        
        try:
            # 1. Dynamically import the module (equivalent to: import muon)
            module = importlib.import_module(module_path)
            
            # 2. Get the specific class from the module (equivalent to: module.Muon)
            optimizer_class = getattr(module, class_name)
            
        except ImportError:
            raise ImportError(f"❌ Failed to import '{name}'. Did you run `tool.install('{name}')` first?")
        except AttributeError:
            raise AttributeError(f"❌ Class '{class_name}' not found inside module '{module_path}'.")
            
        # 3. Instantiate the class with the PyTorch parameters and hyperparameters
        print(f"✅ Successfully loaded {name} (lr={kwargs.get('lr', 'default')})")
        return optimizer_class(params, **kwargs)        
            

# ==========================================
# Test all links in the registry
# ==========================================
def test_install_all():
    """
    Tests the registry by attempting to install every listed optimizer.
    Useful for CI/CD or verifying that no GitHub repos have gone offline.
    """
    print("=== RUNNING REGISTRY TEST: INSTALLING ALL OPTIMIZERS ===")
    tool = OptiHub()
    all_optimizers = list(tool.registry.keys())
    
    print(f"Found {len(all_optimizers)} optimizers in registry: {all_optimizers}")
    print("Warning: This might take a few minutes depending on your network and dependencies.\n")
    
    # Unpack the list to pass as multiple arguments
    tool.install(*all_optimizers)

# ==========================================
# Test instantiating all optimizers
# ==========================================
def test_get_all():
    """
    Tests the dynamic import and instantiation logic for all installed optimizers.
    """
    print("\n=== RUNNING REGISTRY TEST: INSTANTIATING ALL OPTIMIZERS ===")
    import torch 
    tool = OptiHub()
    
    # Create a dummy neural network layer to generate PyTorch parameters
    dummy_model = torch.nn.Linear(128, 128)
    
    success_count = 0
    failed = []

    for name in tool.registry.keys():
        print(f"\nAttempting to load: {name}...")
        try:
            # Note: We pass standard kwargs. Some optimizers might require specific ones.
            opt = tool.get_optimizer(name, dummy_model.parameters(), lr=1e-3)
            print(f"   -> Success! Instantiated: {type(opt)}")
            success_count += 1
        except ImportError as e:
            print(f"   -> Skipped: {e}")
            failed.append((name, "Not Installed"))
        except Exception as e:
            print(f"   -> Failed: {e}")
            failed.append((name, str(e)))
            
    print("-" * 50)
    print(f"Test Summary: {success_count}/{len(tool.registry)} successfully instantiated.")
    if failed:
        print("Issues encountered (Install missing packages or check specific kwargs):")
        for f in failed:
            print(f"  - {f[0]}: {f[1]}")
                
if __name__ == "__main__":
    test_install_all()
    test_get_all()               
