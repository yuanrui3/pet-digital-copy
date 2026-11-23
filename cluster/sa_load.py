"""Load the SuperAnimal-Quadruped checkpoint safely (weights only): its saved training settings use ruamel.yaml
ScalarFloat numbers; a plain float stand-in is allow-listed for them."""
import sys
import types

import torch


def load(path):
    mod = types.ModuleType("ruamel.yaml.scalarfloat")

    class ScalarFloat(float):
        pass
    ScalarFloat.__module__ = "ruamel.yaml.scalarfloat"
    mod.ScalarFloat = ScalarFloat
    for name in ("ruamel", "ruamel.yaml"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["ruamel.yaml.scalarfloat"] = mod
    torch.serialization.add_safe_globals([ScalarFloat])
    return torch.load(path, map_location="cpu", weights_only=True)
