import os
import sys

from .target_axial_lora import JointLoraModel
#from .target_axial_1d_ft_sub import JointModel
from .target_axial_1d_ft import JointModel
from .target_mlp_lora import IndependentLoraModel
from .target_mlp import IndependentModel
from .aggregator import Aggregator
from .baseline import CausalBaseline


def get_model_cls(args):
    """
        Get class only
    """
    if args.model == "baseline":
        cls = CausalBaseline
    elif args.model == "aggregator":
        cls = Aggregator
    elif args.model == "independent":
        cls = IndependentModel
    elif args.model == "joint":
        cls = JointModel
    elif args.model == "lora":
        cls = JointLoraModel
    elif args.model == "mlora":
        cls = IndependentLoraModel
    else:
        raise Exception(f"Invalid model {args.model}")
    return cls


def load_model(args, data_module=None, **kwargs):
    """
        Model factory
    """
    if args.model == "baseline":
        model = CausalBaseline(args, **kwargs)
    elif args.model == "aggregator":
        model = Aggregator(args, **kwargs)
    elif args.model == "independent":
        model = IndependentModel(args, **kwargs)
    elif args.model == "joint":
        model = JointModel(args, **kwargs)
    elif args.model == "lora":
        model = JointLoraModel(args, **kwargs)
    elif args.model == "mlora":
        model = IndependentLoraModel(args, **kwargs)
    else:
        raise Exception(f"Invalid model {args.model}")

    return model

