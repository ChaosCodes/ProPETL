"""Implements the adapters' configurations."""

from collections import OrderedDict

import torch.nn as nn
from dataclasses import dataclass


@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""
    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = False
    non_linearity: str = "swish"
    reduction_factor: float = 16.0
    weight_init_range = 1e-2
    sparsity = 1.0
    share_adapter = False
    share_encoder_decoder_single_adapter = True
    adapter_config_name = "Houlsby"
    mask_extreme_mode = False
    mask_extreme_mode_combine_method = "or"
    use_multilingual = False
    mt_mode = False
    # Whether to use conditional layer norms for adapters.
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.




ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig),
     ])


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}"
                .format(config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())))
