"""T5 model congfiguration"""

from transformers.models.mt5.configuration_mt5 import MT5Config


class MT5Config(MT5Config):
    def __init__(self, train_adapters=False, **kwargs):
        super().__init__(**kwargs)
        self.train_adapters = train_adapters
