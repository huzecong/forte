import logging
from typing import Dict, List, Optional

import yaml
from texar.torch import HParams

from forte.base_pipeline import BasePipeline
from forte.data.multi_pack import MultiPack
from forte.data.selector import Selector
from forte.utils import get_class

logger = logging.getLogger(__name__)

__all__ = [
    "MultiPackPipeline"
]


# pylint: disable=attribute-defined-outside-init

class MultiPackPipeline(BasePipeline[MultiPack]):
    """
    The pipeline consists of a list of predictors.
    """

    def __init__(self):
        super().__init__()
        self._selectors: List[Selector] = []

    @property
    def selectors(self):
        return self._selectors

    def init_from_config(self, configs: Dict):
        """
        Parse the configuration sections from the input config,
            into a list of [processor, config]
        Initialize the pipeline with the configurations
        """
        if "Reader" not in configs or configs["Reader"] is None:
            raise KeyError('No reader in the configuration')

        reader_config = configs["Reader"]

        reader, _ = create_class_with_kwargs(
            class_name=reader_config["type"],
            class_args=reader_config.get("kwargs", {}),
        )

        self.set_reader(reader)

        # HParams cannot create HParams from the inner dict of list
        if "Processors" in configs and configs["Processors"] is not None:
            for processor_configs in configs["Processors"]:
                p, processor_hparams = create_class_with_kwargs(
                    class_name=processor_configs["type"],
                    class_args=processor_configs.get("kwargs", {}),
                    h_params=processor_configs.get("hparams", {}),
                )

                selector_hparams = processor_hparams.selector
                selector_class = get_class(selector_hparams['type'])
                selector_kwargs = selector_hparams["kwargs"]

                selector = selector_class(**selector_kwargs)

                self.add_processor(p, processor_hparams, selector)

            self.initialize()


def create_class_with_kwargs(
        class_name: str, class_args: Dict, h_params: Optional[Dict] = None):
    cls = get_class(class_name)
    if not class_args:
        class_args = {}
    obj = cls(**class_args)

    p_params: Dict = {}

    if h_params is not None and \
            "config_path" in h_params and \
            h_params["config_path"] is not None:
        filebased_hparams = yaml.safe_load(open(h_params["config_path"]))
    else:
        filebased_hparams = {}
    p_params.update(filebased_hparams)

    if h_params is not None:
        p_params.update(h_params.get("overwrite_configs", {}))
    default_processor_hparams = cls.default_hparams()

    processor_hparams = HParams(p_params,
                                default_processor_hparams)

    return obj, processor_hparams
