import logging
from typing import Dict

import yaml
from texar.torch.hyperparams import HParams

from forte.base_pipeline import BasePipeline
from forte.data import DataPack
from forte.utils import get_class

logger = logging.getLogger(__name__)

__all__ = [
    "Pipeline"
]


class Pipeline(BasePipeline[DataPack]):
    """
        The main pipeline class for processing DataPack.
    """

    def init_from_config(self, configs: Dict):
        """
        Initialize the pipeline with the configurations

        Args:
            configs: The configurations used to create the pipeline.

        Returns:

        """
        # HParams cannot create HParams from the inner dict of list

        if "Processors" in configs and configs["Processors"] is not None:
            for processor_configs in configs["Processors"]:

                p_class = get_class(processor_configs["type"])
                if processor_configs.get("kwargs"):
                    processor_kwargs = processor_configs["kwargs"]
                else:
                    processor_kwargs = {}
                p = p_class(**processor_kwargs)

                hparams: Dict = {}

                if processor_configs.get("hparams"):
                    # Extract the hparams section and build hparams
                    processor_hparams = processor_configs["hparams"]

                    if processor_hparams.get("config_path"):
                        filebased_hparams = yaml.safe_load(
                            open(processor_hparams["config_path"]))
                    else:
                        filebased_hparams = {}
                    hparams.update(filebased_hparams)

                    if processor_hparams.get("overwrite_configs"):
                        overwrite_hparams = processor_hparams[
                            "overwrite_configs"]
                    else:
                        overwrite_hparams = {}
                    hparams.update(overwrite_hparams)
                default_processor_hparams = p_class.default_hparams()

                processor_hparams = HParams(hparams,
                                            default_processor_hparams)
                self.add_processor(p, processor_hparams)

            self.initialize()
