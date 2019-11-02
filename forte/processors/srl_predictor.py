import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import texar.torch as tx
from texar.torch.hyperparams import HParams

from forte.common.evaluation import Evaluator
from forte.common.resources import Resources
from forte.common.types import DataRequest
from forte.data.base import Span
from forte.data.data_pack import DataPack
from forte.data.datasets.ontonotes import ontonotes_utils
from forte.data.ontology import ontonotes_ontology
from forte.data.ontology.ontonotes_ontology import (
    PredicateArgument, PredicateMention)
from forte.models.srl.model import LabeledSpanGraphNetwork
from forte.processors.base.batch_processor import FixedSizeBatchProcessor
from forte.utils import shut_up

logger = logging.getLogger(__name__)

__all__ = [
    "SRLPredictor",
]

Prediction = List[
    Tuple[Span, List[Tuple[Span, str]]]
]


class SRLPredictor(FixedSizeBatchProcessor):
    """
    An Semantic Role labeler trained according to `He, Luheng, et al.
    "Jointly predicting predicates and arguments in neural semantic role
    labeling." <https://aclweb.org/anthology/P18-2058>`_.

    Note that to use :class:`SRLPredictor`, the :attr:`ontology` of
    :class:`Pipeline` must be an ontology that includes
    ``forte.data.ontology.ontonotes_ontology``.
    """

    word_vocab: tx.data.Vocab
    char_vocab: tx.data.Vocab
    labels: Optional[List[str]]
    model: LabeledSpanGraphNetwork

    def __init__(self):
        super().__init__()

        self._ontology = ontonotes_ontology
        self.define_context()

        self.batch_size = 4
        self.batcher = self.define_batcher()

        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

    @classmethod
    def load_resources(cls, resource: Resources, configs: HParams,
                       load_weights: bool = False) \
            -> Tuple[LabeledSpanGraphNetwork, tx.data.Vocab, tx.data.Vocab]:
        r"""Try loading stuff from resources, or load them from disk and add
        them to resources. Stuff to load include:

        - Word and character vocabularies.
        - Model weights.
        - Word and head embeddings.
        """
        path = configs.storage_path
        word_vocab = resource.get("word_vocab", None)
        if word_vocab is None:
            word_vocab = tx.data.Vocab(
                os.path.join(path, "embeddings/word_vocab.english.txt"))
            resource.update(word_vocab=word_vocab)
        char_vocab = resource.get("char_vocab", None)
        if char_vocab is None:
            char_vocab = tx.data.Vocab(
                os.path.join(path, "embeddings/char_vocab.english.txt"))
            resource.update(char_vocab=char_vocab)
        labels = resource.get("labels", None)

        model_hparams = tx.HParams(configs.config_model.srl_model,
                                   LabeledSpanGraphNetwork.default_hparams())
        if labels is not None:
            model_hparams["srl_labels"] = labels
        word_embed = resource.get("word_embed", None)
        head_embed = resource.get("head_embed", None)
        model = LabeledSpanGraphNetwork(
            word_vocab, char_vocab,
            word_embed, head_embed, model_hparams)
        if word_embed is None:
            resource.update(word_embed=word_embed)
        if head_embed is None:
            resource.update(head_embed=head_embed)

        if load_weights:
            model_weights = resource.get("model", None)
            if model_weights is None:
                model_path = os.path.join(path, "model.pt")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(
                        f"Model weights not found at{model_path}")
                model_weights = torch.load(model_path)
            model.load_state_dict(model_weights)

        return model, word_vocab, char_vocab

    def initialize(self,
                   resource: Resources,  # pylint: disable=unused-argument
                   configs: HParams):

        model_dir = configs.storage_path
        logger.info("restoring SRL model from %s", model_dir)

        self.model, self.word_vocab, self.char_vocab = \
            self.load_resources(resource, configs)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def define_context(self):
        self.context_type = self._ontology.Sentence

    def _define_input_info(self) -> DataRequest:
        input_info: DataRequest = {
            self._ontology.Token: []
        }
        return input_info

    def predict(self, data_batch: Dict) -> Dict[str, List[Prediction]]:
        text: List[List[str]] = [
            sentence.tolist() for sentence in data_batch["Token"]["text"]]
        text_ids, length = tx.data.padded_batch([
            self.word_vocab.map_tokens_to_ids_py(sentence)
            for sentence in text])
        text_ids = torch.from_numpy(text_ids).to(device=self.device)
        length = torch.tensor(length, dtype=torch.long, device=self.device)
        batch_size = len(text)
        batch = tx.data.Batch(batch_size, text=text, text_ids=text_ids,
                              length=length, srl=[[]] * batch_size)
        batch_srl_spans = self.model.decode(batch)

        # Convert predictions into annotations.
        batch_predictions: List[Prediction] = []
        for idx, srl_spans in enumerate(batch_srl_spans):
            word_spans = data_batch["Token"]["span"][idx]
            predictions: Prediction = []
            for pred_idx, pred_args in srl_spans.items():
                begin, end = word_spans[pred_idx]
                # TODO cannot create annotation here.
                pred_span = Span(begin, end)
                arguments = []
                for arg in pred_args:
                    begin = word_spans[arg.start][0]
                    end = word_spans[arg.end][1]
                    arg_annotation = Span(begin, end)
                    arguments.append((arg_annotation, arg.label))
                predictions.append((pred_span, arguments))
            batch_predictions.append(predictions)
        return {"predictions": batch_predictions}

    def pack(self, data_pack: DataPack,
             inputs: Dict[str, List[Prediction]]) -> None:
        batch_predictions = inputs["predictions"]
        for predictions in batch_predictions:
            for pred_span, arg_result in predictions:

                pred = data_pack.add_entry(
                    PredicateMention(data_pack, pred_span.begin, pred_span.end)
                )

                for arg_span, label in arg_result:
                    arg = data_pack.add_or_get_entry(
                        PredicateArgument(
                            data_pack, arg_span.begin, arg_span.end
                        )
                    )
                    link = self._ontology.PredicateLink(data_pack, pred, arg)
                    link.set_fields(arg_type=label)
                    data_pack.add_or_get_entry(link)

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure
        :return:
        """
        hparams_dict = {
            'storage_path': None,
        }
        return hparams_dict


class SRLEvaluator(Evaluator):
    # TODO: `srl-eval.pl` is now stored in `examples/srl/`, so it only works in
    #   the example. Maybe we can move this to a dedicated folder, and not
    #   hard-code the path here.
    _SRL_EVAL_SCRIPT = "srl-eval.pl"

    def __init__(self, config: Optional[HParams] = None):
        super().__init__(config)
        self._ontology = ontonotes_ontology
        self.test_component = SRLPredictor().component_name
        self.scores: Dict[str, float] = {}

    def consume_next(self, pred_pack: DataPack, refer_pack: DataPack):
        pred_output_file = tempfile.NamedTemporaryFile("w", encoding="utf-8")
        gold_output_file = tempfile.NamedTemporaryFile("w", encoding="utf-8")

        ontonotes_utils.write_tokens_to_file(
            pred_pack, refer_pack, pred_output_file.name, gold_output_file.name)
        # Evaluate twice with official script.
        with shut_up(stdout=True):
            eval_info = subprocess.Popen(
                ["perl", self._SRL_EVAL_SCRIPT,
                 pred_output_file.name, gold_output_file.name],
                stdout=subprocess.PIPE).communicate()[0].decode('utf-8')
            eval_info2 = subprocess.Popen(
                ["perl", self._SRL_EVAL_SCRIPT,
                 gold_output_file.name, pred_output_file.name],
                stdout=subprocess.PIPE).communicate()[0].decode('utf-8')

        pred_output_file.close()
        gold_output_file.close()

        f1 = 0
        try:
            recall = float(eval_info.strip().split("\n")[6].split()[5])
            precision = float(eval_info2.strip().split("\n")[6].split()[5])
            if recall + precision > 0:
                f1 = (2 * recall * precision / (recall + precision))
        except IndexError:
            recall = 0
            precision = 0

        self.scores = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def get_result(self):
        return self.scores
