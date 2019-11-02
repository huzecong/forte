import logging
import os
from typing import Dict, Iterator, List, Optional

import texar.torch as tx
import torch
from torch import nn

from forte import BaseTrainer, Resources
from forte.data.ontology import ontonotes_ontology as Ont
from forte.models.srl.data import SRLExample, SRLSpan
from forte.models.srl.model import LabeledSpanGraphNetwork

logger = logging.getLogger(__name__)


class TensorSizeBatchingStrategy(tx.data.BatchingStrategy):
    _max_size_in_batch: int
    _batch_size: int

    def __init__(self, max_size: int):
        self.max_size = max_size

    def reset_batch(self) -> None:
        self._max_size_in_batch = 0
        self._batch_size = 0

    def add_example(self, example: SRLExample) -> bool:
        new_max = max(self._max_size_in_batch, len(example['text']))
        if new_max * (self._batch_size + 1) <= self.max_size:
            self._max_size_in_batch = new_max
            self._batch_size += 1
            return True
        return False


class SRLSpanData(tx.data.DatasetBase[Dict, SRLExample]):
    def __init__(self, data: Iterator[Dict], word_vocab: tx.data.Vocab,
                 hparams=None):
        source = tx.data.IterDataSource(data)
        self.word_vocab = word_vocab
        super().__init__(source, hparams)

    def process(self, instance: Dict) -> SRLExample:
        words: List[str] = instance["Token"]["text"]
        word_ids = self.word_vocab.map_tokens_to_ids_py(words)

        arg_types = instance["PredicateLink"]["arg_type"]
        parent_ids = instance["PredicateLink"]["parent"]
        child_ids = instance["PredicateLink"]["child"]
        pred_ids = instance["PredicateMention"]["unit_span"][parent_ids][:, 0]
        span_start = instance["PredicateArgument"]["unit_span"][child_ids][:, 0]
        span_end = instance["PredicateArgument"]["unit_span"][child_ids][:, 1]
        srl_spans = [
            SRLSpan(predicate, start, end, label)
            for predicate, start, end, label in zip(
                pred_ids, span_start, span_end, arg_types)]
        return {
            "text": words,
            "text_ids": word_ids,
            "srl": srl_spans,
        }

    def collate(self, examples: List[SRLExample]) -> tx.data.Batch:
        sentences = [ex['text'] for ex in examples]
        tokens, length = tx.data.padded_batch(
            [ex['text_ids'] for ex in examples])
        srl = [ex['srl'] for ex in examples]
        return tx.data.Batch(
            len(examples), srl=srl, text=sentences,
            text_ids=torch.from_numpy(tokens),
            length=torch.tensor(length))


class SRLTrainer(BaseTrainer):
    word_vocab: tx.data.Vocab
    char_vocab: tx.data.Vocab
    labels: Optional[List[str]]
    model: LabeledSpanGraphNetwork

    config_model: tx.HParams
    config_data: tx.HParams
    storage_path: str
    device: torch.device

    optim: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    train_data_cache: List[SRLExample]
    best_dev_result: Optional[float] = None

    def initialize(self, resource: Resources, configs: tx.HParams):
        self.word_vocab = resource.get("word_vocab")
        self.char_vocab = resource.get("char_vocab")
        self.labels = resource.get("labels", None)

        self.config_model = configs.config_model
        self.config_data = configs.config_data
        self.storage_path = configs.storage_path

        model_hparams = tx.HParams(self.config_model.srl_model,
                                   LabeledSpanGraphNetwork.default_hparams())
        if self.labels is not None:
            model_hparams["srl_labels"] = self.labels
        word_embed = resource.get("word_embed", None)
        head_embed = resource.get("head_embed", None)
        self.model = LabeledSpanGraphNetwork(
            self.word_vocab, self.char_vocab,
            word_embed, head_embed, model_hparams)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=float(self.config_model.initial_lr))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optim, self.config_model.lr_decay)

        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.model = self.model.to(self.device)
        if self.config_model.random_seed is not None:
            tx.run.make_deterministic(self.config_model.random_seed)
        self.train_data_cache = []

    def data_request(self):
        return {
            "context_type": Ont.Sentence,
            "request": {
                Ont.Token: [],
                Ont.PredicateMention: {"unit": "Token"},
                Ont.PredicateArgument: {"unit": "Token"},
                Ont.PredicateLink: {"fields": ["arg_type"]},
            }
        }

    def consume(self, instance: Dict):
        # Convert data from the pipeline format to some internal representation.
        # Note that we don't do any training here, but instead just save all
        # the data in the epoch, and do whatever we want by the end.
        self.train_data_cache.append(instance)

    def epoch_finish_action(self, epoch_num: int):
        # Here's where we actually do the training.
        # It's a weird design.
        data_size = len(self.train_data_cache)
        dataset = SRLSpanData(
            self.train_data_cache, self.word_vocab, self.config_data.dataset)
        self.train_data_cache = []
        strategy = TensorSizeBatchingStrategy(
            self.config_data.max_tokens_per_batch)
        iterator = tx.data.DataIterator(dataset, strategy)

        self.model.train()
        avg_loss = tx.utils.AverageRecorder()
        n_examples = 0
        num_iters = 0
        for batch in iterator:
            return_val = self.model(batch)
            n_examples += len(batch)
            loss = return_val['loss']
            avg_loss.add(loss.item())
            del return_val
            loss.backward()
            if self.config_model.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config_model.max_grad_norm)
            self.optim.step()
            self.optim.zero_grad()
            num_iters += 1
            if num_iters % self.config_model.print_frequency == 0:
                # progress.set_postfix(loss=f"{avg_loss.avg():.4f}")
                n_digits = len(str(data_size))
                logger.info(
                    f"Epoch: {epoch_num}, step: {num_iters}, "
                    f"progress: {str(n_examples).rjust(n_digits)}/{data_size}, "
                    f"loss = {avg_loss.avg():.4f}")
                avg_loss.reset()
            if num_iters % self.config_model.lr_decay_frequency == 0:
                self.scheduler.step()

        if epoch_num >= self.config_data.num_epochs:
            self.request_stop_train()

    def _save(self):
        torch.save(self.model.state_dict(),
                   os.path.join(self.storage_path, "model.pt"))
        torch.save(self.optim.state_dict(),
                   os.path.join(self.storage_path, "optimizer.pt"))

    def _load(self):
        self.model.load_state_dict(torch.load(
            os.path.join(self.storage_path, "model.pt")))
        self.optim.load_state_dict(torch.load(
            os.path.join(self.storage_path, "optimizer.pt")))

    def post_validation_action(self, dev_res):
        dev_f1 = dev_res["eval"]["f1"]
        logger.info(f"Dev set F1: %.4f", dev_f1)
        if self.best_dev_result is None or (self.best_dev_result < dev_f1):
            self.best_dev_result = dev_f1
            self._save()
            logger.info("Model checkpoint saved")

    @torch.no_grad()
    def get_loss(self, instances: Iterator[Dict]):
        dataset = SRLSpanData(
            instances, self.word_vocab, self.config_data.dataset)
        iterator = tx.data.DataIterator(dataset)
        total_loss = 0.0
        data_size = 0
        for batch in iterator:
            dataset += len(batch)
            return_val = self.model(batch)
            total_loss += return_val["loss"].item()
        return total_loss / data_size

    def finish(self, resource: Resources):
        # Save resources
        resource.update(model=self.model.state_dict())

    def update_resource(self):
        pass
