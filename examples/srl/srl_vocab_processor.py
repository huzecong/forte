import logging
import tempfile
from collections import Counter
from typing import Counter as CounterType

import texar.torch as tx

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology import ontonotes_ontology as Ont
from forte.processors.vocabulary_processor import VocabularyProcessor

logger = logging.getLogger(__name__)


def _create_vocab(counter: CounterType[str]) -> tx.data.Vocab:
    # A stupid way to programmatically create tx.data.Vocab.
    # This has to remain in-place until Texar adds a way to create vocabularies
    # given a list of words.
    with tempfile.NamedTemporaryFile("w") as f:
        f.write("\n".join(counter))
        vocab = tx.data.Vocab(f.name)
    return vocab


class SRLVocabularyProcessor(VocabularyProcessor):
    r"""Vocabulary Processor for the SRL datasets (presumably OntoNotes).
    Create vocabularies for words, characters, and SRL tags.

    Required configs:

    - ``min_frequency``: Cut-off word frequency. All words with lower frequency
      will be treated as unknown words.
    """

    min_frequency: int

    def __init__(self) -> None:
        super().__init__()
        self.word_cnt: CounterType[str] = Counter()
        self.char_cnt: CounterType[str] = Counter()
        self.label_cnt: CounterType[str] = Counter()

    def initialize(self, resource: Resources, configs: tx.HParams):
        self.min_frequency = configs.min_frequency

    def _process(self, data_pack: DataPack):
        r"""Process the data pack to collect vocabulary information.
        """
        logger.info("vocab processor: _process")
        for token in data_pack.get(Ont.Token):
            self.word_cnt[token.text] += 1
            self.char_cnt.update(ch for ch in token.text)
        for link in data_pack.get(Ont.PredicateLink):
            self.label_cnt[link.arg_type] += 1

    def finish(self, resource: Resources):
        logger.info("vocab processor: finish")
        for word, freq in list(self.word_cnt.items()):
            if freq < self.min_frequency:
                del self.word_cnt[word]

        word_vocab = _create_vocab(self.word_cnt)
        char_vocab = _create_vocab(self.char_cnt)
        label_vocab = list(self.label_cnt)

        # Adding vocabulary information to resource.
        resource.update(
            word_vocab=word_vocab,
            char_vocab=char_vocab,
            label_vocab=label_vocab)
