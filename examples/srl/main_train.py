import logging
import yaml

from texar.torch.hyperparams import HParams
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.processors.srl_predictor import SRLPredictor, SRLEvaluator
from forte.train_pipeline import TrainPipeline
from forte.trainer.srl_trainer import SRLTrainer

from srl_vocab_processor import SRLVocabularyProcessor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s")


def main():
    config_data = yaml.safe_load(open("config_data.yml", "r"))
    config_model = yaml.safe_load(open("config_model.yml", "r"))
    config_preprocess = yaml.safe_load(open("config_preprocessor.yml", "r"))

    config = HParams({}, default_hparams=None)
    config.add_hparam('config_data', config_data)
    config.add_hparam('config_model', config_model)
    config.add_hparam('preprocessor', config_preprocess)
    config.add_hparam('storage_path', ".")

    reader = OntonotesReader(missing_fields=[
        "document_id", "part_number", "pos_tag", "framenet_id",
        "word_sense", "speaker", "entity_label"])

    # Keep the vocabulary processor as a simple counter
    vocab_processor = SRLVocabularyProcessor()

    ner_trainer = SRLTrainer()
    ner_predictor = SRLPredictor()
    ner_evaluator = SRLEvaluator()

    train_pipe = TrainPipeline(train_reader=reader, trainer=ner_trainer,
                               dev_reader=reader, configs=config,
                               preprocessors=[vocab_processor],
                               predictor=ner_predictor, evaluator=ner_evaluator)
    train_pipe.run()


if __name__ == '__main__':
    main()
