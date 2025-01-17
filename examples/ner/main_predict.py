import yaml

from texar.torch import HParams

from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.processors.ner_predictor import CoNLLNERPredictor
from ft.onto.base_ontology import Token, Sentence, EntityMention

config_data = yaml.safe_load(open("config_data.yml", "r"))
config_model = yaml.safe_load(open("config_model.yml", "r"))

config = HParams({}, default_hparams=None)
config.add_hparam('config_data', config_data)
config.add_hparam('config_model', config_model)


pl = Pipeline()
pl.set_reader(CoNLL03Reader())
pl.add_processor(CoNLLNERPredictor(), config=config)

pl.initialize()

for pack in pl.process_dataset(config.config_data.test_path):
    for pred_sentence in pack.get_data(
            context_type=Sentence,
            request={
                Token: {"fields": ["ner"]},
                Sentence: [],  # span by default
                EntityMention: {}
            }):
        print("============================")
        print(pred_sentence["context"])
        print(pred_sentence["Token"]["ner"])
        print("============================")
