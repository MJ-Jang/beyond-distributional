# coding=utf-8
from __future__ import absolute_import, division, print_function

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """\
everyones corpus : spoken
"""
_HOMEPAGE = "NONE"
_LICENSE = "NONE"

# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {'sst2': "NONE"}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class sst(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="sst2",
                               version=VERSION,
                               description="sst2")
    ]

    DEFAULT_CONFIG_NAME = "sst2"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def __init__(self, **config):
        self.data = datasets.load_dataset('glue', 'sst2')
        super(sst, self).__init__(**config)

    def _info(self):
        features = datasets.Features({
            "sentence":
            datasets.Value("string"),
            "label":
            datasets.Value("int32")
            # These are the features of your dataset like images, labels ...
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        training = self.data['train']
        validation = self.data['validation']

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "dataset": training,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "dev",
                    "dataset": validation,
                },
            ),
        ]

    def _generate_examples(self, split, dataset):
        import pandas as pd
        idx = dataset['idx']
        sentence = dataset['sentence']
        labels = dataset['label']

        data_dict = {'idx': idx, 'sentence': sentence, 'label': labels}
        data = pd.DataFrame.from_dict(data_dict)
        print(f"Data point example: {data.iloc[1]}")

        for docid_, datum in data.iterrows():
            yield docid_, {
                    "sentence": datum["sentence"],
                    "label": int(datum['label']),
                }


if __name__ == '__main__':
    pass