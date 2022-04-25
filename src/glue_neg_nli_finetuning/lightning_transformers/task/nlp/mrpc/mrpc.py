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
_URLs = {'mrpc': "NONE"}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class mrpc(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="mrpc",
                               version=VERSION,
                               description="mrpc")
    ]

    DEFAULT_CONFIG_NAME = "mrpc"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def __init__(self, **config):
        self.data = datasets.load_dataset('glue', 'mrpc')
        super(mrpc, self).__init__(**config)

    def _info(self):
        features = datasets.Features({
            "sentence1":
            datasets.Value("string"),
            "sentence2":
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
        sent1 = dataset['sentence1']
        sent2 = dataset['sentence2']
        labels = dataset['label']

        data_dict = {'idx': idx, 'sentence1': sent1, 'sentence2': sent2, 'label': labels}
        data = pd.DataFrame.from_dict(data_dict)
        print(f"Data point example: {data.iloc[1]}")

        for docid_, datum in data.iterrows():
            yield docid_, {
                    "sentence1": datum["sentence1"],
                    "sentence2": datum["sentence2"],
                    "label": int(datum['label']),
                }


if __name__ == '__main__':
    pass