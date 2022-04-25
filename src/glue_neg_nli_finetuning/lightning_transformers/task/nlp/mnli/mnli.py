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
_URLs = {'mnli': "NONE"}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class mnli(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="mnli",
                               version=VERSION,
                               description="mnli")
    ]

    DEFAULT_CONFIG_NAME = "mnli"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def __init__(self, **config):
        self.data = datasets.load_dataset('glue', 'mnli')
        super(mnli, self).__init__(**config)

    def _info(self):
        features = datasets.Features({
            "premise":
            datasets.Value("string"),
            "hypothesis":
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
        validation_matched = self.data['validation_matched']
        validation_mismatched = self.data['validation_mismatched']

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "dataset": training,
                    "dataset_2": None
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "dev",
                    "dataset": validation_matched,
                    # "dataset_2": validation_mismatched
                },
            ),
        ]

    def _generate_examples(self, split, dataset, dataset_2=None):
        import pandas as pd
        idx = dataset['idx']
        premise = dataset['premise']
        hypothesis = dataset['hypothesis']
        labels = dataset['label']

        data_dict = {'idx': idx, 'premise': premise, 'hypothesis': hypothesis, 'label': labels}
        data = pd.DataFrame.from_dict(data_dict)
        print(f"Data point example: {data.iloc[1]}")

        if dataset_2 is not None:
            idx += [len(idx) + i for i in dataset_2['idx']]
            premise += dataset_2['premise']
            hypothesis += dataset_2['hypothesis']
            labels += dataset_2['label']

        for docid_, datum in data.iterrows():
            yield docid_, {
                    "premise": datum["premise"],
                    "hypothesis": datum["hypothesis"],
                    "label": int(datum['label']),
                }


if __name__ == '__main__':
    pass