# coding=utf-8

from __future__ import absolute_import, division, print_function

import datasets
import os
import pandas as pd

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """\
KoBEST BoolQ
"""
_HOMEPAGE = "NONE"
_LICENSE = "NONE"

# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {'boolq': "NONE"}


# KoBEST data repository to load the data
DIR_PATH = os.getcwd()
# DATA_PATH = os.path.join(DIR_PATH, 'KoBEST_JjinFianl')
DATA_PATH = os.path.join(DIR_PATH, '../../../../KoBEST_JjinFianl')


def load_data():
    outp = dict()
    for data_type_ in ['train', 'validation']:
        if data_type_ == 'train':
            df = pd.read_csv(os.path.join(DATA_PATH, 'BoolQ_Train.tsv'), sep='\t', encoding='utf-8')
        else:
            df = pd.read_csv(os.path.join(DATA_PATH, 'BoolQ_Dev.tsv'), sep='\t', encoding='utf-8')
        df = df.rename(columns={
            'ID': 'idx',
            'Answer(FALSE = 0, TRUE = 1)': 'label',
            'Question': 'question',
            'Text': 'text'
        })

        df = datasets.Dataset.from_pandas(df)
        df.features['label'] = datasets.ClassLabel(num_classes=2, names=['False', 'True'])
        outp[data_type_] = df
    return outp


class kobest_boolq(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.5.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="kobest_boolq",
                               version=VERSION,
                               description="kobest_boolq")
    ]

    DEFAULT_CONFIG_NAME = "kobest_boolq"

    def __init__(self, **config):

        self.data = load_data()
        super(kobest_boolq, self).__init__(**config)

    def _info(self):
        features = datasets.Features({
            "input_text":
            datasets.Value("string"),
            # "text":
            # datasets.Value("string"),
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
        # question = [f'질문: {s}' for s in dataset['question']]
        # text = [f'예시문: {s}' for s in dataset['text']]
        labels = dataset['label']

        input_text = [f"예시문: {s1} 질문: {s2}".strip() for s1, s2 in
                      zip(dataset['text'], dataset['question'])]

        data_dict = {'idx': idx, 'input_text': input_text, 'label': labels}
        data = pd.DataFrame.from_dict(data_dict)
        print(f"Data point example: {data.iloc[1]}")

        for docid_, datum in data.iterrows():
            yield docid_, {
                    "input_text": datum["input_text"],
                    # "text": datum["text"],
                    "label": int(datum['label']),
                }


if __name__ == '__main__':
    pass