# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple

from lightning_transformers.core.nlp.seq2seq import Seq2SeqDataModule
from lightning_transformers.task.nlp.translation.config import TranslationDataConfig


class TranslationDataModule(Seq2SeqDataModule):
    """
    Defines the ``LightningDataModule`` for Translation Datasets.

    Args:
        *args: ``Seq2SeqDataModule`` specific arguments.
        cfg: Contains data specific parameters when processing/loading the dataset
            (Default ``TranslationDataConfig``)
        **kwargs: ``Seq2SeqDataModule`` specific arguments.
    """
    cfg: TranslationDataConfig

    def __init__(self, *args, cfg: TranslationDataConfig = TranslationDataConfig(), **kwargs) -> None:
        super().__init__(*args, cfg=cfg, **kwargs)

    @property
    def source_target_column_names(self) -> Tuple[str, str]:
        return self.cfg.source_language, self.cfg.target_language
