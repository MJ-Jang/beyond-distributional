### Meaning-Matching data
- This data is constructed to conduct meaning-matching task

#### 1. Raw data
- Raw data is listed in *meaning_matching_all.jsonl* file.
- File format:
```json
{"idx": 573, "word": "adorable", "definition": ["lovable especially in a childlike or naive way", "charming and easy to love because it is so attractively cute"]}
{"idx": 574, "word": "adored", "definition": ["love intensely"]}
```

#### 2. Build data for training
- run the following code to build train/dev dataset for Meaning-Matching task
```bash
python build_train_dev.py --n_neg_sample 5
```
- Options:
    - n_neg_sample: number of negative samples
- The label is 1 if a word-definition pair is correct and 0 otherwise

#### 3. Data Source
- WordNet
- English word, meaning, and usage dataset [link](https://data.world/idrismunir/english-word-meaning-and-usage-examples/workspace/file?filename=word-meaning-examples.csv)