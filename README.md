# Londinium

### 0. Setup environment and prepare data
#### a. install required packages
```bash
pip install -r requirements.txt
```
#### b. install spacy
```bash
pip install -U spacy
python -m spacy download en_core_web_sm
```


#### b. prepare data
- Fetch LAMA and SNLI data
    ```bash
    bash fetch_data.sh
    ```

### 1. Conduct Experiment 1
This experiment evaluate whether pre-trained model generate antonym for synonym questions and vice versa
1) build dataset for experiment
- set update_thesarsus to True only at the first time (for time efficiency)
    ```bash
    cd data_builder
    python antonym_synonym.py --update_thesarsus True
    ```
2) Conduct experiment
-  supported model type: bert-base, bert-large, electra-base, eletra-large, roberta-base, roberta-large, albert-base, albert-large
    ```bash
    python experiment.py --model_type [model_type]
    ```