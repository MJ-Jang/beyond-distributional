# Londinium

### 0. Setup environment and prepare data
#### a. install required packages
```bash
pip install -r requirements.txt
```

#### b. prepare data
- Fetch LAMA and SNLI data
    ```bash
    bash fetch_data.sh
    ```
- Build dataset for experiments
    - antonym_synonym.py: build dataset for experiment 1
    - negate_lama.py: build dataset for experiment 2
    ```bash
    cd data_builder
    python antonym_synonym.py
    python negate_lama.py
    ```

### 1. Conduct Experiment 1
- This experiment evaluate whether pre-trained model generate antonym for synonym questions and vice versa
    - supported model type: bert-base, bert-large, electra-base, eletra-large, roberta-base, roberta-large, albert-base, albert-large
    ```bash
    python experiment.py --model_type [model_type]
    ```