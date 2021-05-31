# Londinium

### 0. Setup environment and prepare data
#### a. install required packages
- Install all required packages
    ```bash
    pip install -r requirements.txt
    ```
#### b. install spacy
- Install spacy package
    ```bash
    pip install -U spacy
    python -m spacy download en_core_web_sm
    ```
#### c. prepare data
- Fetch LAMA and SNLI data
    ```bash
    bash fetch_data.sh
    ```

### 1. Conduct Experiment 1
This experiment evaluate whether pre-trained model generate antonym for synonym questions and vice versa
##### 1) build dataset for experiment
- set update_thesarsus to True only at the first time (for time efficiency)
    ```bash
    cd data_builder
    python antonym_synonym.py --update_thesarsus True
    ```
- If you already have *cn_thesaursus.json* file, you don't need to set --update_thesarsus to True
##### 2) Conduct experiment
-  supported model type: baseline, bert-base, bert-large, electra-base, eletra-large, roberta-base, roberta-large, albert-base, albert-large
    ```bash
    python experiment_1.py --model_type bert-base --top_k 10 --batch_size 16
    ```
- To conduct experiments for all supported model, run the following shell script.
    ```bash
    bash run_experiment.sh 1 [top_k] [batch_size]
    ```
##### 3) Check results
- The results will be saved in output/exp1 directory
    
### 2. Conduct Experiment 2
This experiment evaluate the discrepency on opposite meaning sentences predicted by  pre-trained model.
##### 1) build dataset for experiment
- negate LAMA data (Only ConceptNet data is used)
    ```bash
    cd data_builder
    python negate_lama.py
    ```
##### 2) Conduct experiment
-  supported model type: bert-base, bert-large, electra-base, eletra-large, roberta-base, roberta-large, albert-base, albert-large
    ```bash
    python experiment_2.py --model_type bert-base --top_k 10 --batch_size 16
    ```
- To conduct experiments for all supported model, run the following shell script.
    ```bash
    bash run_experiment.sh 2 [top_k] [batch_size]
    ```
##### 3) Check results
- The results will be saved in output/exp2 directory
    