# Londinium

## Directory Description
- data: preprocessing and data for our experiment will be placed here
- output: directory where all results (or summary) will be placed
- src: includes all scripts for experiment and analysis


### 0. Setup Environment
#### a. make virtual environment
- Make a new environment
    ```
    virtualenv venv
    source venv/bin/activate
    ```
#### b. install required packages
- Install all required packages
    ```bash
    pip install -r requirements.txt
    ```
#### c. install spacy
- Install spacy package
    ```bash
    pip install -U spacy
    python -m spacy download en_core_web_sm
    ```

### 1. Build Dataset
#### a. fetch data (SNLI and LAMA)
- move to the data folder and fetch data
    ```bash
    cd data
    bash fetch_data.sh
    ```
    
#### b. build data for experiment and analysys
##### 1) build Synonym-Antonym pairs
- set update_thesarsus to True only at the first time (for time efficiency)
- if you already have *cn_thesaursus.json* file, you don't need to set --update_thesarsus to True
    ```bash
    cd data
    python antonym_synonym.py --update_thesarsus True
    ```

##### 2) build Negation pairs
- negate LAMA data (Only ConceptNet data is used)
- if you have *conceptnet_partial.json* file, you don't need to set --build_new_kg to True
    ```bash
    cd data
    python negate_lama.py
    ```
    
### 2. Do Experiment
#### a. Parameters
- exp_type: 1 or Synonym-Antonym pairs and 2 for Negation pairs
--top_k: number of top-k predictions for PLMs
```bash
bash run_experiment 1 [top_k]
bash run_experiment 2 [top_k]
```
- The results will be saved in **output** directory
