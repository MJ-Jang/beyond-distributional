### Meaning-matching Training 

This code is for the intermediate training on meaning-matching task. 

### 0. Build Data
- Before the intermediate training on meaning-matching task, you have to generate negative-sampled data.
- run `build_train_dev.py` script in `/data/meaning_matching` with proper negative sample values.

### 1. Train
- Run `train.sh` script.
    ```bash
    bash train.sh roberta-base 10
    ```
    - Inputs: backbone model name, number of negative samples