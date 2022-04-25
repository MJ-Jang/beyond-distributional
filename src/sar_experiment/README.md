### SAR Experiments

This code is for the SAR experiments.

### 1. Train
- Specify the name of the models your are going to train in `train.sh` shell script.
    ```bash
    models="roberta-base roberta-large albert-base albert-large"
    ```
- Run `train.sh` script.
    ```bash
    bash train.sh
    ```
- Both models with fixed- and non fixed- encoder will be trained.


### 2. Inference
- Run `inference.sh` script.
    ```bash
    bash inference.sh roberta-base 01
    ```
    - The first input is a backbone model name and the second input is the folder name to save.
    
** If you want to use meaning-matching model, they must be first trained and the model binary file should be placed in `../meaning_matching_experiment/model_binary` folder.