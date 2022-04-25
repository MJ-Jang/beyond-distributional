### 1. Dataset processor
- AutoModel, T5Model용으로 분리
- 공용 AutoModel / T5Model dataset을 상속받아서 input_process logic만 추가
```python
class RTEAutoInferenceDataset(GLUEAuToInferenceDataset):
    task_name = 'rte'

    @staticmethod
    def process_input(data):
        input1 = [f'Sentence1: {p}' for p in data['sentence1']]
        input2 = [f'Sentence2: {p}' for p in data['sentence2']]
        return input1, input2
        
class RTEAutoInferenceReverseDataset(GLUEAuToInferenceDataset):
    task_name = 'rte'

    @staticmethod
    def process_input(data):
        input1 = [f'Sentence2: {p}' for p in data['sentence2']]
        input2 = [f'Sentence1: {p}' for p in data['sentence1']]
        return input1, input2
```

### 2. Inferencer
- AutoModel, T5Model용으로 분리
- 날코딩이어서 피드백/리펙토링 필요
- usage
```bash
python inference.py --model_type roberta-base --dataset rte --input_format original --data_type test
```

### 3. 개선점
- inference.py에서 날코딩의 결정체인 *load_model_from_statedict* 처리
- inference.py에서 *prepare_data* 함수 효율화 (테스크 추가되면 답 없을듯..)
