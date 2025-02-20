## main.py
이 파일은 FastAPI를 사용하여 간단한 API 서버를 설정합니다. 주요 기능은 다음과 같습니다:

1. **모델 초기화**: `modelAnd`, `modelXor`, `modelOr`, `modelNot` 모델을 초기화합니다.
2. **루트 엔드포인트**: `/` 경로로 요청이 들어오면 "Hello, World" 메시지를 반환합니다.
3. **아이템 엔드포인트**: `/items/{item_id}` 경로로 요청이 들어오면 아이템 ID와 선택적 쿼리 매개변수를 반환합니다.
4. **예측 엔드포인트**: `/predict{model}/left/{left}/right/{right}` 경로로 요청이 들어오면 지정된 모델을 사용하여 예측을 수행합니다. 모델이 저장된 파일이 있으면 로드하고, 없으면 학습 후 저장합니다.
5. **학습 엔드포인트**: `/train{model}` 경로로 요청이 들어오면 지정된 모델을 학습하고 저장합니다.

## model.py
이 파일은 네 가지 모델(`AndModel`, `XorModel`, `OrModel`, `NotModel`)을 정의합니다. 각 모델은 PyTorch를 사용하여 구현되었습니다. 주요 기능은 다음과 같습니다:

1. **AndModel**:
   - `__init__`: 모델 초기화.
   - `forward`: 순전파 계산.
   - `train`: 모델 학습.
   - `predict`: 예측 수행.
   - `save`: 모델 상태를 파일에 저장.
   - `load`: 파일에서 모델 상태를 로드.

2. **XorModel**:
   - `__init__`: 모델 초기화.
   - `forward`: 순전파 계산.
   - `train`: 모델 학습.
   - `predict`: 예측 수행.
   - `save`: 모델 상태를 파일에 저장.
   - `load`: 파일에서 모델 상태를 로드.

3. **OrModel**:
   - `__init__`: 모델 초기화.
   - `forward`: 순전파 계산.
   - `train`: 모델 학습.
   - `predict`: 예측 수행.
   - `save`: 모델 상태를 파일에 저장.
   - `load`: 파일에서 모델 상태를 로드.

4. **NotModel**:
   - `__init__`: 모델 초기화.
   - `forward`: 순전파 계산.
   - `train`: 모델 학습.
   - `predict`: 예측 수행.
   - `save`: 모델 상태를 파일에 저장.
   - `load`: 파일에서 모델 상태를 로드.