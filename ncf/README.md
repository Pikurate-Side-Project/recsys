# Neural Colaborative Filtering(user-url)

NCF를 적용하여 user에 맞춤형 url을 추천하는 방식
* 대상: 링크를 20개 이상한 저장한 user `247`명
* url: 위의 조건을 만족하는 user가 저장한 `37179`개의 url

## Structure
Pytorch의 Ignite를 사용하여 모델을 구현하였으며 구조는 다음과 같다.

<img src='https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0b02aae4-cdfb-4f5d-9ae9-893b3aa1b066/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20211013%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20211013T064429Z&X-Amz-Expires=86400&X-Amz-Signature=d855f11da6598ec38e39792ebd2fd90f2363f434b988ffd483152ffd1c35438d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22' style="width: 600px;">

모델을 기점으로 Trainer 객체가 학습을 진행하며 Data Loader로 부터 배치 사이즈 만큼의 데이터를 지속적으로 데이터를 공급받도록 한다.
```
    |__ data
    |   |__ url_user_rel.tsv
    |__ models
    |__ modules
    |   |__ FeedForwardNN.py
    |__ utils
    |   |__ dataset.py
    |   |__ trainer.py
    |   |__ utils.py
    |__ ncf_train.py
```
* `FeedForwardNN.py`: hidden layer의 정의 NCF의 구현
* `dataset.py`: 배치 사이즈 만틈 데이터를 로드
* `trainer.py`: trainer의 engine 정의, 학습 및 예측
* `utils.py`: 데이터 읽기, 파라미터 계산
* `ncf_train.ㅔㅛ`: 학습 실행 파일

## Hyper parameter

## 실행

### 실행
```bash
python  ncf_train.py --model_fn ./model/ncf.user-url.gt.20 --train_fn ./data/url_user_rel.tsv --gpu_id 0 --batch_size 80 --n_epochs 2
```

### parameter 정보
```
python ncf_train.py --asdf
```

`ncf_train.py`의 arg parser 객체를 확인하면 더 자세한 정보를 요청할 수 있다.
```python
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    # If you want to use RAdam, I recommend to use LR=1e-4.
    # Also, you can set warmup_ratio=0.
    p.add_argument('--use_radam', action='store_true')
    p.add_argument('--valid_ratio', type=float, default=.2)

    config = p.parse_args()

    return config
```

## Result
