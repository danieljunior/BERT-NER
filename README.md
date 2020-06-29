# BERT NER PT-BR

# Requirements

-  `python3`
- `pip3 install -r requirements.txt`

# Run

- Edit `tags.txt` according your dataset.

### Using DGX
- `docker build -t bert_ner .`
- `NV_GPU=1 nvidia-docker run -itd --rm --shm-size=32g --ulimit memlock=-1 -v ${PWD}:/app bert_ner bash`

- `python run_ner.py --data_dir=data/ --bert_model=bert-base-cased-pt-br --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1 --eval_on=test`

### Otherwise

`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased-pt-br --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1`

### Limiting batch size
`python run_ner.py --data_dir=data/ --bert_model=bert-base-cased-pt-br --train_batch_size=8 --task_name=ner --output_dir=out_base --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.1`

## BERT-BASE Pretrained model download from [here](https://github.com/neuralmind-ai/portuguese-bert)

** To use pytorch with notebook: conda install pytorch torchvision cuda90 -c pytorch


# Inference

```python
from bert import Ner

model = Ner("out_base/")

output = model.predict("Steve went to Paris")

print(output)
'''
    [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
'''
```

# Inference C++

## Pretrained and converted bert-base model download from [here](https://1drv.ms/u/s!Auc3VRul9wo5hgkJjtxZ8FAQGuj2?e=wffJCT)
### Download libtorch from [here](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.2.0.zip)

- install `cmake`, tested with `cmake` version `3.10.2`
- unzip downloaded model and `libtorch` in `BERT-NER`
- Compile C++ App
  ```bash
    cd cpp-app/
    cmake -DCMAKE_PREFIX_PATH=../libtorch
   ```
    ![cmake output image](/img/cmake.png)
    ```bash
    make
    ```
    ![make output image](/img/make.png)


- Runing APP
  ```bash
     ./app ../base
  ```
     ![inference output image](/img/inference.png)

NB: Bert-Base C++ model is split in to two parts.
  - Bert Feature extractor and NER classifier.
  - This is done because `jit trace` don't support `input` depended `for` loop or `if` conditions inside `forword` function of `model`.



# Deploy REST-API
BERT NER model deployed as rest api
```bash
python api.py
```
API will be live at `0.0.0.0:8000` endpoint `predict`
#### cURL request
` curl -X POST http://0.0.0.0:8000/predict -H 'Content-Type: application/json' -d '{ "text": "Steve went to Paris" }'`

Output
```json
{
    "result": [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
}
```
#### cURL 
![curl output image](/img/curl.png)
#### Postman
![postman output image](/img/postman.png)

### C++ unicode support 
- http://github.com/ufal/unilib

### Tensorflow version

- https://github.com/kyzhouhzau/BERT-NER
