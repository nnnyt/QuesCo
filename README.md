# QuesCo
Source code for the AAAI 2023 paper "Towards a Holistic Understanding of Mathematical Questions with Contrastive Pre-training" 



## Environment

Requirments

```
numpy
pandas
torch==1.11.0
transformers==4.18.0
jieba
tqdm
sklearn
scipy
```



## Usage

* Train

  ```bash
  cd src
  # train
  python main.py --batch_size 8 --lr 5e-5 --data all --bert_type bert-base-chinese --name QuesCo --epochs 1 --device 0 --validation_steps 100 --use_same_and_similar_class True --queue_size 1600 --min_tau 0.1 --max_tau 0.6 --ques_dim 128 --mixed_out_in True
  ```

* Test

  ```bash
  cd src
  
  # similarity prediction
  python main.py --mode test --test_task similarity --batch_size 128 --data your_dataset_name --bert_type path_to_your_pretrained_model --name QuesCo --epochs 1 --ques_dim 128
  
  # concept prediction
  python main.py --mode test --test_task concepts --batch_size 32 --data your_dataset_name --test_know_level 1 --valid_batch_size 32 --bert_type path_to_your_pretrained_model --name QuesCo --epochs 10 --save_strategy epoch
  
  # difficulty prediction
  python main.py --mode test --test_task difficulty --batch_size 32 --data your_dataset_name --valid_batch_size 32 --bert_type path_to_your_pretrained_model --name QuesCo  --epochs 10 --save_strategy epoch
  ```

For more running arguments, please refer to [src/parameters.py](https://github.com/nnnyt/QuesCo/blob/main/src/parameters.py).





