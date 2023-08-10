# GPT2 Chinese text generation demo

GPT2 is the first causal model in my study process. With it, I wrote a demo for text generation to feel its ability of generation. Besides, I also learnt how deepspeed works and recorded the results.

## File Introduction

In `data_utils.py`, the dataset is processed with muti-threads and is from the education part of [THUCnews](http://thuctc.thunlp.org/#%E8%8E%B7%E5%8F%96%E9%93%BE%E6%8E%A5)

 `train.py` has codes that how to use gpt2 to train a text generation model without deepspeed.

`train_ds.py` has codes that how to use gpt2 to train a text generation model with deepspeed.

`run_ds.sh` is used for train model with commad `deepspeed`.

`ds_config.json` is a file that contains configuration of deepspeed.

`generate.py` is a file to test the generation result and it is refered to this [repo](https://github.com/Morizeyao/GPT2-Chinese/tree/old_gpt_2_chinese_before_2021_4_22) 

## How to use

`python train.py --raw`: process data first and train model

`python train.py`: use saved data and train model

`sh run_ds.sh`: train model with deepspeed

## The result of DeepSpeed

with deepspeed: 

![with deepspeed](https://github.com/lancorrect/GPT2-Chinese-text-generation/blob/main/with%20deepspeed.jpeg)

without deepspeed:

![without deepspeed](https://github.com/lancorrect/GPT2-Chinese-text-generation/blob/main/without%20deepspeed.jpeg)

## Contact

Please feel free to contact me. It's my honor to make progress with you.

Email address: zhihaolancorrect@gmail.com