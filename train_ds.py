import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import json
import numpy as np
import argparse
from transformers import GPT2LMHeadModel, AutoConfig, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from data_utils import GenerateDataset, Collator
from tqdm import tqdm
import time
import deepspeed

''' 
本文件旨在学习deepspeed，分为安装，训练和效果三个部分 
1.安装：此处踩了很多坑，重要的是显卡中libstdc++.so.6的version CXXABI_1.3.9找不到，根据https://github.com/TimDettmers/bitsandbytes/issues/70的解答，实际上是路径问题，根据issue回答可以解决
    查看是否安装成功,可以用python -m bitsandbytes来检验
2.训练：训练代码上的异同在下面写了，这里主要说deepspeed参数设置的问题。在json文件中对于必须要设定的数值参数不能设置为“auto”，虽然文档中说设定"auto"可以使用默认值，但是实验下来会把auto当做数值处理，也就会报错。
    参数train_batch_size和train_micro_batch_size_per_gpu的关系是前者等于后者乘以梯度累积步数再乘以world_size，实验中定义后者即可
3.效果：根据gpt2的效果上来看，在使用stage-2的条件下，使用deepspeed后显存减少了8GB，内存使用增加了一半，同时明显感觉训练变慢。
    说明在单卡情况下，使用deepspeed stage-2会减少显存的使用，但同时时间和内存相应增加，对于显存不够的情况有所缓解，类似于用时间换空间的做法
'''

def train(args, data_loader, model):
    total_steps = int(len(data_loader) * args.epochs / args.gradient_accumulation)
    print(f'total steps are {total_steps}')
    # optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    # optimizer.zero_grad()
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = total_steps)
    # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

    '''
    使用deepspeed时，训练阶段的代码有所不同。首先第一点是model不再需要单独写代码放到显卡上，在deepspeed.initialize阶段就已经将模型分配好了
    即有什么参数放在gpu上，有什么参数放在cpu上。第二点是不再需要定义optimizer和scheduler，这两个在ds_config.json中定义好了。
    第三点，也是最重要的，损失回传的时候不再是loss.backward()，而是model.backward(loss)。并且没有zero_grad和step了，直接合并成model.step()
    注意的是此处的model指的是已经通过deepspeed初始化后的model engine，不再是单纯的pytorch模块了
    '''
    
    print('start training!')
    model.train()
    overall_step = 0
    running_loss = 0
    for epoch in range(args.epochs):
        print(f'{epoch+1}-th epoch')
        
        for i, (input_ids, label, attention_mask) in enumerate(data_loader):
            input_ids = input_ids.long().to(args.device)
            label = label.long().to(args.device)
            attention_mask = attention_mask.to(args.device)

            outputs = model.forward(input_ids=input_ids, labels=label, attention_mask=attention_mask)
            loss, logits = outputs[:2]

            if args.gradient_accumulation > 1:
                loss = loss/args.gradient_accumulation
            
            running_loss += loss.item()
            # loss.backward()
            model.backward(loss)
            
            if (overall_step+1) % args.gradient_accumulation == 0:
                # optimizer.step()
                # optimizer.zero_grad()
                # scheduler.step()
                model.step()

            if (overall_step + 1) % args.log_step == 0:
                print(f"The loss of {overall_step}-th step is {running_loss}")
                running_loss = 0
            
            overall_step += 1

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_name', default='gpt2', type=str, required=False, help='模型名称')
    parser.add_argument('--raw_data_path', default='./data/education', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='./data/tokenized_data.json', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=30, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=10, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=512, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=5, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--min_length', default=128, type=int, required=False, help='单条数据最短数据')
    parser.add_argument('--max_length', default=512, type=int, required=False, help='单条数据最大长度')
    parser.add_argument('--output_dir', default='./model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--threads', default=10, type=int, required=False, help='处理数据时的线程数量')
    parser.add_argument("--deepspeed", default='./ds_config.json', type=str, help='deepspeed参数文件')
    parser.add_argument('--local_rank', type=int, default=0, help='')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    bos = '<|endoftext|>'
    eos = '<|EOS|>'
    pad = '<|pad|>'
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}
    tokenizer.add_special_tokens(special_tokens_dict)
    collate_fn = Collator(args, tokenizer)

    # process data
    if args.raw:
        dataset = GenerateDataset(args, tokenizer=tokenizer)
        with open(args.tokenized_data_path, 'w') as f:
            json.dump(dataset.dataset, f, indent=4, ensure_ascii=False)
            f.close()
    else:
        if not Path(args.tokenized_data_path).exists():
            raise FileNotFoundError
        with open(args.tokenized_data_path, 'r') as f:
            dataset = json.load(f)
            f.close()
        dataset = GenerateDataset(args, data=dataset)
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # load model
    config = AutoConfig.from_pretrained(args.model_name, bos_token_id=tokenizer.bos_token_id, 
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                        output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(args.model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    
    # deepspeed configuration and initialization
    with open(args.deepspeed, 'r') as fin:
        ds_config = json.load(fin)
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["steps_per_print"] = args.log_step
    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation
    model, _, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())
    model = model.cuda()
    args.device = model.device

    start = time.time()
    train(args, data_loader, model)
    then = time.time()
    print(f'total time: {then-start}')