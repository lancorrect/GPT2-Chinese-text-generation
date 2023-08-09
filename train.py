import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import json
import numpy as np
import argparse
from transformers import GPT2LMHeadModel, AutoConfig, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time/14132912#14132912
from data_utils import GenerateDataset, Collator  # 相对引用只能在该py文件的__name__不是__main__的时候才可以使用，只要在脚本或者命令行中运行一个py文件，那么该文件的名字就已经是__main__了
from tqdm import tqdm
import time


def train(args, data_loader, model):
    total_steps = int(len(data_loader) * args.epochs / args.gradient_accumulation)
    print(f'total steps are {total_steps}')
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.warmup_steps, num_training_steps = total_steps)
    # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

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

            # 这里的input_ids和labels是一样的，在GPT2LMHeadModel源码里面会对input_ids和labels进行偏移操作，input_ids只要前n-1个，labels则是从第二个开始一直到最后。
            # 这样的话input_ids和labels就错了一位。因此input_ids中的第一个token的预测结果，实际上是标签中的第二个token，以此类推，最终仅计算sequence_length-1个token的loss
            outputs = model.forward(input_ids=input_ids, labels=label, attention_mask=attention_mask)
            loss, logits = outputs[:2]

            if args.gradient_accumulation > 1:
                # loss会除以梯度积累的步数，相当于在原始loss前加上了一个小于1的系数，保证在之后的链式求导过程中，梯度会等比例缩小，相当于对梯度进行归一化了
                # 如果要知道梯度积累下损失之和的话，需要额外定义一个变量来记录损失之和
                loss = loss/args.gradient_accumulation
            
            running_loss += loss.item()
            loss.backward()  # 把损失向输入侧进行反向传播，每次调用它都会free掉所有buffers，模型中可能有多次backward()，而前一次backward()存储在buffer中的梯度，会因为后一次调用backward()被free掉
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            if (overall_step+1) % args.gradient_accumulation == 0:
                optimizer.step()  # step可以累积之前的梯度，如果不使用zero_grad()，则会使用之前的梯度
                optimizer.zero_grad()
                scheduler.step()

                print(f"The loss of {overall_step}-th step is {running_loss}")
                running_loss = 0
            
            # if (overall_step+1) % args.log_step == 0:
            #     print(f"The loss of {i+1}-th step in {epoch+1}-th epoch is {running_loss}")
            #     running_loss = 0
            overall_step += 1
        
        # print('saving model for epoch {}'.format(epoch + 1))
        # if not Path(args.output_dir).exists():
        #     Path(args.output_dir).mkdir()
        # model_dir = Path(args.output_dir) / Path(f'epoch_{epoch+1}')
        # model_dir.mkdir(parents=True, exist_ok=True)
        # model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.save_pretrained(str(model_dir))
        # print('epoch {} finished'.format(epoch + 1))
    
    # print('training finished')
    # model_dir = Path(args.output_dir) / Path('final_model')
    # model_dir.mkdir(parents=True, exist_ok=True)
    # model_to_save = model.module if hasattr(model, 'module') else model
    # model_to_save.save_pretrained(model_dir)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--model_name', default='gpt2', type=str, required=False, help='模型名称')
    # parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
    #                     help='选择模型参数')

    # parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--raw_data_path', default='./data/education', type=str, required=False, help='原始训练语料')
    parser.add_argument('--tokenized_data_path', default='./data/tokenized_data.json', type=str, required=False,
                        help='tokenized语料存放位置')
    parser.add_argument('--raw', action='store_true', help='是否先做tokenize')
    parser.add_argument('--epochs', default=30, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=4, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=5, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
    parser.add_argument('--stride', default=512, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    # parser.add_argument('--fp16', action='store_true', help='混合精度')
    # parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    # parser.add_argument('--num_pieces', default=100, type=int, required=False, help='将训练语料分成多少份')
    parser.add_argument('--min_length', default=128, type=int, required=False, help='单条数据最短数据')
    parser.add_argument('--max_length', default=512, type=int, required=False, help='单条数据最大长度')
    parser.add_argument('--output_dir', default='./model/', type=str, required=False, help='模型输出路径')
    parser.add_argument('--threads', default=10, type=int, required=False, help='处理数据时的线程数量')
    # parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
    # parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    # parser.add_argument('--segment', action='store_true', help='中文以词为单位')
    # parser.add_argument('--bpe_token', action='store_true', help='subword')
    # parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    # parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    bos = '<|endoftext|>'
    eos = '<|EOS|>'
    pad = '<|pad|>'
    special_tokens_dict = {'eos_token': eos, 'bos_token': bos, 'pad_token': pad}
    tokenizer.add_special_tokens(special_tokens_dict)
    collate_fn = Collator(args, tokenizer)

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

    config = AutoConfig.from_pretrained(args.model_name, bos_token_id=tokenizer.bos_token_id, 
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                        output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained(args.model_name, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(args.device)

    start = time.time()
    train(args, data_loader, model)
    then = time.time()
    print(f'total time: {then-start}')