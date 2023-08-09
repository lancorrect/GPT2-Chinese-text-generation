from pathlib import Path
import torch
from tqdm import tqdm
from queue import Queue
from threading import Thread
import time

def load_dataset(data_path):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError
    
    data = []
    for txt_file in data_path.iterdir():
        with open(txt_file, 'r', encoding='utf-8') as f:
            data_file = f.readlines()
            data_file = ''.join(data_file)
            data.append(data_file)
            f.close()
    
    return data

def generate_data(args, tokenizer):
    queue_read = Queue()
    queue_pro = Queue()

    if not Path(args.raw_data_path):
        raise FileNotFoundError
    for path in Path(args.raw_data_path).iterdir():
        queue_read.put(path)
    
    l_read = []
    l_pro = []
    for i in range(args.threads):
        l_r = Thread(target=put, args=(i+1, queue_read, queue_pro))
        l_read.append(l_r)
    for l in l_read:
        l.start()
        time.sleep(1)  # 在启动多线程的时候，sleep至关重要，否则会出现最后程序无法结束的问题
    for i in range(args.threads):
        l_p = Thread(target=get, args=(i+1, args, tokenizer, queue_read, queue_pro))
        l_pro.append(l_p)
    for l in l_pro:
        l.start()
        time.sleep(1)
    for l in l_read:
        l.join()
    for l in l_pro:
        l.join()
    
    # data = load_dataset(args.raw_data_path)
    # max_length = args.max_length
    # bos_token = tokenizer.bos_token
    # eos_token = tokenizer.eos_token
    # pad_token = tokenizer.pad_token

    # bar = tqdm(data)
    # num = 1
    # for article in bar:
    #     bar.set_description(f"Processing {num}-th article")
    #     if len(article) < args.min_length:
    #         continue
    #     start_point = 0
    #     stride = args.stride
    #     article_tokens = tokenizer.tokenize(article)
    #     while start_point+max_length-2 < len(article_tokens):
    #         sub_article = article_tokens[start_point:start_point+max_length-2]
    #         sub_article = [bos_token] + sub_article + (max_length-2-len(sub_article))*[pad_token] + [eos_token]
    #         sub_article_ids = tokenizer.convert_tokens_to_ids(sub_article)
    #         # self.dataset.append({"input_ids":sub_article_ids, "label":sub_article_ids})
    #         self.dataset.append(sub_article_ids)
    #         start_point+=stride
    #     sub_article = article_tokens[start_point:]
    #     sub_article = [bos_token] + sub_article + (max_length-2-len(sub_article))*[pad_token] + [eos_token]
    #     sub_article_ids = tokenizer.convert_tokens_to_ids(sub_article)
    #     # self.dataset.append({"input_ids":sub_article_ids, "label":sub_article_ids})
    #     self.dataset.append(sub_article_ids)
    #     num += 1

    # for idx, article in tqdm(enumerate(data), desc="Put data in queue."):
    #     article_tokens = tokenizer.tokenize(article)
    #     if len(article_tokens) < args.min_length:
    #         continue
    #     self.Queue.put(article_tokens)

    
def put(thread_id, queue_read, queue_pro):
    while True:
        if queue_read.empty():
            print(f"{thread_id}-th reading thread ends")
            break
        print(f"{thread_id}-th thread reads, the number of left articles are {queue_pro.qsize()}")
        data_path = queue_read.get()
        with open(data_path, mode='r', encoding='utf-8') as f:
            article = ''.join(f.readlines())
            queue_pro.put(article)
            f.close()

def get(thread_id, args, tokenizer, queue_read, queue_pro):
    while True:
        if queue_read.empty() and queue_pro.empty():
            print(f"{thread_id}-th processing thread ends")
            break
        if queue_pro.empty():
            time.sleep(5)
        else:
            max_length = args.max_length
            bos_token_id = tokenizer.bos_token_id
            eos_token_id = tokenizer.eos_token_id
            pad_token_id = tokenizer.pad_token_id
            print(f"{thread_id}-th thread processes, the number of left articles are {queue_pro.qsize()}")
            article_tokens = tokenizer.encode(queue_pro.get(), return_tensors='pt')
            start_point = 0
            stride = args.stride

            for idx in range(0, len(article_tokens), stride):
                sub_article = article_tokens[idx:idx+max_length-2]
                sub_article = [bos_token_id] + sub_article + [eos_token_id]
                dataset.append(sub_article)
            time.sleep(1)

            # while start_point+max_length-2 < len(article_tokens):
            #     sub_article = article_tokens[start_point:start_point+max_length-2]
            #     sub_article = [bos_token] + sub_article + (max_length-2-len(sub_article))*[pad_token] + [eos_token]
            #     sub_article_ids = tokenizer.convert_tokens_to_ids(sub_article)
            #     dataset.append(sub_article_ids)
            #     start_point+=stride
            # sub_article = article_tokens[start_point:]
            # sub_article = [bos_token] + sub_article + (max_length-2-len(sub_article))*[pad_token] + [eos_token]
            # sub_article_ids = tokenizer.convert_tokens_to_ids(sub_article)
            # dataset.append(sub_article_ids)
            # time.sleep(1)

class GenerateDataset:
    def __init__(self, args, tokenizer=None, data=None):
        

        if args.raw:
            # self.dataset = []
            # self.queue_pro = Queue()
            # data = load_dataset(args.raw_data_path)
            # for idx, article in tqdm(enumerate(data), desc="Put data in queue."):
            #     article_tokens = tokenizer.tokenize(article)
            #     if len(article_tokens) < args.min_length:
            #         continue
            #     self.queue_pro.put(article_tokens)
            
            # l_pro = []
            # for i in range(args.threads):
            #     l_p = Thread(target=self.get, args=(i+1, args, tokenizer))
            #     l_pro.append(l_p)
            # for l in l_pro:
            #     l.start()
            #     time.sleep(1)
            # for l in l_pro:
            #     l.join()

            global dataset
            dataset = []
            print("start processing")
            generate_data(args, tokenizer)
            self.dataset = dataset
            print("processing ends")
        else:
            if not Path(args.tokenized_data_path).exists():
                raise FileNotFoundError
            self.dataset = data
    
    def get(self, thread_id, args, tokenizer):
        while not self.queue_pro.empty():
            max_length = args.max_length
            bos_token = tokenizer.bos_token
            eos_token = tokenizer.eos_token
            pad_token = tokenizer.pad_token
            print(f"{thread_id}-th thread processes, the number of left articles are {self.queue_pro.qsize()}")
            article_tokens = self.queue_pro.get()
            start_point = 0
            stride = args.stride
            while start_point+max_length-2 < len(article_tokens):
                sub_article = article_tokens[start_point:start_point+max_length-2]
                sub_article = [bos_token] + sub_article + (max_length-2-len(sub_article))*[pad_token] + [eos_token]
                sub_article_ids = tokenizer.convert_tokens_to_ids(sub_article)
                self.dataset.append(sub_article_ids)
                start_point+=stride
            sub_article = article_tokens[start_point:]
            sub_article = [bos_token] + sub_article + (max_length-2-len(sub_article))*[pad_token] + [eos_token]
            sub_article_ids = tokenizer.convert_tokens_to_ids(sub_article)
            self.dataset.append(sub_article_ids)
            time.sleep(1)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class Collator:
    def __init__(self, args, tokenizer):
        self.max_length = args.max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        # 在collate_fn中，输入的batch不是torch类型，而是之前在class里面设置好的类型，经过此函数处理后不会转主动成torch类型
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        batch_lens = [len(item) for item in batch]
        batch_max_len = min(max(batch_lens), self.max_length)

        for item in batch:
            item_len = len(item)
            padding_len = batch_max_len - item_len

            item = item + [self.pad_token_id] * padding_len
            attention_mask = [1] * len(item) + [0] * padding_len
            
            input_ids_batch.append(item)
            labels_batch.append(item)
            attention_mask_batch.append(attention_mask)

        return torch.tensor(input_ids_batch), torch.tensor(labels_batch), torch.tensor(attention_mask_batch)