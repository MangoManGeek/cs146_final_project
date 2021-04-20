from datasets import load_dataset
from torch.utils.data import Dataset
from allennlp.modules.elmo import Elmo, batch_to_ids
from transformers import *
from tqdm import tqdm
import numpy as np
import torch

def tokenize(tokenizer, s, model_type, window_size):
    if model_type == 'bert':
        return tokenizer(s, padding = 'max_length', truncation = True, max_length = window_size)['input_ids']
        # , return_tensors="pt"
    else:
        x = batch_to_ids(s)[0]
        x = x[:window_size]
        padding = np.zeros([window_size - x.shape[0], 50])
        rv = np.concatenate([x, padding], axis=0)
        return rv

class AGNewsDataset(Dataset):
    def __init__(self, model_type, dataset_type, window_size, vocab_dict = None):
        self.lm_inputs = []
        self.lm_labels = []
        self.inputs = []
        self.labels = []
        self.vocab_dict = vocab_dict if vocab_dict else dict()
        self.vocab_dict['*PAD*'] = 0
        self.vocab_dict['start'] = 1
        self.vocab_dict['stop'] = 2

        if model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = batch_to_ids

        self.dataset = load_dataset('ag_news', split=dataset_type)
        self.num_classes = self.dataset.info.features['label'].num_classes

        if model_type != 'bert':
            count = len(self.vocab_dict)
            for idx in tqdm(range(len(self.dataset))):
                temp = self.dataset[idx]['text'].split()
                for word in temp:
                    if word not in self.vocab_dict:
                        self.vocab_dict[word] = count
                        count += 1

        if model_type == 'bert':
            self.vocab_size = tokenizer.vocab_size
        else:
            self.vocab_size = len(self.vocab_dict)



        # for idx in tqdm(range(len(self.dataset))):
        #     self.lm_inputs.append(tokenizer(['start'] + self.dataset[idx]['text'].strip().split()))
        #     self.lm_labels.append(tokenizer(self.dataset[idx]['text'].strip().split() + ['stop']))
        #     self.inputs.append(tokenizer(self.dataset[idx]['text'].strip().split()))
        #     self.labels.append(self.dataset[idx]['label'])
        t = 0
        for idx in tqdm(range(len(self.dataset))):
            self.lm_inputs.append(
                tokenize(tokenizer, 'start ' + self.dataset[idx]['text'], model_type, window_size))
            if model_type == 'bert':
                # print("bert label")
                self.lm_labels.append(tokenize(tokenizer, self.dataset[idx]['text'] + ' stop', model_type, window_size))
            else:
                label_words = (self.dataset[idx]['text'] + ' stop').split()
                # print(self.vocab_dict)
                label_ids = [self.vocab_dict[word] for word in label_words]
                label_ids = label_ids[:window_size]
                label_ids = label_ids + [0] * (window_size - len(label_ids))
                self.lm_labels.append(label_ids)
            self.inputs.append(tokenize(tokenizer, self.dataset[idx]['text'], model_type, window_size))
            self.labels.append(self.dataset[idx]['label'])
            if t == 100:
                break
            t += 1
        # print(type(self.lm_inputs))
        if model_type == 'bert':
            self.lm_inputs = torch.tensor(self.lm_inputs)
            self.lm_labels = torch.tensor(self.lm_labels)
            self.inputs = torch.tensor(self.inputs)
            self.labels = torch.tensor(self.labels)
        else:
            self.lm_inputs = torch.tensor(self.lm_inputs).to(torch.int64)
            self.lm_labels = torch.tensor(self.lm_labels).to(torch.int64)
            self.inputs = torch.tensor(self.inputs).to(torch.int64)
            self.labels = torch.tensor(self.labels).to(torch.int64)


    def __len__(self):
        return len(self.lm_inputs)

    def __getitem__(self, idx):
        # item = {
        # 'lm_input': ['start'] + self.dataset[idx]['text'].strip().split(),
        # 'lm_label': self.dataset[idx]['text'].strip().split() + ['stop'],
        # 'input': self.dataset[idx]['text'].strip().split(),
        # 'label': self.dataset[idx]['label']
        # }
        item = {
        'lm_input': self.lm_inputs[idx],
        'lm_label': self.lm_labels[idx],
        'input': self.inputs[idx],
        'label': self.labels[idx]
        }
        return item


# class AGNewsTestDataset(Dataset):
#     def __init__(self, model_type):
#         # self.dataset = load_dataset('ag_news', split='test')
#         self.lm_inputs = []
#         self.lm_labels = []
#         self.inputs = []
#         self.labels = []

#         if model_type == 'bert':
#             tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         else:
#             tokenizer = batch_to_ids

#         self.dataset = load_dataset('ag_news', split='test')

#         t = 0
#         for idx in tqdm(range(len(self.dataset))):
#             self.lm_inputs.append(tokenizer(['start'] + self.dataset[idx]['text'].strip().split()))
#             self.lm_labels.append(tokenizer(self.dataset[idx]['text'].strip().split() + ['stop']))
#             self.inputs.append(tokenizer(self.dataset[idx]['text'].strip().split()))
#             self.labels.append(self.dataset[idx]['label'])
#             t += 1
#             if t == 100:
#                 break

#     def __len__(self):
#         return len(self.lm_inputs)

#     def __getitem__(self, idx):
#         # item = {
#         # 'lm_input': ['start'] + self.dataset[idx]['text'].strip().split(),
#         # 'lm_label': self.dataset[idx]['text'].strip().split() + ['stop'],
#         # 'input': self.dataset[idx]['text'].strip().split(),
#         # 'label': self.dataset[idx]['label']
#         # }
#         item = {
#         'lm_input': self.lm_inputs[idx],
#         'lm_label': self.lm_labels[idx],
#         'input': self.inputs[idx],
#         'label': self.labels[idx]
#         }
#         return item