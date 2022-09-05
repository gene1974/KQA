import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer

def load_choice_vocab(dataset):
    choice_list = [] # choice token
    choice_dict = {}
    for item in dataset:
        for choice in item['choices']:
            if choice not in choice_dict:
                choice_list.append(choice)
                choice_dict[choice] = len(choice_dict)
    print('Choices: ', len(choice_dict)) # 81629
    return choice_list, choice_dict

# 树型结构，后序遍历
def get_program_seq(program):
    seq = []
    for item in program:
        func = item['function']
        inputs = item['inputs']
        args = ''
        for input in inputs:
            args += ' <arg> ' + input
        seq.append(func + args)
        # seq.append(func + '(' + '<c>'.join(inputs) + ')')
    seq = ' <func> '.join(seq)
    return seq

def encode_dataset(dataset, tokenizer, choice_dict, mode = 'train'):
    programs = []
    answers = []
    choices = []
    questions = []
    for item in dataset:
        questions.append(item['question'])
        choices.append([choice_dict[i] for i in item['choices']])
        if mode == 'train':
            programs.append(get_program_seq(item['program']))
            answers.append(choice_dict[item['answer']])
            
    inputs = programs + questions
    tokens = tokenizer(inputs, padding = True, return_tensors = 'pt') # [188752, 164]
    question_ids, question_masks = tokens['input_ids'][len(programs):], tokens['attention_mask'][len(programs):]
    choice_ids = torch.tensor(choices)
    if mode == 'train':
        program_ids, program_masks = tokens['input_ids'][:len(programs)], tokens['attention_mask'][:len(programs)]
        answer_ids = torch.tensor(answers)
        return program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids
    else:
        return question_ids, question_masks, choice_ids

def load_datasets(args, debug = False):
    with open(args['path'] + 'train.json', 'r') as f:
        train_data = json.load(f) # ['program', 'sparql', 'answer', 'choices', 'question']
    with open(args['path'] + 'val.json', 'r') as f:
        valid_data = json.load(f) # ['program', 'sparql', 'answer', 'choices', 'question']
    with open(args['path'] + 'test.json', 'r') as f:
        test_data = json.load(f) # ['choices', 'question']
    print('Load data: ', len(train_data), len(valid_data), len(test_data))
    
    # choice list 是所有 choice 选项的 list，不是 word list
    choice_list, choice_dict = load_choice_vocab(train_data + valid_data + test_data)
    tokenizer = BartTokenizer.from_pretrained('/data/pretrained/bart-base/')
    
    tokenizer.add_tokens('<func>', special_tokens = True)
    tokenizer.add_tokens('<arg>', special_tokens = True)

    program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids = \
        encode_dataset(train_data, tokenizer, choice_dict, 'train')
    train_set = KQADataset(args, program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids, 'train')
    # if debug:
    #     return train_set
    program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids = \
        encode_dataset(valid_data, tokenizer, choice_dict, 'train')
    valid_set = KQADataset(args, program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids, 'train')
    question_ids, question_masks, choice_ids = \
        encode_dataset(test_data, tokenizer, choice_dict, 'test')
    test_set = KQADataset(args, None, None, question_ids, question_masks, choice_ids, None, 'test')

    vocabs = {
        'tokenizer': tokenizer,
        'choice_list': choice_list,
        'choice_dict': choice_dict,
    }
    # print(len(tokenizer))
    # print(torch.max(program_ids), torch.min(program_ids))
    
    return train_set, valid_set, test_set, vocabs
    

class KQADataset(Dataset):
    def __init__(self, args, program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids, mod = 'test'):
        super().__init__()
        self.args = args
        self.mod = mod
        self.batch_size = args['batch']

        self.program_ids = program_ids
        self.program_masks = program_masks
        self.question_ids = question_ids
        self.question_masks = question_masks
        self.choice_ids = choice_ids
        self.answer_ids = answer_ids
    
    def __len__(self):
        return int(np.ceil(len(self.question_ids) / self.batch_size))
    
    def __getitem__(self, index):
        return self.get_batch(index)

    def get_batch(self, index):
        begin = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.program_ids))
        
        question_ids = self.question_ids[begin: end].cuda()
        question_masks = self.question_masks[begin: end].cuda()
        choice_ids = self.choice_ids[begin: end].cuda()
        if self.mod == 'train':
            program_ids = self.program_ids[begin: end].cuda()
            program_masks = self.program_masks[begin: end].cuda()
            answer_ids = self.answer_ids[begin: end].cuda()
        else:
            program_ids, program_masks, answer_ids = None, None, None
        return program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids
        # else:
        #     return question_ids, question_masks, choice_ids
        


if __name__ == '__main__':
    args = {
        'path': './data/',
        'batch': 16,

    }
    train_set, valid_set, test_set, vocabs = load_datasets(args)
    train_set = load_datasets(args, True)
    print(train_set[0][0])
    
    # with open('demo_train.json', 'r') as f:
    #     demo_data = json.load(f)
    # program_seq = gen_program(demo_data['program'])
    # print(program_seq)


