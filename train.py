import os
import sys

from query import query_kb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BartForConditionalGeneration

from data import load_datasets
assert(torch.cuda.is_available())

def train_epoch(model, vocabs, optimizer, train_set, dev_set = None):
    tokenizer = vocabs['tokenizer']
    train_losses = []
    valid_losses = []
    model.train()
    for i in range(len(train_set)):
        program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids = train_set[i]
        labels = program_ids[:, 1:].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        program_ids = program_ids[:, :-1].contiguous()
        optimizer.zero_grad()
        outputs = model(
            input_ids = question_ids, attention_mask = question_masks, 
            decoder_input_ids = program_ids, labels = labels
        ) # odict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])
        loss = outputs[0]
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    if dev_set is None:
        return np.average(train_losses), -1
    
    model.eval()
    with torch.no_grad():
        for i in range(len(dev_set)):
            program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids = train_set[i]
            labels = program_ids[:, 1:].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            program_ids = program_ids[:, :-1].contiguous()
            outputs = model(
                input_ids = question_ids, attention_mask = question_masks, 
                decoder_input_ids = program_ids, labels = labels
            ) # odict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])
            loss = outputs[0]
            valid_losses.append(loss.item())
    return np.average(train_losses), np.average(valid_losses)

def train(args):
    args['time'] = '{}'.format(time.strftime('%m%d%H%M', time.localtime()))
    train_set, valid_set, test_set, vocabs = load_datasets(args)
    model = BartForConditionalGeneration.from_pretrained('/data/pretrained/bart-base/').to('cuda')
    model.resize_token_embeddings(len(vocabs['tokenizer']))
    optimizer = optim.Adam(model.parameters(), lr = args['lr'])
    for epoch in range(args['epoch']):
        train_loss, valid_loss = train_epoch(model, vocabs, optimizer, train_set, valid_set)
        time_stamp = time.strftime("%m-%d %H:%M:%S", time.localtime())
        print('[{}][epoch {:d}] Loss: Train: {:.4f} Dev: {:.4f}'.format(time_stamp, epoch + 1, train_loss, valid_loss))
        sys.stderr.write('[{}][epoch {:d}] Loss: Train: {:.4f} Dev: {:.4f}\n'.format(time_stamp, epoch + 1, train_loss, valid_loss))
    save_model(model, vocabs, args)

def save_model(model, vocab, args):
    print('Save model: {}'.format(args.time))
    torch.save(model.state_dict(), './savemodel/model_' + args.time)
    with open('./savemodel/vocab_' + args.time, 'wb') as f:
        pickle.dump(vocab, f)
        pickle.dump(args, f)

def do_test(model, dataset):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i in range(len(dataset)):
            program_ids, program_masks, question_ids, question_masks, choice_ids, answer_ids = dataset[i]
            program_seq = model.generate(
                input_ids = question_ids, attention_mask = question_masks
            )
            result = query_kb(program_seq)
    return 

def test(args):
    model_time = args.time
    with open('./savemodel/vocab_' + model_time, 'rb') as f:
        vocab = pickle.load(f)
        args = pickle.load(f)
    train_set, dev_set, test_set, _ = load_datasets(args)
    model = BartForConditionalGeneration.from_pretrained('/data/pretrained/bart-base/').to('cuda')
    model.resize_token_embeddings(len(vocab['tokenizer']))
    model.load_state_dict(torch.load('./savemodel/model_' + model_time))
    do_test(model, train_set)
    do_test(model, dev_set)
    do_test(model, test_set)
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default = 'train')
    parser.add_argument('-path', default = './data/')
    # parser.add_argument('-model', default = 'bart')
    # parser.add_argument('-time', default = '05131154', help = 'Test model')
    parser.add_argument('-epoch', type = int, default = 50)
    parser.add_argument('-lr', type = float, default = 1e-5)
    parser.add_argument('-batch', type = int, default = 8)
    # parser.add_argument('-savemodel', type = bool, default = True)
    # parser.add_argument('-hidden', type = int, default = 300)
    
    args = parser.parse_args()
    args = args.__dict__
    print(args)
    
    if args['mode'] == 'train':
        train(args)
    # # elif args.mode == 'test':
    # #     test(args)
    
