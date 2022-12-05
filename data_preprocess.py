import torch
import nlp
from transformers import MT5Tokenizer
import os
import functools
import pandas as pd
from nlp import Dataset
import json
import shutil

import numpy as np
import random


TASK2LANGS = {
  "pawsx": "de,en,es,fr,ja,ko,zh".split(","),
  "xnli": "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh".split(","),
  "panx": "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu".split(","),
  "udpos": "af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh".split(","),
  "bucc2018": "de,fr,ru,zh".split(","),
  "tatoeba": "ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(","),
  "xquad": "en,es,de,el,ru,tr,ar,vi,th,zh,hi".split(","),
  "mlqa": "en,es,de,ar,hi,vi,zh".split(","),
  "tydiqa": "en,ar,bn,fi,id,ko,ru,sw,te".split(","),
  "marc": "en,de,es,fr,ja,zh".split(","),
  "xlsum": "en,ar,vi,ko,es,zhCN,ru,fr,tr,hi,id,fa,pt,mr,th,az,bn,np,srcy,sw,ta,te,ur,cy,am,my,gu,ha,ig,pa,si,yo".split(
    ','),
  "unifiedQA": "newsqa,winograndeXL,mctest,openbookqa,quoref,boolq,commonsenseqa,narrativeqa,social_iqa,ropes,boolq_np,qasc,squad2,race".split(','),
  "mldoc": "en,de,es,fr,ja,zh,it,ru".split(","),
  "panx": "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu".split(","),
}

# process the examples in input and target text format and the eos token at the end
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')
def add_eos_to_examples_original(example):
    example['input_text'] = 'question: %s  context: %s </s>' % (example['question'], example['context'])
    example['target_text'] = '%s </s>' % example['answers'][0]
    return example


def add_eos_to_examples_prompt(example):
    example['input'] = '%s </s>' % (example['input'])
    example['target'] = '%s </s>' % (example['target'])
    return example

# tokenize the examples
def convert_to_features(example_batch,in_len=512, out_len=64):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input'], pad_to_max_length=True, max_length=in_len)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target'], pad_to_max_length=True, max_length=out_len)
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings


def preprocess(datas,save_path,in_len,out_len):
    print('in_len: ',in_len)
    print('out_len: ', out_len)
    df = pd.DataFrame(data=datas)
    dataset = Dataset.from_pandas(df)
    dataset_map = dataset.map(add_eos_to_examples_prompt)
    dataset_convert = dataset_map.map(functools.partial(convert_to_features,in_len=in_len, out_len=out_len), batched=True)

    print('dataset_convert: ',dataset_convert)
    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    dataset_convert.set_format(type='torch', columns=columns)
    print('dataset_convert[0]: ', dataset_convert[0])

    torch.save(dataset_convert,  save_path)


def read_outtasks_train(tasknames, data_dir,split='train'):
    new_samps = []
    for taskname in tasknames:
        filename = split+'-'+taskname+'.json'
        path_task = data_dir+'/'+filename

        read_samples = json.load(open(path_task, 'r'))
        print('data: %s; n_samples: %d'%(taskname,len(read_samples)))
        for samp in read_samples:
            input = samp["input"]
            target = samp["target"]
            answers = samp["answers"]
            samp_id = samp["samp_id"]
            new_samp = {"input": input, "target": target, "answers": answers, "samp_id": samp_id}
            new_samps.append(new_samp)
    print('num of outtasks: ', len(tasknames))
    print('num of outtasks samples: ',len(new_samps))

    return new_samps

def read_intasks_train(TASK2LANGS, tasknames, data_dir, split='train'):
    new_samps = []
    for task in tasknames:
        langs = TASK2LANGS[task]
        for lang in langs:
            file_name = split + '-' + lang + '.json'
            file_path = os.path.join(data_dir, task, file_name)
            read_samples = json.load(open(file_path, 'r'))
            for samp in read_samples:
                input = samp["input"]
                target = samp["target"]
                answers = samp["answers"]
                samp_id = samp["samp_id"]

                new_samp = {"input": input, "target": target, "answers": answers, "samp_id": samp_id}
                new_samps.append(new_samp)
    print('num of intasks: ',len(tasknames))
    print('num of samples in intasks: ',len(new_samps))
    return new_samps


def generate_big_train(TASK2LANGS,save_dir, inT_tasks,outT_tasks,inT_datadir,outT_datadir,save_filename,
                       split='train',in_len=512,out_len=64):
    outT_samps = []
    if len(outT_tasks)!=0:
        outT_samps = read_outtasks_train(outT_tasks, outT_datadir, split='train')
    print('len(outT_samps): ',len(outT_samps))
    inT_samps = read_intasks_train(TASK2LANGS,inT_tasks, inT_datadir, split='train')
    outT_samps +=inT_samps
    all_samps = outT_samps
    print('the num of samples for out-tasks and in-tasks: ', len(all_samps))

    random.shuffle(all_samps)
    print('all_samps[0]: ', all_samps[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir,save_filename)
    print('save_path: ', save_path)
    preprocess(all_samps, save_path, in_len=in_len, out_len=out_len)



def read_intasks_train_zeroshot(TASK2LANGS, tasknames, data_dir, split='train'):
    new_samps = []
    for task in tasknames:
        langs = ['en']
        if task in ['xlsum','panx']:
            langs = TASK2LANGS[task]
        for lang in langs:
            file_name = split + '-' + lang + '.json'
            file_path = os.path.join(data_dir, task, file_name)
            read_samples = json.load(open(file_path, 'r'))
            for samp in read_samples:
                input = samp["input"]
                target = samp["target"]
                answers = samp["answers"]
                samp_id = samp["samp_id"]

                new_samp = {"input": input, "target": target, "answers": answers, "samp_id": samp_id}
                new_samps.append(new_samp)
        print('task: %s; samples: %d; langs: %d.'%(task, len(read_samples), len(langs)))
        print()
    print('num of intasks: ',len(tasknames))
    print('num of samples in intasks: ',len(new_samps))
    return new_samps



def generate_big_train_zeroshot(TASK2LANGS,save_dir, inT_tasks,outT_tasks,inT_datadir,outT_datadir,save_file_name,
                       split='train',in_len=512,out_len=64):
    outT_samps = []
    if len(outT_tasks)!=0:
        outT_samps = read_outtasks_train(outT_tasks, outT_datadir, split='train')

    inT_samps = read_intasks_train_zeroshot_fullen(TASK2LANGS, inT_tasks, inT_datadir, split='train')

    print('outT_tasks: ', outT_tasks)
    print('inT_tasks: ',inT_tasks)
    print('len(outT_samps): ', len(outT_samps))
    print('len(inT_samps): ', len(inT_samps))
    all_samps = outT_samps+inT_samps
    print('the num of samples for out-tasks and in-tasks: ', len(all_samps))



    random.shuffle(all_samps)
    print('all_samps[0]: ', all_samps[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir,save_file_name)
    print('save_path: ', save_path)
    preprocess(all_samps, save_path, in_len=in_len, out_len=out_len)



if __name__ == '__main__':
    # build a big training-set based on the target datasets...
    inT_tasks = ['tydiqa', 'xquad', 'mlqa', 'pawsx', 'xnli', 'marc', 'mldoc']
    inT_datadir = './datas/datas_CL'
    save_path = './datas/datas_CL_pt/7datas_train.pt'
    # new_samps = read_intasks_train(TASK2LANGS, inT_tasks, inT_datadir, split='train')
    # preprocess(new_samps, save_path, in_len=512, out_len=64)

    # build a big training-set based on the target datasets and non-target (e.g., mctest dataset) datasets..
    outT_tasks = ['quoref', 'newsqa', 'ropes', 'squad2', 'mctest', 'social_iqa', 'quora', 'rte', 'snli', 'imdb',
                  'amazon_polarity', 'sst2','yahoo_answers_topics', 'dbpedia_14','ag_news']
    outT_datadir ='./datas/datas_expand'
    save_dir = './datas/datas_CL_pt'
    save_filename='polyprompt_7CLdatas15expand.pt'
    generate_big_train(TASK2LANGS, save_dir, inT_tasks, outT_tasks, inT_datadir, outT_datadir, save_filename,
                       split='train', in_len=512, out_len=64)


