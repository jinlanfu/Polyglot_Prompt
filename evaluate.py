import logging
import os
import sys
import numpy as np
import torch
from tqdm.auto import tqdm
from utils.squad_eval import evaluate as evaluate_squad
from utils.mlqa_eval import evaluate as evaluate_mlqa
from utils.tc_eval import evaluate as evaluate_tc
import json
logger = logging.getLogger(__name__)

TASK2LANGS = {
"pawsx": "en,es".split(","),
"tydiqa": "en,bn".split(","),
# "mlqa": "en,es".split(","),
    "pawsx": "de,en,es,fr,ja,ko,zh".split(","),
    "xnli": "ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh".split(","),
    "panx": "en,ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu".split(
        ","),
    "udpos": "af,ar,bg,de,el,en,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,kk,ko,mr,nl,pt,ru,ta,te,th,tl,tr,ur,vi,yo,zh".split(
        ","),
    "bucc2018": "de,fr,ru,zh".split(","),
    "tatoeba": "ar,he,vi,id,jv,tl,eu,ml,ta,te,af,nl,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,zh,kk,tr,et,fi,hu".split(
        ","),
    "xquad": "en,es,de,el,ru,tr,ar,vi,th,zh,hi".split(","),
    "mlqa": "en,es,de,ar,hi,vi,zh".split(","),
    # "tydiqa": "en,ar,bn,fi,id,ko,ru,sw,te".split(","),
    "marc": "en,de,es,fr,ja,zh".split(","),
    "mldoc": "ja,en,de,es,fr,zh,it,ru".split(","),
}


def print_qa_res(taskname, results,model_path='null'):
    print('-------------------------------------')
    print('model: %s; eval task: %s' % (model_path, taskname))
    print('lang, f1, em')
    ems = []
    f1s = []
    for lang, res in results.items():
        em = res['exact_match']
        f1 = res['f1']
        ems.append(em)
        f1s.append(f1)
        print(lang, f1, em)
    avg_em = np.mean(ems)
    avg_f1 = np.mean(f1s)
    print('%s, %f, %f' % ('avg', avg_f1, avg_em))

    return  avg_f1

def print_tc_res(taskname, results,model_path='null'):
    print('model: %s; eval task: %s' % (model_path, taskname))
    print('lang, f1, em')
    accs = []
    for lang, res in results.items():
        acc = res['acc']
        accs.append(acc)
        print(lang, acc)
    avg_acc = np.mean(accs)
    print('%s %f' % ('avg', avg_acc))

    return avg_acc

def model_evaluate(save_dir,taskname,tokenizer,model,prompt_dir,eval_batch_size,data_dir):
    # get the test datasets..
    data_dir = os.path.join(data_dir,taskname)
    results = {}
    langs = TASK2LANGS[taskname]
    print('##Evaluate data_dir: ', data_dir)
    for lang in langs:
        file_name = 'test-' + lang + '.pt'
        file_path = os.path.join(data_dir, file_name)
        print('##Evaluate file_path: ', file_path)
        columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
        valid_dataset = torch.load(file_path)
        valid_dataset.set_format(type='torch', columns=columns)
        dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size)

        answers = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                outs = model.generate(input_ids=batch['input_ids'].cuda(),
                                      attention_mask=batch['attention_mask'].cuda(),
                                      max_length=64,
                                      early_stopping=True
                                      )

                outs = [tokenizer.decode(ids) for ids in outs]
                answers.extend(outs)

        predictions = []
        references = valid_dataset['answers']
        candidates_list = []
        if taskname in ['pawsx', 'xnli', 'marc', 'mldoc']:
            prompt_path= ''
            if os.path.isdir(prompt_dir):
                prompt_filename = taskname + '_ansPrompt.json'
                prompt_path = os.path.join(prompt_dir, prompt_filename)
            elif os.path.isfile(prompt_dir):
                prompt_path = prompt_dir
            else:
                print('prompt path error! prompt_path: ',prompt_path)
            prompt_dic = json.load(open(prompt_path))
            print('evaluate prompt_path: ', prompt_path)
            print('prompt_dic: ', prompt_dic)
            candidates_list = prompt_dic[lang]["answers"]

        print('references[0]: ', references[0])
        references_ids = []
        for ref, pred in zip(valid_dataset, answers):
            pred_words = pred.split(' ')
            pred_keep = []
            for pred_word in pred_words:
                if pred_word != '<pad>':
                    pred_keep.append(pred_word)
            pred_text = ' '.join(pred_keep).replace('</s>', '').strip()
            predictions.append(pred_text)

            # begin{get reference answer from target_ids...}
            ans_len = ref['target_attention_mask'].sum(dim=0)
            ans_ids = ref['target_ids'][:ans_len]
            ans_text = tokenizer.decode(ans_ids).replace('</s>', '').strip()
            references_ids.append([ans_text])
            # end{get reference answer from target_ids...}

        print('predictions, len: ', len(predictions))
        print('references, len: ', len(references))
        print()


        # Evaluate the overall results...
        save_perm = 0.0
        res = {}
        if taskname in ['tydiqa', 'xquad']:
            print('evaluate squad')
            res = evaluate_squad(references_ids, references, predictions)
        elif taskname == 'mlqa':
            print('evaluate mlqa')
            res = evaluate_mlqa(references, predictions, lang)
        elif taskname in ['pawsx', 'xnli', 'marc', 'mldoc']:
            print('evaluate tc')
            res = evaluate_tc(references, candidates_list, predictions)
        else:
            print('The evaluated TASK isnt exist.')

        # Print the overall results...
        if taskname in ['tydiqa', 'xquad', 'mlqa']:
            print('exact_match: ', res['exact_match'])
            print('f1: ', res['f1'])
            print()
            save_perm = str(int(res['f1'] * 100))
        elif taskname in ['pawsx', 'xnli', 'marc', 'mldoc']:
            print(res)
            print()
            save_perm = str(int(res['acc'] * 100))
        else:
            print('The Printing TASK isnt exist.')
            print()

        results[lang] = res
        save_result(save_dir, taskname, valid_dataset, predictions, lang, save_perm)

    if taskname in ['tydiqa', 'xquad', 'mlqa']:
        print_qa_res(taskname, results)

    elif taskname in ['pawsx', 'xnli', 'marc', 'mldoc']:
        print_tc_res(taskname, results)



    return results

def save_result(save_dir,taskname,valid_dataset, predictions,lang,save_perm):
    print('save_dir: ',save_dir)
    print('taskname: ', taskname)
    print('lang: ', lang)
    save_filename = 'test-'+ lang + '_'+save_perm+ '.json'
    save_path = os.path.join(save_dir, save_filename)
    save_results = []
    answers = valid_dataset["answers"]
    input = valid_dataset["input"]

    for inp,gans,pans in zip(input,answers,predictions):
        res = {
            "input": inp,
            "answers": gans,
            "pred_answer":pans
        }
        save_results.append(res)
    with open(save_path, 'w') as f:
        print('save result: ',save_path)
        json.dump(save_results, f, ensure_ascii=False, indent=4)


def evaluate_save(tasknames,
                tokenizer,
                model,
                model_name_or_path,
                model_name,
                prompt_dir,
                eval_batch_size,
                data_dir):

    tasks_eval_ress = {}
    for taskname in tasknames:
        print('###Eval model_name_or_path: ',model_name_or_path)
        save_dir = "./results/"+ model_name + '_'+taskname
        print('###save_dir: ',save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        eval_ress = model_evaluate(save_dir,taskname, tokenizer, model,prompt_dir=prompt_dir,eval_batch_size=eval_batch_size,data_dir=data_dir)
        # print('eval_ress: ',eval_ress)
        tasks_eval_ress[taskname] = eval_ress
    print('eval model_name_or_path: ',model_name_or_path)
    print('tasks_eval_ress: ',tasks_eval_ress)

    return tasks_eval_ress
