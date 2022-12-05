import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

from evaluate import evaluate_save
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, EvalPrediction,HfArgumentParser,Trainer,TrainingArguments,set_seed
logger = logging.getLogger(__name__)


@dataclass
class T2TDataCollator():
    def __call__(self, batch):
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': lm_labels,
            'decoder_attention_mask': decoder_attention_mask
        }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    model_name: Optional[str] = field(
        default=None, metadata={"help": "model name"},)
    model_path: Optional[str] = field(
        default=None, metadata={"help": "model path"},)
    taskname: Optional[str] = field(
        default=None, metadata={"help": "Task names for evaluation"},)
    model_dir: Optional[str] = field(
        default=None, metadata={"help": "Path for model"}, )
    prompt_dir: Optional[str] = field(
        default=None, metadata={"help": "Path for prompt template"},)
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path for train dataset"},)
    eval_batch_size: Optional[int] = field(
        default=50, metadata={"help": "Max input length for the source text"},)
    train_file_path: Optional[str] = field(
        default='./datas/datas_CL_pt/tydiqa/test-en.pt', metadata={"help": "Path for train dataset"},)
    valid_file_path: Optional[str] = field(
        default='./datas/datas_CL_pt/tydiqa/test-en.pt', metadata={"help": "Path for valid dataset"},)
    valid_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Path for valid dataset"},)
    test_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory path for test dataset"},)
    source_max_len: Optional[int] = field(
        default=512, metadata={"help": "Max input length for the source text"},)
    target_max_len: Optional[int] = field(
        default=64, metadata={"help": "Max input length for the target text"},)
    test_source_max_len: Optional[int] = field(
        default=512, metadata={"help": "Max input length for the source text"},)
    test_target_max_len: Optional[int] = field(
        default=64, metadata={"help": "Max input length for the target text"},)



def main():
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    TrainingArguments.prediction_loss_only=True
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.ignore_data_skip = True
    print('model_args, print: ',model_args)
    print('data_args, print: ', data_args)
    print('training_args, print: ', training_args)
    logger.info('model_args: ',model_args)
    logger.info('data_args: ', data_args)
    logger.info('training_args: ', training_args)


    if (os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    tokenizer = MT5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = MT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('train_file_path: ',data_args.train_file_path)
    print('encoder max_len: ', data_args.source_max_len)
    print('target_max_len: ', data_args.target_max_len)
    print('loading data')
    columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
    train_dataset = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    valid_dataset.set_format(type='torch', columns=columns)

    print('train_dataset[0]: ',train_dataset[0])
    print('loading done')


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator()
    )

    # Training
    if training_args.do_train:
        trainer.train(
            resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        logger.info(data_args.taskname)
        tasknames = data_args.taskname.split(',')
        logger.info(tasknames)

        tasks_eval_ress = evaluate_save(tasknames,
                      tokenizer,
                      model,
                      model_args.model_name_or_path,
                      data_args.model_name,
                      data_args.prompt_dir,
                      data_args.eval_batch_size,
                      data_args.data_dir)
    return tasks_eval_ress

if __name__ == '__main__':
    main()

