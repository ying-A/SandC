# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import json
from sklearn.metrics import classification_report

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sketch_label_process import load_and_cache_examples

logger = None
SPECIAL_TOKENS = ["<premise>", "<raw>", "<cf>"]
ATTR_TO_SPECIAL_TOKEN = {
    'additional_special_tokens': ['<premise>', '<raw>', '<cf>']
}
TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]


def set_seed(args):
    """Set the seed for reproducibility."""

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def create_logger(args, filename):
    """Creat the logger to record the trainging process."""

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def cal_loss(labels, logits, attention_mask, uni_w, ske_w):
    """Calculate the token classification loss.
    
    Args:
        uni_w: float. The weight for causal words.
        ske_w: float. The weight for background words.
    """
    w = torch.tensor([uni_w, ske_w]).cuda()
    loss_fct = CrossEntropyLoss(weight=w)
    if attention_mask is not None:
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, 2)
        active_labels = torch.where(
            active_loss, labels.view(-1),
            torch.tensor(loss_fct.ignore_index).type_as(labels))
        loss = loss_fct(active_logits, active_labels)
    else:
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
    return loss


def train(args, model, tokenizer, pad_token_label_id, logger):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = load_and_cache_examples(args,
                                            tokenizer,
                                            pad_token_label_id,
                                            mode="train",
                                            logger=logger)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(
            train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(
            args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(
                args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) //
                                         args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (
            len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info(
            "  Continuing training from checkpoint, will skip to saved global_step"
        )
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    evaluation_loss = 100000000000000
    f1_0 = -100000000000000
    for _ in train_iterator:

        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids

            outputs = model(**inputs)
            labels = batch[3]
            loss = cal_loss(labels=labels,
                            logits=outputs[0],
                            attention_mask=inputs['attention_mask'],
                            uni_w=args.uni_w,
                            ske_w=args.ske_w)

            if args.n_gpu > 1:
                loss = loss.mean(
                )  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                logger.info("global_step:%d, step_tr_loss:%.4f", global_step,
                            loss.item())
                if args.local_rank in [
                        -1, 0
                ] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1
                            and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args,
                                              model,
                                              tokenizer,
                                              pad_token_label_id,
                                              mode="dev")
                        for key, value in results.items():
                            # tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            logger.info(
                                "global_step:%d,eval_key:%s,eval_value:%s",
                                global_step, str(key), "\n" + str(value))

                        if results['f1_0'] > f1_0 or results[
                                'eval_loss'] < evaluation_loss:

                            f1_0 = results['f1_0']
                            evaluation_loss = results['eval_loss']

                            output_dir = os.path.join(
                                args.output_dir,
                                "checkpoint-{}".format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module
                                if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(
                                args,
                                os.path.join(output_dir, "training_args.bin"))
                            logger.info(
                                "Saving best model till now checkpoint to %s",
                                output_dir)

                            torch.save(
                                optimizer.state_dict(),
                                os.path.join(output_dir, "optimizer.pt"))
                            torch.save(
                                scheduler.state_dict(),
                                os.path.join(output_dir, "scheduler.pt"))
                            logger.info(
                                "Saving optimizer and scheduler states to %s",
                                output_dir)

                    # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("global_step:%d, lr:%.4f", global_step,
                                scheduler.get_lr()[0])
                    logger.info("global_step:%d, loss:%.4f", global_step,
                                (tr_loss - logging_loss) / args.logging_steps)
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args,
                                           tokenizer,
                                           pad_token_label_id,
                                           mode=mode,
                                           logger=logger)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(
            eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            labels = batch[3]
            logits = outputs[0]
            tmp_eval_loss = cal_loss(labels=labels,
                                     logits=logits,
                                     attention_mask=inputs['attention_mask'],
                                     uni_w=args.uni_w,
                                     ske_w=args.ske_w)

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean(
                )  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      labels.detach().cpu().numpy(),
                                      axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    out_label_ = []
    preds_ = []
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                preds_list[i].append(preds[i][j])
                out_label_.append(out_label_ids[i][j])
                preds_.append(preds[i][j])
    report = classification_report(out_label_, preds_, output_dict=True)
    scores_0 = report['0']
    scores_1 = report['1']
    results = {
        "eval_loss": eval_loss,
        "report": "\n" + classification_report(out_label_, preds_),
        "precision_0": scores_0['precision'],
        "recall_0": scores_0['recall'],
        "f1_0": scores_0['f1-score']
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="The input data dir.",
    )
    parser.add_argument(
        "--file_paths",
        default={
            'train_skeletons_path':
            "data/train_skeletons_supervised_large.json",
            'dev_skeletons_path': "data/dev_skeletons.json",
            'test_skeletons_path': "data/test_skeletons.json"
        },
        type=dict,
        help="The skeletons files paths.",
    )
    parser.add_argument(
        "--log_path",
        default="data/",
        type=str,
        help="The training log path.",
    )
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="Model type bert",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained bert model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default="sketch_model/",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=300,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict_on_test",
                        action="store_true",
                        help="Whether to run predictions on the test set.")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--keep_accents",
                        action="store_const",
                        const=True,
                        help="Set this flag if model is trained with accents.")
    parser.add_argument(
        "--strip_accents",
        action="store_const",
        const=True,
        help="Set this flag if model is trained without accents.")
    parser.add_argument("--use_fast",
                        action="store_const",
                        const=True,
                        help="Set this flag to use fast tokenization.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=5.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps",
                        default=2000,
                        type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps",
                        type=int,
                        default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir",
                        action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help=
        "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip",
                        type=str,
                        default="",
                        help="For distant debugging.")
    parser.add_argument("--server_port",
                        type=str,
                        default="",
                        help="For distant debugging.")

    parser.add_argument("--unique_flag",
                        type=str,
                        default="",
                        help="type",
                        required=True)  # 2080 5050 8020
    parser.add_argument("--uni_w",
                        type=float,
                        default=0.5,
                        help="unique(causal) words weight",
                        required=True)
    parser.add_argument("--ske_w",
                        type=float,
                        default=0.5,
                        help="skeleton(background) words weight",
                        required=True)
    parser.add_argument("--pred_dir_path",
                        type=str,
                        default="sketch_pred_results/",
                        help="predicted skeletons results file path")
    args = parser.parse_args()

    global logger
    if args.do_train:
        filename = args.log_path + args.unique_flag + "_sketch_train.log"
    elif args.do_predict_on_test:
        filename = args.log_path + args.unique_flag + "_sketch_predict.log"
    logger = create_logger(args, filename)

    args.output_dir = args.unique_flag + "_" + args.output_dir

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, )
    tokenizer_args = {
        k: v
        for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS
    }
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        **tokenizer_args,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    orig_num_tokens = tokenizer.vocab_size
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens +
                                      num_added_tokens)

    if args.local_rank == 0:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args,
                                     model,
                                     tokenizer,
                                     pad_token_label_id,
                                     logger=logger)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1
                          or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, "module") else model
                         )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir,
                                                  **tokenizer_args)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME,
                              recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForTokenClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args,
                                 model,
                                 tokenizer,
                                 pad_token_label_id,
                                 mode="dev",
                                 prefix=global_step)
            if global_step:
                result = {
                    "{}_{}".format(global_step, k): v
                    for k, v in result.items()
                }
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))

    if args.do_predict_on_test and args.local_rank in [-1, 0]:

        if not os.path.exists(args.pred_dir_path):
            os.makedirs(args.pred_dir_path)

        if args.unique_flag == "2080":
            args.output_dir = "2080_sketch_model/checkpoint-12500/"
        if args.unique_flag == "5050":
            args.output_dir = "5050_sketch_model/checkpoint-10000/"
        if args.unique_flag == "8020":
            args.output_dir = "8020_sketch_model/checkpoint-6500/"

        tokenizer = AutoTokenizer.from_pretrained(args.output_dir,
                                                  **tokenizer_args)
        model = AutoModelForTokenClassification.from_pretrained(
            args.output_dir)
        model.to(args.device)

        result, predictions = evaluate(args,
                                       model,
                                       tokenizer,
                                       pad_token_label_id,
                                       mode="test")
        # Save results
        output_test_results_file = os.path.join(args.output_dir,
                                                "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir,
                                                    "test_predictions.txt")
        output_test_predictions_json = os.path.join(
            args.pred_dir_path,
            args.unique_flag + "_test_skeletons_predictions.json")
        f = open("data/test_skeletons.json", "r")
        all_data = json.load(f)

        all_pred = []
        g = open(output_test_predictions_json, "w")

        with open(output_test_predictions_file, "w") as writer:
            i = 0
            for item in all_data:
                pre = item['premise']
                con = item['raw_condition']
                c_con = item['counterfactual_condition']
                ske = item['raw_skeletons_endings'][0]
                c_ske = item['counterfactual_skeletons_endings'][0]
                end = item['ending']
                c_end = item['c_ending']
                end_words = end.strip().split()
                end_words_cf = c_end.strip().split()
                words_raw = ske.strip().split()
                labels_raw = item['label_raw'][1][1:]
                words_cf = c_ske.strip().split()
                labels_cf = item['label_cf'][1][1:]

                assert len(labels_raw) == len(predictions[i])
                pred_ske_raw = ""
                for word, pred in zip(end_words, predictions[i]):
                    if pred == 1:
                        pred_ske_raw = (pred_ske_raw + " " + word)
                    else:
                        # append "__" and merge the consecutive blanks into one blank
                        if not pred_ske_raw.endswith(" __ "):
                            pred_ske_raw = pred_ske_raw + " __ "
                pred_ske_raw = pred_ske_raw.strip()

                words_raw = " ".join([w for w in words_raw])
                labels_raw = " ".join([str(w) for w in labels_raw])
                preds_raw = " ".join([str(w) for w in predictions[i]])

                i += 1
                assert len(labels_cf) == len(predictions[i])
                pred_ske_cf = ""
                for word, pred in zip(end_words_cf, predictions[i]):
                    if pred == 1:
                        pred_ske_cf = (pred_ske_cf + " " + word)
                    else:
                        # append "__" and merge the consecutive blanks into one blank
                        if not pred_ske_cf.endswith(" __ "):
                            pred_ske_cf = pred_ske_cf + " __ "
                pred_ske_cf = pred_ske_cf.strip()

                words_cf = " ".join([w for w in words_cf])
                labels_cf = " ".join([str(w) for w in labels_cf])
                preds_cf = " ".join([str(w) for w in predictions[i]])
                i += 1

                writer.write(words_raw + "\n")
                writer.write(labels_raw + "\n")
                writer.write(preds_raw + "\n")

                writer.write(words_cf + "\n")
                writer.write(labels_cf + "\n")
                writer.write(preds_cf + "\n")

                res = {}
                res['premise'] = pre
                res['raw_condition'] = con
                res['ending'] = end
                res['gt_raw_skeletons_ending'] = ske
                # pred for next customize gpt2 preprocess
                res['raw_skeletons_endings'] = [pred_ske_cf]
                res['counterfactual_condition'] = c_con
                res['c_ending'] = c_end
                res['gt_counterfactual_skeletons_ending'] = c_ske
                # pred for next customize gpt2 preprocess
                res['counterfactual_skeletons_endings'] = [pred_ske_raw]
                all_pred.append(res)

        json.dump(all_pred, g)

    return results


if __name__ == "__main__":
    main()
