import os
from os.path import join
import numpy as np
import argparse
from datetime import datetime

import torch
from torch.nn import DataParallel, CrossEntropyLoss
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup, AdamW

from customize_data_process import MyDataset, collate_fn, preprocess_customize_data, SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN, PAD, PAD_ID
from sketch_main import set_seed, create_logger
logger = None


def calculate_loss_and_accuracy(outputs, lens, labels, device):
    """Calculate the generation loss and token accuracy for GPT2."""
    def chose_inner_mask(lengths_left, lengths_full, device, maxlen=None):
        """Get the mask matrix where only the tokens of the target sentence are 1,
        the tokens of the source sentence and the padding part are 0.

        Args:
            lengths_left, the lengths of the source part of sentences,
            lengths_full, the lengths of the full sentences,
            maxlen, the setted max length.
        Example:
            sentences, [s,s,s,t,t,t,t]
                       [s,s,s,s,s,s,s,t,t,t]
                       [s,s,s,s,t,t]
            lengths_left, tensor([3, 7, 4], device='cuda:0')
            lengths_full, tensor([7, 10, 6], device='cuda:0')
            then the returned mask matrix is,
                tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], 
                        device='cuda:0')
        """
        if maxlen is None:
            maxlen = lengths_full.max()
        row_vector = torch.arange(0, maxlen, 1).to(device)
        matrix_all = torch.unsqueeze(lengths_full, dim=-1)
        matrix_left = torch.unsqueeze(lengths_left, dim=-1)
        mask_all = (row_vector < matrix_all).type(torch.cuda.LongTensor)
        mask_left = (~(row_vector < matrix_left)).type(torch.cuda.LongTensor)
        mask = mask_all * mask_left
        return mask

    lengths_left = torch.stack([x[0] - 1 for x in lens])
    lengths_full = torch.stack([x[1] - 1 for x in lens])
    maxlen = lens[0][-1] - 1
    mask = chose_inner_mask(lengths_left, lengths_full, device, maxlen)

    logits = outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous() * mask.to(device)

    loss_fct = CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(PAD_ID)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets
    return loss, accuracy


def create_model(args, tokenizer):
    """Create the customize generation model."""
    if args.pretrained_model:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens +
                                      num_added_tokens)
    logger.info('model config:\n{}'.format(model.config.to_json_string()))
    return model, model.config.to_dict().get("n_ctx")


def train(model, device, train_list, train_lens_list, dev_list, dev_lens_list,
          multi_gpu, args):
    """Train the customize generation model.

    Args:
        model: The model.
        train_list: List of the training token ids.
        train_lens_list: List of the length of the training source and full sentences.
        dev_list: List of the dev token ids.
        dev_lens_list: List of the length of the dev source and full sentences.
        multi_gpu: Whether use multiple gpus to train.
        args: The args.
    """
    train_dataset = MyDataset(train_list, train_lens_list, args.batch_size)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    model.train()
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size /
                      args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps)
    logger.info('starting training')
    running_loss = 0
    overall_step = 0
    oom_time = 0
    eval_loss = 10000000000000

    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        for batch_idx, (input_ids, lens) in enumerate(train_dataloader):
            input_ids = input_ids.to(device)
            lens = lens.to(device)
            try:
                outputs = model.forward(input_ids=input_ids)
                loss, accuracy = calculate_loss_and_accuracy(outputs,
                                                             lens,
                                                             labels=input_ids,
                                                             device=device)
                if multi_gpu:
                    loss = loss.mean()
                    accuracy = accuracy.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    accuracy = accuracy / args.gradient_accumulation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    overall_step += 1
                    logger.info(
                        "batch {} of epoch {}, loss {}, accuracy {}".format(
                            batch_idx + 1, epoch + 1, loss, accuracy))
                    if (overall_step + 1) % args.log_step == 0:
                        if args.evaluate_during_training:
                            print("## EVAL DURING TRAINING ##")
                            tmp_eval_loss = evaluate(model, device, dev_list,
                                                     dev_lens_list, multi_gpu,
                                                     args)
                            if tmp_eval_loss < eval_loss:
                                eval_loss = tmp_eval_loss
                                model_path = join(args.customize_model_path,
                                                  'best_eval_model')
                                if not os.path.exists(model_path):
                                    os.mkdir(model_path)
                                model_to_save = model.module if hasattr(
                                    model, 'module') else model
                                model_to_save.save_pretrained(model_path)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(
                        oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
        logger.info('saving model for epoch {}'.format(epoch + 1))

        model_path = join(args.customize_model_path,
                          'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time -
                                                    epoch_start_time))
    logger.info('training finished')


def evaluate(model, device, td_list, td_lens_list, multi_gpu, args):
    """Evaluate the customize model.

    Args:
        model: The model.
        td_list: List of the test or dev token ids.
        td_lens_list: List of the length of the source and full sentences.
        multi_gpu: Whether use multiple gpus.
        args: The args.
    """
    logger.info("start evaluating model")
    model.eval()
    logger.info('starting evaluating')
    td_dataset = MyDataset(td_list, td_lens_list, args.batch_size)
    td_dataloader = DataLoader(td_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               collate_fn=collate_fn)
    all_td_loss = []
    with torch.no_grad():
        for batch_idx, (input_ids, lens) in enumerate(td_dataloader):
            input_ids = input_ids.to(device)
            lens = lens.to(device)
            outputs = model.forward(input_ids=input_ids)
            loss, accuracy = calculate_loss_and_accuracy(outputs,
                                                         lens,
                                                         labels=input_ids,
                                                         device=device)
            if multi_gpu:
                loss = loss.mean()
                accuracy = accuracy.mean()
            if args.gradient_accumulation > 1:
                loss = loss / args.gradient_accumulation
                accuracy = accuracy / args.gradient_accumulation
            all_td_loss.append(loss.item())
            logger.info("evaluate batch {} ,loss {} ,accuracy {}".format(
                batch_idx, loss, accuracy))
        all_td_loss = np.array(all_td_loss).mean()
        logger.info("finishing evaluating, eval_loss:{}".format(all_td_loss))
    return all_td_loss


def setup_train_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument(
        '--train_raw_path',
        default='data/merge_train_skeletons_supervised_large.json',
        type=str,
        required=False)
    parser.add_argument(
        '--train_tokenized_path',
        default='data/merge_train_tokenized_supervised_large.txt',
        type=str,
        required=False)
    parser.add_argument('--train_lens_path',
                        default='data/merge_train_lens_supervised_large.txt',
                        type=str,
                        required=False)
    parser.add_argument('--dev_raw_path',
                        default='data/merge_dev_skeletons.json',
                        type=str,
                        required=False)
    parser.add_argument('--dev_tokenized_path',
                        default='data/merge_dev_tokenized.txt',
                        type=str,
                        required=False)
    parser.add_argument('--dev_lens_path',
                        default='data/merge_dev_lens.txt',
                        type=str,
                        required=False)
    parser.add_argument('--log_path',
                        default='data/',
                        type=str,
                        required=False)
    parser.add_argument("--unique_flag",
                        type=str,
                        default="without_aug",
                        required=True,
                        help="The flag for distinguish different settings")
    parser.add_argument('--raw',
                        action='store_true',
                        help="Use this arg in the first run.")
    parser.add_argument('--epochs', default=10, type=int, required=False)
    parser.add_argument('--batch_size', default=8, type=int, required=False)
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False)
    parser.add_argument('--warmup_steps',
                        default=2000,
                        type=int,
                        required=False)
    parser.add_argument('--log_step', default=500, type=int, required=False)
    parser.add_argument('--gradient_accumulation',
                        default=1,
                        type=int,
                        required=False)
    parser.add_argument('--max_grad_norm',
                        default=1.0,
                        type=float,
                        required=False)
    parser.add_argument('--customize_model_path',
                        default='_customize_model/',
                        type=str,
                        help="Customize model saved path.",
                        required=False)
    parser.add_argument('--pretrained_model',
                        default='gpt2-medium',
                        type=str,
                        required=False)
    parser.add_argument('--writer_dir',
                        default='tensorboard_summary/',
                        type=str,
                        required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Bool, whether do evaluation in the training stage.")
    return parser.parse_args()


def main():
    args = setup_train_args()
    global logger
    logname = args.log_path + args.unique_flag + "_customize_train.log"
    logger = create_logger(args, logname)
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    if args.seed:
        set_seed(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    global PAD_ID
    PAD_ID = tokenizer.convert_tokens_to_ids(PAD)
    args.customize_model_path = args.unique_flag + args.customize_model_path
    if not os.path.exists(args.customize_model_path):
        os.mkdir(args.customize_model_path)
    model, n_ctx = create_model(args, tokenizer)
    model.to(device)
    if args.raw:
        preprocess_customize_data("train", args, tokenizer, n_ctx)
        preprocess_customize_data("dev", args, tokenizer, n_ctx)
    multi_gpu = False
    if args.cuda and torch.cuda.device_count() > 1:
        logger.info("Let's use GPUs to train")
        model = DataParallel(
            model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))
    logger.info("loading traing data")

    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        train_data = f.read()
    with open(args.train_lens_path, "r", encoding="utf8") as g:
        train_lens_data = g.read()
    train_list = train_data.split("\n")
    train_lens_list = train_lens_data.split("\n")

    with open(args.dev_tokenized_path, "r", encoding="utf8") as f:
        dev_data = f.read()
    with open(args.dev_lens_path, "r", encoding="utf8") as g:
        dev_lens_data = g.read()
    dev_list = dev_data.split("\n")
    dev_lens_list = dev_lens_data.split("\n")

    train(model, device, train_list, train_lens_list, dev_list, dev_lens_list,
          multi_gpu, args)


if __name__ == '__main__':
    main()
