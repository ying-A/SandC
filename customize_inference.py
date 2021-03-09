import os
import json
import argparse
from tqdm import tqdm

import torch
# from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from customize_data_process import SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN, PAD, PAD_ID
from sketch_main import set_seed


def setup_test_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1', type=str, required=False)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--customize_model_path',
                        default='_customize_model/best_eval_model/',
                        type=str,
                        required=False)
    parser.add_argument('--sketch_pred_results_path',
                        default='sketch_pred_results/random_test_skes.json',
                        type=str,
                        required=False)
    parser.add_argument('--save_results_dir',
                        default="customize_pred_results/",
                        type=str,
                        required=False)
    parser.add_argument('--customize_pred_results_name',
                        default="random_and_c.json",
                        type=str,
                        required=False)
    parser.add_argument('--batch_size', default=8, type=int, required=False)
    parser.add_argument('--pretrained_model',
                        default='gpt2-medium',
                        type=str,
                        required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument('--temperature',
                        default=0.7,
                        type=float,
                        required=False)
    parser.add_argument(
        '--repetition_penalty',
        default=1.0,
        type=float,
        required=False,
    )
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--use_lcs_skeletons",
        action="store_true",
        help="Bool, whether use LCS skeletons to generate counterfactual endings.")

    return parser.parse_args()


def main():
    args = setup_test_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    print('using device:{}'.format(device))
    if args.seed:
        set_seed(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
    model = GPT2LMHeadModel.from_pretrained(args.customize_model_path)
    model.to(device)

    global PAD_ID
    PAD_ID = tokenizer.convert_tokens_to_ids(PAD)
    multi_gpu = False
    model.eval()
    print("loading test data")
    with open(args.sketch_pred_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("there are {} story in raw test dataset".format(len(data)))
    premise, condition, ending, skeleton, c_condition, c_ending, c_skeleton, bos, eos = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])

    def tk2id(tokenizer, text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    if not os.path.exists(args.save_results_dir):
        os.makedirs(args.save_results_dir)
    g = open(args.save_results_dir + args.customize_pred_results_name,
             "w",
             encoding="utf-8")
    gen_results = []

    i = -1
    with torch.no_grad():
        for story_index, story in enumerate(tqdm(data)):
            i += 1
            if i % 3 != 0:
                continue
            pre = story['premise']
            con = story['raw_condition']
            c_con = story['counterfactual_condition']
            ske = story['gt_raw_skeletons_ending']
            c_ske = story['gt_counterfactual_skeletons_ending']
            pred_ske = story['raw_skeletons_endings'][0]
            pred_c_ske = story['counterfactual_skeletons_endings'][0]
            end = story['ending']
            c_end = story['c_ending']

            if args.use_lcs_skeletons:
                pred_c_ske = c_ske

            # pre_ccon_pred_cske: premise + counterfactual condition + predicted counterfactual skeleton
            pre_ccon_pred_cske = [bos] + [premise] + tk2id(tokenizer, pre) + [
                c_condition
            ] + tk2id(tokenizer, c_con) + [c_skeleton] + tk2id(
                tokenizer, pred_c_ske) + [c_ending]
            pre_ccon_pred_cske = torch.tensor(pre_ccon_pred_cske).unsqueeze(
                0).cuda()

            pc_output_sequences = model.generate(
                input_ids=pre_ccon_pred_cske,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=1,
            )

            pgenerated_sequence = pc_output_sequences[0].tolist()
            pc_text = tokenizer.decode(pgenerated_sequence)
            pc_text = pc_text[len(
                tokenizer.decode(pre_ccon_pred_cske[0],
                                 clean_up_tokenization_spaces=True)):]
            pc_text = pc_text[:pc_text.find("<eos>")]

            res = {}
            res['premise'] = pre
            res['condition'] = con
            res['ending'] = end
            res['cf_condition'] = c_con
            res['cf_ending'] = c_end
            res['cf_skeleton'] = c_ske
            res['cf_pred_skeleton'] = pred_c_ske
            res['cf_pred_gen_ending'] = pc_text

            gen_results.append(res)

    json.dump(gen_results, g)


if __name__ == '__main__':
    main()
