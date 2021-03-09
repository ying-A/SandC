import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import logging

SPECIAL_TOKENS = [
    "<premise>", "<condition>", "<ending>", "<skeleton>", "c_condition",
    "c_ending", "c_skeleton", "<bos>", "<eos>", "<pad>"
]
ATTR_TO_SPECIAL_TOKEN = {
    'bos_token': '<bos>',
    'eos_token': '<eos>',
    'pad_token': '<pad>',
    'additional_special_tokens': [
        '<premise>', '<condition>', '<ending>', '<c_condition>', '<c_ending>',
        '<skeleton>', 'c_skeleton'
    ]
}
PAD = SPECIAL_TOKENS[-1]
PAD_ID = 0

logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    def __init__(self, data_list, data_lens_list, batch_size):
        data_size = len(data_list)
        data_lens_size = len(data_lens_list)
        assert data_size == data_lens_size
        n = data_size % batch_size
        if n != 0:
            data_list = data_list[:-n]
        self.data_list = data_list
        self.data_lens_list = data_lens_list

    def __getitem__(self, index):
        input_ids = self.data_list[index].strip()
        input_ids = [int(token_id) for token_id in input_ids.split()]
        lens = self.data_lens_list[index].strip()
        lens = [int(len_) for len_ in lens.split()]
        return (input_ids, lens)

    def __len__(self):
        return len(self.data_list)


def collate_fn(batch):
    global PAD_ID
    input_ids = []
    lens = []
    btc_size = len(batch)
    max_input_len = 0
    for btc_idx in range(btc_size):
        if max_input_len < len(batch[btc_idx][0]):
            max_input_len = len(batch[btc_idx][0])
    for btc_idx in range(btc_size):
        input_len = batch[btc_idx][1][1]
        input_ids.append(batch[btc_idx][0])
        input_ids[btc_idx].extend([PAD_ID] * (max_input_len - input_len))
        lens.append(batch[btc_idx][1] + [max_input_len])
    return torch.tensor(input_ids,
                        dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def preprocess_customize_data(data_name, args, tokenizer, n_ctx):
    """Process the raw string data into tokenized ids and restore the lengths.
       Each line restore two lengths, the first one is the length of the source sentence for GPT (like encoder),
       the second one is the length of the full sentence (source + target) for GPT.
    """
    def tk2id(tokenizer, text):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    premise, condition, ending, skeleton, c_condition, c_ending, c_skeleton, bos, eos = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])

    if data_name == "test":
        data_raw_path = args.test_raw_path
        data_tokenized_path = args.test_tokenized_path
        data_lens_path = args.test_lens_path
    elif data_name == "dev":
        data_raw_path = args.dev_raw_path
        data_tokenized_path = args.dev_tokenized_path
        data_lens_path = args.dev_lens_path
    elif data_name == "train":
        data_raw_path = args.train_raw_path
        data_tokenized_path = args.train_tokenized_path
        data_lens_path = args.train_lens_path

    logger.info(
        "tokenizing raw data,raw data path:{}, token output path:{}，token lens path:{}"
        .format(data_raw_path, data_tokenized_path, data_lens_path))
    with open(data_raw_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info("there are {} story in raw {} dataset".format(
        len(data), data_name))

    g = open(data_lens_path, "w", encoding="utf-8")

    with open(data_tokenized_path, "w", encoding="utf-8") as f:
        for story_index, story in enumerate(tqdm(data)):
            pre = story['premise']
            con = story['raw_condition']
            c_con = story['counterfactual_condition']
            skes = story['raw_skeletons_endings']
            c_skes = story['counterfactual_skeletons_endings']
            end = story['ending']
            c_end = story['c_ending']

            for ske in skes:
                pre_con_ske = [bos] + [premise] + tk2id(tokenizer, pre) + [
                    condition
                ] + tk2id(tokenizer, con) + [skeleton] + tk2id(tokenizer, ske)
                end_ = [ending] + tk2id(tokenizer, end) + [eos]
                story_ids_raw = pre_con_ske + end_
                len_a = len(pre_con_ske)
                len_b = len(end_)
                len_all_raw = len_a + len_b
                story_ids_raw = story_ids_raw[:n_ctx]
                for story_id_raw in story_ids_raw:
                    f.write(str(story_id_raw) + ' ')
                if story_index < len(data) - 1:
                    f.write("\n")
                g.write(str(len_a) + " " + str(len_all_raw))
                if story_index < len(data) - 1:
                    g.write("\n")

            for c_ske in c_skes:
                pre_ccon_cske = [bos] + [premise] + tk2id(tokenizer, pre) + [
                    c_condition
                ] + tk2id(tokenizer, c_con) + [c_skeleton] + tk2id(
                    tokenizer, c_ske)
                cend = [c_ending] + tk2id(tokenizer, c_end) + [eos]
                story_ids_cf = pre_ccon_cske + cend
                len_c = len(pre_ccon_cske)
                len_d = len(cend)
                len_all_cf = len_c + len_d
                story_ids_cf = story_ids_cf[:n_ctx]
                for story_id_cf in story_ids_cf:
                    f.write(str(story_id_cf) + ' ')
                if story_index < len(data) - 1:
                    f.write("\n")
                g.write(str(len_c) + " " + str(len_all_cf))
                if story_index < len(data) - 1:
                    g.write("\n")

    logger.info(
        "finish preprocessing raw skeleton {} data,the result is stored in {},{}"
        .format(data_name, data_tokenized_path, data_lens_path))
