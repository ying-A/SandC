import os
import json
import jsonlines
import random
import numpy as np
from tqdm import tqdm
from transformers.tokenization_bert import BasicTokenizer

PAD_LABEL_ID = -100


def basic_tokenize(string):
    """Use Bert BasicTokenizer as the tokenizer."""

    return BasicTokenizer().tokenize(string)


def get_vocab(paths):
    """Get the vocab from the training file."""

    if os.path.isfile(paths['train_vocab_path']):
        g = open(paths['train_vocab_path'], 'r', encoding='utf-8')
        voc = [s.strip() for s in g.readlines()]
    else:
        vocab = {}
        g = open(paths['train_vocab_path'], 'w', encoding='utf-8')
        with open(paths['train_raw_path'], 'r', encoding='utf-8') as f:
            json_data = jsonlines.Reader(f)
            train_data = []
            for item in json_data:
                train_data.append(item)
            for story_index, story in enumerate(tqdm(train_data)):
                pre = basic_tokenize(story['premise'])
                con = basic_tokenize(story['initial'])
                end = basic_tokenize(story['original_ending'])
                c_con = basic_tokenize(story['counterfactual'])
                c_end = basic_tokenize(story['edited_ending'][0] + " " +
                                       story['edited_ending'][1] + " " +
                                       story['edited_ending'][2])
                a = pre + con + end + c_con + c_end
                for wd in a:
                    if wd in vocab:
                        vocab[wd] += 1
                    else:
                        vocab[wd] = 1
            voc = [
                v[0] for v in sorted(
                    vocab.items(), key=lambda item: item[1], reverse=True)
            ]
        for x in voc:
            g.write(x + "\n")
    return voc


def bottom_up_dp_lcs(str_a, str_b, do_merge, mask_rate, replace_rate, vocab,
                     mode):
    """Get LCS skeletons using the bottom up DP algorithm.

    Args:
        str_a, str_b: string. Two raw strings.
        do_merge: bool. Wether merge the consecutive blanks into one blank.
        mask_rate: float. The rate of background words to be masked.
        replace_rate: float. The rate of background words to be replaced with the random words.
        vocab: list. The vocab for getting random words.
        mode: string. "train", "dev" or "test".
    """

    str_a = basic_tokenize(str_a)
    str_b = basic_tokenize(str_b)
    str_a.insert(0, "_str_")
    str_b.insert(0, "_str_")
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max([dp[i - 1][j], dp[i][j - 1]])

    i, j = len(str_a), len(str_b)
    LCS_a = ""
    LCS_b = ""
    a_unique = []
    b_unique = []
    while i > 0 and j > 0:
        if str_a[i - 1] == str_b[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            LCS_a = str_a[i - 1] + " " + LCS_a
            LCS_b = str_a[i - 1] + " " + LCS_b
            i, j = i - 1, j - 1
            continue
        if dp[i][j] == dp[i - 1][j]:
            i, j = i - 1, j
            if LCS_a.startswith(" __ ") and do_merge:
                LCS_a = LCS_a
            else:
                LCS_a = " __ " + LCS_a
            a_unique.append((i - 1, str_a[i]))
            continue
        if dp[i][j] == dp[i][j - 1]:
            i, j = i, j - 1
            if LCS_b.startswith(" __ ") and do_merge:
                LCS_b = LCS_b
            else:
                LCS_b = " __ " + LCS_b
            b_unique.append((j - 1, str_b[j]))
            continue
    a_unique = a_unique[::-1]
    b_unique = b_unique[::-1]
    LCS_a = LCS_a[6:]
    LCS_b = LCS_b[6:]

    if not do_merge:
        lcsas = [LCS_a]
        lcsbs = [LCS_b]
    else:
        if mode == "train_aug":
            split_LCS_a = LCS_a.split()
            split_LCS_b = LCS_b.split()
            lcsas = [LCS_a] + get_aug_skeletons(split_LCS_a, mask_rate,
                                                replace_rate, vocab)
            lcsbs = [LCS_b] + get_aug_skeletons(split_LCS_b, mask_rate,
                                                replace_rate, vocab)
        else:
            lcsas = [LCS_a]
            lcsbs = [LCS_b]

    return lcsas, lcsbs, a_unique, b_unique, str_a[1:], str_b[1:]


def get_aug_skeletons(s, mask_rate, replace_rate, vocab):
    """ This function is used for generating augmented skeletons.

    Args:
        s: list. The raw skeleton.
        mask_rate: The rate of background words to be masked.
        replace_rate: The rate of background words to be replaced with random words.
        vocab: The vocab for getting random words.
    """
    # a for mask, b for replace, c for shuffle
    a = s.copy()
    b = s.copy()
    c = s.copy()
    # mask
    for i in range(len(s)):
        if s[i] != "__":
            rand = random.random()
            if rand < mask_rate:
                a[i] = "__"
    # replace
    for i in range(len(s)):
        if s[i] != "__":
            rand = random.random()
            if rand < replace_rate:
                rand_word_id = np.random.randint(0, len(vocab))
                b[i] = vocab[rand_word_id]
    # shuffle
    no_blank_word_id_in_sen = []
    no_blank_word_in_sen = []
    for i in range(len(s)):
        if s[i] != "__":
            no_blank_word_id_in_sen.append(i)
            no_blank_word_in_sen.append(s[i])
    random.shuffle(no_blank_word_in_sen)
    j = 0
    for i in range(len(s)):
        if s[i] != "__":
            c[i] = no_blank_word_in_sen[j]
            j += 1

    str_mask = " ".join([s for s in a]).strip()
    str_replace = " ".join([s for s in b]).strip()
    str_shuffle = " ".join([s for s in c]).strip()

    return [str_mask, str_replace, str_shuffle]


def get_train_skeletons(paths, mode, do_merge, mask_rate, replace_rate):
    """Get the skeletons and word labels for the training dataset."""

    vocab = get_vocab(paths)

    with open(paths['train_raw_path'], 'r', encoding='utf-8') as f:
        json_data = jsonlines.Reader(f)
        train_data = []
        for item in json_data:
            train_data.append(item)

    skeletons = []

    with open(paths['train_skeletons_path'], "w", encoding="utf-8") as f:
        for story_index, story in enumerate(tqdm(train_data)):

            pre = story['premise']
            con = story['initial']
            end = story['original_ending']
            c_con = story['counterfactual']
            c_end = story['edited_ending'][0] + " " + story['edited_ending'][
                1] + " " + story['edited_ending'][2]
            skeleton_ends, skeleton_c_ends, end_unique, c_end_uinque, basic_toked_end, basic_toked_c_end = bottom_up_dp_lcs(
                end, c_end, do_merge, mask_rate, replace_rate, vocab, mode)

            skeleton = {}
            skeleton['premise'] = pre
            skeleton['raw_condition'] = con
            skeleton['counterfactual_condition'] = c_con
            skeleton['raw_skeletons_endings'] = skeleton_ends
            skeleton['counterfactual_skeletons_endings'] = skeleton_c_ends
            skeleton['raw_ending_unique_words'] = end_unique
            skeleton['counterfactual_ending_unique_words'] = c_end_uinque
            skeleton['ending'] = end
            skeleton['c_ending'] = c_end

            basic_toked_pre = basic_tokenize(pre)
            basic_toked_con = basic_tokenize(con)
            basic_toked_c_con = basic_tokenize(c_con)

            sen_pre_1 = ["<premise>"] + basic_toked_pre + ["<raw>"] + basic_toked_con + ["<cf>"] \
                + basic_toked_c_con
            sen_after_1 = ["<raw>"] + basic_toked_end

            sen_pre_2 = ["<premise>"] + basic_toked_pre + ["<cf>"] + basic_toked_c_con + ["<raw>"] \
                + basic_toked_con
            sen_after_2 = ["<cf>"] + basic_toked_c_end

            len_pre = len(sen_pre_1)
            label_pre = [PAD_LABEL_ID for i in range(len(sen_pre_1))]
            label_end = [1 for i in range(len(basic_toked_end))]
            label_c_end = [1 for i in range(len(basic_toked_c_end))]
            for x in end_unique:
                label_end[x[0]] = 0
            for x in c_end_uinque:
                label_c_end[x[0]] = 0
            label_end = [PAD_LABEL_ID] + label_end
            label_c_end = [PAD_LABEL_ID] + label_c_end

            label_raw = [label_pre, label_end]
            label_cf = [label_pre, label_c_end]

            ex_raw = [sen_pre_1, sen_after_1]
            ex_cf = [sen_pre_2, sen_after_2]

            assert len(label_raw[1]) == len(ex_raw[1])
            assert len(label_cf[1]) == len(ex_cf[1])
            skeleton['label_raw'] = label_raw
            skeleton['label_cf'] = label_cf
            skeleton['ex_raw'] = ex_raw
            skeleton['ex_cf'] = ex_cf
            skeletons.append(skeleton)

        json.dump(skeletons, f)


def get_dev_test_skeletons(paths, mode, do_merge):
    """Get the skeletons and word labels for the dev and test dataset."""

    if mode == "dev":
        raw_path = paths['dev_raw_path']
        skeletons_path = paths['dev_skeletons_path']
    elif mode == "test":
        raw_path = paths['test_raw_path']
        skeletons_path = paths['test_skeletons_path']

    with open(raw_path, 'r', encoding='utf-8') as f:
        json_data = jsonlines.Reader(f)
        data = []
        for item in json_data:
            data.append(item)

    skeletons = []

    with open(skeletons_path, "w", encoding="utf-8") as f:
        for story_index, story in enumerate(tqdm(data)):

            pre = story['premise']
            con = story['initial']
            end = story['original_ending']
            c_con = story['counterfactual']
            c_end_0 = story['edited_endings'][0][0] + " " + story[
                'edited_endings'][0][1] + " " + story['edited_endings'][0][2]
            c_end_1 = story['edited_endings'][1][0] + " " + story[
                'edited_endings'][1][1] + " " + story['edited_endings'][1][2]
            c_end_2 = story['edited_endings'][2][0] + " " + story[
                'edited_endings'][2][1] + " " + story['edited_endings'][2][2]

            basic_toked_pre = basic_tokenize(pre)
            basic_toked_con = basic_tokenize(con)
            basic_toked_c_con = basic_tokenize(c_con)

            for c_end in [c_end_0, c_end_1, c_end_2]:
                skeleton_ends, skeleton_c_ends, end_unique, c_end_uinque, basic_toked_end, basic_toked_c_end = bottom_up_dp_lcs(
                    end, c_end, do_merge, 0, 0, [], mode)
                skeleton = {}
                skeleton['premise'] = pre
                skeleton['raw_condition'] = con
                skeleton['counterfactual_condition'] = c_con
                skeleton['raw_skeletons_endings'] = skeleton_ends
                skeleton['counterfactual_skeletons_endings'] = skeleton_c_ends
                skeleton['raw_ending_unique_words'] = end_unique
                skeleton['counterfactual_ending_unique_words'] = c_end_uinque
                skeleton['ending'] = end
                skeleton['c_ending'] = c_end

                sen_pre_1 = ["<premise>"] + basic_toked_pre + ["<raw>"] + basic_toked_con + ["<cf>"] \
                    + basic_toked_c_con
                sen_after_1 = ["<raw>"] + basic_toked_end

                sen_pre_2 = ["<premise>"] + basic_toked_pre + ["<cf>"] + basic_toked_c_con + ["<raw>"] \
                    + basic_toked_con
                sen_after_2 = ["<cf>"] + basic_toked_c_end

                len_pre = len(sen_pre_1)
                label_pre = [PAD_LABEL_ID for i in range(len(sen_pre_1))]
                label_end = [1 for i in range(len(basic_toked_end))]
                label_c_end = [1 for i in range(len(basic_toked_c_end))]
                for x in end_unique:
                    label_end[x[0]] = 0
                for x in c_end_uinque:
                    label_c_end[x[0]] = 0
                label_end = [PAD_LABEL_ID] + label_end
                label_c_end = [PAD_LABEL_ID] + label_c_end

                label_raw = [label_pre, label_end]
                label_cf = [label_pre, label_c_end]

                ex_raw = [sen_pre_1, sen_after_1]
                ex_cf = [sen_pre_2, sen_after_2]

                assert len(label_raw[1]) == len(ex_raw[1])
                assert len(label_cf[1]) == len(ex_cf[1])
                skeleton['label_raw'] = label_raw
                skeleton['label_cf'] = label_cf
                skeleton['ex_raw'] = ex_raw
                skeleton['ex_cf'] = ex_cf
                skeletons.append(skeleton)

        json.dump(skeletons, f)


if __name__ == '__main__':
    paths = {}
    paths['train_vocab_path'] = "data/train_vocab.txt"
    paths['train_raw_path'] = "data/train_supervised_large.json"
    paths['dev_raw_path'] = "data/dev_data.json"
    paths['test_raw_path'] = "data/test_data.json"

    # for sketch:
    do_merge = False
    paths[
        'train_skeletons_path'] = "data/train_skeletons_supervised_large.json"
    paths['dev_skeletons_path'] = "data/dev_skeletons.json"
    paths['test_skeletons_path'] = "data/test_skeletons.json"
    get_train_skeletons(paths, "train", do_merge, 0, 0)
    get_dev_test_skeletons(paths, "dev", do_merge)
    get_dev_test_skeletons(paths, "test", do_merge)

    # for customize, not do augmentation for the skeletons of the train set.
    do_merge = True
    paths[
        'train_skeletons_path'] = "data/merge_train_skeletons_supervised_large.json"
    paths['dev_skeletons_path'] = "data/merge_dev_skeletons.json"
    paths['test_skeletons_path'] = "data/merge_test_skeletons.json"
    get_train_skeletons(paths, "train_wo_aug", do_merge, 0, 0)
    get_dev_test_skeletons(paths, "dev", do_merge)
    get_dev_test_skeletons(paths, "test", do_merge)

    # for customize, do augmentation for the skeletons of the train set.
    paths[
        'train_skeletons_path'] = "data/aug_merge_train_skeletons_supervised_large.json"
    get_train_skeletons(paths, "train_aug", do_merge, 0.2, 0.2)
