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

import os
import json
import torch
from torch.utils.data import TensorDataset


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(file_paths, mode):
    """Read data from the file paths and convert them into examples."""

    guid_index = 1
    examples = []
    if mode == "train":
        file_path = file_paths['train_skeletons_path']
    elif mode == "dev":
        file_path = file_paths['dev_skeletons_path']
    elif mode == "test":
        file_path = file_paths['test_skeletons_path']

    with open(file_path, encoding="utf-8") as f:
        all_data = json.load(f)
        for item in all_data:
            words_raw = item['ex_raw']
            labels_raw = item['label_raw']
            words_cf = item['ex_cf']
            labels_cf = item['label_cf']
            examples.append(
                InputExample(guid="{}-{}".format(mode, guid_index),
                             words=words_raw,
                             labels=labels_raw))
            guid_index += 1
            examples.append(
                InputExample(guid="{}-{}".format(mode, guid_index),
                             words=words_cf,
                             labels=labels_cf))
            guid_index += 1

    return examples


def convert_examples_to_features(
    logger,
    mode,
    input_dropout_rate,
    examples,
    max_seq_length,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=1,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []

        for word, label in zip(example.words[0], example.labels[0]):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([pad_token_label_id] * (len(word_tokens)))

        segment_ids = [sequence_a_segment_id] * len(tokens)

        # [CLS]
        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        # first [SEP]
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids += [sequence_a_segment_id]

        for word, label in zip(example.words[1], example.labels[1]):

            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label] + [pad_token_label_id] *
                             (len(word_tokens) - 1))
            segment_ids.extend([sequence_b_segment_id] * len(word_tokens))

        # sencond [SEP]
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids += [sequence_b_segment_id]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s",
                        " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s",
                        " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids))
    return features


def load_and_cache_examples(args, tokenizer, pad_token_label_id, mode, logger):

    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length)),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating %s features from dataset file at %s", mode,
                    args.data_dir)
        examples = read_examples_from_file(args.file_paths, mode)
        features = convert_examples_to_features(
            logger,
            mode,
            args.input_dropout_rate,
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token
                                                       ])[0],
            pad_token_segment_id=1,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features],
                                 dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_label_ids)
    return dataset
