from bert_score import score as bert_score
from typing import List
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
import json
import spacy
import tqdm
import numpy as np
import rouge
import pandas as pd
import jsonlines
from wmd import WMD

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(WMD.SpacySimilarityHook(nlp), last=True)


def _clean_text(txt):
    return txt.lower()


class CFRInstance(object):
    def __init__(
        self,
        original_context: str,
        cf_context: str,
        original_ending: str,
        predicted_ending: str,
        gold_cf_endings: List[str],
    ):
        self.original_context = original_context
        self.cf_context = cf_context

        self.predicted_ending = predicted_ending
        self.original_ending = original_ending
        self.gold_cf_endings = gold_cf_endings

        self.spacy_docs = {
            'original_context': nlp(_clean_text(self.original_context)),
            'original_ending': nlp(_clean_text(self.original_ending)),
            'cf_context': nlp(_clean_text(self.cf_context)),
            'predicted_ending': nlp(_clean_text(self.predicted_ending)),
            'gold_cf_endings':
            [nlp(_clean_text(g)) for g in self.gold_cf_endings]
        }

        self.original_context_tokens = [
            t.text for t in self.spacy_docs['original_context']
        ]
        self.original_ending_tokens = [
            t.text for t in self.spacy_docs['original_ending']
        ]
        self.cf_context_tokens = [
            t.text for t in self.spacy_docs['cf_context']
        ]
        self.predicted_ending_tokens = [
            t.text for t in self.spacy_docs['predicted_ending']
        ]
        self.gold_cf_endings_tokens = [[
            t.text for t in _spacy_doc
        ] for _spacy_doc in self.spacy_docs['gold_cf_endings']]


def eval_bleu(instances: List[CFRInstance]):
    references = []
    hypotheses = []
    for instance in tqdm.tqdm(instances):
        references.append(instance.gold_cf_endings_tokens)
        hypotheses.append(instance.predicted_ending_tokens)

    corpus_bleu_scores = corpus_bleu(
        references, hypotheses, smoothing_function=SmoothingFunction().method4)

    sentence_bleu_scores = []
    total_skipped = 0
    for r, h in tqdm.tqdm(zip(references, hypotheses)):
        if len(h) == 0:
            sentence_bleu_scores.append(0)
            continue
        else:
            try:
                sentence_bleu_scores.append(
                    sentence_bleu(
                        r, h, smoothing_function=SmoothingFunction().method4))
            except:
                sentence_bleu_scores.append(0.0)
                total_skipped += 1

    print("Total skipped = {}".format(total_skipped))

    metrics = {
        'corpus_bleu': corpus_bleu_scores,
        'mean_sentence_bleu': np.mean(sentence_bleu_scores)
    }
    return metrics


def eval_bert_score(instances: List[CFRInstance],
                    bert_model="bert-base-uncased"):
    references = []
    hypotheses = []
    for instance in instances:
        # clean_reference = _clean_text(instance.original_context + ' ' + instance.original_ending)
        # clean_hypothesis = _clean_text(instance.cf_context + ' ' + instance.predicted_ending)
        clean_reference = [_clean_text(x) for x in instance.gold_cf_endings]
        clean_hypothesis = _clean_text(instance.predicted_ending)
        if len(clean_hypothesis) == 0:
            continue
        references.append(clean_reference)
        hypotheses.append(clean_hypothesis)

    P, R, F1 = bert_score(hypotheses,
                          references,
                          model_type=bert_model,
                          verbose=True)
    return {
        "bert_score_P": P.mean().item(),
        "bert_score_R": R.mean().item(),
        "bert_score_F1": F1.mean().item()
    }


def eval_rouge(instances: List[CFRInstance]):
    references = []
    hypotheses = []

    evaluator = rouge.Rouge(
        metrics=['rouge-n', 'rouge-l', 'rouge-w'],
        max_n=4,
        limit_length=True,
        length_limit=100,
        length_limit_type='words',
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True)

    by_instance = []
    for instance in instances:
        _r = [_clean_text(g) for g in instance.gold_cf_endings]
        _h = _clean_text(instance.predicted_ending)
        references.append(_r)
        hypotheses.append(_h)
        try:
            by_instance.append(evaluator.get_scores(_h, _r))
        except:
            by_instance.append({})

    scores = evaluator.get_scores(hypotheses, references)
    return {
        'rouge_all': scores
    }


if __name__ == "__main__":

    f_raw_data = "data/test_data.json"
    f_seq2seq_gpt = "customize_pred_results/test_m_supervised_x1x2yx1xx2.tsv"
    # Result of the Seq2Seq-GPT are got using the Qin's code.
    # https://github.com/qkaren/Counterfactual-StoryRW
    f_sandc_8020 = open("customize_pred_results/sandc_8020.json", "r")
    f_sandc_5050 = open("customize_pred_results/sandc_5050.json", "r")
    f_sandc_wo_aug = open("customize_pred_results/sandc_wo_aug.json", "r")
    f_random_and_c = open("customize_pred_results/random_and_c.json", "r")
    f_lcs_and_c = open("customize_pred_results/lcs_and_c.json", "r")

    with open(f_raw_data, 'r', encoding='utf-8') as dd:
        json_data = jsonlines.Reader(dd)
        data = []
        for item in json_data:
            data.append(item)

    res_sandc_8020 = json.load(f_sandc_8020)
    res_random_and_c = json.load(f_random_and_c)
    data_seq2seq_gpt = pd.read_csv(f_seq2seq_gpt, sep='\t',
                                   header=None).iloc[0:1873, 0:2].values
    res_seq2seq_gpt = []
    for item in data_seq2seq_gpt:
        t = item[1].strip().split(".")[0:3]
        ex = ".".join([e for e in t])
        ex = ex + "."
        res_seq2seq_gpt.append([item[0], ex])
    res_sandc_wo_aug = json.load(f_sandc_wo_aug)
    res_lcs_and_c = json.load(f_lcs_and_c)
    res_sandc_5050 = json.load(f_sandc_5050)

    seq2seq_gpt_instances = []
    random_and_c_instances = []
    sandc_8020_instances = []
    human_instances = []
    sanc_wo_aug_instances = []
    lcs_and_c_instances = []
    sandc_5050_instances = []
    alld = [
        seq2seq_gpt_instances, random_and_c_instances, sandc_8020_instances,
        human_instances, sanc_wo_aug_instances, lcs_and_c_instances,
        sandc_5050_instances
    ]

    j = -1
    for [
            story, item_sandc_8020, item_random_and_c, item_seq2seq_gpt,
            item_sandc_wo_aug, item_lcs_and_c, item_sandc_5050
    ] in zip(data, res_sandc_8020, res_random_and_c, res_seq2seq_gpt,
             res_sandc_wo_aug, res_lcs_and_c, res_sandc_5050):
        j += 1
        print(j)
        assert story['initial'] == item_sandc_8020['condition']
        assert item_sandc_8020['premise'] == item_random_and_c['premise']
        assert item_sandc_8020['condition'] == item_random_and_c['condition']
        assert item_sandc_8020['premise'] in item_seq2seq_gpt[
            0] and item_sandc_8020['condition'] in item_seq2seq_gpt[0]
        premise = item_sandc_8020['premise']
        condition = item_sandc_8020['condition']
        ending = item_sandc_8020['ending']
        cf_condition = item_sandc_8020['cf_condition']

        c_end_0 = story['edited_endings'][0][0] + " " + story[
            'edited_endings'][0][1] + " " + story['edited_endings'][0][2]
        c_end_1 = story['edited_endings'][1][0] + " " + story[
            'edited_endings'][1][1] + " " + story['edited_endings'][1][2]
        c_end_2 = story['edited_endings'][2][0] + " " + story[
            'edited_endings'][2][1] + " " + story['edited_endings'][2][2]

        seq2seq_gpt = item_seq2seq_gpt[1]
        random_and_c = item_random_and_c['cf_pred_gen_ending']
        sandc_8020 = item_sandc_8020['cf_pred_gen_ending']
        human = item_sandc_8020['cf_ending']
        sandc_wo_aug = item_sandc_wo_aug['cf_pred_gen_ending']
        lcs_and_c = item_lcs_and_c['cf_pred_gen_ending']
        sandc_5050 = item_sandc_5050['cf_pred_gen_ending']

        for i, pred in enumerate([
                seq2seq_gpt, random_and_c, sandc_8020, human, sandc_wo_aug,
                lcs_and_c, sandc_5050
        ]):
            instance = CFRInstance(
                original_context=premise + " " + condition,
                cf_context=premise + " " + cf_condition,
                predicted_ending=pred,
                original_ending=ending,
                # gold_cf_endings=[ending]
                gold_cf_endings=[c_end_0, c_end_1, c_end_2])

            alld[i].append(instance)

    for i, instances in enumerate(alld):
        print(i)
        print("Eval GT ROUGE ... ")
        print(eval_rouge(instances))
        print("Eval GT BertScore ... ")
        print(eval_bert_score(instances))
        print("Eval BLEU ... ")
        print(eval_bleu(instances))
        print("----------")
