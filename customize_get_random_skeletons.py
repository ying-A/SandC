import json
import random

f = open("data/merge_test_skeletons.json", "r")
all_data = json.load(f)
random_test_skes_json = "sketch_pred_results/random_test_skeletons_predictions.json"
all_pred = []
g = open(random_test_skes_json, "w")
random.seed(42)

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

    len_labels_raw_0 = 0
    len_labels_raw_1 = 0
    for x in labels_raw:
        if x == 0:
            len_labels_raw_0 += 1
        else:
            len_labels_raw_1 += 1
    len_labels_raw = len_labels_raw_0 + len_labels_raw_1

    ids = [i for i in range(len_labels_raw)]
    sampled_id = random.sample(ids, len_labels_raw_0)
    random_skes_raw = [1 for i in range(len_labels_raw)]
    for x in sampled_id:
        random_skes_raw[x] = 0

    assert len(labels_raw) == len(random_skes_raw)
    pred_ske_raw = ""
    for word, pred in zip(end_words, random_skes_raw):
        if pred == 1:
            pred_ske_raw = (pred_ske_raw + " " + word)
        else:
            if not pred_ske_raw.endswith(" __ "):
                pred_ske_raw = pred_ske_raw + " __ "
    pred_ske_raw = pred_ske_raw.strip()
    words_raw = " ".join([w for w in words_raw])
    labels_raw = " ".join([str(w) for w in labels_raw])
    preds_raw = " ".join([str(w) for w in random_skes_raw])
    len_labels_cf_0 = 0
    len_labels_cf_1 = 0
    for x in labels_cf:
        if x == 0:
            len_labels_cf_0 += 1
        else:
            len_labels_cf_1 += 1
    len_labels_cf = len_labels_cf_0 + len_labels_cf_1

    ids = [i for i in range(len_labels_cf)]
    sampled_id = random.sample(ids, len_labels_cf_0)
    random_skes_cf = [1 for i in range(len_labels_cf)]
    for x in sampled_id:
        random_skes_cf[x] = 0

    assert len(labels_cf) == len(random_skes_cf)
    pred_ske_cf = ""
    for word, pred in zip(end_words_cf, random_skes_cf):
        if pred == 1:
            pred_ske_cf = (pred_ske_cf + " " + word)
        else:
            if not pred_ske_cf.endswith(" __ "):
                pred_ske_cf = pred_ske_cf + " __ "
    pred_ske_cf = pred_ske_cf.strip()
    words_cf = " ".join([w for w in words_cf])
    labels_cf = " ".join([str(w) for w in labels_cf])
    preds_cf = " ".join([str(w) for w in random_skes_cf])

    res = {}
    res['premise'] = pre
    res['raw_condition'] = con
    res['ending'] = end
    res['gt_raw_skeletons_ending'] = ske
    res['raw_skeletons_endings'] = [pred_ske_cf]
    res['counterfactual_condition'] = c_con
    res['c_ending'] = c_end
    res['gt_counterfactual_skeletons_ending'] = c_ske
    res['counterfactual_skeletons_endings'] = [pred_ske_raw]
    all_pred.append(res)

json.dump(all_pred, g)
