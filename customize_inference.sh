#!/bin/bash

# Use the skeletons predicted by the sketch model.

# Inference of the model Random&C
python customize_get_random_skeletons.py
python customize_inference.py --customize_model_path with_aug_customize_model/best_eval_model/\
    --sketch_pred_results_path sketch_pred_results/random_test_skeletons_predictions.json\
    --customize_pred_results_name random_and_c.json\

# Inference of the model LCS&C
python customize_inference.py --customize_model_path with_aug_customize_model/best_eval_model/\
    --sketch_pred_results_path sketch_pred_results/8020_test_skeletons_predictions.json\
    --customize_pred_results_name lcs_and_c.json\
    --use_lcs_skeletons

# Inference of the model S&C-0.5
python customize_inference.py --customize_model_path with_aug_customize_model/best_eval_model/\
    --sketch_pred_results_path sketch_pred_results/5050_test_skeletons_predictions.json\
    --customize_pred_results_name sandc_5050.json

# Inference of the model S&C-0.8
python customize_inference.py --customize_model_path with_aug_customize_model/best_eval_model/\
    --sketch_pred_results_path sketch_pred_results/8020_test_skeletons_predictions.json\
    --customize_pred_results_name sandc_8020.json

# Inference of the model S&C-w/o-Aug
python customize_inference.py --customize_model_path without_aug_customize_model/best_eval_model/\
    --sketch_pred_results_path sketch_pred_results/8020_test_skeletons_predictions.json\
    --customize_pred_results_name sandc_wo_aug.json