#!/bin/bash

# weights for causal words and background words are (0.8,0.2)
python sketch_main.py --do_train\
    --evaluate_during_training\
    --do_lower_case\
    --unique_flag 8020\
    --uni_w 0.8\
    --ske_w 0.2

# weights for causal words and background words are (0.5,0.5)
python sketch_main.py --do_train\
    --evaluate_during_training\
    --do_lower_case\
    --unique_flag 5050\
    --uni_w 0.5\
    --ske_w 0.5

# weights for causal words and background words are (0.2,0.8)
python sketch_main.py --do_train\
    --evaluate_during_training\
    --do_lower_case\
    --unique_flag 2080\
    --uni_w 0.2\
    --ske_w 0.8