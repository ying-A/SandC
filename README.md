# Implementation of the paper *Sketch and Customize: A Counterfacutal Story Generator* .

## 1. Preparation.
```
pip install -r requierements.txt
```

## 2. Get the labels of the causal and background words, and the various types of skeletons.
```
python get_augment_skeletons.py
```

## 3. Training of the sketch model.
```
bash sketch_train.sh
```

## 4. Inference of the sketch model, results will be saved in sketch_pred_results dir.
```
bash sketch_inference.sh
```

## 5. Training of the customize model.
```
bash customize_train.sh
```

## 6. Inference of the customzie models with various types of skeletons.
```
bash customize_inference.sh
```

## 7. Evaluation of the models.
```
python evaluation.py
```






