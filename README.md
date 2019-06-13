# End-to-End Knowledge-Routed Relational Dialogue System for Automatic Diagnosis
By Lin Xu, Qixian Zhou, Ke Gong, Xiaodan Liang, Liang Lin (AAAI19)
This is a simple code of KR-DQN for medical diagnosis diaglogue system. The model is trained and evaluated on [MuZhi dataset](http://www.aclweb.org/anthology/P18-2033)

## Requirements
Python3, pytorch 0.4.1


## Training
```Bash
# Training on MZ dataset. 
./scripts/train.sh
```
The checkpoint models are saved at ./checkpoints/exp_models. Tensorboard logs are saved at ./runs

## Inference and Evaluation
```Bash
# Inference and Evaluation
./scripts/predict.sh
```
The visible dialogue results and quantitative matrics are showed on terminal.

## DX dataset
The DX dataset is available in [Google Could](https://drive.google.com/file/d/19WrAIH1fyJ8BTdsahU-LycTsYnPhUt2n/view?usp=sharing)


## Reference
```
@inproceedings{xu2018,
    Author = {Lin Xu, Qixian Zhou, Ke Gong, Xiaodan Liang, Liang Lin},
    Title = {End-to-End Knowledge-Routed Relational Dialogue System for Automatic Diagnosis},
    Booktitle = {AAAI},
    Year = {2019}
} 
```
