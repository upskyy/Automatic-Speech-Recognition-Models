# Automatic Speech Recognition Models
![Codestyle Status](https://img.shields.io/badge/build-passing-blue) ![Codestyle Status](https://img.shields.io/badge/license-MIT-blue) ![Codestyle Status](https://img.shields.io/badge/framework-PyTorch-blue) ![Codestyle Status](https://img.shields.io/badge/codestyle-PEP--8-blue)  

End-to-end (E2E) automatic speech recognition (ASR) models were implemented with Pytorch.   
We used KsponSpeech dataset for training and [Hydra](https://github.com/facebookresearch/hydra) to control all the training configurations.

## Installation
```   
pip install -e .   
```   

## Preparation  
You can download dataset at [AI-Hub](https://www.aihub.or.kr/aidata/105). Anyone can download this dataset just by applying. Then, the KsponSpeech dataset was preprocessed through [here](https://github.com/sooftware/ksponspeech).  


## Usage  
### _Training_  
You can choose from several models and training options.
- **Deep Speech2** _Training_
```
python main.py model=deepspeech2 train=deepspeech2_train train.dataset_path=$DATASET_PATH train.audio_path=$AUDIO_PATH train.label_path=$LABEL_PATH
```  
- **Listen, Attend and Spell** _Training_
```
python main.py model=las train=las_train train.dataset_path=$DATASET_PATH train.audio_path=$AUDIO_PATH train.label_path=$LABEL_PATH
```  
- **Joint CTC-Attention Listen, Attend and Spell** _Training_
```
python main.py model=joint_ctc_attention_las train=las_train train.dataset_path=$DATASET_PATH train.audio_path=$AUDIO_PATH train.label_path=$LABEL_PATH
```  
### _Evaluation_
```
python eval.py eval.dataset_path=$DATASET_PATH eval.audio_path=$AUDIO_PATH eval.label_path=$LABEL_PATH eval.model_path=$MODEL_PATH
```  



## Reference  
- [IBM/Pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)  
- [KoSpeech](https://github.com/sooftware/KoSpeech)  
- [ClovaCall](https://github.com/clovaai/ClovaCall)

## Author
- seomk9896@naver.com  

## License  
```
# MIT License
#
# Copyright (c) 2021 Sangchun Ha
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
```


