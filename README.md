# Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration
This repository contains code for AAAI2025 paper "Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration". This paper introduce a novel CCNet, comprising two core modules: the Cross-Modal Consistency Collaboration (CMCC) and the Multi-Temporal Granularity Collaboration (MTGC), resulting in a new state-of-the-art performance in dense audio-visual event localization.

[Project Webpage](https://github.com/zzhhfut/CCNet-AAAI2025)

[arvix page](https://arxiv.org/abs/2412.12628)

## Data preparation
<!-- #### Download features and annotations -->
- Download UnAV-100 from [this link](https://pan.baidu.com/s/1uBRdq6mXTfnRODMbZ0-QnA?pwd=zyfm) (pwd: zyfm). For visual features, fps=16, sliding window size=16 and stride=4. For audio features, sample rate=16kHZ, sliding window size=1s and stride=0.25s.  

After downloading, unpack files under `./data`. The folder structure should look like:
```
This folder
│   README.md
│   ...  
└───data/
│    └───unav100/
│    	 └───annotations
│    	 └───av_features  
└───libs
│   ...
```

## Training 
Run `./train.py` to jointly train CCNet on UnAV-100.
For UnAV-100 dataset, train using the following command:
```
python ./train.py ./configs/CCNet_unav100.yaml --output reproduce
```

## Evaluation
Run ```eval.py``` to evaluate the trained model. 
```
python ./eval.py ./configs/CCNet_unav100.yaml ./ckpt/CCNet_unav100_reproduce
```

