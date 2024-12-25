# Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration
This repository contains code for AAAI2025 paper "Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration". This paper introduce a novel CCNet, comprising two core modules: the Cross-Modal Consistency Collaboration (CMCC) and the Multi-Temporal Granularity Collaboration (MTGC), resulting in a new state-of-the-art performance in dense audio-visual event localization.

[arvix page](https://arxiv.org/abs/2412.12628)

## Requirements
The implemetation is based on PyTorch. Follow [INSTALL.md](INSTALL.md) to install required dependencies.

## Data preparation
<!-- #### Download features and annotations -->
- Download UnAV-100 from [this link](https://pan.baidu.com/s/1uBRdq6mXTfnRODMbZ0-QnA?pwd=zyfm) (pwd: zyfm).  The audio and visual features are extracted from the audio and visual encoder of [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE), respectively, where the visual encoder is finetuned on Kinetics-400.  

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

## Citation
@ARTICLE{2024arXiv241212628Z,
       author = {{Zhou}, Ziheng and {Zhou}, Jinxing and {Qian}, Wei and {Tang}, Shengeng and {Chang}, Xiaojun and {Guo}, Dan},
        title = "{Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration}",
      journal = {arXiv e-prints},
         year = 2024,
        pages = {arXiv:2412.12628},
          doi = {10.48550/arXiv.2412.12628},
}

## Acknowledgement
Thanks for the poineering work [UnAV](https://github.com/ttgeng233/UnAV). The video and audio features were extracted using [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE), following [UniAV](https://github.com/ttgeng233/UniAV). Our model was implemented based on [ActionFormer](https://github.com/happyharrycn/actionformer_release) and [UnAV](https://github.com/ttgeng233/UnAV). We thank the authors for their excellent efforts. If you use our code, please also consider to cite their works.
