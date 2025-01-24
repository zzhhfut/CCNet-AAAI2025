# Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration
This repository contains the official code for our AAAI-2025 paper "Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration". 

[arXiv page](https://arxiv.org/abs/2412.12628)

## Requirements
The implementation is based on PyTorch. Please follow [INSTALL.md](INSTALL.md) to install the required dependencies.

## Data Preparation
<!-- #### Download features and annotations -->
Please follow the repo. of [UnAV](https://github.com/ttgeng233/UnAV) to download the audiovisual features and annotations of UnAV-100 dataset. You may download it directly using [this link](https://pan.baidu.com/s/1uBRdq6mXTfnRODMbZ0-QnA?pwd=zyfm) (pwd: zyfm). Notably, our SOTA model utilizes the [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE) to extract audiovisual features.

After downloading the dataset, unpack the files under `./data`. The folder structure should look like this:
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
Run `./train.py` to train CCNet on UnAV-100.
For the UnAV-100 dataset, train using the following command:
```
python ./train.py ./configs/CCNet_unav100.yaml --output reproduce
```

## Evaluation
Run ```eval.py``` to evaluate the trained model. You can download our pre-trained model from [this link](https://pan.baidu.com/s/1tLAf-H70ngRLGeC6lZho9g?pwd=5pcs)(pwd: 5pcs) or [this link](https://drive.google.com/file/d/11_kmeXwmMMZixh8TAJ2n1TRfRmZLm0UY/view?usp=sharing)(Google drive).
```
python ./eval.py ./configs/CCNet_unav100.yaml ./ckpt/CCNet_unav100_reproduce
```

## Citation
```
@inproceedings{zhou2024dense,
  title={Dense Audio-Visual Event Localization under Cross-Modal Consistency and Multi-Temporal Granularity Collaboration},
  author={Zhou, Ziheng and Zhou, Jinxing and Qian, Wei and Tang, Shengeng and Chang, Xiaojun and Guo, Dan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2025}
}
```

## Acknowledgement
Our codes are developed based on the codebase of pioneering work [UnAV](https://github.com/ttgeng233/UnAV). We thank the authors for their excellent efforts. The computation of our model is completed on the HPC Platform of Hefei University of Technology.
