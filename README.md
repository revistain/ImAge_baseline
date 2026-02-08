<img src="figures/ImAge.jpg" width="1000px">

🆕 **[Feb 2026]** The code for obtaining the unified dataset have been released at [HERE](https://github.com/Tong-Jin01/Unified_dataset).

This is the official repository for the NeurIPS 2025 paper "Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era".

[[Paper on ArXiv](https://arxiv.org/pdf/2511.06024) | [Paper on HF](https://huggingface.co/papers/2511.06024) | [Model on HF](https://huggingface.co/fenglu96/ImAge4VPR)]

ImAge is an implicit aggregation method to get robust global image descriptors for visual place recognition, which neither modifies the backbone nor needs an extra aggregator. It only adds some aggregation tokens before a specific block of the transformer backbone, leveraging the inherent self-attention mechanism to implicitly aggregate patch features. Our method provides a novel perspective different from the previous paradigm, effectively and efficiently achieving SOTA performance. 

The difference between ImAge and the previous paradigm is shown in this figure:

<img src="figures/pipeline.jpg" width="800px">

To quickly use our model, you can use Torch Hub:
```
import torch
model = torch.hub.load("Lu-Feng/ImAge", "ImAge")
```

## Getting Started

This repo follows the framework of [GSV-Cities](https://github.com/amaralibey/gsv-cities) for training, and the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) for evaluation. You can download the GSV-Cities datasets [HERE](https://www.kaggle.com/datasets/amaralibey/gsv-cities), and refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:

```
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

Before training, you should download the pre-trained foundation model DINOv2-register(ViT-B/14) [HERE](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth).

## Train
```
python3 train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=20 --patience=20 --initialization_dataset=msls_train --training_dataset=gsv_cities --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_reg4_pretrain.pth
```

If you don't have the MSLS-train dataset, you can also set `--initialization_dataset=gsv_cities`.
Additionally, `--training_dataset` can be chosen as gsv_cities or unified_dataset (See [Here](https://github.com/Tong-Jin01/Unified_dataset) to get it).

## Test
```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --resume=/path/to/trained/model/ImAge_GSV.pth
```

## Trained Model

<table style="margin: auto">
  <thead>
    <tr>
      <th>Training set</th>
      <th>Pitts30k</th>
      <th>MSLS-val</th>
      <th>Nordland</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">GSV-Cities</td>
      <td align="center">94.0</td>
      <td align="center">93.0</td>
      <td align="center">93.2</td>
      <td><a href="https://huggingface.co/fenglu96/ImAge4VPR/resolve/main/ImAge_GSV.pth">LINK</a></td>
    </tr>
    <tr>
      <td align="center">Unified dataset</td>
      <td align="center">94.1</td>
      <td align="center">94.5</td>
      <td align="center">97.7</td>
      <td><a href="https://huggingface.co/fenglu96/ImAge4VPR/resolve/main/ImAge_Merged.pth">LINK</a></td>
    </tr>
  </tbody>
</table>

## Others

This repository also supports training NetVLAD, SALAD, and BoQ on the GSV-Cities dataset with PyTorch (not pytorch-lightning in other repos) and using Automatic Mixed Precision.

## Acknowledgements

Parts of this repo are inspired by the following repositories:

[GSV-Cities](https://github.com/amaralibey/gsv-cities)

[Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

[DINOv2](https://github.com/facebookresearch/dinov2)

## Citation

If you find this repo useful for your research, please consider leaving a star⭐️ and citing the paper

```
@inproceedings{ImAge,
title={Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era},
author={Feng Lu and Tong Jin and Canming Ye and Xiangyuan Lan and Yunpeng Liu and Chun Yuan},
booktitle={The Annual Conference on Neural Information Processing Systems},
year={2025}
}
```

```
@ARTICLE{selavprpp,
author={Lu, Feng and Jin, Tong and Lan, Xiangyuan and Zhang, Lijun and Liu, Yunpeng and Wang, Yaowei and Yuan, Chun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2025.3629287}}
```
