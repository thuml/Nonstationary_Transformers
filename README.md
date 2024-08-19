# Non-stationary Transformers

This is the codebase for the paper:
[Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting](https://arxiv.org/abs/2205.14415), NeurIPS 2022. [[Slides]](https://cloud.tsinghua.edu.cn/f/8d6ce7b18d3c468190e7/), [[Poster]](https://cloud.tsinghua.edu.cn/f/6eea66909aa7465ca9a4/).

:triangular_flag_on_post: **News** (2023.02) Non-stationary Transformer has been included in [[Time-Series-Library]](https://github.com/thuml/Time-Series-Library), which covers long- and short-term forecasting, imputation, anomaly detection, and classification.

## Discussions

There are already several discussions about our paper, we appreciate a lot for their valuable comments and efforts: [[Official]](https://mp.weixin.qq.com/s/LkpkTiNBVBYA-FqzAdy4dw), [[OpenReview]](https://openreview.net/forum?id=ucNDIDRNjjv), [[Zhihu]](https://zhuanlan.zhihu.com/p/535931701).

## Architecture

![arch](./figures/arch.png)

### Series Stationarization

Series Stationarization unifies the statistics of each input and converts the output with restored statistics for better predictability. 

![arch](./figures/ss.png)

### De-stationary Attention

De-stationary Attention is devised to recover the intrinsic non-stationary information into temporal dependencies by approximating distinguishable attentions learned from unstationarized series. 

![arch](./figures/da.png)


## Showcases

![arch](./figures/showcases.png)

## Preparation

1. Install Python 3.7 and neccessary dependencies.
```
pip install -r requirements.txt
```
2. All the six benchmark datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1CC4ZrUD4EKncndzgy5PSTzOPSqcuyqqj/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b8f4a78a39874ac9893e/?dl=1).

## Training scripts

### Non-stationary Transformer

We provide the Non-stationary Transformer experiment scripts and hyperparameters of all benchmark dataset under the folder `./scripts`.

```bash
# Transformer with our framework
bash ./scripts/ECL_script/ns_Transformer.sh
bash ./scripts/Traffic_script/ns_Transformer.sh
bash ./scripts/Weather_script/ns_Transformer.sh
bash ./scripts/ILI_script/ns_Transformer.sh
bash ./scripts/Exchange_script/ns_Transformer.sh
bash ./scripts/ETT_script/ns_Transformer.sh
```

```bash
# Transformer baseline
bash ./scripts/ECL_script/Transformer.sh
bash ./scripts/Traffic_script/Transformer.sh
bash ./scripts/Weather_script/Transformer.sh
bash ./scripts/ILI_script/Transformer.sh
bash ./scripts/Exchange_script/Transformer.sh
bash ./scripts/ETT_script/Transformer.sh
```

### Non-stationary framework to promote other Attention-based models 

We also provide the scripts for other Attention-based models (Informer, Autoformer), for example:

```bash
# Informer promoted by our Non-stationary framework
bash ./scripts/Exchange_script/Informer.sh
bash ./scripts/Exchange_script/ns_Informer.sh

# Autoformer promoted by our Non-stationary framework
bash ./scripts/Weather_script/Autoformer.sh
bash ./scripts/Weather_script/ns_Autoformer.sh
```

## Experiment Results

### Main Results

For multivariate forecasting results, the vanilla Transformer equipped with our framework consistently achieves state-of-the-art performance in all six benchmarks and prediction lengths.

![arch](./figures/main_results.png)

### Model Promotion

By applying our framework to six mainstream Attention-based models. Our method consistently improves the forecasting ability. Overall, it achieves averaged **49.43%** promotion on Transformer, **47.34%** on Informer, **46.89%** on Reformer, **10.57%** on Autoformer, **5.17%** on ETSformer and **4.51%** on FEDformer, making each of them surpass previous state-of-the-art.

![arch](./figures/promotion.png)

## Future Work

We will keep equip the following models with our proposed Non-stationary Transformers framework:

- [x] Transformer
- [x] iTransformer
- [x] Informer
- [x] Autoformer
- [x] FEDformer
- [x] Crossformer
- [x] Reformer
- [x] ......

Note: Series Stationarization as an architecture-free module has been widely applied for addressing non-stationarity in time series. Please refer to [time-series-library](https://github.com/thuml/Time-Series-Library/tree/main/models) for the implementationdetails.

## Citation

If you find this repo useful, please cite our paper. 

```
@article{liu2022non,
  title={Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting},
  author={Liu, Yong and Wu, Haixu and Wang, Jianmin and Long, Mingsheng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Contact

If you have any questions or want to use the code, please contact liuyong21@mails.tsinghua.edu.cn.


## Acknowledgement

This repo is built on the [Autoformer repo](https://github.com/thuml/Autoformer), we appreciate the authors a lot for their valuable code and efforts.
