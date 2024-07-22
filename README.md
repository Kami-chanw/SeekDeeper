# SeekDeeper: Minimal Implementations of Popular AI Models
[\[📖中文ReadMe\]](./README_zh.md)

## Motivation
Official code repositories often include many engineering details, which can be overwhelming for beginners. This repository aims to implement various models with as little code as possible using PyTorch, making it easier for learners to understand and reproduce results. Additionally, most tutorials lack a complete workflow, focusing only on the model without considering data loading and training. This makes it difficult for beginners to apply their knowledge in practice.

## Models

| Model       | Paper                                                                                                                                                         | Official or Reference Repository                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                                                                                 | https://github.com/hyunwoongko/transformer        |
| GPT         | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language_models_are_unsupervised_multitask_learners.pdf) | https://github.com/openai/gpt-2                   |
| GAN         | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)                                                                                            | https://github.com/goodfeli/adversarial           |
| DCGAN       | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434)                              | https://github.com/Newmu/dcgan_code               |
| WGAN-GP     | [Improved Training of Wasserstein GANs](https://arxiv.org/pdf/1704.00028)                                                                                     | https://github.com/igul222/improved_wgan_training |

## Directory Structure

For each model, the typical directory structure is as follows:

```
<model name>/
├── checkpoints/
├── modules/
├── datasets/
├── images/
├── README.md
├── data.py
├── config.py
├── train.ipynb
└── inference.ipynb
```

- **checkpoints/**: Contains pre-trained model weights for direct use in `inference.ipynb`. Sometimes, pre-trained parameters from official repositories are loaded directly.
- **modules/**: Contains modules necessary for model implementation.
- **datasets/**: Contains datasets required for training or inference validation, which may sometimes be downloaded to this directory via code.
- **images/**: Contains datasets required for training or inference validation, which may sometimes be downloaded to this directory via code.
- **README.md**: Introduces the implemented task and describes the implementation details.
- **data.py**: Defines `Dataset`, `Dataloader`, or data preprocessing.
- **config.py**: Defines hyperparameters needed for the experiment.
- **train.ipynb**: Clearly presents the process from data loading, preprocessing, to training and evaluation.
- **inference.ipynb**: Loads model parameters from the `checkpoints/` directory for inference.

## License

This project uses the MIT License.
