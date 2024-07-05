# SeekDeeper: 最小化的流行人工智能模型实现
[\[📖English ReadMe\]](./README.md)

## 动机
官方的代码库往往考虑了太多工程化的细节，对于初学者来说可能会感到负担过重。这个仓库旨在基于PyTorch使用尽可能少的代码实现各种模型，使学习者更容易理解和复现结果。此外，大部分的教程都缺乏一套完整的流程讲解，即只含有模型的内容而忽视了数据的加载、训练，这同样导致初学者走马观花，难以动手实操。

## 模型

| 模型       | 论文                                                                                                                                                        | 官方或引用仓库                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ |
| Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                                                                                | https://github.com/hyunwoongko/transformer |
| GPT         | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | https://github.com/karpathy/minGPT         |
| GPT-2       | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | https://github.com/openai/gpt-2            |
| VAE         | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)                                                                                          | https://github.com/pytorch/examples/tree/main/vae  |
| GAN         | [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)                                                                                           | https://github.com/goodfeli/adversarial    |

## 目录结构

对于每个模型而言，其典型目录结构如下：

```
<model name>/
├── checkpoints/
├── components/
├── dataset/
├── README.md
├── download.py
├── data.py
├── config.py
├── train.ipynb
└── inference.ipynb
```

- **checkpoints/**：包含已训练好的模型权重，以供`inference.ipynb`中载入直接使用。有时也会直接从官方代码库中载入预训练参数。
- **components/**：包含实现模型所需的组件。
- **dataset/**：包含训练或推理验证时所需的数据集，有时可能会通过代码下载到该目录中。
- **README.md**：介绍实现的任务，并描述实现的细节。
- **download.py**：下载数据集的脚本。
- **data.py**：定义了`Dataset`、`Dataloader`或预处理数据。
- **config.py**：定义了实验所需的超参数。
- **train.ipynb**：以清晰的方式展现从数据加载、预处理，到训练、评估的一系列过程。
- **inference.ipynb**：加载`checkpoints/`目录下的模型参数并进行推理。

## 许可协议

本项目使用MIT许可证。
