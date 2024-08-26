[📖中文 ReadMe](./README_zh.md)
## Introduction

In this implementation of GPT-2, I will demonstrate how to pre-train on the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset, then load the official pre-trained weights from [Hugging Face](https://huggingface.co/openai-community/gpt2/tree/main) into our model, and evaluate directly on the [Children's Book Test (CBT)](https://arxiv.org/pdf/1511.02301) dataset without fine-tuning, replicating the results mentioned in the paper.

## Model details

### Key differences with GPT-1
1. GPT-2 moved `LayerNorm` to the input of each sub-block and added an extra `LayerNorm` at the end. For analysis on `Pre-LN`, refer to the paper [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).
2. GPT-2 expanded the vocabulary size from 40,478 to 50,257 and increased the context length from 512 to 1,024.
3. GPT-2 used a modified initialization method, considering the accumulation of residual paths as the model depth increases. During initialization, the weights of the residual mapping layer are scaled by $1/\sqrt{N}$, where $N$ is the number of residual layers.

### [Byte-pair encoding (BPE)](./modules/bpe.py)

BPE is a tokenization method that builds larger subword units by merging the most frequently occurring pairs of characters, reducing the vocabulary size and addressing the issue of rare words. It requires training on a corpus to obtain the vocabulary before encoding and decoding.

My implementation is largely based on Karpathy's [minGPT](https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py). If you're interested in the BPE training process, you can also refer to his [minbpe](https://github.com/karpathy/minbpe).

The BPE used in GPT-2 has some improvements over the version used in GPT-1. Unlike GPT-1, which was based on Unicode characters, GPT-2's BPE operates at the byte level. This means GPT-2 can handle various character sets and special symbols more flexibly, especially non-ASCII characters and emojis, which is particularly helpful for multilingual support and processing non-English text.

## [Pre-training](./pretrain.ipynb)

The general steps are similar to GPT-1's pre-training.

[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) Sec. 2.1 provides a detailed description of how OpenWebText was generated, but does not mention much about the training hyperparameters in the subsequent paragraphs.

## [Inference](./inference.ipynb)

GPT-2 has made significant improvements in text generation capabilities compared to GPT-1. I mainly replicated the evaluation on the CN subset of the Children's Book Test (CBT) mentioned in the paper. This subset requires the model to choose a suitable noun from ten options to fill in a blank in a context paragraph and calculate the prediction accuracy. For this, we need to substitute each option into the blank, calculate the conditional probability of the entire sentence given the candidate word, and select the one with the highest probability as the model's prediction.

## Appendix

### How to download pretrained GPT-2?

Run the following commands in the terminal:

```bash
pip install -U huggingface-cli
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download openai-community/gpt-2 --local-dir path/to/pretrained_dir
```