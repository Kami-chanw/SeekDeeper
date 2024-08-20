[$$📖English ReadMe\]](./README.md)
## Introduction
在这个GPT的实现中，我将展示如何在[BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus)数据集上进行预训练，然后从[huggingface](https://huggingface.co/openai-community/openai-gpt)上加载官方的预训练权重到我们的模型中，在[Corpus of Linguistic Acceptability(CoLA)](https://nyu-mll.github.io/CoLA/)数据集上进行微调，并复现论文中提到的效果。

## Model details
### Key differences with vanilla transformer
1. GPT只使用了原始transformer的Decoder。由于不再需要Encoder，因此删去了与Encoder交互的交叉自注意力，只使用因果自注意力层。
2. GPT的Decoder层中的前馈神经网络层去掉了一次dropout。此外，还将激活函数从ReLU替换为了GeLU。
3. GPT的模型规模略微扩大了一些，还固定了词表的大小（因为GPT是预训练模型）。
4. GPT使用了[Weight Tying](https://arxiv.org/abs/1608.05859)，令language modelling head与token embedding层共享权重。
5. GPT使用positional embedding而不是正余弦位置编码。
   
汇总如下表所示：
<table>
  <thead>
    <tr style="font-weight: bold; border-bottom: 2px solid">
      <th></th>
      <th style="text-align: center">GPT</th>
      <th style="text-align: center">Transformer</th>
    </tr>
  </thead>
  <tbody style="text-align:center">
    <tr>
    <td>Positonal encoding</td>
    <td> learnable </td>
    <td> sinusoidal(mainstream) </td>
    </tr>
    <tr>
      <td>n_head</td>
      <td>12</td>
      <td>8</td>
    </tr>
    <tr>
      <td>n_layer</td>
      <td>12</td>
      <td>6 encoder layers, 6 decoder alyers</td>
    </tr>
    <tr>
      <td>d_model</td>
      <td>768</td>
      <td>512</td>
    </tr>
    <tr>
      <td>vocab_size</td>
      <td>40468</td>
      <td>depends on dataset</td>
    </tr>
    <tr>
      <td>FFN path</td>
      <td style="text-align:left">
      <pre>
      <code>
      mlpf = lambda x: dropout(fc2(gelu(fc1(x))))
      x = x + layer_norm(mlpf(x))
      </code>
      </pre>
      </td>
      <td style="text-align:left">
      <pre>
      <code>
      mlpf = lambda x: dropout(fc2(dropout(relu(fc1(x)))))
      x = x + layer_norm(mlpf(x))
      </code>
      </pre>
      </td>
    </tr>
  </tbody>
</table>

### [Byte-pair encoding (BPE)](./modules/bpe.py)
BPE 是一种tokenize的方法，其核心思想是通过合并最频繁出现的字符对来构建更大的子词单元，从而减少词汇表的大小，处理稀有词问题。它需要现在一个语料库上进行训练，得到词表后才能进行编码和解码。

由于Huggingface提供的[BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus)数据集已经经过了细致的后处理，因此我们无法完全复现出[原始GPT代码](https://github.com/openai/finetune-transformer-lm)的结果。我仅基于原始实现[text_utils.py](https://github.com/openai/finetune-transformer-lm/blob/master/text_utils.py)完成了编码和解码部分的工作。如果你对BPE的训练流程感兴趣，可以参考Karpathy的[minbpe](https://github.com/karpathy/minbpe)。

#### Training
1. 预分词：用[`ftfy`](https://github.com/rspeer/python-ftfy)规范化Unicode字符，把非标准标点统一，并替换所有的空白字符为`\n`，然后使用spacy的[en_core_web_sm](https://spacy.io/models/en#en_core_web_sm)模型进行分词（见[bpe.py](./modules/bpe.py)）。
2. 初始化词汇表：将整个文本语料库拆分成单字符的子词单元，最后一个字符添加`</w>`。在训练后的词表[encoder_bpe_40000.json](./datasets/bookcorpus/encoder_bpe_40000.json)中可以看出，id从1~238都为单个字符，239~476为单个字符+`</w>`的形式。这里的`</w>`代表一个token的结尾。例如在单词`bamboo`中，最后一个`o`会被视作`o</w>`以与倒数第二个`o`区分。
3. 统计bi-gram字符对的频率。
4. 合并最频繁出现的字符对，并形成一个新的子词单元。更新语料库中的词汇表，并记录该合并操作。
5. 重复步骤3-4 40000次，于是在476个单个词元的基础上获得了40000个新的字词单元。再加上`<unk>`和`\n</w>`共计40478个词元。

#### Encoding
0. 加载训练好的词表。
1. 预分词：对输入的文本进行预分词，同训练阶段。
2. 将每个子词拆分成单字符的子词单元，最后一个字符添加`</w>`。
3. 统计bi-gram字符对的频率。
4. 选择在词表中最早被合并的字符对，并形成一个新的子词单元。将目前文本中出现的字符对以新子词单元进行替换。
5. 重复步骤3-4 直到没有更多的有效 bigram 或者只剩一个字符单元。
6. 缓存结果，将子词单元映射到词表中对应token的id。

#### Decoding
0. 加载训练好的词表。
1. 根据词表建立反向映射，将给定token id映射回原子词即可。
  
## Pretraining
根据[论文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) Sec. 4.1的设置，在BooksCorpus数据集上，使用AdamW ($w=0.01, \text{max\_lr}=2.4\times 10^{-4}$)作为优化器，使用先使学习率从0分2000步线性增加到$\text{max\_lr}$，而后余弦退火的学习率调整策略，对随机采样的 64 个连续 512 个token sequences 的 minibatch 进行 100 个epoch的训练。

为了采样连续指定数量的token sequence，我们需要先用bpe把原数据集的文本转换为token id。而我们自己实现的tokenize速度非常慢，因此只能在小部分数据集上进行实验。如果你想要尝试更多的数据，可以修改[pretrain.ipynb](./pretrain.ipynb)中`load_data`的`loading_ratio`参数。