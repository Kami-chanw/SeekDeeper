[\[📖English ReadMe\]](./README.md)
## Introduction
在这里，我实现了一个Transformer，并使用其Encoder在IMDB数据集上进行了文本情感分类任务（见[此](./train_imdb.ipynb)）。

## Model details
### [Transformer](./modules/transformer.py)
Transformer最初提出被用于解决翻译任务。如果要实现中文到英文的翻译，那么我们称中文为源语言，英文为目标语言。Transformer的结构如下图所示，源文本的embedding与positional encoding相加后输入到Encoder，经过N层Encoder layer后，输出在Decoder的cross attention中进行交互。目标文本的embedding同样与positional encoding相加后输入到Decoder，Dncoder的输出通常会再经过一个线性层（具体取决于任务要求）。
<div style="text-align: center;">
  <img src="./images/transformer.png" alt="Transformer" style="width: 300px; height: auto;">
</div>

Encoder和Decoder分别使用了两种mask，`src_mask`和`tgt_mask`。`src_mask`用于遮盖所有的PAD token，避免它们在attention计算中产生影响。`tgt_mask`除了遮盖所有PAD token，还要防止模型在进行next word prediction时访问未来的词。

### [Positional Encoding](./modules/layers.py)
由于Transformer不像RNN那样具有天然的序列特性，在计算attention时会丢失顺序信息，因此需要引入位置编码。在原始论文中，位置编码的计算公式如下：

- 对于偶数维度：
  $$
   \text{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
  $$

- 对于奇数维度：
  $$ 
  \text{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) 
  $$

为了数值稳定性，我们对div term取指数和对数，即：
$$
\text{div\_term} = 10000^{2i/d_{\text{model}}} = \exp\left(\frac{2i \cdot -\log(10000)}{d_{\text{model}}}\right)
$$

位置编码对任何序列都是相同的，因此positional encoding的shape为`[seq_len, d_model]`。然后根据广播机制与shape为`[batch_size, seq_len, d_model]`的input embedding相加，得到Encoder的输入，记作$x_0$。

## [Encoder](./modules/encoder.py)
Encoder包含多个相同的层。上一层的输出$x_i$以如下途径经过该层（省略了dropout）：
```python
# attention mechanism
residual = x
x = multihead_attention(q=x, k=x, v=x, mask=src_mask)
x = layer_norm(x + residual)

# position-wise feed forward
residual = x
x = feed_forward(x)
x = layer_norm(x + residual)
```

## [Attention](./modules/layers.py)
Attention的计算流程如下：
<div style="text-align: center;">
  <img src="./images/attention.png" alt="Attention" style="width: 400px; height: auto;">
</div>
在Encoder的self-attention中，K、Q、V均为上一层的输出经过不同线性层得到的。在Decoder的cross-attention中，K和V来自Encoder最后一层的输出，而Q是Decoder上一层的输出。

为了使模型关注不同位置的不同特征子空间信息，我们需要使用多头注意力。具体来说，将shape为`[batch_size, seq_len, d_model]`的K、Q、V分为`[batch_size, seq_len, n_head, d_key]`，再交换`seq_len`和`n_head`两个维度，以便进行attention机制中的矩阵乘法。计算了attention之后再将结果合并，并通过一个线性层映射到与输入相同的维度。算法的流程如下：
```python
# projection
K, Q, V = W_k(x), W_q(x), W_v(x)

# split
d_key = d_model // n_head
K, Q, V = (K, Q, V).view(batch_size, seq_len, n_head, d_key).transpose(1, 2)
out = scaled_dot_product_attention(K, Q, V)

#concatenate
out = out.transpose(1, 2).view(batch_size, seq_len, d_model)
out = W_cat(out)
```

Scaled Dot-Product Attention用公式表示为：
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_{key}}}\right) \cdot V
$$

## [Decoder](./modules/decoder.py)
Decoder相较于Encoder除了多了一层cross-attention之外，还使用了masked multi-head attention。由于模型在此处不能访问未来信息，因此这种注意力机制也称为causal self-attention。
Decoder同样包含多个相同的层，Encoder最后一层的输出`enc`和Decoder上一层的输出`dec`以如下途径经过该层（省略了dropout）：
```python
# causal self-attention
residual = dec
x = multihead_attention(q=dec, k=dec, v=dec, mask=tgt_mask)
x = layer_norm(x + residual)

# cross-attention
x = multihead_attention(q=x, k=enc, v=enc, mask=tgt_mask)
x = layer_norm(x + residual)

# position-wise feed forward
residual = x
x = feed_forward(x)
x = layer_norm(x + residual)
```
