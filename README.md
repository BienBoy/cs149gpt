# 作业 4：NanoGPT149

**截止日期：12月4日11:59pm PST**

**总分100分 + 额外12分**

## 概览

在本作业中，你将实现并优化一个基于 transformer 的深度神经网络的关键组件，该网络能合成莎士比亚文本。虽然你将处理的 DNN 模型相对较小，但模型的基本组件与构成如今 ChatGPT 等技术基础的大型语言模型（LLM）中的组件相同。具体来说，你将用 C++ 实现此模型的注意力层，并专注于提高算术强度、减少内存占用，并利用 CPU 上的多核及可能的 SIMD 并行性的优化。你的实现将作为完整的[NanoGPT](https://github.com/karpathy/nanoGPT)模型的一部分，用来生成新颖的类似莎士比亚的文本。

总的来说，这项作业将：

- 让你体验实现 DNN 层的底层细节，换句话说，就是像 NVIDIA 的 CuDNN 或 Intel 的 One API 等供应商库的“内部”。

- 展示保持关键局部性的优化（如循环嵌套优化和循环合并）的价值。

## 环境设置

我们将为你提供 SSH 访问权限，以便在共享机器集群上测试你的代码。（我们不再使用像在编程作业 3 中的 AWS Lightsail）。你将直接通过 ssh 登录这些机器。如何访问集群的详细信息将在 Ed 帖子中提供。

首先，从 github 克隆仓库：

```
git clone https://github.com/stanford-cs149/cs149gpt.git
```

运行以下命令以使用由 CS149 工作人员训练的模型进行推理。你将看到一些随机生成的莎士比亚文本。

```
python3 gpt149.py part0 --inference -m shakes128
```

注意，你第一次运行程序时，将进行编译步骤，可能需要几秒钟，你会看到`Compiling code into a PyTorch module...`的文字。

完成后，你将看到类似这样的文本：

```
Running inference using dnn model shakes128
number of parameters: 0.80M
Loading meta from data/shakespeare_char/meta.pkl...

BOTtaps along my lord.

DUKE OF AUMERLE:
The this is needs! Camillo, put I will make be strong.

QUEEN MARGARET:
My lord, yet t
-------------------------------------------------------------
CAMILLO:
The shadows men sweet thy will burn comes.

FLORIZEL:
But of appear, good from thy heart
As I be come of repeal of a w
-------------------------------------------------------------
```

当然，NanoGPT 的输出可能不是文学上的杰作，但它仍然相当精彩！你在屏幕上看到的是 NanoGPT 的标准 PyTorch 实现的输出。如果你想看到更长序列的表现，可以通过改变`-m`参数到更大的模型，如`shakes256`、`shakes1024`或`shakes2048`。你会发现，随着模型变大，NanoGPT 生成 token 的性能明显变慢。

### My Compilation Hangs

有些学生遇到过问题：即使之前可以正常工作，但他们的编译却随机开始挂起。当 Python JIT 编译你的代码时，它使用锁来允许多个线程同时编译，以提高效率。如果你在编译代码时遇到挂起，这意味着由于某种原因，Python 认为你的文件锁被占用了。为了解决这个问题，你可以运行：

```
rm ~/.cache/torch_extensions/py310_cpu/custom_module/lock
```

## 注意力模块

你在本次作业中执行的 NanoGPT 模块是一个 sequence-to-sequence 模型。输入是一系列单词，比如短语 *"The course of true love never did run smooth"*。模型的输出是可能在输入之后的新单词序列，这是由在大量莎士比亚文本上训练的模型决定的。例如，给定上述前缀，模型的输出可能是 *"whispered cs149 students whilst coding on assignments"*。

NanoGPT 模型使用了一种受欢迎的 DNN 模块，称为 *transformer*，而 transformer 模块的关键组件是称为 *注意力机制* 的块。在这次作业中，你的任务是实现注意力机制。你将从注意力的简单串行实现开始，然后在作业过程中，我们将引导你完成循环嵌套优化、循环合并和基本并行化等优化。

在本节中，我们将描述注意力机制的数学原理（你应该计算的）。你可以参考[第10讲第52张幻灯片](https://gfxcourses.stanford.edu/cs149/fall23/lecture/dnneval/slide_52)获得可视化演示以便跟进。对于想要更直观地了解注意力机制为何如此的学生，我们推荐你查看关于这一流行 DNN 架构的许多在线教程，例如：

- [What is the intuition behind the attention mechanism?](https://ai.stackexchange.com/questions/21389/what-is-the-intuition-behind-the-attention-mechanism)
- [Transformer Neural Networks: A Step-by-Step Breakdown](https://builtin.com/artificial-intelligence/transformer-neural-network)
- [How Transformers Work](https://towardsdatascience.com/transformers-141e32e69591)

注意力机制需要输入三个矩阵 `Q`、`K` 和 `V`，分别称为“query”、“key”和“value”向量。这些矩阵的大小都是 `Nxd`。`N` 是输入序列中的 token（word）数，所以这些矩阵中的每一行都是一个长度为 `d` 的向量，包含一个输入词的嵌入（embedding，神经编码）。换句话说，`Q`、`K` 和 `V`都包含了输入标记的不同 `d` 维嵌入。

**重要注意事项：** 因为一个 batch 中存在多个注意力头和多个输入，为了提高模型的效率和表达能力，这个注意力模块通常会并行运行多次。确切地理解为什么会出现这种情况不重要，但是要知道，在你的实现中，这些矩阵将显示为 **4 维** 张量，你将只关心四个维度中的两个（对应于 $N\times d$ 的大小）。

注意力模块的第一步是计算单词之间的所有交互对。这是通过将查询矩阵 $Q$ 与键矩阵 $K$ 相乘来计算的：

$$S = QK^T.$$

接下来的计算是对 $S$ 的每一行执行的 [softmax 操作](https://machinelearningmastery.com/softmax-activation-function-with-python/)。softmax 生成每行的归一化概率。

对于矩阵的每一行，softmax 操作执行以下计算。请注意，我们为你提供了在一维向量 $X$ 上计算 softmax 的数学公式。你需要对上面的矩阵 $S$ 的每一行执行此数学计算。

$$\text{softmax}(x) = \frac{\mathbf f(x)}{l(x)}$$

其中

$$\mathbf f(x) = \begin{bmatrix}e^{x_1} & e^{x_2} &\cdots & e^{x_N} \end{bmatrix}\qquad \text{and} \qquad l(x) = \sum_{i=1}^N f(x)_i.$$

请注意，上述数学公式与你在讲座中看到的方程不同，因为没有从分子中减去`max(x)`。讲座中使用的版本在实践中是实践中用于数值稳定性的版本，但在这次作业中你可以使用上述数学公式。（如果使用上述数学公式，实现 FlashAttention 会容易得多。对于有抱负的人，如果你希望使用讲座中的版本，也可以……你可能会在“正确性检查”中看到差异。）

这将生成一个注意力权重矩阵 $P$，其中

$$P = \texttt{softmax}(\texttt{each row of }S).$$

最后，注意力权重用于聚合一组学习到的**值**向量，这些向量作为矩阵 $V$ 提供，形状为 $N \times d$，以产生最终输出 $O$：

$$O = PV.$$

总之，注意力层由一个昂贵的矩阵乘法组成，后面跟着一个 softmax 层，然后是另一个矩阵乘法。这三个组成部分将是你实现的主体——继续阅读以获取详细信息！

## 热身：访问张量（3分）

张量是 Pytorch 中使用的一种数据抽象。尽管这个名字听起来有点吓人，但它们不过是多维数组。通过将这些多维数组抽象成张量数据类型，普通的 PyTorch 程序员不再需要担心访问值或矩阵乘法等内部机制是如何工作的。此外，Pytorch 的张量允许轻松地在 GPU 上移植，因此它们可以在专门的硬件上运行，比如 Tensor 核心。然而，对于这个作业，我们将仅使用 CPU，并且我们希望你使用大家都熟悉的数据类型：C++ vector。

理解张量的关键是知道如何访问它们。这就是为什么我们希望你为一个 4 维张量编写访问器作为热身。对于第 1-4 部分，我们已经为你提供了一个名为`formatTensor`的函数，该函数将张量转换为 C++ vector。这为你提供了一个张量值的连续内存布局，类似于 Pytorch 存储张量数据的方式。对于第 1-4 部分，我们还会将输出 vector 转换回张量。

### 步骤1：理解二维访问器

在完成作业 3 后，你应该相对熟悉如何展平一个二维数组了。你的第一个任务是理解如何访问多维数组的元素，多维数组在内存中实际上只是作为一个展平的一维缓冲区存储的。我们为你提供了用于读写二维数组的示例访问器。你可以在`module.cpp`的顶部找到它们，名字叫做`twoDimRead`和`twoDimWrite`。给定的二维访问器将向你展示我们如何访问这个展平数组中的任意元素`(i, j)`。**注意，公式如下：对于数组内的任何给定元素 (i, j)，你可以使用 [i * 列数 + j] 来访问它。**

### 步骤2：实现一个四维访问器

在我们的 LLM 模型中，我们的数组是四维的，所以我们实际上需要一个四维访问器来访问它的元素！现在，扩展访问二维张量的概念，以便你可以访问四维张量。步骤 2 的任务是在文件`module.cpp`中实现函数`fourDimRead`和`fourDimWrite`。

### 测试：

运行以下命令来测试你的四维访问器：

```
python3 gpt149.py 4Daccess
```

在运行测试时，如果你正确实现了你的访问器，预期值和结果值应该是相同的，将产生如下输出。

```
Expected: 0.0006
Result: 0.0006
```

### 提交内容

- 在文件`module.cpp`中实现`fourDimRead`和`fourDimWrite`。

- 接下来，在你的报告中回答以下问题：
	- 简要描述四维张量/数组在内存中的布局方式。你认为为什么选择这种约定，它是如何利用硬件的？

## 第 1 部分：简单（但不太高效）的注意力机制实现（10分）

现在你已经拥有了访问器，是时候开始着手实现你的自定义注意力层了。作为这个作业的第一步，你将在 C++ 中实现一个无优化的串行注意力层。在 `myNaiveAttention` 中，我们为你提供了两个示例。第一个示例展示了如何将一个四维张量填充为 0，第二个示例展示了如何将一个二维张量填充为 0。扩展这些概念，使你能够实现注意力机制。你应该：

```
1) 对每个Batch:
2) 对每个Head:
    a) 遍历Q和K并将Q与K^t相乘，将结果存储在QK^t中 
    QK^t已为你预先分配，并作为参数传递给myNaiveAttention。
    （你不应该为这个作业的任何部分分配任何pytorch张量）
    
    注意，在索引Batch和Head后，你将得到Q和K的二维矩阵，形状为 (N, d)。同时注意K的维度为 (N, d)，而你想要的K^t的维度为 (d, N)。不转置K^t，如何直接通过改变相乘的顺序来得到QK^t呢？思考如何重新排序你的`for`循环以脱离传统的矩阵乘法。
   
    b) 在得到QK^t后——形状应该是(N, N) ——你应该遍历每一行。对于每一行，你应该获取每个元素的幂，可以使用 C++ 内建的`exp`函数。现在，将这些结果的幂除以其行中所有幂的总和，然后将它存回QK^t。
   
    c) 最后，你应该将QK^t与V进行矩阵乘法，并将结果存储在O中。
    注意，与Q和K类似，当你索引Batch和Head后，V和O的形状将为 (N, d)。因此，在你将QK^t (N, N) 与V (N, d) 相乘后，你可以直接将所得形状 (N, d) 存储回O。
```

### 测试

运行以下测试以检查你的程序的正确性：

```
python3 gpt149.py part1
```

在运行测试时，我们会展示 pytorch profiler 的结果——这些信息会以表格形式呈现，显示测试中所有函数调用的详细统计数据。导出的表格将如下所示：

```
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
              aten::empty         0.01%      23.000us         0.01%      23.000us       3.286us       5.00 Mb       5.00 Mb             7  
              aten::zeros         0.14%     321.000us         0.18%     408.000us     102.000us       4.50 Mb           4 b             4  
STUDENT - NAIVE ATTENTION        99.56%     229.600ms        99.97%     230.538ms     230.538ms       4.50 Mb      -1.00 Mb             1  
              aten::clone         0.02%      37.000us         0.10%     231.000us     115.500us       1.00 Mb           0 b             2  
            aten::flatten         0.02%      48.000us         0.07%     153.000us      30.600us     512.00 Kb           0 b             5  
         aten::empty_like         0.00%       3.000us         0.00%       8.000us       8.000us     512.00 Kb           0 b             1  
      aten::empty_strided         0.01%      16.000us         0.01%      16.000us      16.000us     512.00 Kb     512.00 Kb             1  
          model_inference         0.02%      38.000us        99.98%     230.578ms     230.578ms     512.00 Kb      -4.00 Mb             1  
              aten::zero_         0.02%      42.000us         0.15%     354.000us      88.500us           0 b           0 b             4  
              aten::fill_         0.14%     312.000us         0.14%     312.000us     156.000us           0 b           0 b             2  
-----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
```

在表格导出后，我们还会显示两个相关的统计数据：CPU时间（以毫秒为单位）和内存使用量（以字节为单位）。如果你正确实现了你的函数，你应该会看到如下输出：

```
REFERENCE - NAIVE ATTENTION statistics
cpu time:  230.724ms
mem usage:  4718588 bytes

STUDENT - NAIVE ATTENTION statistics
cpu time:  232.561ms
mem usage:  4718588 bytes
```

如果你的注意力机制没有产生正确的输出，你将看到以下消息：

```
YOUR ATTENTION PRODUCED INCORRECT RESULTS
```

注意，你不需要完全匹配参考的 `cpu time`，只要你仍然产生正确的结果。你应该尽量接近 CPU 时间。我们将为你提供 15 毫秒的缓冲时间。如果你的时间比参考解法慢 <= 15ms，那就可以了，你仍然会得到满分。当然，我们鼓励你超过参考的 CPU 时间，速度更快不会受到惩罚。

注意，即使你分配的中间变量比我们给你的更多的，内存使用值也不会改变。此内存使用量仅根据作为参数传递的变量进行分析。对于每个部分（1-4），我们为你提供了产生正确结果的最少变量。**你还可以假设我们传递的所有临时中间张量都被初始化为包含零。**我们这样做是因为我们希望你看到操作合并后的内存使用量下降，并且会有基于这些内存值的写作问题。添加任何更多的高内存数据结构可能只会损害你的性能，但你可以尝试在`module.cpp`文件中添加额外的变量，你不会因此受到惩罚。

你应该确保你的函数在不同的 N 值下工作正常，因为这将是我们评估你正确性的方式。我们提供了一个命令行参数：

```
python3 gpt149.py part1 -N <val>
```

如果你已经实现了你的注意力层，你还可以看到 DNN 使用你的注意力层生成文本，如果你希望输出更多文本，可以选择将模型更改为`shakes256`、`shakes1024`或`shakes2048`：

```
python3 gpt149.py part1 --inference -m shakes128
```

请注意，在推理过程中不会自动评分，这纯粹是为了好玩。请注意，由于溢出错误，模型`shakes1024`和`shakes2048`将无法与我们在本文中描述的 softmax 一起使用。如果你希望它们正常工作，你必须实现课堂上描述的“安全” softmax。这完全是可选的，因为我们在评分时总是确保给你合理的值。第 1-3 部分都遵循本节中列出的相同评分程序。

### 提交内容

- 在`module.cpp`中实现`myNaiveAttention`。

## 第 2 部分：块矩阵乘法和 Unfused Softmax（20分）

现在我们已经有了基线的矩阵乘法，让我们看看如何优化它。目前，我们的矩阵乘法行为如下：

<p align="center">
  <img src="./assets/current_matmul.png" width=40% height=40%>
</p>

请注意这个操作的缓存表现有多差。对于 C 的每个元素，我们都从 A 和 B 中加载了多个缓存行。然而，需要注意的是，这些矩阵的大小远大于我们的缓存大小。因此，当我们想处理 C 的下一个元素时，我们将重新加载已经被驱逐的缓存行。但是，如果我们重用这些缓存行呢？我们的代码效率低下的主要原因是我们一次处理一个C的元素，但如果我们一次处理一块元素呢？特别是，如果我们处理一个缓存行大小的元素呢？

你的任务是进一步扩展你的矩阵乘法，使其采用在[讲座](https://gfxcourses.stanford.edu/cs149/fall23/lecture/perfopt2/slide_43)中讨论的分块技术。把大矩阵分解成较小的缓存大小的子矩阵。然后你的乘法将在从缓存中驱逐它们之前处理。其行为应如下图所示：

<p align="center">
  <img src="./assets/blocked_matmul.png" width=40% height=40%>
</p>

再举一个例子，假设我有 3 个 NxN 矩阵和一个大小为 L 的缓存行。然后我会将我的 3 个 NxN 矩阵分解成 (N/L)x(N/L) 子矩阵。这是怎样改进缓存利用的？

然而，需要牢记的是，我们的矩阵并不是完全的方阵。我们的 $Q$ 和 $K^{T}$ 矩阵的形状分别是 Nxd 和 dxN。在你尝试将矩阵分解成块时，请记住这一点。此外，你的块大小并不总是能均匀地整除 N，这意味着你将有一些“剩余”块没有充满数据。在这种情况下，你不想遍历整个“剩余”块，而只想遍历一个“子块”，其维度为`min(tile_size, N-tileIndex*tileSize)`。

还请记住，如前所述，需要的临时内存已预先分配，并传递给你需要实现的函数（`myUnfusedAttentionBlocked`）——你不需要自己分配任何东西，尽管这样做不会受到惩罚。

**请注意，你在这里有两次机会进行块矩阵乘法：QK^t 和 PV。你应该对两者都使用块矩阵乘法，以实现参考的加速效果。**

### 测试：

运行以下测试以检查你的程序的正确性：

```
python3 gpt149.py part2
```

正确的实现应产生以下输出：

```
REFERENCE - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  156.271ms
mem usage:  4718588 bytes

STUDENT - BLOCKED MATMUL + UNFUSED SOFTMAX statistics
cpu time:  160.891ms
mem usage:  4718588 bytes
```

不正确的实现将产生以下输出：

```
YOUR ATTENTION PRODUCED INCORRECT RESULTS
```

和第 1 部分一样，我们将自动评估函数输出的正确性及其 CPU 时间。你同样有与参考解法相比的 <=15ms 缓冲时间。如果你的程序更快，你不会受到惩罚。

你应该确保你的函数在不同的 N 值下工作正常，因为这将是我们评估你正确性的方式。我们提供了一个命令行参数：

```
python3 gpt149.py part2 -N <val>
```

你可以看到 DNN 使用你的注意力层生成文本，如果你希望输出更多文本，可以选择将模型更改为`shakes256`、`shakes1024`或`shakes2048`：

```
python3 gpt149.py part2 --inference -m shakes128
```

请注意，在推理过程中不会自动评分，这纯粹是为了好玩。请注意，由于溢出错误，模型`shakes1024`和`shakes2048`将无法与我们在本文中描述的 softmax 一起使用。如果你希望它们正常工作，你必须实现课堂上描述的“安全” softmax。这完全是可选的，因为我们在评分时总是确保给你合理的值。

### 提交内容

- 在`module.cpp`中实现`myUnfusedAttentionBlocked`。

- 然后，在你的报告中回答以下问题：
	- 分享一些当 N=1024 时你尝试的块大小的数据，以及每个块大小的性能时间。矩阵乘法的最佳块大小是什么？解释你为什么认为这个块大小最适合你的实现。这里没有错误的答案，我们只是想看到你进行了实验并尝试得出结论。
	- 对于 $Q$ (Nxd) 和 $K^{T}$ (dxN) 的矩阵乘法，第 2 部分中的 DRAM 访问与 第 1 部分中的 DRAM 访问的比率是多少？（假设 4 字节浮点基元，64 字节缓存行，以及 N 和 d 非常大）。

## 第 3 部分：融合注意力机制（25分）

到目前为止，我们已经看到 $Q * K^{T}$ 产生了一个巨大的 NxN 矩阵。在不同的函数中执行矩阵乘法和 softmax 需要我们写入 NxN 矩阵的每一行，然后在随后的 softmax 中再次遍历这个 NxN 矩阵，然后在与 V 相乘时第三次遍历 softmax 后的矩阵。这不仅对缓存性能不好，而且对程序的内存占用也非常不利。

幸运的是，我们可以通过“融合”计算来解决这两个问题，这样我们只需要一个 Nx1 的临时向量，而不是 NxN 的临时矩阵。

你可以通过观察以下事实来做到这一点。一旦我们计算了 $Q * K^t$ NxN 矩阵的一行，我们实际上就可以对整行进行 softmax，而不必计算 NxN 矩阵的其余部分。

一旦该行进行了 softmax 处理，我们就可以立即将经过 softmax 处理后的行乘以 V，以完全计算我们的注意力输出的第一行（这是一个合理的大小：Nxd）。换句话说，我们可以计算 $Q * K^{t}$ 的一行后，对其进行 softmax，然后将 softmax 后的行乘以V。这样做不需要创建 NxN 矩阵……只需要创建一个 Nx1 大小的中间向量来保存$Q*K^{t}$ 的第一行及其 softmax。然后我们可以重新使用这个 Nx1 数组来计算第二行注意力，接着是第三行，依此类推。这意味着我们从未实际生成 NxN 矩阵，这很好，因为该矩阵在网络的后续计算中也从未被再次使用。

### 使用 OpenMP 进行并行化

你可能注意到，随着我们融合了矩阵乘法和 softmax，我们使计算的很大一部分变得非常容易并行化。例如，我们可以独立计算输出矩阵的 Batch、Head 和行。这是一个多线程的绝佳机会！这次我们将使用 OpenMP，因此你不必实现自己的线程池。OpenMP 语法相对简单。如果你想并行化一段代码，它看起来像下面这样：

```
#pragma omp parallel for collapse()

-- code is here --
```

如果你发现循环直接嵌套在另一个循环之上并希望并行化它们，你会发现 `#pragma omp parallel for collapse()` 非常有用。例如，对于一个完美嵌套的三重循环：

```
#pragma omp parallel for collapse(3)

for ()

    for()

        for()
```

注意：在使用 OpenMP 时，写入一个 Nx1 的临时行，通常需要小心，因为这会引发竞争。为了解决这个问题，我们为你提供了前三个循环的框架（你将需要更多的循环），其中每个 OpenMP 线程都被分配了自己的 Nx1 临时数组的副本，以避免竞争。这个局部副本的数组是我们为你分配的临时内存的一部分，并作为参数传递给函数（`myFusedAttention`）。请记住，在你尝试并行化的循环内声明的任何变量都将是每个线程私有的。

### 测试：

运行以下测试以检查你的程序的正确性：

```
python3 gpt149.py part3
```

正确的实现应产生以下输出：

```
REFERENCE - FUSED ATTENTION statistics
cpu time:  32.361ms
mem usage:  557052 bytes

STUDENT - FUSED ATTENTION statistics
cpu time:  33.209ms
mem usage:  557052 bytes
```

不正确的实现将产生以下输出：

```
YOUR ATTENTION PRODUCED INCORRECT RESULTS
```

和第 1 和第 2 部分一样，我们将自动评估函数输出的正确性及其 CPU 时间。你同样有与参考解法相比的 <=15ms 缓冲时间。如果你的程序更快，你不会受到惩罚。

你应该确保你的函数在不同的 N 值下工作正常，因为这将是我们评估你正确性的方式。我们提供了一个命令行参数：

```
python3 gpt149.py part3 -N <val>
```

现在，你可以看到 DNN 使用你的注意力层生成文本，如果你希望输出更多文本，可以选择将模型更改为`shakes256`、`shakes1024`或`shakes2048`：

```
python3 gpt149.py part3 --inference -m shakes128
```

请注意，在推理过程中不会自动评分，这纯粹是为了好玩。请注意，由于溢出错误，模型`shakes1024`和`shakes2048`将无法与我们在本文中描述的 softmax 一起使用。如果你希望它们正常工作，你必须实现课堂上描述的“安全” softmax。这完全是可选的，因为我们在评分时总是确保给你合理的值。

### 提交内容

- 在`module.cpp`中实现`myFusedAttention`。

- 然后，在你的报告中回答以下问题：
	- 为什么与第 1 和第 2 部分相比，我们在第 3 部分使用的内存量少得多？
	- 注释掉你的`#pragma omp ...`语句，你的 CPU 时间会发生什么变化？在你的报告中记录 CPU 时间。为什么与第 1 部分相比，融合注意力使我们更容易充分利用多线程？

## 第 4 部分：整合——Flash Attention（35分）

### 为什么矩阵乘法和 Softmax 难以作为块进行融合？

注意力公式由于几个原因而难以融合。注意到公式由一个矩阵乘法组成，接着是来自 softmax 的逐行计算，最后是另一个矩阵乘法。真正使得将这三种操作作为块融合变得困难的是 softmax 必须对整行进行操作。所以，如果我们想绕过这种依赖性，我们真的需要跳出思维框架。这就是 Flash Attention 的作用所在。

### 将 Softmax 分解成块

假设我们有一个大小为 BLOCKSIZE 的向量，我们将其表示为 $x \in \mathbb{R}^{B}$。 $x$ 的 softmax 可以表示为：

<p align="center">
  <img src="./assets/Softmax_decomp1.png" width=55% height=55%>
</p>

由此可见，如果我们有两个大小为 BLOCKSIZE 的向量，分别表示为 $x \in \mathbb{R}^{B}$ 和 $y \in \mathbb{R}^{B}$，那么我们可以将 $softmax([x\ y])$ 分解为：

<p align="center">
  <img src="./assets/Softmax_decomp2.png" width=55% height=55%>
</p>

### 实现 Flash Attention

你的任务是将 softmax 分解成块，以便与块矩阵乘法融合。因此，对于每个块，你将 $Q$ (BLOCKROWSIZE x d) 与 $K^{t}$ (d x BLOCKCOLUMNSIZE) 相乘，得到 $QK^t$ (BLOCKROWSIZE x BLOCKCOLUMNSIZE)。然后计算 $\texttt{softmax}(QK^t)$ (BLOCKROWSIZE x BLOCKCOLUMNSIZE) 并将其与 $V$ (BLOCKCOLUMNSIZE x d) 相乘，得到 $O$ (BLOCKROWSIZE x d)。记住，这是一种累积过程，就像块矩阵乘法一样！

通过这样做，我们可显著减少内存占用。 我们将能够将其减少到 $O(N)$ 线性缩放的内存占用，而非 $O(N^{2})$ 的内存占用。

### Flash Attention 伪代码

下面显示的 Flash Attention 算法，将矩阵 $Q$、$K$ 和 $V$ 的块导入到较小的物理块中。然后在每个块中计算局部 softmax，并将此结果块写回到完整的输出矩阵 $O$ 中。例如，对于 $Q$，每个块的大小为 (Br x d)，而 $K$ 的块大小为 (Bc x d)。计算 Br 和 Bc（如下伪代码所示）需要知道你的 SRAM/cache 的大小 $M$，在本例中，$M=131072$ 个浮点数。对于此编程作业，你的程序应能够处理我们给定的任何 Br/Bc。

<p align="center">
  <img src="./assets/FlashAttentionPseudo.png" width=65% height=65%>
</p>

### 测试：

运行以下测试以检查你的程序的正确性：

```
python3 gpt149.py part4
```

**确保在不同的块大小上测试你的实现。** 运行此测试时，$N$ 和 $d$ 的默认值分别为 $1024$ 和 $32$。确保你的程序能够处理任意块大小，无论你的块大小是否能整除这些 $N/d$ 值。我们为你提供了命令行标志来更改注意力算法的 $Br$ 和 $Bc$ 参数。你可以使用 `-br <value>` 和 `-bc <value>` 标志来进行更改。每个的默认值为 $256$。例如，如果想将 $Br$ 更改为 $128$ 并将 $Bc$ 更改为 $512$，运行：

```
python3 gpt149.py part4 -br 128 -bc 512
```

正确的实现应产生以下输出：

```
REFERENCE - FLASH ATTENTION statistics
cpu time:  435.709ms
mem usage:  524284 bytes

STUDENT - FLASH ATTENTION statistics
cpu time:  435.937ms
mem usage:  524284 bytes
```

不正确的实现将产生以下输出：

```
YOUR ATTENTION PRODUCED INCORRECT RESULTS
```

请注意，CPU 速度实际上比第 3 部分低。为什么会这样？在报告中回答这个问题。

你应该确保你的函数在不同的 N 值下工作正常，因为这将是我们评估你正确性的方式。你应该测试不同的 N 值以及不同的块大小。请注意，参考解法会先运行，所以如果参考解法失败，那么你不必担心该 N/Br/Bc 组合。 要更改 N、Br 和 Bc 的值，请使用以下命令行参数：

```
python3 gpt149.py part4 -N <val> -br <val> -bc <val>
```

你可以看到 DNN 使用你的注意力层生成文本，如果你希望输出更多文本，可以选择将模型更改为`shakes256`、`shakes1024`或`shakes2048`：

```
python3 gpt149.py part4 --inference -m shakes128
```

请注意，在推理过程中不会自动评分，这纯粹是为了好玩。请注意，由于溢出错误，模型`shakes1024`和`shakes2048`将无法与我们在本文中描述的 softmax 一起使用。如果你希望它们正常工作，你必须实现课堂上描述的“安全” softmax。这完全是可选的，因为我们在评分时总是确保给你合理的值。

### 提交内容

- 在`module.cpp`中实现`myFlashAttention`。

- 然后，在报告中回答以下问题：
	- 第 4 部分的内存使用情况与之前各部分相比如何？为什么会这样？
	- 注意第 4 部分的性能比之前各部分要慢。我们是否完全优化了第 4 部分？还可以进行哪些性能改进？请列出它们并描述为什么它们会提高性能。

## 额外加分：进一步优化（总计12分 - 每部分3分）

### 使用 ISPC 内置函数进行向量化

你可能注意到有许多基于循环的无分支浮点操作。这是使用向量内置函数的绝佳机会！我们为你提供了 ISPC 支持，以编写自己的向量化函数，例如矩阵乘法和行求和。代码库中包含一个名为`module.ispc`的文件。你可以在这里自由编写自己的 ISPC 函数，并使用以下命令编译它们：

```
ispc -O3 --target=avx2-i32x8 --arch=x86-64 --pic module.ispc -h module_ispc.h -o module_ispc.o 
```

要在你的`module.cpp`文件中启用它们，只需取消文件顶部的以下两行注释：

```
#include "module_ispc.h"
using namespace ispc;
```

### 报告问题

- 请在`writeup.pdf`中记录向量化后的加速效果和你的实现。

## 分数分配：（总计100分 + 12分额外加分）

- 实现`fourDimRead`：1.5分
- 实现`fourDimWrite`：1.5分
- 实现`myNaiveAttention`：10分
- 实现`myUnfusedAttentionBlocked`：20分
- 实现`myFusedAttention`：25分
- 实现`myFlashAttention`：35分
- 回答报告问题：7分
	- 1 个热身问题
	- 2 个第 2 部分问题
	- 2 个第 3 部分问题
	- 2 个第 4 部分问题
- 额外加分：对第 1-4 部分进行向量化：每部分 3 分

## 提交说明

请使用 [Gradescope](https://www.gradescope.com/) 提交你的作业。如果你与合作伙伴一起完成，请记得在 Gradescope 上标记你的合作伙伴。

请将你的报告问题的回答放在一个名 `writeup.pdf`的文件中。记住在 Gradescope 上将页面映射到问题。如果你完成了额外加分，请在报告的结尾注明，因为我们将手动运行这些部分。此外，请记录使用向量化加速的每部分预期性能数据。

- 请将以下文件提交到 Assignment 4 (Code)：
	- module.cpp
	- module.ispc（如果你尝试了额外加分）
- 请将你的报告文件`writeup.pdf`提交到 Assignment 4 (Write-Up)。
