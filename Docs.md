# CS336 作业 1（基础）：构建 Transformer 语言模型

## 1 作业概述

在本次作业中，你将从头构建训练一个标准 Transformer 语言模型（LM）所需的所有组件，并训练一些模型。

**你将实现的内容**
1. 字节对编码 (BPE) 分词器（Tokenizer）(§2)
2. Transformer 语言模型 (LM) (§3)
3. 交叉熵（Cross-entropy）损失函数和 AdamW 优化器 (§4)
4. 训练循环，支持序列化和加载模型及优化器状态 (§5)

**你将运行的内容**
1. 在 TinyStories 数据集上训练一个 BPE 分词器。
2. 在该数据集上运行你训练好的分词器，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练一个 Transformer LM。
4. 使用训练好的 Transformer LM 生成样本并评估困惑度（perplexity）。
5. 在 OpenWebText 上训练模型，并将你获得的困惑度提交到排行榜。

**你可以使用的内容** 
我们希望你从头开始构建这些组件。特别地，你**不得**使用来自 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 的任何定义，但以下情况除外：
* `torch.nn.Parameter`
* `torch.nn` 中的容器类（例如，`Module`、`ModuleList`、`Sequential` 等）[^1]
* `torch.optim.Optimizer` 基类

你可以使用任何其他的 PyTorch 定义。如果你想使用某个函数或类但不确定是否被允许，请随时在 Slack 上提问。如果有疑问，请考虑使用它是否违背了本作业“从头开始构建”的宗旨。

[^1]: 完整列表请参阅 PyTorch.org/docs/stable/nn.html#containers。

---

**关于 AI 工具的声明** 
允许使用 ChatGPT 等 LLM 来询问底层的编程问题或关于语言模型的高层次概念问题，但禁止直接使用它们来解决作业问题。
我们强烈建议你在完成作业时在 IDE 中禁用 AI 自动补全（例如 Cursor Tab、GitHub Copilot）（不过非 AI 自动补全，例如自动补全函数名是完全没问题的）。我们发现，AI 自动补全会让人很难深入参与到内容中。

**代码结构说明** 
所有作业代码以及本文档均可在 GitHub 上获取：
`github.com/stanford-cs336/assignment1-basics`
请使用 `git clone` 克隆该仓库。如果有任何更新，我们会通知你，以便你可以通过 `git pull` 获取最新版本。

1. `cs336_basics/*`: 这是你编写代码的地方。注意这里面没有预置代码——你可以从头开始做任何你想做的事！
2. `adapters.py`: 这是一组你的代码必须具备的接口功能。对于每一个功能块（例如，缩放点积注意力），只需通过调用你的代码来填写其实现（例如 `run_scaled_dot_product_attention`）。注意：你在 `adapters.py` 中的修改不应包含任何实质性逻辑；这只是粘合代码（glue code）。
3. `test_*.py`: 这里包含了你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），这些测试将调用在 `adapters.py` 中定义的钩子（hooks）。**不要修改测试文件**。

**如何提交** 
你需要向 Gradescope 提交以下文件：
* `writeup.pdf`: 回答所有书面问题。请对你的回答进行排版。
* `code.zip`: 包含你编写的所有代码。

要提交到排行榜，请提交一个 PR（Pull Request）到：
`github.com/stanford-cs336/assignment1-basics-leaderboard`
有关详细的提交说明，请参阅排行榜仓库中的 `README.md`。

**何处获取数据集** 
本次作业将使用两个预处理过的数据集：TinyStories [Eldan and Li, 2023] 和 OpenWebText [Gokaslan et al., 2019]。这两个数据集都是单个大型纯文本文件。如果你正与全班一起完成作业，可以在任何非头节点机器的 `/data` 目录下找到这些文件。
如果你在家里跟着做，你可以使用 `README.md` 中的命令下载这些文件。

> **低资源/降级提示：初始化**
> 在整个课程的作业文档中，我们将提供在使用较少或没有 GPU 资源的情况下完成作业各部分的建议。例如，我们有时会建议**缩小（downscaling）**你的数据集或模型大小，或者解释如何在 MacOS 集成 GPU 或 CPU 上运行训练代码。你会在一个蓝色框中找到这些“低资源提示”（就像这个一样）。即使你是注册了课程且有权限访问课程机器的斯坦福学生，这些提示也可能帮助你更快地迭代并节省时间，因此我们建议你阅读它们！

> **低资源/降级提示：在 Apple Silicon 或 CPU 上进行作业 1**
> 使用教学团队的解决方案代码，我们可以在配备 36 GB RAM 的 Apple M3 Max 芯片上训练一个能生成相当流畅文本的语言模型（LM），在 Metal GPU (MPS) 上耗时不到 5 分钟，在 CPU 上约需 30 分钟。如果这些词汇对你来说意义不大，不用担心！只要知道，如果你有一台相当新的笔记本电脑，并且你的实现既正确又高效，你就能够训练出一个能生成带有不错流畅度的小型儿童故事的语言模型。
> 在作业的后半部分，我们将解释如果你在使用 CPU 或 MPS，需要做哪些更改。

---

## 2 字节对编码 (BPE) 分词器 (Tokenizer)

在作业的第一部分，我们将训练并实现一个字节级字节对编码 (BPE) 分词器 [Sennrich et al., 2016, Wang et al., 2019]。具体来说，我们将任意 (Unicode) 字符串表示为字节序列，并在该字节序列上训练我们的 BPE 分词器。稍后，我们将使用这个分词器将文本（字符串）编码为词元（token，整数序列），以用于语言建模。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它将字符映射到整数**代码点（code points）**。截至 Unicode 16.0（发布于 2024 年 9 月），该标准定义了 168 种书写系统中的 154,998 个字符。例如，字符“s”的代码点是 115（通常记为 `U+0073`，其中 `U+` 是常规前缀，`0073` 是 115 的十六进制），而字符“牛”的代码点是 29275。在 Python 中，你可以使用 `ord()` 函数将单个 Unicode 字符转换为其整数表示。`chr()` 函数则将整数的 Unicode 代码点转换为包含对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

> **问题 (unicode1)：理解 Unicode (1 分)**
> 
> (a) `chr(0)` 返回的是什么 Unicode 字符？
> **交付物**：一句话的回答。
> 
> (b) 该字符的字符串表示（`__repr__()`）与其打印出的表示有何不同？
> **交付物**：一句话的回答。
> 
> (c) 当该字符出现在文本中时会发生什么？在 Python 解释器中运行以下代码并查看是否符合你的预期，这可能会有所帮助：
> ```python
> >>> chr(0)
> >>> print(chr(0))
> >>> "this is a test" + chr(0) + "string"
> >>> print("this is a test" + chr(0) + "string")
> ```
> **交付物**：一句话的回答。

### 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到代码点（整数）的映射，但直接在 Unicode 代码点上训练分词器是不切实际的，因为词汇表（vocabulary）将会非常大（大约 15 万项）且稀疏（因为许多字符非常罕见）。相反，我们将使用一种 Unicode 编码，它将一个 Unicode 字符转换为一个字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网的主导编码（占所有网页的 98% 以上）。

要将 Unicode 字符串编码为 UTF-8，我们可以使用 Python 中的 `encode()` 函数。要访问 Python `bytes` 对象的底层字节值，我们可以遍历它（例如，调用 `list()`）。最后，我们可以使用 `decode()` 函数将 UTF-8 字节字符串解码回 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # 获取编码字符串的字节值 (0 到 255 的整数).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # 一个字节不一定对应一个 Unicode 字符！
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过将我们的 Unicode 代码点转换为字节序列（例如，通过 UTF-8 编码），我们实际上是将代码点序列（范围在 0 到 154,997 之间的整数）转换为字节值序列（范围在 0 到 255 之间的整数）。长度为 256 的字节词汇表处理起来要容易得多。在使用字节级分词时，我们不需要担心词汇表外（out-of-vocabulary, OOV）的 token，因为我们知道*任何*输入文本都可以表示为 0 到 255 之间的整数序列。

> **问题 (unicode2)：Unicode 编码 (3 分)**
> 
> (a) 相比于 UTF-16 或 UTF-32，为什么我们更倾向于在 UTF-8 编码的字节上训练我们的分词器？对比这些编码对各种输入字符串的输出可能会有所帮助。
> **交付物**：一到两句话的回答。
> 
> (b) 考虑以下（不正确的）函数，该函数旨在将 UTF-8 字节字符串解码为 Unicode 字符串。为什么这个函数是不正确的？提供一个产生错误结果的输入字节字符串示例。
> ```python
> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
>     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
> 
> >>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
> 'hello'
> ```
> **交付物**：一个会导致 `decode_utf8_bytes_to_str_wrong` 产生错误输出的示例输入字节字符串，以及用一句话解释为什么该函数不正确。
> 
> (c) 给出一个不能解码为任何 Unicode 字符（们）的两个字节的序列。
> **交付物**：一个例子，并附带一句话解释。

### 2.3 子词（Subword）分词

虽然字节级分词可以缓解词级（word-level）分词器面临的词汇表外（OOV）问题，但将文本分词为字节会导致输入序列极长。这会减慢模型训练的速度，因为在词级语言模型中只占 10 个 token 的包含 10 个单词的句子，在字符级模型中可能长达 50 个或更多的 token（取决于单词的长度）。处理这些更长的序列需要在模型的每一步进行更多的计算。此外，由于较长的输入序列在数据中产生了长距离的依赖关系，在字节序列上进行语言建模是困难的。

子词分词是词级分词器和字节级分词器之间的折中方案。请注意，字节级分词器的词汇表有 256 个条目（字节值为 0 到 255）。子词分词器通过更大的词汇表大小来换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在我们的原始文本训练数据中经常出现，将其作为词汇表中的一个条目，将把这个 3 个 token 的序列缩减为单个 token。

我们如何选择这些子词单元加入词汇表？Sennrich 等人 [2016] 提出使用字节对编码（BPE; Gage, 1994），这是一种压缩算法，它迭代地将最高频的字节对替换（“合并”）为一个新的、未使用的单一索引。注意，该算法将子词 token 添加到我们的词汇表中，以最大化我们输入序列的压缩率——如果一个词在我们的输入文本中出现足够多次，它将被表示为单个子词单元。

词汇表通过 BPE 构建的子词分词器通常被称为 **BPE 分词器**。在本次作业中，我们将实现一个字节级 BPE 分词器，其中词汇表项是字节或字节的合并序列，这让我们在处理词汇表外单词和可控输入序列长度两方面两全其美。构建 BPE 分词器词汇表的过程称为“训练” BPE 分词器。

### 2.4 BPE 分词器训练

BPE 分词器训练过程包括三个主要步骤。

**词汇表初始化（Vocabulary initialization）**：分词器词汇表是字节串 token 到整数 ID 的一对一映射。由于我们正在训练字节级 BPE 分词器，我们的初始词汇表只是所有字节的集合。因为存在 256 种可能的字节值，所以我们的初始词汇表大小为 256。

**预分词（Pre-tokenization）**：一旦有了词汇表，原则上你可以统计你的文本中字节相邻出现的频率，并合并它们。然而，这在计算上是相当昂贵的，因为每次合并我们都需要完整遍历一次语料库。此外，直接在整个语料库中合并字节可能会导致仅仅在标点符号上有所不同的 token（例如，`dog!` 与 `dog.`）。这些 token 将获得完全不同的 token ID，即使它们可能具有很高的语义相似度（因为它们仅在标点符号上有所不同）。

为了避免这种情况，我们对语料库进行**预分词（pre-tokenize）**。你可以将此视为对语料库的一种粗粒度分词，它有助于我们统计字符对出现的频率。例如，单词 `'text'` 可能是一个出现了 10 次的预分词 token。在这种情况下，当我们统计字符 ‘t’ 和 ‘e’ 相邻出现的频率时，我们会看到单词 ‘text’ 中 ‘t’ 和 ‘e’ 是相邻的，因此我们可以直接将其计数增加 10，而不需要重新遍历语料库。由于我们正在训练一个字节级别的 BPE 模型，每个预分词 token 都被表示为 UTF-8 字节的序列。

Sennrich 等人 [2016] 的原始 BPE 实现通过简单地按空格拆分（即 `s.split(" ")`）来进行预分词。相比之下，我们将使用一个基于正则表达式的预分词器（被 GPT-2 使用；Radford 等人，2019），来自 `github.com/openai/tiktoken/pull/234/files`：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

使用这个预分词器交互式地拆分一些文本，可能会有助于你更好地了解其行为：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

然而，在你的代码中使用它时，你应该使用 `re.finditer`，以避免在构建从预分词 token 到其计数的映射时将预分词后的单词存储下来。

**计算 BPE 合并（Compute BPE merges）** 既然我们已经将输入文本转换为了预分词 token，并将每个预分词 token 表示为 UTF-8 字节序列，我们就可以计算 BPE 合并（即训练 BPE 分词器）。在宏观层面上，BPE 算法迭代地统计每一对字节，并找出频率最高的一对（“A”，“B”）。然后将这对最高频组合（“A”，“B”）的每次出现进行**合并**，即替换为一个新的 token “AB”。这个新合并的 token 会被添加到我们的词汇表中；因此，BPE 训练后的最终词汇表大小等于初始词汇表的大小（在我们的例子中是 256），加上训练期间执行的 BPE 合并操作的次数。为了在 BPE 训练期间提高效率，我们不考虑跨越预分词 token 边界的字节对。[^2] 在计算合并时，如果出现字节对频率相同的情况，请通过*优先选择字典序较大的字节对*来确定性地打破平局。例如，如果字节对 ("A", "B")、("A", "C")、("B", "ZZ") 和 ("BA", "A") 都具有最高频率，我们将合并 ("BA", "A")：

[^2]: 注意，最初的 BPE 公式 [Sennrich 等人，2016] 指定了要包含一个词尾（end-of-word）token。在训练字节级 BPE 模型时，我们不添加词尾 token，因为所有字节（包括空格和标点符号）都已包含在模型的词汇表中。由于我们显式地表示了空格和标点符号，学习到的 BPE 合并将自然地反映这些词的边界。

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

**特殊 token（Special tokens）** 通常，一些字符串（例如 `<|endoftext|>`）被用于编码元数据（例如文档之间的边界）。在编码文本时，我们通常希望将某些字符串视为“特殊 token”，它们绝对不应该被拆分成多个 token（即始终作为单个 token 保留）。例如，序列结束字符串 `<|endoftext|>` 应始终作为单个 token（即单个整数 ID）保留，这样我们就知道何时停止从语言模型生成文本。这些特殊 token 必须被添加到词汇表中，以便它们具有对应的固定 token ID。

Sennrich 等人 [2016] 的算法 1 包含了一个效率较低的 BPE 分词器训练实现（基本上遵循了我们上面概述的步骤）。作为第一个练习，实现并测试这个函数以检验你的理解可能会很有帮助。

> **示例 (bpe_example)：BPE 训练示例**
> 
> 以下是 Sennrich 等人 [2016] 中的一个简化示例。考虑一个由以下文本组成的语料库：
> ```text
> low low low low low
> lower lower widest widest widest
> newest newest newest newest newest newest
> ```
> 且词汇表中包含一个特殊 token `<|endoftext|>`。
> 
> **词汇表（Vocabulary）** 我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化我们的词汇表。
> 
> **预分词（Pre-tokenization）** 为了简单起见并聚焦于合并过程，我们在这个例子中假设预分词只是按空格拆分。当我们进行预分词和计数时，我们得到了如下的频率表：
> `{'low': 5, 'lower': 2, 'widest': 3, 'newest': 6}`
> 
> 在 Python 中将其表示为 `dict[tuple[bytes], int]` 会很方便，例如 `{('l','o','w'): 5 ...}`。请注意，即使是单个字节在 Python 中也是 `bytes` 对象。Python 中没有 `byte` 类型来表示单个字节，就像 Python 中没有 `char` 类型来表示单个字符一样。
> 
> **合并（Merges）** 我们首先查看每一对相邻的字节，并对它们出现的单词频率求和 `{'lo': 7, 'ow': 7, 'we': 8, 'er': 2, 'wi': 3, 'id': 3, 'de': 3, 'es': 9, 'st': 9, 'ne': 6, 'ew': 6}`。字节对 `('es')` 和 `('st')` 出现了平局，因此我们取字典序较大的字节对 `('st')`。然后我们将预分词 token 合并，使得我们最终得到 `{('l','o','w'): 5, ('l','o','w','e','r'): 2, ('w','i','d','e','st'): 3, ('n','e','w','e','st'): 6}`。
> 
> 在第二轮中，我们看到 `('e', 'st')` 是最常见的字节对（计数为 9），我们将合并得到 `{('l','o','w'): 5, ('l','o','w','e','r'): 2, ('w','i','d','est'): 3, ('n','e','w','est'): 6}`。继续这个过程，我们最终得到的合并序列将是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。
> 
> 如果我们进行 6 次合并，我们会得到 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，并且我们的词汇表元素将包含 `[<|endoftext|>, [...256 个字节字符], st, est, ow, low, west, ne]`。
> 
> 使用这个词汇表和这组合并规则，单词 `newest` 将被分词为 `[ne, west]`。

### 2.5 BPE 分词器训练实验

让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明可以在第 1 节中找到。在开始之前，我们建议你看一下 TinyStories 数据集，以了解数据中包含的内容。

**并行化预分词（Parallelizing pre-tokenization）** 你会发现预分词步骤是一个主要瓶颈。你可以通过使用内置库 `multiprocessing` 并行化你的代码来加速预分词。具体来说，我们建议在预分词的并行实现中，对语料库进行分块（chunk），同时确保你的块边界出现在特殊 token 的开头。你可以随意原封不动地使用以下链接处的初始代码来获取块边界，然后你可以使用这些边界将工作分配给你的进程：

`https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py`

这种分块方法将始终有效，因为我们永远不想跨越文档边界进行合并。就本作业而言，你可以始终以这种方式进行拆分。不要担心收到一个非常大且不包含 `<|endoftext|>` 的语料库这种边缘情况。

**在预分词之前移除特殊 token** 在使用正则表达式模式（通过 `re.finditer`）运行预分词之前，你应该从你的语料库（如果你使用了并行实现，则是你的文本块）中剥离所有的特殊 token。确保你在特殊 token 处进行拆分，这样在它们分隔的文本之间就不会发生合并。例如，如果你有一个像 `[Doc 1]<|endoftext|>[Doc 2]` 这样的语料库（或文本块），你应该在特殊 token `<|endoftext|>` 处拆分，并分别对 `[Doc 1]` 和 `[Doc 2]` 进行预分词，这样文档边界上就不会发生合并。这可以通过使用 `re.split` 并以 `"|".join(special_tokens)` 作为分隔符来完成（请谨慎使用 `re.escape`，因为特殊 token 中可能会出现 `|`）。测试 `test_train_bpe_special_tokens` 将对此进行测试。

**优化合并步骤** 上面简化示例中 BPE 训练的朴素实现很慢，因为对于每次合并，它都会遍历所有字节对以找出最高频的组合。然而，每次合并后，唯一会发生改变的字节对计数是那些与已合并组合重叠的计数。因此，可以通过索引所有字节对的计数并递增更新这些计数，而不是显式地遍历每对字节来统计频率，从而提高 BPE 训练的速度。你可以通过这种缓存过程获得显著的加速，不过我们要指出的是，BPE 训练的合并部分在 Python 中是无法并行化的。

> **低资源/降级提示：性能分析 (Profiling)**
> 
> 你应该使用像 `cProfile` 或 `scalene` 这样的性能分析工具来识别你实现中的瓶颈，并集中精力优化这些地方。

> **低资源/降级提示：“降级 (Downscaling)”**
> 
> 我们建议你不要一上来就在完整的 TinyStories 数据集上训练分词器，而是先在一小部分数据上训练：即“调试数据集”。例如，你可以选择在 TinyStories 的验证集上训练分词器，该验证集有 22K 个文档，而不是 2.12M 个。这展示了在开发过程中尽可能使用降级策略来加速开发的通用方法：例如，使用更小的数据集、更小的模型规模等。选择调试集的大小或超参数配置需要仔细考量：你希望你的调试集足够大，以至于能够暴露出与完整配置相同的瓶颈（这样你做的优化才能泛化），但又不能大到需要运行无尽的时间。

> **问题 (train_bpe)：BPE 分词器训练 (15 分)**
> 
> **交付物**：编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。你的 BPE 训练函数应该处理（至少）以下输入参数：
> 
> `input_path: str` 包含 BPE 分词器训练数据的文本文件路径。
> 
> `vocab_size: int` 一个正整数，定义最大最终词汇表大小（包括初始字节词汇表、合并产生的词汇表项以及任何特殊 token）。
> 
> `special_tokens: list[str]` 要添加到词汇表中的特殊 token 字符串列表。这些特殊 token 之外不会影响 BPE 训练。
> 
> 你的 BPE 训练函数应返回生成的词汇表和合并列表：
> 
> `vocab: dict[int, bytes]` 分词器词汇表，从 `int`（词汇表中的 token ID）到 `bytes`（token 字节）的映射。
> 
> `merges: list[tuple[bytes, bytes]]` 训练产生的 BPE 合并列表。每个列表项是一个字节元组 `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 被合并。合并项应按创建顺序排序。
> 
> 为了根据我们提供的测试来测试你的 BPE 训练函数，你需要首先在 `[adapters.run_train_bpe]` 处实现测试适配器。然后，运行 `uv run pytest tests/test_train_bpe.py`。你的实现应该能够通过所有测试。作为可选项（这可能需要投入大量时间），你可以使用某种系统级语言来实现训练方法的核心部分，例如 C++（考虑使用 `cppyy`）或 Rust（使用 `PyO3`）。如果你这样做，请注意哪些操作需要复制内存，哪些可以直接读取 Python 内存，并确保留下构建说明，或者确保它仅使用 `pyproject.toml` 构建。另外请注意，GPT-2 的正则表达式在大多数正则引擎中并未得到很好的支持，即使支持通常也会太慢。我们已经验证了 Oniguruma 相当快且支持负向先行断言（negative lookahead），但 Python 中的 `regex` 包甚至比它还要快。

> **问题 (train_bpe_tinystories)：在 TinyStories 上进行 BPE 训练 (2 分)**
> 
> (a) 在 TinyStories 数据集上训练一个字节级 BPE 分词器，使用 10,000 的最大词汇表大小。确保将 TinyStories 的 `<|endoftext|>` 特殊 token 添加到词汇表中。将结果的词汇表和合并项序列化到磁盘以供进一步检查。训练花费了多少小时和内存？词汇表中最长的 token 是什么？它合理吗？
> 
> **资源需求**：≤ 30 分钟（无 GPU），≤ 30GB RAM
> 
> **提示** 通过在预分词时使用 `multiprocessing` 以及以下两个事实，你应该能够将 BPE 训练时间控制在 2 分钟以内：
> (a) `<|endoftext|>` token 分隔数据文件中的文档。
> (b) `<|endoftext|>` token 在应用 BPE 合并之前作为一个特殊情况处理。
> 
> **交付物**：一到两句话的回答。
> 
> (b) 对你的代码进行性能分析。分词器训练过程的哪一部分耗时最长？
> 
> **交付物**：一到两句话的回答。

接下来，我们将尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。如前所述，我们建议你查看一下该数据集以更好地理解其内容。

> **问题 (train_bpe_expts_owt)：在 OpenWebText 上进行 BPE 训练 (2 分)**
> 
> (a) 在 OpenWebText 数据集上训练一个字节级 BPE 分词器，使用 32,000 的最大词汇表大小。将结果的词汇表和合并项序列化到磁盘以供进一步检查。词汇表中最长的 token 是什么？它合理吗？
> 
> **资源需求**：≤ 12 小时（无 GPU），≤ 100GB RAM
> 
> **交付物**：一到两句话的回答。
> 
> (b) 比较并对比在 TinyStories 与在 OpenWebText 上训练得到的分词器。
> 
> **交付物**：一到两句话的回答。

### 2.6 BPE 分词器：编码和解码

在作业的前一部分中，我们实现了一个在输入文本上训练 BPE 分词器的函数，以获得分词器词汇表和 BPE 合并列表。现在，我们将实现一个 BPE 分词器，它加载所提供的词汇表和合并列表，并使用它们将文本编码为 token ID，或将 token ID 解码回文本。

#### 2.6.1 编码文本

BPE 编码文本的过程与我们训练 BPE 词汇表的方式相对应。它包括几个主要步骤。
**步骤 1：预分词（Pre-tokenize）。** 我们首先对序列进行预分词，并将每个预分词 token 表示为 UTF-8 字节的序列，就像我们在 BPE 训练中所做的那样。我们将在每个预分词 token 内部将这些字节合并为词汇表元素，独立处理每个预分词 token（不跨越预分词 token 边界进行合并）。
**步骤 2：应用合并。** 然后，我们取出在 BPE 训练期间创建的词汇表元素合并序列，并按照**与创建时相同的顺序**将其应用于我们的预分词 token。

> **示例 (bpe_encoding)：BPE 编码示例**
> 
> 例如，假设我们的输入字符串是 `'the cat ate'`，我们的词汇表是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b'at'}`，我们学习到的合并列表是 `[(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'), (b' a', b't')]`。首先，我们的预分词器会将该字符串拆分为 `['the', ' cat', ' ate']`。然后，我们将查看每个预分词 token 并应用 BPE 合并。
> 
> 第一个预分词 token `'the'` 最初表示为 `[b't', b'h', b'e']`。查看我们的合并列表，我们确定第一个适用的合并是 `(b't', b'h')`，并利用它将预分词 token 转换为 `[b'th', b'e']`。然后，我们回到合并列表并确定下一个适用的合并是 `(b'th', b'e')`，它将预分词 token 转换为 `[b'the']`。最后，回顾合并列表，我们看到没有更多的合并规则适用于该字符串（因为整个预分词 token 已合并为单一 token），所以我们已经完成了应用 BPE 合并的过程。对应的整数序列是 `[9]`。
> 
> 对剩余的预分词 token 重复此过程，我们看到在应用 BPE 合并后，预分词 token `' cat'` 被表示为 `[b' c', b'a', b't']`，其转换为整数序列 `[7, 1, 5]`。最后的预分词 token `' ate'` 在应用 BPE 合并后为 `[b' at', b'e']`，变为整数序列 `[10, 3]`。因此，编码我们输入字符串的最终结果是 `[9, 7, 1, 5, 10, 3]`。

**特殊 token。** 在编码文本时，你的分词器应该能够正确处理用户定义的特殊 token（在构建分词器时提供）。

**内存考量。** 假设我们想要对一个无法装入内存的大型文本文件进行分词。为了高效地对这个大文件（或任何其他数据流）进行分词，我们需要将其分解为可管理的块并逐一处理每个块，使得内存复杂度是常数，而不是随文本大小线性增长。在此过程中，我们需要确保 token 不会跨越块边界，否则我们得到的分词结果将会与在内存中对整个序列进行分词的朴素方法不同。

#### 2.6.2 解码文本

要将整数 token ID 序列解码回原始文本，我们只需查找词汇表中每个 ID 对应的条目（字节序列），将它们拼接在一起，然后将这些字节解码为 Unicode 字符串。请注意，输入 ID 并不能保证映射到有效的 Unicode 字符串（因为用户可以输入任何整数 ID 序列）。如果输入 token ID 没有生成有效的 Unicode 字符串，你应该使用官方的 Unicode 替换字符 `U+FFFD` 替换格式错误的字节。[^3] `bytes.decode` 的 `errors` 参数控制如何处理 Unicode 解码错误，使用 `errors='replace'` 将自动把格式错误的数据替换为替换标记。

[^3]: 关于 Unicode 替换字符的更多信息，请参阅 `en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character`。

> **问题 (tokenizer)：实现分词器 (15 分)**
> 
> **交付物**：实现一个 `Tokenizer` 类，给定词汇表和合并列表，将文本编码为整数 ID，并将整数 ID 解码为文本。你的分词器还应支持用户提供的特殊 token（如果它们尚未存在于词汇表中，则将它们追加进去）。我们推荐以下接口：
> 
> `def __init__(self, vocab, merges, special_tokens=None)`：根据给定的词汇表、合并列表以及（可选的）特殊 token 列表构建一个分词器。该函数应接受以下参数：
> `vocab: dict[int, bytes]`
> `merges: list[tuple[bytes, bytes]]`
> `special_tokens: list[str] | None = None`
> 
> `def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`：类方法，从序列化的词汇表和合并列表（格式与你的 BPE 训练代码输出格式相同）以及（可选的）特殊 token 列表中构造并返回一个 `Tokenizer`。此方法应接受以下附加参数：
> `vocab_filepath: str`
> `merges_filepath: str`
> `special_tokens: list[str] | None = None`
> 
> `def encode(self, text: str) -> list[int]`：将输入文本编码为 token ID 序列。
> 
> `def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`：给定一个字符串的可迭代对象（例如，Python 文件句柄），返回一个惰性生成 token ID 的生成器。这是对无法直接加载到内存中的大文件进行内存高效分词所必需的。
> 
> `def decode(self, ids: list[int]) -> str`：将 token ID 序列解码为文本。
> 
> 为了根据我们提供的测试来测试你的 `Tokenizer`，你需要首先在 `[adapters.get_tokenizer]` 处实现测试适配器。然后，运行 `uv run pytest tests/test_tokenizer.py`。你的实现应该能够通过所有测试。

### 2.7 实验

> **问题 (tokenizer_experiments)：分词器实验 (4 分)**
> 
> (a) 从 TinyStories 和 OpenWebText 中抽取 10 个文档。使用你之前训练好的 TinyStories 和 OpenWebText 分词器（词汇表大小分别为 10K 和 32K），将这些抽样的文档编码为整数 ID。每个分词器的压缩率（字节/token）是多少？
> 
> **交付物**：一到两句话的回答。
> 
> (b) 如果你用 TinyStories 的分词器对 OpenWebText 的样本进行分词会发生什么？比较压缩率和/或定性描述发生的情况。
> 
> **交付物**：一到两句话的回答。
> 
> (c) 估算你的分词器的吞吐量（例如，以字节/秒为单位）。对 Pile 数据集（825GB 的文本）进行分词需要多长时间？
> 
> **交付物**：一到两句话的回答。
> 
> (d) 使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练和开发数据集编码为整数 token ID 的序列。稍后我们将使用它来训练我们的语言模型。我们建议将 token ID 序列化为数据类型为 `uint16` 的 NumPy 数组。为什么 `uint16` 是一个合适的选择？
> 
> **交付物**：一到两句话的回答。

---

## 3 Transformer 语言模型架构

语言模型接收一个批次化的整数 token ID 序列（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`）作为输入，并返回一个词汇表上的（批次化的）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch 张量），其中预测分布针对的是每个输入 token 的**下一个词**。在训练语言模型时，我们使用这些对下一个词的预测来计算实际下一个词与预测下一个词之间的交叉熵损失。在推理阶段从语言模型生成文本时，我们从最后一个时间步（即序列的最后一个项）取出预测的下一个词分布，以生成序列中的下一个 token（例如，通过获取具有最高概率的 token、从分布中采样等），将生成的 token 添加到输入序列中，然后重复此过程。

在作业的这一部分，你将从头开始构建这个 Transformer 语言模型。我们将首先对模型进行高层次的描述，然后逐步详细说明各个组件。

### 3.1 Transformer 语言模型

给定一个 token ID 序列，Transformer 语言模型使用一个输入嵌入层将 token ID 转换为稠密向量，将嵌入后的 token 传递通过 `num_layers` 个 Transformer 块（blocks），然后应用一个学习到的线性投影（“输出嵌入”或“LM 头”）以产生预测的下一个 token 的对数几率（logits）。请参见图 1 以获取直观的结构表示。

#### 3.1.1 Token 嵌入（Token Embeddings）

在第一步中，Transformer 将（批次化的）token ID 序列嵌入为一个包含 token 身份信息的向量序列（图 1 中的红色方块）。
更具体地说，给定一个 token ID 序列，Transformer 语言模型使用 token 嵌入层来生成向量序列。每个嵌入层接收一个形状为 `(batch_size, sequence_length)` 的整数张量，并产生一个形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

#### 3.1.2 前置归一化 (Pre-norm) Transformer 块

在嵌入之后，激活值由几个结构相同的神经网络层进行处理。一个标准的仅包含解码器（decoder-only）的 Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer“块”）组成。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块在序列中聚合信息（通过自注意力机制），并对其进行非线性变换（通过前馈层）。

### 3.2 输出归一化与嵌入

经过 `num_layers` 个 Transformer 块之后，我们将提取最终的激活值，并将其转换为词汇表上的分布。
我们将实现“前置归一化（pre-norm）”的 Transformer 块（在 §3.5 中详述），这要求在最后的 Transformer 块之后额外使用层归一化（详见下文），以确保其输出被正确缩放。
在此次归一化之后，我们将使用一个标准的、学习得到的线性变换，将 Transformer 块的输出转换为预测的下一个 token 的对数几率（logits）（例如，参见 Radford 等人 [2018] 的等式 2）。

### 3.3 注解：批处理、Einsum 和高效计算

在整个 Transformer 中，我们将对许多类似批处理的输入执行相同的计算。这里有几个例子：
* **批次（batch）中的元素**：我们对每个批次元素应用相同的 Transformer 前向操作。
* **序列长度**：像 RMSNorm 和前馈网络这样的“逐位置（position-wise）”操作，在序列的每个位置上都以相同的方式运行。
* **注意力头**：注意力操作在“多头”注意力机制中跨多个注意力头进行批处理。

拥有符合人体工程学的方式来执行此类操作是非常有用的，这样可以充分利用 GPU，并且易于阅读和理解。许多 PyTorch 操作可以在张量的开头接收多余的“类似批处理”维度，并高效地跨这些维度重复/广播操作。

例如，假设我们正在执行一个逐位置的批处理操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的“数据张量” `D`，并且我们希望将其与形状为 `(d_model, d_model)` 的矩阵 `A` 进行批量的向量-矩阵乘法。在这种情况下，`D @ A` 将执行批处理矩阵乘法，这是 PyTorch 中的一个高效原语，其中 `(batch_size, sequence_length)` 维度被当作批次处理。

正因为如此，假设你的函数可能会被赋予额外的类似批处理的维度，并将这些维度保持在 PyTorch 形状的开头是有帮助的。为了组织张量以便能够以这种方式进行批处理，可能需要使用多个 `view`、`reshape` 和 `transpose` 步骤来改变形状。这可能有点痛苦，并且往往很难读懂代码到底在做什么以及你的张量形状是什么。

一个更符合人体工程学的选项是在 `torch.einsum` 中使用 `einsum` 表示法，或者使用像 `einops` 或 `einx` 这样的框架无关库。这两个关键操作是 `einsum`（它可以对具有任意维度的输入张量进行张量收缩）和 `rearrange`（它可以对任意维度进行重新排序、拼接和拆分）。事实证明，机器学习中的几乎所有操作都是维度操作和张量收缩的某种组合，偶尔还会带有（通常是逐点的）非线性函数。这意味着如果使用 `einsum` 表示法，你的大量代码将更具可读性和灵活性。

我们**强烈建议**在学习本课程时学习并使用 `einsum` 表示法。之前未接触过 `einsum` 表示法的学生应该使用 `einops`（文档在此），而已经熟悉 `einops` 的学生应该学习更通用的 `einx`（在此）。[^4] 这两个包都已经安装在我们提供的环境中。

这里我们提供一些如何使用 `einsum` 表示法的示例。这些是对 `einops` 文档的补充，你应该先阅读 `einops` 的文档。

[^4]: 值得注意的是，虽然 `einops` 有很好的支持，但 `einx` 还没有经过那么多的实战检验。如果你在 `einx` 中发现任何限制或漏洞，随时可以退回到使用带有纯 PyTorch 的 `einops`。

> **示例 (einstein_example1)：使用 `einops.einsum` 进行批处理矩阵乘法**
> ```python
> import torch
> from einops import rearrange, einsum
> 
> ## 基本实现
> Y = D @ A.T
> # 很难看出输入和输出形状以及它们的含义。
> # D 和 A 可以有哪些形状，它们中有没有出现意外行为的风险？
> 
> ## Einsum 是自我文档化的且健壮的
> #           D                          A           ->          Y
> Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
> 
> ## 或者，一个批处理版本，其中 D 可以有任意的领先维度，但 A 是受限的。
> Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
> ```

> **示例 (einstein_example2)：使用 `einops.rearrange` 进行广播操作**
> 我们有一批图像，针对每张图像，我们想基于某个缩放因子生成 10 个变暗的版本：
> ```python
> images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
> dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
> 
> ## 重塑并相乘
> dim_value = rearrange(dim_by,       "dim_value -> 1 dim_value 1 1 1")
> images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
> dimmed_images = images_rearr * dim_value
> 
> ## 或者一步到位：
> dimmed_images = einsum(
>     images, dim_by,
>     "batch height width channel, dim_value -> batch dim_value height width channel"
> )
> ```

> **示例 (einstein_example3)：使用 `einops.rearrange` 进行像素混合**
> 假设我们有一批图像，表示为形状为 `(batch, height, width, channel)` 的张量，我们希望对图像的所有像素进行线性变换，但这种变换应该对每个通道独立发生。我们的线性变换表示为一个形状为 `(height * width, height * width)` 的矩阵 B。
> ```python
> channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
> B = torch.randn(32*32, 32*32)
> 
> ## 重排图像张量以便跨所有像素进行混合
> channels_last_flat = channels_last.view(
>     -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
> )
> channels_first_flat = channels_last_flat.transpose(1, 2)
> 
> channels_first_flat_transformed = channels_first_flat @ B.T
> 
> channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
> 
> channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
> 
> ### 相比之下，使用 einops：
> height = width = 32
> ## Rearrange 取代了笨重的 torch view + transpose
> channels_first = rearrange(
>     channels_last,
>     "batch height width channel -> batch channel (height width)"
> )
> channels_first_transformed = einsum(
>     channels_first, B,
>     "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
> )
> channels_last_transformed = rearrange(
>     channels_first_transformed,
>     "batch channel (height width) -> batch height width channel",
>     height=height, width=width
> )
> 
> ### 或者，如果你觉得疯狂一点：使用 einx.dot（einx 相当于 einops.einsum）一步到位
> height = width = 32
> channels_last_transformed = einx.dot(
>     "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
>     "-> batch row_out col_out channel",
>     channels_last, B,
>     col_in=width, col_out=width
> )
> ```
> 
> 第一个实现本来可以通过在前后加上注释来指出输入和输出形状是什么来改进，但这很笨拙且容易出 bug。而使用 einsum 表示法时，*文档就是代码实现本身*！

Einsum 表示法可以处理任意的输入批处理维度，但也有成为自文档化（self-documenting）的关键优势。在使用 einsum 表示法的代码中，你的输入和输出张量的相关形状是什么要清楚得多。对于其余张量，你可以考虑使用张量类型提示，例如使用 `jaxtyping` 库（并非专属于 Jax）。
我们将在作业 2 中进一步讨论使用 einsum 表示法对性能的影响，但目前只要知道，它们几乎总是优于其他替代方案即可！

#### 3.3.1 数学符号和内存排序

许多机器学习论文在其符号体系中使用行向量，这会导致其表示能够很好地与 NumPy 和 PyTorch 默认使用的行主序（row-major）内存排序相吻合。使用行向量时，线性变换看起来像
$$y = xW^\top,$$
其中 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 为行主序矩阵，行向量 $x \in \mathbb{R}^{1 \times d_{\text{in}}}$。

在线性代数中，使用列向量通常更为常见，其中线性变换看起来像
$$y = W x,$$
假设给定行主序 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 以及列向量 $x \in \mathbb{R}^{d_{\text{in}}}$。**在本次作业的数学推导中，我们将使用列向量**，因为通常以这种方式跟进数学推导更加容易。你应该记住，如果你想使用普通的矩阵乘法记号，你必须使用行向量约定来应用矩阵，因为 PyTorch 使用的是行主序的内存排序。如果你在矩阵操作中使用了 `einsum`，那么这应该不是问题。

### 3.4 基础构建块：Linear 和 Embedding 模块

#### 3.4.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化可能会导致梯度消失或梯度爆炸等不良行为。前置归一化（Pre-norm）Transformer 对初始化异常健壮，但它们仍然会对训练速度和收敛性产生显著影响。由于本次作业已经很长，我们将细节留到作业 3，并在本作业中为你提供一些在大多数情况下都效果不错的大致初始化设定。目前，请使用：

*   线性层权重（Linear weights）：$\mathcal{N}\left(\mu=0, \sigma^2=\frac{2}{d_{\text{in}}+d_{\text{out}}}\right)$，截断于 $[-3\sigma, 3\sigma]$。
*   嵌入层（Embedding）：$\mathcal{N}(\mu=0, \sigma^2=1)$，截断于 $[-3, 3]$。
*   RMSNorm：1

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态分布权重。

#### 3.4.2 Linear 模块

线性层是 Transformer 乃至整个神经网络的基本构建模块。首先，你将实现自己的 `Linear` 类，该类继承自 `torch.nn.Module` 并执行线性变换：
$$y = W x$$
请注意，遵循大多数现代 LLM 的做法，我们不包含偏置（bias）项。

> **问题 (linear)：实现 linear 模块 (1 分)**
> 
> **交付物**：实现一个继承自 `torch.nn.Module` 并执行线性变换的 `Linear` 类。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，除了没有 `bias` 参数以外。我们推荐使用以下接口：
> 
> `def __init__(self, in_features, out_features, device=None, dtype=None)` 构造线性变换模块。此函数应接受以下参数：
> `in_features: int` 输入的最终维度
> `out_features: int` 输出的最终维度
> `device: torch.device | None = None` 存储参数的设备
> `dtype: torch.dtype | None = None` 参数的数据类型
> 
> `def forward(self, x: torch.Tensor) -> torch.Tensor` 对输入应用线性变换。
> 
> 确保做到以下几点：
> * 继承 `nn.Module`
> * 调用超类构造函数
> * 出于内存排序的考虑，将参数构造并存储为 $W$（而不是 $W^\top$），并将其放入 `nn.Parameter` 中。
> * 当然，不要直接使用 `nn.Linear` 或 `nn.functional.linear`
> 
> 对于初始化，请使用上述的设置，结合 `torch.nn.init.trunc_normal_` 来初始化权重。
> 为了测试你的 `Linear` 模块，请在 `[adapters.run_linear]` 处实现测试适配器。适配器应该将给定的权重加载到你的 `Linear` 模块中。为此你可以使用 `Module.load_state_dict`。然后，运行 `uv run pytest -k test_linear`。

#### 3.4.3 Embedding 模块

如上所述，Transformer 的第一层是一个嵌入层，它将整数 token ID 映射到维度为 `d_model` 的向量空间中。我们将实现一个自定义的 `Embedding` 类，继承自 `torch.nn.Module`（因此你不应使用 `nn.Embedding`）。其 `forward` 方法应该通过使用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token ID，从形状为 `(vocab_size, d_model)` 的嵌入矩阵中提取对应的索引，从而为每个 token ID 选择嵌入向量。

> **问题 (embedding)：实现 embedding 模块 (1 分)**
> 
> **交付物**：实现继承自 `torch.nn.Module` 并执行嵌入查找（embedding lookup）的 `Embedding` 类。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。我们推荐以下接口：
> 
> `def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)` 构造嵌入模块。此函数应接受以下参数：
> `num_embeddings: int` 词汇表的大小
> `embedding_dim: int` 嵌入向量的维度，即 `d_model`
> `device: torch.device | None = None` 存储参数的设备
> `dtype: torch.dtype | None = None` 参数的数据类型
> 
> `def forward(self, token_ids: torch.Tensor) -> torch.Tensor` 为给定的 token ID 查找嵌入向量。
> 
> 确保做到：
> * 继承 `nn.Module`
> * 调用超类构造函数
> * 将嵌入矩阵初始化为 `nn.Parameter`
> * 以 `d_model` 为最后维度来存储嵌入矩阵
> * 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`
> 
> 同样，使用上述的初始化设置，并使用 `torch.nn.init.trunc_normal_` 来初始化权重。
> 要测试你的实现，在 `[adapters.run_embedding]` 处实现测试适配器。然后，运行 `uv run pytest -k test_embedding`。

### 3.5 前置归一化 (Pre-Norm) Transformer 块

每个 Transformer 块都有两个子层：多头自注意力（multi-head self-attention）机制和逐位置的前馈网络（position-wise feed-forward network）（Vaswani 等人，2017，第 3.1 节）。

在最初的 Transformer 论文中，模型在两个子层的周围使用了残差连接，然后跟上层归一化（layer normalization）。这种架构通常被称为“后置归一化（post-norm）”Transformer，因为层归一化应用于子层输出。然而，许多研究发现，将层归一化从每个子层的输出移至每个子层的输入（在最终的 Transformer 块之后还有一次额外的层归一化）可以提高 Transformer 训练的稳定性 [Nguyen and Salazar, 2019, Xiong et al., 2020]——参见图 2 以获取这种“前置归一化（pre-norm）”Transformer 块的直观展示。然后每个 Transformer 块子层的输出通过残差连接加到子层的输入上（Vaswani 等人，2017，第 5.4 节）。采用 pre-norm 的一个直觉原因是，从输入嵌入到 Transformer 最终输出的这段路径有一条干净的“残差流（residual stream）”，没有任何归一化，据称这可以改善梯度的流动。这种 pre-norm Transformer 现在是目前语言模型（例如 GPT-3、LLaMA、PaLM 等）中使用的标准，因此我们将实现此变体。我们将逐步走查 pre-norm Transformer 块的每个组件，并按顺序实现它们。

#### 3.5.1 均方根层归一化 (Root Mean Square Layer Normalization)

Vaswani 等人 [2017] 的原始 Transformer 实现使用层归一化 [Ba 等人，2016] 来归一化激活值。跟随 Touvron 等人 [2023] 的步伐，我们将使用均方根层归一化（RMSNorm; Zhang 和 Sennrich, 2019，等式 4）进行层归一化。给定一个激活值向量 $a \in \mathbb{R}^{d_{\text{model}}}$，RMSNorm 将如下重新缩放每个激活值 $a_i$：

$$RMSNorm(a_i) = \frac{a_i}{RMS(a)} g_i,$$

其中 $RMS(a) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} a_i^2 + \varepsilon}$。这里，$g_i$ 是一个可学习的“增益（gain）”参数（总共有 `d_model` 个这样的参数），而 $\varepsilon$ 是一个超参数，通常固定为 `1e-5`。

你应该将输入向上转型（upcast）到 `torch.float32`，以防止在对输入求平方时发生溢出。整体而言，你的 `forward` 方法应该如下所示：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# 在此编写执行 RMSNorm 的代码
...
result = ...

# 将结果以原始数据类型返回
return result.to(in_dtype)
```

> **问题 (rmsnorm)：均方根层归一化 (1 分)**
> 
> **交付物**：实现作为 `torch.nn.Module` 的 RMSNorm。我们推荐以下接口：
> 
> `def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)`
> 构造 RMSNorm 模块。此函数应接受以下参数：
> `d_model: int` 模型的隐藏维度
> `eps: float = 1e-5` 用于数值稳定性的 Epsilon 值
> `device: torch.device | None = None` 存储参数的设备
> `dtype: torch.dtype | None = None` 参数的数据类型
> 
> `def forward(self, x: torch.Tensor) -> torch.Tensor` 处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，并返回相同形状的张量。
> 
> **注意**：如上文所述，切记在执行归一化之前将输入向上转型为 `torch.float32`（之后再向下转型回原始 `dtype`）。
> 要测试你的实现，在 `[adapters.run_rmsnorm]` 处实现测试适配器。然后，运行 `uv run pytest -k test_rmsnorm`。

#### 3.5.2 逐位置前馈网络 (Position-Wise Feed-Forward Network)

在原始的 Transformer 论文中（Vaswani 等人 [2017] 第 3.3 节），Transformer 的前馈网络由两个线性变换组成，它们之间夹杂着 ReLU 激活函数（$\text{ReLU}(x) = \max(0, x)$）。内部前馈层的维度通常是输入维度的 4 倍。

然而，与这种原始设计相比，现代语言模型倾向于包含两个主要变化：它们使用另一种激活函数并采用门控（gating）机制。具体而言，我们将实现被 Llama 3 [Grattafiori 等人，2024] 和 Qwen 2.5 [Yang 等人，2024] 等 LLM 所采用的“SwiGLU”激活函数，它将 SiLU（通常称为 Swish）激活与称为门控线性单元（GLU）的门控机制结合在一起。遵循自 PaLM [Chowdhery 等人，2022] 和 LLaMA [Touvron 等人，2023] 以来的大多数现代 LLM 的做法，我们也将省略线性层中有时会使用的偏置项。

SiLU 或 Swish 激活函数 [Hendrycks 和 Gimpel, 2016, Elfwing 等人, 2017] 定义如下：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

正如图 3 所示，SiLU 激活函数类似于 ReLU 激活函数，但在零点处是平滑的。

门控线性单元（GLU）最初由 Dauphin 等人 [2017] 定义为一个线性变换通过 sigmoid 函数后的结果与另一个线性变换进行逐元素乘积：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1x) \odot W_2x,$$

其中 $\odot$ 表示逐元素乘法。据提出，门控线性单元能够“通过为梯度保留线性路径，同时保持非线性能力，从而减少深层架构中的梯度消失问题”。

将 SiLU/Swish 和 GLU 结合起来，我们就得到了 SwiGLU，我们将它用于我们的前馈网络：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2(\text{SiLU}(W_1x) \odot W_3x),$$

其中 $x \in \mathbb{R}^{d_{\text{model}}}$，$W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，一般按惯例有 $d_{\text{ff}} = \frac{8}{3}d_{\text{model}}$。

Shazeer [2020] 首先提出了将 SiLU/Swish 激活与 GLU 结合的想法，并进行的实验表明，SwiGLU 在语言建模任务上的表现优于 ReLU 和 SiLU（不带门控）等基准模型。在作业的后面部分，你将对比 SwiGLU 和 SiLU。虽然我们提到了一些关于这些组件的启发式论据（而且论文中提供了更多支持证据），但保持经验主义的视角是有好处的：Shazeer 论文中有一句现在很出名的话：

> 我们无法解释为什么这些架构看起来有效；我们将它们的成功（以及其他一切）归功于神圣的恩宠（divine benevolence）。

> **问题 (positionwise_feedforward)：实现逐位置前馈网络 (2 分)**
> 
> **交付物**：实现 SwiGLU 前馈网络，它由 SiLU 激活函数和 GLU 组成。
> **注意**：在这种特殊情况下，为了数值稳定性，你可以在实现中随意使用 `torch.sigmoid`。
> 在你的实现中，你应该将 $d_{\text{ff}}$ 设置为大约 $\frac{8}{3} \times d_{\text{model}}$，同时确保内部前馈层的维度是 64 的倍数，以便充分利用你的硬件。要根据我们提供的测试来测试你的实现，你需要实现 `[adapters.run_swiglu]` 处的测试适配器。然后，运行 `uv run pytest -k test_swiglu`。

#### 3.5.3 相对位置嵌入 (Relative Positional Embeddings)

为了向模型注入位置信息，我们将实现旋转位置嵌入（Rotary Position Embeddings [Su 等人，2021]），通常称为 RoPE。对于给定 token 位置 $i$ 处的查询 token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们将应用一个成对旋转矩阵 $R^i$，得到 $q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$。在这里，$R^i$ 将把嵌入元素的配对 $q^{(i)}_{2k-1:2k}$ 视为 2D 向量，并旋转角度 $\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$，其中 $k \in \{1, \dots, d/2\}$，$\Theta$ 为某个常数。因此，我们可以认为 $R^i$ 是一个大小为 $d \times d$ 的分块对角矩阵，其分块为 $R^i_k$（对于 $k \in \{1, \dots, d/2\}$）：

$$R^i_k = \begin{bmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{bmatrix}. \quad (8)$$

从而我们得到完整的旋转矩阵：

$$R^i = \begin{bmatrix} R^i_1 & 0 & 0 & \dots & 0 \\ 0 & R^i_2 & 0 & \dots & 0 \\ 0 & 0 & R^i_3 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & R^i_{d/2} \end{bmatrix}, \quad (9)$$

其中 0 表示 $2 \times 2$ 的零矩阵。虽然可以构造完整的 $d \times d$ 矩阵，但一个好的解决方案应该是利用该矩阵的特性更高效地实现变换。由于我们只关心给定序列内 token 的相对旋转，我们可以在不同层和不同批次之间重用为 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 计算的值。如果你想优化它，你可以使用一个被所有层引用的单一 RoPE 模块，它可以拥有一个在初始化期间使用 `self.register_buffer(persistent=False)` 创建的 $\sin$ 和 $\cos$ 值的 2D 预计算缓存，而不是使用 `nn.Parameter`（因为我们不想学习这些固定的余弦和正弦值）。对我们的 $q^{(i)}$ 执行的完全相同的旋转过程也会对 $k^{(j)}$ 执行，即旋转对应的 $R^j$。请注意，该层没有可学习的参数。

> **问题 (rope)：实现 RoPE (2 分)**
> 
> **交付物**：实现一个 `RotaryPositionalEmbedding` 类，将 RoPE 应用于输入张量。推荐以下接口：
> 
> `def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)` 构造 RoPE 模块，如果需要，创建缓存。
> `theta: float` RoPE 的 $\Theta$ 值
> `d_k: int` 查询（query）和键（key）向量的维度
> `max_seq_len: int` 将会输入的最大序列长度
> `device: torch.device | None = None` 存储缓存的设备
> 
> `def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor`
> 处理形状为 `(..., seq_len, d_k)` 的输入张量，并返回相同形状的张量。注意，你应该容忍 $x$ 具有任意数量的批处理维度。你应该假设 `token_positions` 是一个形状为 `(..., seq_len)` 的张量，指定了 $x$ 沿序列维度的 token 位置。
> 你应该使用 `token_positions` 来对你（可能预计算的）沿序列维度的 $\cos$ 和 $\sin$ 张量进行切片。
> 
> 要测试你的实现，完成 `[adapters.run_rope]` 并确保它通过 `uv run pytest -k test_rope`。

#### 3.5.4 缩放点积注意力 (Scaled Dot-Product Attention)

我们现在将实现 Vaswani 等人 [2017]（第 3.2.1 节）中描述的缩放点积注意力。作为初步步骤，注意力操作的定义将使用 `softmax`，这是一个将未归一化的分数向量转换为归一化分布的操作：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^n \exp(v_j)}. \quad (10)$$

请注意，对于较大的值，$\exp(v_i)$ 可能会变成 `inf`（此时，`inf/inf = NaN`）。我们可以通过注意到 `softmax` 操作对所有输入增加任何常数 $c$ 具有不变性来避免这种情况。我们可以利用这一性质来提高数值稳定性——通常，我们会从 $v_i$ 的所有元素中减去 $v_i$ 的最大项，使新的最大项为 0。你现在将使用这个技巧来实现 `softmax` 以获得数值稳定性。

> **问题 (softmax)：实现 softmax (1 分)**
> 
> **交付物**：编写一个函数，对张量应用 softmax 操作。你的函数应接收两个参数：一个张量和一个维度 $i$，并对输入张量的第 $i$ 维应用 softmax。输出张量应与输入张量具有相同的形状，但其第 $i$ 维现在将具有归一化的概率分布。使用减去第 $i$ 维最大值的技巧来避免数值稳定性问题。
> 要测试你的实现，完成 `[adapters.run_softmax]` 并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

我们现在可以从数学上定义注意力（Attention）操作如下：

$$\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q^\top K}{\sqrt{d_k}} \right) V \quad (11)$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，以及 $V \in \mathbb{R}^{m \times d_v}$。这里，$Q, K$ 和 $V$ 都是该操作的输入——注意这些不是可学习的参数。如果你想知道为什么这不是 $QK^\top$，请参阅 3.3.1 节。

**掩码 (Masking)**：有时屏蔽注意力操作的输出是很方便的。掩码应该具有形状 $M \in \{\text{True, False}\}^{n \times m}$，并且该布尔矩阵的每一行 $i$ 表示查询 $i$ 应该关注哪些键。规范地（且有点令人困惑地），位置 $(i, j)$ 处的 `True` 值表示查询 $i$ *确实*关注键 $j$，而 `False` 值表示查询 $i$ *不*关注键 $j$。换句话说，“信息流”发生在值为 `True` 的 $(i, j)$ 对。例如，考虑一个具有项 `[[True, True, False]]` 的 $1 \times 3$ 掩码矩阵。单个查询向量仅关注前两个键。

计算上，使用掩码比在子序列上计算注意力要高效得多，我们可以通过获取 softmax 之前的分数 $\left( \frac{Q^\top K}{\sqrt{d_k}} \right)$ 并在掩码矩阵中为 `False` 的任何项添加 $-\infty$ 来实现这一点。

> **问题 (scaled_dot_product_attention)：实现缩放点积注意力 (5 分)**
> 
> **交付物**：实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 代表任意数量的其他类似批次的维度（如果提供）。该实现应返回一个形状为 `(batch_size, ..., d_v)` 的输出。关于批处理维度的讨论请参见 3.3 节。
> 你的实现还应支持一个可选的用户提供的形状为 `(seq_len, seq_len)` 的布尔掩码。掩码值为 `True` 的位置的注意力概率总和应为 1，掩码值为 `False` 的位置的注意力概率应为零。
> 要根据我们提供的测试来测试你的实现，你需要实现 `[adapters.run_scaled_dot_product_attention]` 处的测试适配器。
> `uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试你的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试你的实现。

#### 3.5.5 因果多头自注意力 (Causal Multi-Head Self-Attention)

我们将实现 Vaswani 等人 [2017] 第 3.2.2 节中描述的多头自注意力。回想一下，在数学上，应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \quad (12)$$
$$\text{对于 } \text{head}_i = \text{Attention}(Q_i, K_i, V_i) \quad (13)$$

其中 $Q_i, K_i, V_i$ 是查询、键和值的嵌入维度分别为 $d_k$ 或 $d_v$ 的第 $i$ 个切片（$i \in \{1, \dots, h\}$）。由于注意力是在 §3.5.4 中定义的缩放点积注意力操作，由此我们可以形成多头自注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W_O \text{MultiHead}(W_Q x, W_K x, W_V x) \quad (14)$$

在这里，可学习参数为 $W_Q \in \mathbb{R}^{hd_k \times d_{\text{model}}}$，$W_K \in \mathbb{R}^{hd_k \times d_{\text{model}}}$，$W_V \in \mathbb{R}^{hd_v \times d_{\text{model}}}$，以及 $W_O \in \mathbb{R}^{d_{\text{model}} \times hd_v}$。由于 $Q, K, V$ 在多头注意力操作中是被切分的，我们可以认为 $W_Q, W_K$ 和 $W_V$ 在输出维度上针对每个头是分开的。当你实现了这个功能后，你应该在总共三个矩阵乘法中计算键、值和查询的投影。[^5]

[^5]: 作为一个进阶目标，尝试将键、查询和值投影合并到一个权重矩阵中，这样你只需要进行一次矩阵相乘。

**因果掩码 (Causal masking)**：你的实现应该防止模型关注序列中的未来 token。换句话说，如果给模型一个 token 序列 $t_1, \dots, t_n$，并且我们要计算前缀 $t_1, \dots, t_i$（其中 $i < n$）的下一个词预测，模型应该*不能*访问（关注）位置 $t_{i+1}, \dots, t_n$ 处的 token 表示，因为在推理期间生成文本时它将无法访问这些 token（而且这些未来的 token 会泄露真实下一个词的身份信息，使语言建模预训练目标变得毫无意义）。对于输入 token 序列 $t_1, \dots, t_n$，我们可以通过运行多头自注意力 $n$ 次（针对序列中的 $n$ 个唯一前缀）来朴素地防止访问未来的 token。相反，我们将使用**因果注意力掩码**，它允许 token $i$ 关注序列中所有位置 $j \leq i$。你可以使用 `torch.triu` 或广播索引比较来构造此掩码，并且你应该利用 §3.5.4 中的缩放点积注意力实现已经支持注意力掩码这一事实。

**应用 RoPE**：RoPE 应应用于查询和键向量，但不应应用于值向量。此外，头维度应被视为批处理维度，因为在多头注意力中，注意力是针对每个头独立应用的。这意味着应该对每个头的查询和键向量应用完全相同的 RoPE 旋转。

> **问题 (multihead_self_attention)：实现因果多头自注意力 (5 分)**
> 
> **交付物**：将因果多头自注意力实现为一个 `torch.nn.Module`。你的实现应（至少）接受以下参数：
> `d_model: int` Transformer 块输入的维度。
> `num_heads: int` 在多头自注意力中使用的头数。
> 
> 遵循 Vaswani 等人 [2017]，设置 $d_k = d_v = d_{\text{model}}/h$。要根据我们提供的测试来测试你的实现，请在 `[adapters.run_multihead_self_attention]` 处实现测试适配器。然后，运行 `uv run pytest -k test_multihead_self_attention` 以测试你的实现。

### 3.6 完整的 Transformer LM

让我们从组装 Transformer 块开始（回顾图 2 将会有所帮助）。一个 Transformer 块包含两个“子层”，一个是多头自注意力，另一个是前馈网络。在每个子层中，我们首先执行 RMSNorm，然后是主要操作（MHA/FF），最后加入残差连接。
具体来说，Transformer 块的前一半（第一个“子层”）应该实现以下更新集，以从输入 $x$ 产生输出 $y$：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x)). \quad (15)$$

> **问题 (transformer_block)：实现 Transformer 块 (3 分)**
> 
> 实现如 §3.5 中所述并如图 2 所示的前置归一化 Transformer 块。你的 Transformer 块应（至少）接受以下参数。
> `d_model: int` Transformer 块输入的维度。
> `num_heads: int` 在多头自注意力中使用的头数。
> `d_ff: int` 逐位置前馈内部层的维度。
> 
> 要测试你的实现，请实现适配器 `[adapters.run_transformer_block]`。然后运行 `uv run pytest -k test_transformer_block` 来测试你的实现。
> **交付物**：通过所提供测试的 Transformer 块代码。

现在我们将这些块放在一起，遵循图 1 中的高级图示。遵循我们在第 3.1.1 节中对嵌入的描述，将此输入到 `num_layers` 个 Transformer 块中，然后将其通过三个输出层，以获得词汇表上的分布。

> **问题 (transformer_lm)：实现 Transformer LM (3 分)**
> 
> 是时候把所有东西组合在一起了！实现如 §3.1 中所述并如图 1 所示的 Transformer 语言模型。你的实现应（至少）接受上述 Transformer 块的所有构造参数，以及以下附加参数：
> `vocab_size: int` 词汇表的大小，用于确定 token 嵌入矩阵的维度。
> `context_length: int` 最大上下文长度，用于确定位置嵌入矩阵的维度（如果需要）。
> `num_layers: int` 使用的 Transformer 块的数量。
> 
> 要根据我们提供的测试来测试你的实现，你首先需要实现 `[adapters.run_transformer_lm]` 处的测试适配器。然后，运行 `uv run pytest -k test_transformer_lm` 来测试你的实现。
> **交付物**：一个通过上述测试的 Transformer LM 模块。

**资源核算 (Resource accounting)**：能够理解 Transformer 的各个部分如何消耗计算和内存是有用的。我们将通过一些步骤来进行基础的“FLOPs 核算”。Transformer 中绝大多数的 FLOPs 都花在了矩阵乘法上，所以我们的核心方法很简单：
1. 写下 Transformer 前向传递中所有的矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实将很有用：
**规则**：给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵乘积 $AB$ 需要 $2mnp$ FLOPs。
要理解这一点，请注意 $(AB)[i, j] = A[i, :] \cdot B[:, j]$，这个点积需要 $n$ 次加法和 $n$ 次乘法（$2n$ FLOPs）。然后，由于矩阵乘积 $AB$ 有 $m \times p$ 个条目，FLOPs 的总数为 $(2n)(mp) = 2mnp$。
现在，在做下一个题目之前，走查你的 Transformer 块和 Transformer LM 的每个组件，并列出所有的矩阵乘法及其相关的 FLOPs 成本，可能会有所帮助。

> **问题 (transformer_accounting)：Transformer LM 资源核算 (5 分)**
> 
> (a) 考虑 GPT-2 XL，它具有以下配置：
> `vocab_size` : 50,257
> `context_length` : 1,024
> `num_layers` : 48
> `d_model` : 1,600
> `num_heads` : 25
> `d_ff` : 6,400
> 
> 假设我们使用此配置构建模型。我们的模型将有多少个可学习参数？假设每个参数使用单精度浮点数表示，仅加载此模型需要多少内存？
> **交付物**：一到两句话的回答。
> 
> (b) 识别完成我们的 GPT-2 XL 规模模型的前向传递所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设我们的输入序列具有 `context_length` 个 token。
> **交付物**：矩阵乘法列表（带有描述），以及所需的 FLOPs 总数。
> 
> (c) 根据你上面的分析，模型的哪些部分需要的 FLOPs 最多？
> **交付物**：一到两句话的回答。
> 
> (d) 对 GPT-2 small（12 层，768 `d_model`，12 头）、GPT-2 medium（24 层，1024 `d_model`，16 头）和 GPT-2 large（36 层，1280 `d_model`，20 头）重复你的分析。随着模型尺寸的增加，Transformer LM 的哪些部分占总 FLOPs 的比例越来越大或越来越小？
> **交付物**：对于每个模型，提供模型组件及其相关 FLOPs 的细分（作为前向传递所需总 FLOPs 的比例）。此外，提供一到两句话的描述，说明模型尺寸的变化如何改变每个组件的比例 FLOPs。
> 
> (e) 取 GPT-2 XL 并将上下文长度增加到 16,384。一次前向传递的总 FLOPs 如何变化？模型组件的 FLOPs 相对贡献如何变化？
> **交付物**：一到两句话的回答。

---

## 4 训练 Transformer LM

我们现在已经有了预处理数据（通过分词器）和模型（Transformer）的步骤。剩下的就是构建所有支持训练的代码。这包括以下内容：
* **损失函数 (Loss)**：我们需要定义损失函数（交叉熵）。
* **优化器 (Optimizer)**：我们需要定义优化器来最小化此损失（AdamW）。
* **训练循环 (Training loop)**：我们需要所有支持训练的基础设施，用于加载数据、保存检查点并管理训练。

### 4.1 交叉熵损失 (Cross-entropy loss)

回想一下，Transformer 语言模型为长度为 $m+1$ 的每个序列 $x$ 和 $i = 1, \dots, m$ 定义了一个分布 $p_\theta(x_{i+1} | x_{1:i})$。给定一个由长度为 $m$ 的序列组成的训练集 $D$，我们将标准交叉熵（负对数似然）损失函数定义为：

$$\ell(\theta; D) = \frac{1}{|D|m} \sum_{x \in D} \sum_{i=1}^m -\log p_\theta(x_{i+1} | x_{1:i}). \quad (16)$$

（注意，Transformer 中的单次前向传递会对所有 $i = 1, \dots, m$ 产生 $p_\theta(x_{i+1} | x_{1:i})$。）
特别地，Transformer 为每个位置 $i$ 计算对数几率 $o_i \in \mathbb{R}^{\text{vocab\_size}}$，结果为：[^6]

$$p(x_{i+1} | x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])}. \quad (17)$$

交叉熵损失通常是针对对数几率向量 $o_i \in \mathbb{R}^{\text{vocab\_size}}$ 和目标 $x_{i+1}$ 定义的。[^7]
实现交叉熵损失需要注意数值问题，就像 softmax 的情况一样。

[^6]: 注意 $o_i[k]$ 指的是向量 $o_i$ 索引 $k$ 处的值。
[^7]: 这对应于 $x_{i+1}$ 上的狄拉克 δ 分布与预测的 $\text{softmax}(o_i)$ 分布之间的交叉熵。

> **问题 (cross_entropy)：实现交叉熵 (Cross entropy)**
> 
> **交付物**：编写一个函数来计算交叉熵损失，该函数接收预测的对数几率 $(o_i)$ 和目标 $(x_{i+1})$，并计算交叉熵 $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$。你的函数应处理以下内容：
> * 为了数值稳定性减去最大元素。
> * 尽可能抵消 $\log$ 和 $\exp$。
> * 处理任何额外的批处理维度，并返回整个批次的*平均值*。与第 3.3 节一样，我们假设类似批次的维度总是排在第一位，位于词汇表大小维度之前。
> 
> 实现 `[adapters.run_cross_entropy]`，然后运行 `uv run pytest -k test_cross_entropy` 以测试你的实现。

**困惑度 (Perplexity)**：交叉熵对于训练来说已经足够了，但当我们评估模型时，我们还希望报告困惑度。对于一个长度为 $m$ 且我们遭受交叉熵损失 $\ell_1, \dots, \ell_m$ 的序列：

$$\text{perplexity} = \exp\left( \frac{1}{m} \sum_{i=1}^m \ell_i \right). \quad (18)$$

### 4.2 SGD 优化器

现在我们有了损失函数，我们将开始探索优化器。最简单的基于梯度的优化器是随机梯度下降（SGD）。我们从随机初始化的参数 $\theta_0$ 开始。然后对于每一步 $t = 0, \dots, T-1$，我们执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; B_t), \quad (19)$$

其中 $B_t$ 是从数据集 $D$ 中采样的随机数据批次，学习率 $\alpha_t$ 和批次大小 $|B_t|$ 是超参数。

#### 4.2.1 在 PyTorch 中实现 SGD

为了实现我们的优化器，我们将继承 PyTorch 的 `torch.optim.Optimizer` 类。`Optimizer` 子类必须实现两个方法：
`def __init__(self, params, ...)`：应该初始化你的优化器。这里，`params` 将是要优化的参数集合（或参数组，以防用户想要针对模型的不同部分使用不同的超参数，例如学习率）。确保将 `params` 传递给基类的 `__init__` 方法，基类会将这些参数存储起来供在 `step` 中使用。你可以根据优化器的不同（例如，学习率是一个常见的参数）接受额外的参数，并将它们作为字典传递给基类构造函数，其中键是你为这些参数选择的名称（字符串）。

`def step(self)`：应该对参数进行一次更新。在训练循环期间，这将在反向传递之后被调用，因此你可以访问上一个批次的梯度。此方法应迭代每个参数张量 $p$ 并在原地（in-place）修改它们，即设置 `p.data`，它根据梯度 `p.grad`（如果存在）保存与该参数关联的张量，`p.grad` 是表示损失对该参数梯度的张量。

PyTorch 优化器 API 有一些细微之处，所以用一个例子来解释会更容易。为了使我们的例子更丰富，我们将实现 SGD 的一个微调版本，其中学习率随着训练的进行而衰减，从初始学习率 $\alpha$ 开始，随着时间的推移采取越来越小的步长：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; B_t) \quad (20)$$

让我们看看这个版本的 SGD 如何实现为一个 PyTorch 优化器：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # 获取与 p 关联的状态。
                t = state.get("t", 0) # 从状态中获取迭代次数，或使用初始值。
                grad = p.grad.data # 获取损失对 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad # 原地更新权重张量。
                state["t"] = t + 1 # 增加迭代次数。
        return loss
```

在 `__init__` 中，我们将参数以及默认超参数传递给基类构造函数（参数可能会分成组，每组具有不同的超参数）。如果参数只是 `torch.nn.Parameter` 对象的单一集合，基类构造函数将创建一个单一组并分配默认超参数。然后，在 `step` 中，我们遍历每个参数组，然后遍历该组中的每个参数，并应用等式 20。在这里，我们将迭代次数保持为与每个参数关联的状态：我们首先读取该值，在梯度更新中使用它，然后更新它。API 规定用户可以传入一个可调用的 `closure` 以在优化器步骤之前重新计算损失。我们使用的优化器不需要这个，但我们添加它以符合 API。

为了看它如何工作，我们可以使用以下训练循环的最小示例：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad() # 重置所有可学习参数的梯度。
    loss = (weights**2).mean() # 计算一个标量损失值。
    print(loss.cpu().item())
    loss.backward() # 运行反向传递，计算梯度。
    opt.step() # 运行优化器步骤。
```

这是训练循环的典型结构：在每次迭代中，我们将计算损失并运行优化器的步骤。在训练语言模型时，我们的可学习参数将来自模型（在 PyTorch 中，`m.parameters()` 给了我们这个集合）。损失将在采样的数据批次上计算，但训练循环的基本结构将是相同的。

> **问题 (learning_rate_tuning)：调节学习率 (1 分)**
> 
> 正如我们将看到的，对训练影响最大的超参数之一是学习率。让我们在我们的玩具示例中实际观察这一点。运行上面的 SGD 示例，使用另外三个学习率值：1e1, 1e2 和 1e3，仅进行 10 次训练迭代。每个学习率的损失发生了什么变化？它是衰减得更快、更慢，还是发散（即在训练过程中增加）？
> **交付物**：一到两句话的回答，描述你观察到的行为。

### 4.3 AdamW

现代语言模型通常使用更复杂的优化器进行训练，而不是 SGD。最近使用的大多数优化器都是 Adam 优化器 [Kingma and Ba, 2015] 的衍生品。我们将使用 AdamW [Loshchilov and Hutter, 2019]，它在最近的工作中被广泛使用。AdamW 提出对 Adam 进行修改，通过添加权重衰减（在每次迭代中，我们将参数向 0 拉动）来改善正则化，这种方式与梯度更新解耦。我们将按照 Loshchilov 和 Hutter [2019] 的算法 2 中所述实现 AdamW。

AdamW 是**有状态的 (stateful)**：对于每个参数，它都会跟踪其一阶矩和二阶矩的运行估计。因此，AdamW 使用额外的内存来换取改善的稳定性和收敛性。除了学习率 $\alpha$ 之外，AdamW 还有一对控制矩估计更新的超参数 $(\beta_1, \beta_2)$，以及权重衰减率 $\lambda$。典型的应用将 $(\beta_1, \beta_2)$ 设置为 $(0.9, 0.999)$，但像 LLaMA [Touvron 等人，2023] 和 GPT-3 [Brown 等人，2020] 这样的大型语言模型通常使用 $(0.9, 0.95)$ 进行训练。该算法可以写成如下形式，其中 $\epsilon$ 是一个小值（例如 $10^{-8}$），用于提高数值稳定性，以防 $v$ 中的值变得极小：

---
**算法 1 AdamW 优化器**
---
$\text{init}(\theta)$ (初始化可学习参数)
$m \leftarrow 0$ (一阶矩向量的初始值；与 $\theta$ 形状相同)
$v \leftarrow 0$ (二阶矩向量的初始值；与 $\theta$ 形状相同)
**for** $t = 1, \dots, T$ **do**
    采样数据批次 $B_t$
    $g \leftarrow \nabla_\theta \ell(\theta; B_t)$ (计算当前时间步损失的梯度)
    $m \leftarrow \beta_1 m + (1 - \beta_1) g$ (更新一阶矩估计)
    $v \leftarrow \beta_2 v + (1 - \beta_2) g^2$ (更新二阶矩估计)
    $\alpha_t \leftarrow \alpha \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$ (计算迭代 $t$ 调整后的 $\alpha$)
    $\theta \leftarrow \theta - \alpha_t \frac{m}{\sqrt{v} + \epsilon}$ (更新参数)
    $\theta \leftarrow \theta - \alpha \lambda \theta$ (应用权重衰减)
**end for**
---

请注意，$t$ 从 1 开始。你现在将实现这个优化器。

> **问题 (adamw)：实现 AdamW (2 分)**
> 
> **交付物**：将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。你的类应在 `__init__` 中接收学习率 $\alpha$，以及 $\beta, \epsilon$ 和 $\lambda$ 超参数。为了帮助你保持状态，基类 `Optimizer` 为你提供了一个字典 `self.state`，它将 `nn.Parameter` 对象映射到一个字典，该字典存储了该参数所需的任何信息（对于 AdamW，这将是矩估计）。实现 `[adapters.get_adamw_cls]` 并确保它通过 `uv run pytest -k test_adamw`。

> **问题 (adamwAccounting)：使用 AdamW 训练的资源核算 (2 分)**
> 
> 让我们计算一下运行 AdamW 需要多少内存和计算量。假设我们为每个张量使用 `float32`。
> 
> (a) 运行 AdamW 需要多少峰值内存？根据参数、激活值、梯度和优化器状态的内存使用情况来分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`）来表达你的回答。假设 $d_{\text{ff}} = 4 \times d_{\text{model}}$。
> 为了简化，在计算激活值的内存使用量时，仅考虑以下组件：
> *   Transformer 块
>     *   RMSNorm(s)
>     *   多头自注意力子层：QKV 投影，$Q^\top K$ 矩阵乘法，softmax，加权值之和，输出投影。
>     *   逐位置前馈：$W_1$ 矩阵乘法，SiLU，$W_2$ 矩阵乘法
> *   最后的 RMSNorm
> *   输出嵌入
> *   对数几率上的交叉熵
> 
> **交付物**：参数、激活值、梯度和优化器状态各自的代数表达式，以及总和。
> 
> (b) 将你的答案代入 GPT-2 XL 规模的模型，得到一个仅取决于 `batch_size` 的表达式。在 80GB 内存限制内，你可以使用的最大批次大小是多少？
> **交付物**：一个看起来像 $a \cdot \text{batch\_size} + b$ 的表达式（其中 $a, b$ 为数值），以及代表最大批次大小的一个数字。
> 
> (c) 运行一步 AdamW 需要多少 FLOPs？
> **交付物**：一个代数表达式，并附带简短的解释。
> 
> (d) 模型 FLOPs 利用率 (MFU) 被定义为观测到的吞吐量（每秒 token 数）相对于硬件理论峰值 FLOP 吞吐量的比率 [Chowdhery 等人，2022]。NVIDIA A100 GPU 对于 `float32` 操作的理论峰值为 19.5 teraFLOP/s。假设你能够获得 50% 的 MFU，在单个 A100 上训练一个 GPT-2 XL 模型 400K 步且批次大小为 1024 需要多长时间？遵循 Kaplan 等人 [2020] 和 Hoffmann 等人 [2022]，假设反向传递的 FLOPs 是前向传递的两倍。
> **交付物**：训练所需的天数，并附带简短的解释。

### 4.4 学习率调度 (Learning rate scheduling)

导致损失下降最快的学习率值通常在训练过程中会有所变化。在训练 Transformer 时，通常使用学习率调度（schedule），我们在开始时使用较大的学习率，进行更快的更新，然后随着模型训练的进行，慢慢将其衰减到较小的值。[^8] 在本次作业中，我们将实现用于训练 LLaMA [Touvron 等人，2023] 的余弦退火（cosine annealing）调度。

[^8]: 有时在训练初期让学习率上升（热身/restarts）以帮助模型跳出局部最小值也很常见。

调度器只是一个接收当前步数 $t$ 和其他相关参数（如初始和最终学习率）并返回用于该步梯度更新的学习率的函数。最简单的调度是恒定（constant）函数，无论 $t$ 是多少都返回相同的学习率。

余弦退火学习率调度接收 (i) 当前迭代 $t$，(ii) 最大学习率 $\alpha_{\text{max}}$，(iii) 最小（最终）学习率 $\alpha_{\text{min}}$，(iv) 热身迭代次数 $T_w$，以及 (v) 余弦退火迭代次数 $T_c$。在第 $t$ 次迭代时的学习率定义如下：

*   **(热身阶段/Warm-up)**：如果 $t < T_w$，则 $\alpha_t = \frac{t}{T_w} \alpha_{\text{max}}$。
*   **(余弦退火阶段/Cosine annealing)**：如果 $T_w \leq t \leq T_c$，则 $\alpha_t = \alpha_{\text{min}} + \frac{1}{2} \left( 1 + \cos\left( \frac{t - T_w}{T_c - T_w} \pi \right) \right) (\alpha_{\text{max}} - \alpha_{\text{min}})$。
*   **(退火后阶段/Post-annealing)**：如果 $t > T_c$，则 $\alpha_t = \alpha_{\text{min}}$。

> **问题 (learning_rate_schedule)：实现带热身的余弦学习率调度**
> 
> 编写一个函数，接收 $t, \alpha_{\text{max}}, \alpha_{\text{min}}, T_w$ 和 $T_c$，并根据上面定义的调度器返回学习率 $\alpha_t$。然后实现 `[adapters.get_lr_cosine_schedule]` 并确保它通过 `uv run pytest -k test_get_lr_cosine_schedule`。

### 4.5 梯度裁剪 (Gradient clipping)

在训练过程中，我们有时会遇到产生极大梯度的训练样本，这可能会破坏训练的稳定性。为了缓解这种情况，实践中经常采用的一种技术是梯度裁剪（gradient clipping）。其想法是在每次反向传递后、采取优化器步骤前，对梯度的范数强制执行一个限制。

给定所有参数的梯度 $g$，我们计算其 $\ell_2$ 范数 $\|g\|_2$。如果该范数小于最大值 $M$，则保持 $g$ 不变；否则，我们将 $g$ 按因子 $\frac{M}{\|g\|_2 + \epsilon}$ 缩小（其中添加了一个较小的 $\epsilon$，如 $10^{-6}$，以保证数值稳定性）。请注意，结果范数将略低于 $M$。

> **问题 (gradient_clipping)：实现梯度裁剪 (1 分)**
> 
> 编写一个实现梯度裁剪的函数。你的函数应接收一个参数列表和一个最大 $\ell_2$ 范数。它应该原地修改每个参数的梯度。使用 $\epsilon = 10^{-6}$（PyTorch 默认值）。然后，实现适配器 `[adapters.run_gradient_clipping]` 并确保它通过 `uv run pytest -k test_gradient_clipping`。

## 5 训练循环

现在我们终于要将目前为止构建的主要组件整合在一起了：分词后的数据、模型和优化器。

### 5.1 数据加载器 (Data Loader)

分词后的数据（例如，你在 `tokenizer_experiments` 中准备的数据）是一个单一的 token 序列 $x = (x_1, \dots, x_n)$。尽管源数据可能由不同的文档组成（例如，不同的网页或源代码文件），通用的做法是将所有这些文档连接成一个单一的 token 序列，并在它们之间添加分隔符（例如 `<|endoftext|>` token）。

数据加载器将其转换为批次流，其中每个批次由 $B$ 个长度为 $m$ 的序列组成，并配有相应的长度为 $m$ 的下一个 token 目标。例如，对于 $B = 1, m = 3$，$([x_2, x_3, x_4], [x_3, x_4, x_5])$ 可能是一个潜在的批次。

以这种方式加载数据简化训练的原因有很多。首先，任何 $1 \leq i < n - m$ 都会给出一个有效的训练序列，因此采样序列非常简单。由于所有的训练序列都具有相同的长度，因此不需要对输入序列进行填充（padding），这通过增加批次大小 $B$ 提高了硬件利用率。最后，我们也不需要将整个数据集完全加载到内存中来采样训练数据，这使得处理大型数据集变得容易，否则这些数据集可能无法装入内存。

> **问题 (data_loading)：实现数据加载 (2 分)**
> 
> **交付物**：编写一个函数，接收一个 numpy 数组 $x$（带有 token ID 的整数数组）、一个 `batch_size`、一个 `context_length` 和一个 PyTorch 设备字符串（例如 `'cpu'` 或 `'cuda:0'`），并返回一对张量：采样的输入序列和相应的下一个 token 目标。两个张量都应具有形状 `(batch_size, context_length)` 且包含 token ID，并且两者都应放置在请求的设备上。要根据我们提供的测试来测试你的实现，你需要首先在 `[adapters.run_get_batch]` 处实现测试适配器。然后，运行 `uv run pytest -k test_get_batch`。

> **低资源/降级提示：在 CPU 或 Apple Silicon 上加载数据**
> 
> 如果你计划在 CPU 或 Apple Silicon 上训练你的 LM，你需要将你的数据移动到正确的设备上（同样，你稍后也应该为你的模型使用相同的设备）。如果你使用的是 CPU，可以使用 `'cpu'` 设备字符串；如果你使用的是 Apple Silicon（M* 芯片），可以使用 `'mps'` 设备字符串。
> 关于 MPS 的更多信息，请查看以下资源：
> * https://developer.apple.com/metal/pytorch/
> * https://pytorch.org/docs/main/notes/mps.html

如果数据集太大而无法加载到内存中怎么办？我们可以使用名为 `mmap` 的 Unix 系统调用，它将磁盘上的文件映射到虚拟内存中，并在访问该内存位置时延迟加载文件内容。因此，你可以“假装”你在内存中拥有整个数据集。Numpy 通过 `np.memmap`（或者在使用 `np.load` 加载最初用 `np.save` 保存的数组时使用 `mmap_mode='r'` 标志）实现了这一点，它将返回一个类似于 numpy 数组的对象，在你访问条目时按需加载它们。当从你的数据集（即 numpy 数组）采样以进行训练时，**请确保在内存映射模式下加载数据集**。还要确保指定一个与你正在加载的数组相匹配的 `dtype`。显式验证内存映射后的数据看起来是否正确（例如，不包含超出预期词汇表大小的值）可能会有所帮助。

### 5.2 检查点 (Checkpointing)

除了加载数据外，我们还需要在训练时保存模型。在运行任务时，我们经常希望能够恢复由于某种原因中途停止的训练运行（例如，由于任务超时、机器故障等）。即使一切顺利，我们稍后也可能希望访问中间模型（例如，研究训练动力学、从训练的不同阶段采样模型等）。

一个检查点应该包含我们恢复训练所需的所有状态。我们当然希望至少能够恢复模型权重。如果使用的是有状态优化器（如 AdamW），我们还需要保存优化器的状态（例如，在 AdamW 的情况下，即矩估计）。最后，为了恢复学习率调度，我们需要知道停止时的迭代次数。PyTorch 使得保存所有这些变得容易：每个 `nn.Module` 都有一个 `state_dict()` 方法，它返回一个包含所有可学习权重的字典；我们可以稍后使用同类的 `load_state_dict()` 方法恢复这些权重。对于任何 `nn.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以将一个对象（例如，一个包含张量值的字典，但也包含像整数这样的普通 Python 对象）转储到一个文件（路径）或类文件对象中，然后可以使用 `torch.load(src)` 将其重新加载到内存中。

> **问题 (checkpointing)：实现模型检查点 (1 分)**
> 
> 实现以下两个函数来加载和保存检查点：
> 
> `def save_checkpoint(model, optimizer, iteration, out)`：应将前三个参数中的所有状态转储到类文件对象 `out` 中。你可以使用模型和优化器的 `state_dict` 方法来获取它们的相关状态，并使用 `torch.save(obj, out)` 将 `obj` 转储到 `out` 中。典型的选择是让 `obj` 成为一个字典，但只要你以后能加载你的检查点，你可以使用任何你想要的格式。
> 
> `def load_checkpoint(src, model, optimizer)`：应从 `src`（路径或类文件对象）加载检查点，然后从该检查点恢复模型和优化器状态。你的函数应返回保存到检查点的迭代次数。你可以使用 `torch.load(src)` 来恢复你在 `save_checkpoint` 实现中保存的内容，并使用模型和优化器中的 `load_state_dict` 方法将它们恢复到之前的状态。
> 
> 实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` 适配器，并确保它们通过 `uv run pytest -k test_checkpointing`。

### 5.3 训练循环 (Training loop)

现在，终于可以将你实现的所有组件整合到你的主训练脚本中了。让训练运行能够轻松地以不同的超参数开始（例如，通过将它们作为命令行参数）将会有所回报，因为你以后会多次这样做来研究不同的选择如何影响训练。

> **问题 (training_together)：整合在一起 (4 分)**
> 
> **交付物**：编写一个脚本，运行训练循环以在用户提供的输入上训练你的模型。特别是，我们建议你的训练脚本允许（至少）以下操作：
> * 能够配置和控制各种模型和优化器超参数。
> * 使用 `np.memmap` 内存高效地加载训练和验证大型数据集。
> * 将检查点序列化到用户提供的路径。
> * 定期记录训练和验证性能（例如，记录到控制台和/或像 Weights and Biases (wandb.ai) 这样的外部服务）。

---

## 6 生成文本

既然我们可以训练模型了，最后一块拼图就是从我们的模型中生成文本的能力。回想一下，语言模型接收一个（可能是批次化的）长度为 `sequence_length` 的整数序列，并产生一个大小为 `(sequence_length x vocab_size)` 的矩阵，其中序列的每个元素都是一个预测该位置之后下一个词的概率分布。我们现在将编写几个函数，将此转换为新序列的采样方案。

**Softmax**：按照标准惯例，语言模型输出是最后一个线性层的输出（“对数几率”），因此我们必须通过 `softmax` 操作将其转换为归一化概率，正如我们在等式 10 中看到的。

**解码 (Decoding)**：要从我们的模型生成文本（解码），我们将为模型提供一个前缀 token 序列（“提示词/prompt”），并要求它产生一个预测序列中下一个词的词汇表上的概率分布。然后，我们将从该词汇表分布中采样以确定下一个输出 token。
具体来说，解码过程的一个步骤应该接收一个序列 $x_{1 \dots t}$ 并通过以下等式返回一个 token $x_{t+1}$：

$$P(x_{t+1} = i | x_{1 \dots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}$$
$$v = \text{TransformerLM}(x_{1 \dots t})_t \in \mathbb{R}^{\text{vocab\_size}}$$

其中 `TransformerLM` 是我们的模型，它接收 `sequence_length` 的输入序列并产生 `(sequence_length x vocab_size)` 大小的矩阵，我们取该矩阵的最后一个元素，因为我们正在寻找在第 $t$ 个位置的下一个词预测。
这通过反复采样这些单步条件分布（将我们之前生成的输出 token 追加到下一个解码时间步的输入中），直到生成序列结束 token `<|endoftext|>`（或达到用户指定的生成 token 最大数量），为我们提供了一个基础解码器。

**解码技巧 (Decoder tricks)**：我们将尝试使用小型模型，而小型模型有时会生成非常低质量的文本。两个简单的解码器技巧可以帮助解决这些问题。首先，在**温度缩放（temperature scaling）**中，我们使用温度参数 $\tau$ 修改我们的 softmax，新的 softmax 是：

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i/\tau)}{\sum_{j=1}^{|\text{vocab\_size}|} \exp(v_j/\tau)} \quad (24)$$

注意设置 $\tau \to 0$ 如何使得 $v$ 的最大元素占据主导地位，并且 softmax 的输出变成一个集中在该最大元素上的 one-hot 向量。
其次，另一个技巧是 **nucleus 或 top-p 采样**，即我们通过截断低概率词来修改采样分布。设 $q$ 是我们从大小为 `vocab_size` 的（经过温度缩放的）softmax 中获得的概率分布。具有超参数 $p$ 的 Nucleus 采样根据以下等式产生下一个 token：

$$P(x_{t+1} = i | q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{如果 } i \in V(p) \\ 0 & \text{否则} \end{cases}$$

其中 $V(p)$ 是满足 $\sum_{j \in V(p)} q_j \geq p$ 的**最小**索引集合。你可以通过先按大小对概率分布 $q$ 进行排序，并选择最大的词汇表元素直到达到目标水平 $\alpha$ 来轻松计算此值。

> **问题 (decoding)：解码 (3 分)**
> 
> **交付物**：实现一个从你的语言模型解码的函数。我们建议你支持以下功能：
> * 为用户提供的提示生成补全（即，接收一些 $x_{1 \dots t}$ 并采样补全，直到遇到 `<|endoftext|>` token）。
> * 允许用户控制生成 token 的最大数量。
> * 给定所需的温度值，在采样前对预测的下一个词分布应用 softmax 温度缩放。
> * 给定用户指定的阈值，进行 Top-p 采样（也称为 nucleus 采样）。

---

## 7 实验

现在是时候把所有东西整合在一起，并在预训练数据集上训练（小型）语言模型了。

### 7.1 如何运行实验和提交物

理解 Transformer 架构组件背后原理的最佳方法是亲自修改并运行它。没有什么能替代亲自动手的经验。
为此，重要的是能够**快速、一致地进行实验并记录**你的所作所为。为了快速实验，我们将在小规模模型（17M 参数）和简单数据集（TinyStories）上运行许多实验。为了保持一致，你将进行组件消融（ablate）并有系统地改变超参数，为了记录，我们将要求你提交与每个实验相关的实验日志和学习曲线。
为了能够提交损失曲线，请确保定期评估验证损失，并记录**步数和墙钟时间（wallclock times）**。你可能会发现像 Weights and Biases 这样的日志基础设施很有帮助。

> **问题 (experiment_log)：实验日志 (3 分)**
> 
> 对于你的训练和评估代码，创建实验跟踪基础设施，允许你跟踪实验以及相对于梯度步数和墙钟时间的损失曲线。
> **交付物**：实验日志的基础设施代码，以及一份实验日志（记录了你在本节下面的作业问题中所尝试的所有事情的文档）。

### 7.2 TinyStories

我们将从一个非常简单的基准数据集开始（TinyStories; Eldan and Li, 2023），模型在该数据集上可以快速训练，并且我们可以看到一些有趣的行为。获取此数据集的说明在第 1 节。

> **示例 (tinystories_example)：TinyStories 的一个例子**
> 
> 从前有一个叫 Ben 的小男孩。Ben 喜欢探索周围的世界。他在一家商店里看到了许多神奇的东西，比如陈列着的漂亮花瓶。有一天，Ben 正在店里走着，突然发现了一个非常特别的花瓶。Ben 看到它时惊讶极了！他说：“哇，这真是一个神奇的花瓶！我能买它吗？”店主微笑着说：“当然可以。你可以把它带回家，给你的所有朋友展示它是多么神奇！”于是 Ben 把花瓶带回了家，他为此感到非常自豪！他叫来他的朋友们，向他们展示了那个神奇的花瓶。他的所有朋友都觉得这个花瓶很漂亮，不敢相信 Ben 是多么幸运。这就是 Ben 在商店里发现神奇花瓶的故事！

**超参数调节**：我们将告诉你一些非常基础的超参数作为开始，并要求你寻找其他一些能良好运作的超参数。
`vocab_size`：10000。典型的词汇表大小在几万到几十万之间。你应该改变这个值，观察词汇表和模型行为如何变化。
`context_length`：256。像 TinyStories 这样简单的数据集可能不需要很长的序列长度，但对于后来的 OpenWebText 数据，你可能想要改变这个值。尝试改变这个值，并观察其对每轮迭代运行时间和最终困惑度的影响。
`d_model`：512。这比许多小型 Transformer 论文中使用的 768 维略小，但这会使运行速度更快。
`d_ff`：1344。这大约是 $\frac{8}{3} d_{\text{model}}$，同时是 64 的倍数，这有利于 GPU 性能。
`RoPE theta 参数` $\Theta$：10000。
`层数和头数`：4 层，16 头。总共将产生约 17M 个非嵌入参数，这是一个相当小的 Transformer。
`处理的总 token 数`：327,680,000（你的 `batch_size` × `总步数` × `context_length` 应大致等于此值）。

你应该通过反复尝试来为以下其他超参数找到良好的默认值：**学习率、学习率热身、其他 AdamW 超参数（$\beta_1, \beta_2, \epsilon$）以及权重衰减**。你可以在 Kingma 和 Ba [2015] 中找到此类超参数的一些典型选择。

**整合在一起**：现在你可以通过获得训练好的 BPE 分词器、对训练数据集进行分词，并在你编写的训练循环中运行它，将所有东西整合在一起。**重要提示**：如果你的实现正确且高效，上述超参数应该在 1 台 H100 GPU 上产生约 30-40 分钟的运行时间。如果你的运行时间长得多，请检查并确保你的数据加载、检查点或验证损失代码没有成为运行时间的瓶颈，并且你的实现已正确批次化。

**调试模型架构的提示和技巧**：我们强烈建议你习惯于使用 IDE 的内置调试器（例如 VSCode/PyCharm），这将比使用 print 语句调试节省大量时间。如果你使用文本编辑器，可以使用像 `pdb` 这样的工具。调试模型架构的一些其他良好做法包括：
* 开发任何神经网路架构的一个常见的第一步是过拟合（overfit）单个微批次。如果你的实现是正确的，你应该能够迅速将训练损失降至接近零。
* 在各种模型组件中设置调试断点，并检查中间张量的形状，以确保它们符合你的预期。
* 监控激活值、模型权重和梯度的范数，以确保它们没有爆炸或消失。

> **问题 (learning_rate)：调节学习率 (3 分) (4 H100 小时)**
> 
> 学习率是要调节的最重要的超参数之一。拿你训练的基础模型，回答以下问题：
> (a) 对学习率进行超参数搜索，并报告最终损失（或者如果优化器发散，则记录发散情况）。
> **交付物**：与多个学习率相关的学习曲线。解释你的超参数搜索策略。
> **交付物**：一个在 TinyStories 上的验证损失（每 token）至多为 1.45 的模型。

> **低资源/降级提示：在 CPU 或 Apple Silicon 上训练少量步数**
> 
> 如果你是在 `cpu` 或 `mps` 上运行，你应该将处理的总 token 数减少到 40,000,000，这将足以产生相当流畅的文本。你也可以将目标验证损失从 1.45 增加到 2.00。
> 使用调节后的学习率在 M3 Max 芯片和 36 GB RAM 上运行我们的解决方案代码，我们使用的 `batch_size × total step count × context length = 32 × 5000 × 256 = 40,960,000` 个 token，在 `cpu` 上耗时 1 小时 22 分钟，在 `mps` 上耗时 36 分钟。在第 5000 步时，我们达到了 1.80 的验证损失。
> 一些额外的提示：
> * 当使用 $X$ 个训练步数时，我们建议调整余弦学习率衰减调度，使其衰减恰好在第 $X$ 步终止（即达到最小学习率）。
> * 当使用 `mps` 时，**不要**使用 TF32 内核，即不要像在 `cuda` 设备上那样设置 `torch.set_float32_matmul_precision('high')`。我们尝试在使用 `mps` 时（torch 版本 2.6.0）启用 TF32 内核，发现后端会静默使用损坏的内核，导致训练不稳定。
> * 你可以使用 `torch.compile` 进行 JIT 编译来加速训练。具体来说：
>   - 在 `cpu` 上，使用 `model = torch.compile(model)` 编译你的模型。
>   - 在 `mps` 上，你可以使用 `model = torch.compile(model, backend="aot_eager")` 来稍微优化反向传递。截止到 torch 版本 2.6.0，`mps` 上不支持使用 Inductor 进行编译。

> (b) 民间智慧认为，最好的学习率是“处于稳定性的边缘”。调查学习率发散的点与你的最佳学习率有何关系。
> **交付物**：学习率增加的学习曲线，其中至少包含一次发散运行，以及关于这与收敛速度关系的分析。

现在让我们改变批次大小，看看训练会发生什么。批次大小很重要——它们让我们通过执行更大的矩阵乘法来从 GPU 中获得更高的效率，但批次大小真的总是越大越好吗？让我们运行一些实验来找出答案。

> **问题 (batch_size_experiment)：批次大小变化 (1 分) (2 H100 小时)**
> 
> 将你的批次大小从 1 一直改变到 GPU 内存限制。尝试至少几个中间的批次大小，包括典型的大小如 64 和 128。
> **交付物**：具有不同批次大小的运行学习曲线。如果有必要，应该再次优化学习率。
> **交付物**：几句话讨论你的发现以及它们对训练的影响。

有了手中的解码器，我们现在可以生成文本了！我们将从模型中生成文本，看看它有多好。作为参考，你应该获得至少与下面示例一样好的输出。

> **示例 (ts_generate_example)：TinyStories 语言模型的输出示例**
> 
> 从前，有一个叫 Lily 的漂亮女孩。她喜欢吃口香糖，尤其是那个大黑色的。有一天，Lily 的妈妈请她帮忙做晚饭。Lily 非常兴奋！她喜欢帮妈妈。Lily 的妈妈做了一大锅汤作为晚餐。Lily 非常开心并说：“谢谢你，妈妈！我爱你。”她帮妈妈把汤倒进一个大碗里。晚饭后，Lily 的妈妈做了一些美味的汤。Lily 很喜欢它！她说：“谢谢你，妈妈！这汤真好喝！”她妈妈微笑着说：“我很高兴你喜欢它，Lily。”他们做完了饭并继续一起做饭。结束。

> **低资源/降级提示：在 CPU 或 Apple Silicon 上生成文本**
> 
> 如果你改用处理了 40M token 的低资源配置，你应该会看到仍然类似于英语但不如上述流利的生成结果。例如，我们训练了 40M token 的 TinyStories 语言模型的采样输出如下：
> 
> 从前，有一个叫 Sue 的小女孩。Sue 有一颗她非常喜欢的牙齿。那是他最好的头。有一天，Sue 出去散步，遇到了一只瓢虫！他们成了好朋友并在小路上一起玩耍。
> “嘿，Polly！我们出去玩吧！”Tim 说。Sue 抬头看着天空，看到很难找到跳舞发光的方法。她微笑并同意帮说话！”
> 当 Sue 看着天空移动时，它是什么。她

这是精确的问题陈述以及我们的要求：

> **问题 (generate)：生成文本 (1 分)**
> 
> 使用你的解码器和训练好的检查点，报告你的模型生成的文本。你可能需要操作解码器参数（温度、top-p 等）来获得流利的输出。
> **交付物**：至少 256 个 token 的文本转储（或直到遇到第一个 `<|endoftext|>` token），以及对该输出流利度的简短评论，以及至少两个影响该输出质量好坏的因素。

### 7.3 消融实验与架构修改

理解 Transformer 的最佳方法是实际修改它并观察它的行为。我们现在将进行一些简单的消融和修改。

**消融 1：层归一化**。人们常说层归一化对于 Transformer 训练的稳定性很重要。但也许我们想活得冒险一点。让我们从每个 Transformer 块中移除 RMSNorm，看看会发生什么。

> **问题 (layer_norm_ablation)：移除 RMSNorm 并训练 (1 分) (1 H100 小时)**
> 
> 从你的 Transformer 中移除所有的 RMSNorms 并进行训练。在之前的最佳学习率下发生了什么？你能通过使用较低的学习率来获得稳定性吗？
> **交付物**：移除 RMSNorms 并训练时的学习曲线，以及针对最佳学习率的学习曲线。
> **交付物**：关于 RMSNorm 影响的几句话评论。

现在让我们调查另一个起初看起来有些武断的层归一化选择。**前置归一化 (Pre-norm)** Transformer 块定义为：

$$z = x + \text{MultiHeadedSelfAttention}(\text{RMSNorm}(x))$$
$$y = z + \text{FFN}(\text{RMSNorm}(z))$$

这是对原始 Transformer 架构为数不多的“共识”修改之一，原始架构使用的是**后置归一化 (post-norm)** 方法：

$$z = \text{RMSNorm}(x + \text{MultiHeadedSelfAttention}(x))$$
$$y = \text{RMSNorm}(z + \text{FFN}(z))$$

让我们回到后置归一化方法，看看会发生什么。

> **问题 (pre_norm_ablation)：实现后置归一化并训练 (1 分) (1 H100 小时)**
> 
> 将你的前置归一化 Transformer 实现修改为后置归一化。使用后置归一化模型进行训练，看看会发生什么。
> **交付物**：后置归一化 Transformer 与前置归一化 Transformer 的学习曲线对比。

我们看到层归一化对 Transformer 的行为有重大影响，甚至层归一化的位置也很重要。

**消融 2：位置嵌入**。我们接下来将调查位置嵌入对模型性能的影响。具体来说，我们将比较我们的基础模型（带有 RoPE）和完全不包含位置嵌入（NoPE）的情况。事实证明，仅包含解码器的 Transformer，即我们实现的具有因果掩码的 Transformer，理论上可以在不被显式提供位置嵌入的情况下推断出相对或绝对位置信息 [Tsai et al., 2019, Kazemnejad et al., 2023]。我们现在将实证测试 NoPE 与 RoPE 相比表现如何。

> **问题 (no_pos_emb)：实现 NoPE (1 分) (1 H100 小时)**
> 
> 修改你的 Transformer 实现，将 RoPE 替换为完全移除位置嵌入信息，看看会发生什么。
> **交付物**：RoPE 与 NoPE 性能对比的学习曲线。

**消融 3：SwiGLU vs. SiLU**。接下来，我们将跟随 Shazeer [2020]，通过比较 SwiGLU 前馈网络与使用 SiLU 激活但不带门控线性单元 (GLU) 的前馈网络的性能，测试前馈网络中门控的重要性：

$$\text{FFN}_{\text{SiLU}}(x) = W_2 \text{SiLU}(W_1 x). \quad (25)$$

回想一下，在我们的 SwiGLU 实现中，我们将内部前馈层的维度设置为大约 $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$（同时确保 $d_{\text{ff}} \mod 64 = 0$）。在你的 $\text{FFN}_{\text{SiLU}}$ 实现中，你应该设置 $d_{\text{ff}} = 4 \times d_{\text{model}}$，以大致匹配 SwiGLU 前馈网络（它有三个权重矩阵而不是两个）的参数量。

> **问题 (swiglu_ablation)：SwiGLU vs. SiLU (1 分) (1 H100 小时)**
> 
> **交付物**：SwiGLU 与 SiLU 前馈网络在参数量大致匹配的情况下的性能对比学习曲线。
> **交付物**：讨论你的发现的几句话。

> **低资源/降级提示：受限 GPU 资源的在线学生应在 TinyStories 上测试修改**
> 
> 在作业的剩余部分，我们将转向更大规模、更多噪声的网络数据集 (OpenWebText)，实验架构修改并（可选地）向课程排行榜提交。
> 在 OpenWebText 上将 LM 训练到流利需要很长时间，因此我们建议 GPU 访问受限的在线学生继续在 TinyStories 上测试修改（使用验证损失作为评估性能的指标）。

### 7.4 在 OpenWebText 上运行

我们现在将转向一个由网页爬取创建的更标准预训练数据集。还提供了一个 OpenWebText [Gokaslan et al., 2019] 的小样本作为单个文本文件：参见第 1 节了解如何访问此文件。
这里有一个来自 OpenWebText 的例子。注意文本更加写实、复杂且多样。你可能想要通读训练数据集，以了解网络抓取语料库的训练数据是什么样的。

> **示例 (owt_example)：来自 OWT 的一个例子**
> 
> 棒球招募网站 (Baseball Prospectus) 的技术总监 Harry Pavlidis 在雇佣 Jonathan Judge 时冒了风险。Pavlidis 知道，正如 Alan Schwarz 在《数字游戏》中所写，“美国文化的任何角落都不如棒球运动员的表现那样被更精确地计数，更热情地量化。”只需点击几下，你就可以发现 Noah Syndergaard 的快球在飞向本垒板的途中每分钟旋转超过 2100 次，Nelson Cruz 在 2016 年具有最高平均击球初速，以及无数其他看起来像是从视频游戏或科幻小说中摘取的琐碎细节。日益增长的数据海洋赋予了棒球文化中一个日益重要的参与者权力：分析爱好者。
> 这种赋权也带来了额外的审视——不仅是对数据的测量，也是对数据背后的人员和出版物的审视。通过棒球招募网站，Pavlidis 了解了伴随量化不完善而来的所有负面反应。他也知道该网站的接球指标需要重新设计，这需要一个有学识的人——一个能够处理复杂统计建模问题的人——来完成这项工作。
> “他吓坏了我们。”Harry Pavlidis 说。
> Pavlidis 有种直觉，Judge 凭借其后期写作以及他们在网站赞助的球场活动中的互动而“搞定了它”。此后不久，两人在喝酒时进行了交谈。Pavlidis 的直觉得到了证实。Judge 适合这个职位——更好的是，他是一个乐意的适合者。“我告诉了很多人，”Pavlidis 说，“他是唯一一个敢于承担这项任务的人。” [...]

**注意**：你可能需要为此实验重新调节你的超参数，如学习率或批次大小。

> **问题 (main_experiment)：在 OWT 上实验 (2 分) (3 H100 小时)**
> 
> 使用与 TinyStories 相同的模型架构和总训练迭代次数在 OpenWebText 上训练你的语言模型。该模型表现如何？
> **交付物**：你在 OpenWebText 上训练的语言模型的学习曲线。描述与 TinyStories 损失的差异——我们应该如何解释这些损失？
> **交付物**：在与 TinyStories 输出相同的格式下，由 OpenWebText LM 生成的文本。该文本流利度如何？为什么尽管我们拥有相同的模型和计算预算，输出质量却比 TinyStories 差？

### 7.5 你自己的修改 + 排行榜

恭喜你进行到这一步。你快完成了！你现在将尝试改进 Transformer 架构，看看你的超参数和架构与班上其他同学相比如何。

**排行榜规则**：除了以下几点外，没有其他限制：
**运行时间**：你的提交物在 1 台 H100 上运行的时间不能超过 1.5 小时。你可以在 slurm 提交脚本中设置 `--time=01:30:00` 来强制执行此操作。
**数据**：你只能使用我们提供的 OpenWebText 训练数据集。
否则，你可以随心所欲地做任何事情。
如果你正在寻找关于实现什么的建议，可以查看以下一些资源：
* 最先进的开源 LLM 家族，如 Llama 3 [Grattafiori et al., 2024] 或 Qwen 2.5 [Yang et al., 2024]。
* NanoGPT speedrun 仓库 (https://github.com/KellerJordan/modded-nanogpt)，社区成员在那里发布了许多关于“加速运行”小规模语言模型预训练的有趣修改。例如，一个可以追溯到原始 Transformer 论文的常见修改是将输入和输出嵌入的权重绑定（tie）在一起（参见 Vaswani et al. [2017] (第 3.4 节) 和 Chowdhery et al. [2022] (第 2 节)）。如果你确实尝试了权重绑定，你可能必须减小嵌入/LM 头初始化的标准差。

在尝试完整的 1.5 小时运行之前，你会想要在 OpenWebText 的一小部分子集或 TinyStories 上测试这些修改。
作为一个警示，我们确实注意到你在此排行榜中发现效果良好的一些修改可能无法推广到更大规模的预训练。我们将在本课程的缩放法则 (scaling laws) 单元进一步探讨这个想法。

> **问题 (leaderboard)：排行榜 (6 分) (10 H100 小时)**
> 
> 你将在上述排行榜规则下训练一个模型，目标是在 1.5 小时 H100 时间内最小化你语言模型的验证损失。
> **交付物**：记录的最终验证损失、明确显示小于 1.5 小时的墙钟时间 x 轴的相关学习曲线，以及你所做工作的描述。我们期望排行榜提交物至少击败 5.0 损失的朴素基准。在此处提交至排行榜：https://github.com/stanford-cs336/assignment1-basics-leaderboard。

---

## 参考文献

（参考文献列表省略，请参阅原文第 47-50 页）