# Handout

## Task 2

1. Answer

(a) `chr(0)` 返回的是什么 Unicode 字符？

`'\x00'`

(b) 该字符的字符串表示（`__repr__()`）与其打印出的表示有何不同？


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