项目2 流压缩指导
========================

提交截止日期：**2024年9月17日晚上11:59**。

**概述：** 在这个项目中，你将从头开始用CUDA实现GPU流压缩。这个算法被广泛使用，对于加速你的路径追踪项目也很重要。

在本项目中，你的流压缩实现将简单地从整数数组中移除`0`。在路径追踪器中，你将从光线数组中移除已终止的路径。

除了对你的路径追踪器有用之外，这个项目旨在让你重新思考GPU的算法思维方式。在GPU上，许多算法可以从大规模并行和数据并行中受益：同时使用不同的数据执行相同的代码多次。

你将实现几个不同版本的*扫描*（*前缀和*）算法。首先，你将实现算法的CPU版本以加深理解。然后，你将编写几个GPU实现："朴素"版本和"工作效率"版本。最后，你将使用其中的一些来实现GPU流压缩。

**算法概述和细节：** 关于扫描和流压缩实现的细节，主要有两个参考资料：

* [并行算法幻灯片](https://docs.google.com/presentation/d/1ETVONA7QDM-WqsEj4qVOGD6Kura5I6E9yqH-7krnwZ0/edit#slide=id.p126)，
  包含扫描、流压缩和工作效率并行扫描的内容。
* GPU Gems 3，第39章 - [使用CUDA实现并行前缀和（扫描）](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)。
    - 在线版本包含一些小错误（上标、缺少括号、缩进错误等）
    - 我们在[GPU Gem 3 Ch 39 补丁](https://github.com/CIS565-Fall-2017/Project2-Stream-Compaction/blob/master/INSTRUCTION.md#gpu-gem-3-ch-39-patch)中维护了修复。如果你在章节中发现更多错误，欢迎提交新的pull request贡献。
* 如果阅读步骤后仍不清楚，请查看最后一章 - [算法示例](https://github.com/CIS565-Fall-2017/Project2-Stream-Compaction/blob/master/INSTRUCTION.md#algorithm-examples)。
* [复习课幻灯片](https://docs.google.com/presentation/d/1daOnWHOjMp1sIqMdVsNnvEU1UYynKcEMARc_W6bGnqE/edit?usp=sharing)

你的GPU流压缩实现将位于`stream_compaction`子项目中。这样，你就可以轻松地将其复制到GPU路径追踪器中使用。

## 第0部分：基本要求

本项目（以及本课程中的所有其他CUDA项目）需要具有CUDA功能的NVIDIA显卡。任何具有计算能力2.0（`sm_20`）或更高版本的显卡都可以使用。在这个[兼容性表](https://developer.nvidia.com/cuda-gpus)中检查你的GPU。
如果你没有符合这些规格的个人电脑，可以使用Moore 100或SIG实验室中配备支持GPU的CETS计算机。

### 有用的现有代码

* `stream_compaction/common.h`
  * `checkCUDAError`宏：检查CUDA错误并在出现错误时退出。
  * `ilog2ceil(x)`：计算log2(x)的上限，作为整数。
* `main.cpp`
  * 用于测试你的实现的一些代码。

**注意1：** 测试将简单地与你的CPU实现进行比较。
先完成它！

**注意2：** 测试默认使用大小为256的数组。
也要测试更大的数组（10,000？1,000,000？）！

## 第1部分：CPU扫描和流压缩

这个流压缩方法将从整数数组中移除`0`。

首先完成这部分，并仔细检查输出！它将作为其他测试的预期值。

在`stream_compaction/cpu.cu`中实现：

* `StreamCompaction::CPU::scan`：计算独占前缀和。为了性能比较，这应该是一个简单的`for`循环。但为了在开始转向GPU之前更好地理解，你可以先在这个函数中模拟GPU扫描。
* `StreamCompaction::CPU::compactWithoutScan`：不使用`scan`函数的流压缩。
* `StreamCompaction::CPU::compactWithScan`：使用`scan`函数的流压缩。将输入数组映射到0和1的数组，对其进行扫描，并使用散射来生成输出。你需要一个**CPU**散射实现（参见幻灯片或GPU Gems章节的解释）。

这些实现应该只有几行长。

## 第2部分：朴素GPU扫描算法

在`stream_compaction/naive.cu`中实现`StreamCompaction::Naive::scan`

这使用了GPU Gems 3第39.2.1节中的"朴素"算法。示例39-1使用共享内存。在这个项目中不需要这样做。你可以简单地使用全局内存。因此，你将不得不进行`ilog2ceil(n)`次单独的内核调用。

由于你的单个GPU线程不能保证同时运行，你通常不能在GPU上就地操作数组；这会导致竞争条件。相反，创建两个设备数组。在每次迭代时交换它们：从A读取并写入B，从B读取并写入A，依此类推。

注意第39章中示例39-1的错误；在线版本中的伪代码和CUDA代码都有一些小错误（上标、缺少括号、缩进错误等）。

确保测试非2的幂大小的数组。

## 第3部分：工作效率GPU扫描和流压缩

### 3.1. 扫描

在`stream_compaction/efficient.cu`中实现
`StreamCompaction::Efficient::scan`

第2部分的大部分文本都适用。

* 这使用了GPU Gems 3第39.2.2节中的"工作效率"算法。
* 这可以就地完成 - 它不会受到朴素方法的竞争条件的影响，因为不会出现一个线程写入而另一个线程从数组的同一位置读取的情况。
* 注意示例39-2中的错误。
* 测试非2的幂大小的数组。

由于工作效率扫描在二叉树结构上操作，它最适合长度为2的幂的数组。确保你的实现适用于非2的幂大小的数组（参见`ilog2ceil`）。这需要额外的内存 - 你的中间数组大小需要向上舍入到下一个2的幂。

### 3.2. 流压缩

这个流压缩方法将从整数数组中移除`0`。

在`stream_compaction/efficient.cu`中实现
`StreamCompaction::Efficient::compact`

对于压缩，你还需要实现幻灯片和GPU Gems章节中介绍的散射算法。

在`stream_compaction/common.cu`中实现这些用于`compact`：

* `StreamCompaction::Common::kernMapToBoolean`
* `StreamCompaction::Common::kernScatter`

## 第4部分：使用Thrust的实现

在`stream_compaction/thrust.cu`中实现：

* `StreamCompaction::Thrust::scan`

这应该是一个非常短的函数，它包装了对Thrust库函数`thrust::exclusive_scan(first, last, result)`的调用。

为了测量时间，确保通过传递`thrust::device_vector`（已在GPU上分配）来排除内存操作。你可以通过从给定指针创建`thrust::host_vector`，然后进行转换来创建`thrust::device_vector`。

对于thrust流压缩，请查看[thrust::remove_if](https://thrust.github.io/doc/group__stream__compaction.html)。分析`thrust::remove_if`不是必需的，但我们鼓励你这样做。

## 第5部分：为什么我的GPU方法这么慢？（额外学分）（+5）

如果你严格按照幻灯片实现高效扫描版本，很可能会得到一个实际上并不那么高效的"高效"GPU扫描 -- 它比CPU方法更慢？

虽然这对于本作业来说完全可以接受，
除了解释这种现象的原因外，我们鼓励你尝试升级你的工作效率GPU扫描。

思考这些可能会让你有一个顿悟的时刻：
- 在上/下扫描的更深层次上，占用率是多少？大多数线程真的在工作吗？
- 你是否在上/下扫描的每个层次都启动相同数量的块？
- 如果某些线程很懒，我们能否提前终止它们？
- 我如何压缩线程？我应该修改什么来保持剩余线程仍然正确工作？

请记住，这个优化不需要你改变很多代码结构。
这都是关于一些索引计算技巧。

如果你没有遇到更慢的GPU方法。
恭喜！你已经领先了，你自动获得这个额外学分。

## 第6部分：额外学分

### 额外学分1：基数排序（+10）

向`stream_compaction`子项目添加一个额外模块。使用你的一个扫描实现来实现基数排序。添加测试以检查其正确性。

### 额外学分2：使用共享内存的GPU扫描和硬件优化（额外学分）（+10）

实现[GPU Gem第39章](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)示例39.1，39.2。

注意共享内存的大小是动态的，并且与块大小相关。由于每个SM的共享内存有限，你设置的块大小将影响每个SM中块的占用率。例如，假设你的显卡每个SM有N Kb的共享内存，如果你每个块使用最大的N Kb共享内存，那么每个SM的最大占用率将是1个块。这可能不是最佳性能。

此外，我们可以通过改变内存访问模式来避免bank冲突，从而优化效率。参见[GPU Gem第39章](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html)第39.2.3节。这在课程中没有涉及，但我们鼓励你挑战自己。

## 第7部分：报告

1. 更新你的`README.md`顶部的所有TODO。
2. 添加这个项目的描述，包括其功能列表。
3. 添加你的性能分析（见下文）。

所有额外学分功能必须在你的`README.md`中记录，解释其价值（如果适用，包括性能比较！）并展示其工作方式的示例。对于基数排序，展示如何调用它以及其输出的示例。

始终使用Release模式构建进行性能分析，并在不调试的情况下运行。

### 问题

* 粗略优化每个实现的块大小，以在你的GPU上获得最小运行时间。
  * （你不应该比较未优化的实现！）

* 将所有这些GPU扫描实现（朴素、工作效率和Thrust）与串行CPU版本的扫描进行比较。绘制比较图（以数组大小为自变量）。
  * 我们为你封装了CPU和GPU计时函数作为性能计时器类，以方便测量时间成本。
    * 我们使用`std::chrono`提供CPU高精度计时，使用CUDA事件测量CUDA性能。
    * 对于CPU，将你的CPU代码放在`timer().startCpuTimer()`和`timer().endCpuTimer()`之间。
    * 对于GPU，将你的CUDA代码放在`timer().startGpuTimer()`和`timer().endGpuTimer()`之间。确保**不要**在性能测量中包含任何*初始/最终*内存操作（`cudaMalloc`，`cudaMemcpy`），以保持可比性。
    * 不要混淆`CpuTimer`和`GpuTimer`。
  * 要猜测Thrust实现内部可能发生的情况（例如分配、内存复制），请查看其执行的Nsight时间线。由于你甚至没有查看实现的代码，这里的分析不需要很详细。

* 简要解释你在这里看到的现象。
  * 你能找到性能瓶颈吗？是内存I/O？计算？对每个实现是否不同？

* 将测试程序的输出粘贴到你的README中的三重反引号块中。
  * 如果你添加了自己的测试（例如基数排序或测试其他边界情况），请确保明确提到它。

这些问题也应该帮助指导你在未来的作业中进行性能分析。

## 提交

如果你修改了任何`CMakeLists.txt`文件（除了`SOURCE_FILES`列表），请明确提到。
注意Ed Discussion上讨论的任何构建问题。

打开GitHub pull request，以便我们可以看到你已完成。
标题应为"Project 2: 你的名字"。
你的pull request评论部分的模板如下，你可以复制粘贴：

* [仓库链接](到你的仓库的链接)
* （简要）提到你已完成的功能。特别是那些你想要突出的额外功能
    * 功能0
    * 功能1
    * ...
* 对项目本身的反馈（如果有）。

## GPU Gem 3第39章补丁

* 示例1
![](img/example-1.png)

* 示例2
![](img/example-2.jpg)

* 图39-4
![](img/figure-39-4.jpg)

* 图39-2。这张图显示了一个朴素的包含扫描。我们应该将其转换为用于压缩的独占扫描。
![](img/figure-39-2.jpg)

## 算法示例

* 扫描：
  - 目标：生成给定数组的前缀和数组（这里我们只关心独占扫描）
  - 输入
    - [1 5 0 1 2 0 3]
  - 输出
    - [0 1 6 6 7 9 9]
* 压缩：
  - 目标：紧密整齐地打包非零元素
  - 输入
    - [1 5 0 1 2 0 3]
  - 输出
    - [1 5 1 2 3]
* 不使用扫描的压缩（CPU）
  - 压缩的一种实现。因此目标、输入和输出都应与压缩相同
  - 简单地遍历输入数组，同时维护一个指针，指示我们应该将下一个非零元素放在哪个地址
* 使用扫描的压缩（CPU/GPU）
  - 压缩的一种实现。因此目标、输入和输出都应与压缩相同
  - 3个步骤
    - 映射
      + 目标：将我们的原始数据数组（整数、光线等）映射到布尔数组
      + 输入
        - [1 5 0 1 2 0 3]
      + 输出
        - [1 1 0 1 1 0 1]
    - 扫描
        + 将上一步的输出作为输入
        + 输入
          - [1 1 0 1 1 0 1]
        + 输出
          - [0 1 2 2 3 4 4]
    - 散射
        + 保留非零元素并将它们压缩到新数组中
        + 输入：
          + 原始数组
            - [1 5 0 1 2 0 3]
          + 映射数组
            - [1 1 0 1 1 0 1]
          + 扫描数组
            - [0 1 2 2 3 4 4]
        + 输出：
          - [1 5 1 2 3]
        + 这可以在GPU上并行完成
        + 如果你愿意，可以在CPU上尝试多线程（不是必需的，也不是我们的重点）
        + 对于原始数组中的每个元素input[i]
          - 如果它是非零的（由映射数组给出）
          - 则将其放在output[index]中，其中index = scanned[i]
