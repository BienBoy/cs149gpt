# 作业4 CS149-GPT

## Warm-Up：访问张量

张量/数组都是按行存储的，四维数组可以看作元素为三维数组的数组，元素大小即为三维数组内元素总数，以此类推。

## 第 1 部分：简单（但不太高效）的注意力机制实现

主要实现两个矩阵乘法和一个 softmax 运算。

![part1](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F11%2F22-01-03-part1.png)

## 第 2 部分：块矩阵乘法和 Unfused Softmax

通过对矩阵进行分块，有效提高缓存的利用率，减少 miss。

N=1024 时，块大小设为 $8 \times 8$、$16 \times 16$、$32 \times 32$、$64 \times 64$ 对应的时间为 $185.069ms$、$154.328ms$、$159.785ms$、$170.758 \text{ms}$，缓存行的大小为 $64B$，`float` 类型的大小为 $4B$，因此块大小为 $16 \times 16$ 时，一个缓存行恰好可以装下块内的一行。

![part2](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F11%2F22-01-10-part2.png)

## 第 3 部分：Fused Attention

Fused Attention 使得 $N \times N$ 的临时矩阵减小为长度为 $N$ 向量，虽然因为多线程要使用的多个长度为 $N$ 向量，但线程数往往远小于 $N$，因此减少了内存占用。

注释掉 `#pragma omp ...` 语句后，时间为 $217.351 \text{ms}$。在单线程的情况下，Fused Attention 虽然减小了内存占用，但性能有所降低，同时对缓存的利用率也比第 2 部分的低很多。

与第 1 部分相比，由于使用 Fused Attention，可并行化的循环变成了 $3$ 个：batch、head、**row** 易于并行计算。

![part3](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F11%2F22-01-13-part3.png)

## 第 4 部分：Flash Attention

第 4 部分的内存使用最少。Flash Attention 是对 Fused Attention 的改进，在减少内存使用的同时，通过分块，提高了对缓存的利用。第 4 部分的性能比之前各部分要慢，但比单线程的 Fused Attention 稍快。

目前的 Flash Attention 可以通过使用多线程、使用 CPU 向量化硬件单元等方式提高性能。

![part4](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F11%2F22-01-15-part4.png)

## ISPC加速

使用 ISPC 对各部分加速效果如下：

- part1：运行时间在 $60 \text{ms}$ 左右
- part2：运行时间在 $85 \text{ms}$ 左右
- part3：运行时间大致在 $$15-55 \text{ms}$$
- part4：运行时间在 $55 \text{ms}$ 左右

主要使用 ISPC 加速了矩阵乘法，实现主要参考 [ISPC Examples](https://github.com/ispc/ispc/blob/main/examples/cpu/sgemm/SGEMM_kernels.ispc)。

![optimized part1](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F12%2F22-04-44-optimized%20part1.png)

![optimized part2](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F12%2F22-04-52-optimized%20part2.png)

![optimized part3](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F12%2F22-05-01-optimized%20part3.png)

![optimized part4](https://cdn.jsdelivr.net/gh/BienBoy/images/images/2024%2F10%2F12%2F22-05-09-optimized%20part4.png)