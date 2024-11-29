# 图像处理优化说明

本项目为SYSU 2024年秋体系结构课程大作业B的个人事项，针对图像处理程序进行了多层次的性能优化，开源仅供参考，请勿抄袭，助教将会进行查重检测。

本实现在AMD EPYC 7763 CPU上使用 4 核得到了对比单核串行 baseline 加速约 **47.5** 倍的加速比。

如果对你有所帮助，请为我点一个star~

## 优化内容

- 使用`alignas(32)`确保数据对齐，提高内存访问效率
- 使用 openmp 进行并行

### gaussianFilter函数

- 使用AVX2指令集进行向量化处理
- 每次处理16个像素（BLOCK_SIZE = 16）
- 使用`__m256i`寄存器存储中间计算结果，减少result数组store次数
- 更改循环逻辑，减少figure数组load次数

### powerLawTransformation函数

- 实现了查找表(LUT)优化，避免重复计算幂运算
- 使用AVX2的gather指令进行向量化查表操作
- 每次处理32个字节的数据
- 通过`_mm256_i32gather_epi32`实现高效的并行查表

## 编译要求

- 需要支持AVX2指令集的CPU
- 支持OpenMP的编译器
- 建议使用优化标志：`-O0 -march=native -fopenmp`
