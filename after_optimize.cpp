#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <immintrin.h>
#include <cstring>

using std::vector;

class FigureProcessor {
private:
  alignas(32) vector<unsigned char> figure_data;  // 使用一维数组存储数据
  alignas(32) vector<unsigned char> result_data;
  vector<unsigned char*> figure;  // 存储每行的指针
  vector<unsigned char*> result;
  const size_t size;
  alignas(32) static int powerLUT32[256];
  static bool lutInitialized;

  static void initializeLUT() {
    if (lutInitialized) return;
    constexpr float gamma = 0.5f;
    
    // 特殊处理0
    powerLUT32[0] = 0;
    
    // 对其他值保持与原始计算完全一致
    for (int i = 1; i < 256; ++i) {
      float normalized = static_cast<float>(i) / 255.0f;
      float powered = std::pow(normalized, gamma);
      powerLUT32[i] = static_cast<unsigned char>(255.0f * powered + 0.5f);
    }
    lutInitialized = true;
  }

public:
  FigureProcessor(size_t size, size_t seed = 0) : size(size) {
    // 初始化LUT

    initializeLUT();
    
    // 预分配对齐的内存
    figure_data.resize(size * size);
    result_data.resize(size * size);
    
    // 初始化行指针
    figure.resize(size);
    result.resize(size);
    for (size_t i = 0; i < size; ++i) {
      figure[i] = &figure_data[i * size];
      result[i] = &result_data[i * size];
    }

    // !!! Please do not modify the following code !!!
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 保持相同的初始化顺序
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        figure[i][j] = static_cast<unsigned char>(distribution(gen));
      }
    }

    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        result[i][j] = 0;
      }
    }
  }

  ~FigureProcessor() = default;

  // Gaussian filter
  // [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16
  //FIXME: Feel free to optimize this function
  //Hint: You can use SIMD instructions to optimize this function
  void gaussianFilter() {

     // 处理内部区域
    for (size_t i = 1; i < size - 1; ++i) {
      for (size_t j = 1; j < size - 1; ++j) {
        result[i][j] =
            static_cast<unsigned char>((float)(figure[i - 1][j - 1] + 2 * figure[i - 1][j] +
             figure[i - 1][j + 1] + 
             2 * figure[i][j - 1] + 
             4 * figure[i][j] +
             2 * figure[i][j + 1] + 
             figure[i + 1][j - 1] +
             2 * figure[i + 1][j] + figure[i + 1][j + 1]) /
            16.0);
      }
    }

    // 处理边界
    for (size_t i = 1; i < size - 1; ++i) {
      result[i][0] = static_cast<unsigned char>(
          (float)(figure[i-1][0] + 2*figure[i-1][0] + figure[i-1][1] +
           2*figure[i][0] + 4*figure[i][0] + 2*figure[i][1] +
           figure[i+1][0] + 2*figure[i+1][0] + figure[i+1][1]) / 16.0f);

      result[i][size-1] = static_cast<unsigned char>(
          (float)(figure[i-1][size-2] + 2*figure[i-1][size-1] +
           figure[i-1][size-1] + 2*figure[i][size-2] +
           4*figure[i][size-1] + 2*figure[i][size-1] +
           figure[i+1][size-2] + 2*figure[i+1][size-1] +
           figure[i+1][size-1]) / 16.0f);
    }

    // 处理四个角点
    // 左上角
    result[0][0] = static_cast<unsigned char>(
                (float)(4 * figure[0][0] + 2 * figure[0][1] + 2 * figure[1][0] +
                    figure[1][1]) /
                   9.0f); 

    // 右上角
    result[0][size - 1] = static_cast<unsigned char>(
                  (float)(4 * figure[0][size - 1] + 2 * figure[0][size - 2] +
                           2 * figure[1][size - 1] + figure[1][size - 2]) /
                          9.0f);

    // 左下角
    result[size - 1][0] = static_cast<unsigned char>(
                  (float)(4 * figure[size - 1][0] + 2 * figure[size - 1][1] +
                           2 * figure[size - 2][0] + figure[size - 2][1]) /
                          9.0f);

    // 右下角
    result[size - 1][size - 1] = static_cast<unsigned char>(
                (float)(4 * figure[size - 1][size - 1] + 2 * figure[size - 1][size - 2] +
         2 * figure[size - 2][size - 1] + figure[size - 2][size - 2]) /
        9.0f);
    }

    /*
    */

  // Power law transformation
  // FIXME: Feel free to optimize this function
  // Hint: LUT to optimize this function?
  void powerLawTransformation() {
    const size_t vectorSize = 32;  // 一次处理32个字节
    
    for (size_t i = 0; i < size; ++i) {
      size_t j = 0;
      // 使用AVX2处理每行的主要部分
      for (; j + vectorSize <= size; j += vectorSize) {
        // 加载32个字节
        __m256i input = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&figure[i][j]));
        
        // 处理低16个字节
        __m128i input_low = _mm256_extracti128_si256(input, 0);
        __m256i input_32_low1 = _mm256_cvtepu8_epi32(input_low);
        __m256i result_32_low1 = _mm256_i32gather_epi32(
            powerLUT32,
            input_32_low1,
            4
        );
        
        __m128i input_low_high = _mm_unpackhi_epi64(input_low, input_low);
        __m256i input_32_low2 = _mm256_cvtepu8_epi32(input_low_high);
        __m256i result_32_low2 = _mm256_i32gather_epi32(
            powerLUT32,
            input_32_low2,
            4
        );
        
        // 处理高16个字节
        __m128i input_high = _mm256_extracti128_si256(input, 1);
        __m256i input_32_high1 = _mm256_cvtepu8_epi32(input_high);
        __m256i result_32_high1 = _mm256_i32gather_epi32(
            powerLUT32,
            input_32_high1,
            4
        );
        
        __m128i input_high_high = _mm_unpackhi_epi64(input_high, input_high);
        __m256i input_32_high2 = _mm256_cvtepu8_epi32(input_high_high);
        __m256i result_32_high2 = _mm256_i32gather_epi32(
            powerLUT32,
            input_32_high2,
            4
        );
        
        // 打包结果
        __m256i result_16_low = _mm256_packus_epi32(result_32_low1, result_32_low2);
        __m256i result_16_high = _mm256_packus_epi32(result_32_high1, result_32_high2);
        
        __m128i result_8_low = _mm_packus_epi16(
            _mm256_castsi256_si128(result_16_low),
            _mm256_extracti128_si256(result_16_low, 1)
        );
        
        __m128i result_8_high = _mm_packus_epi16(
            _mm256_castsi256_si128(result_16_high),
            _mm256_extracti128_si256(result_16_high, 1)
        );
        
        // 存储结果
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(&result[i][j]),
            _mm256_set_m128i(result_8_high, result_8_low)
        );
      }
      
      // 处理剩余的元素
      for (; j < size; ++j) {
        result[i][j] = static_cast<unsigned char>(powerLUT32[figure[i][j]]);
      }
    }
  }

  // Run benchmark
  unsigned int calcChecksum() {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        sum += result[i][j];
        sum %= mod;
      }
    }
    return sum;
  }
  void runBenchmark() {
    auto start = std::chrono::high_resolution_clock::now();
    gaussianFilter();
    auto middle = std::chrono::high_resolution_clock::now();

    unsigned int sum = calcChecksum();

    auto middle2 = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    auto end = std::chrono::high_resolution_clock::now();

    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";

    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(middle - start) +
        std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2);
    std::cout << "Benchmark time: " << milliseconds.count() << " ms\n";
    std::cout << "gaussianFilter: " << std::chrono::duration_cast<std::chrono::milliseconds>(middle - start).count() << " ms\n";
    std::cout << "powerLawTransformation: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2).count() << " ms\n";
  }
};

// 在类外部定义静态成员
alignas(32) int FigureProcessor::powerLUT32[256] = {0};
bool FigureProcessor::lutInitialized = false;

// Main function
// !!! Please do not modify the main function !!!
int main(int argc, const char **argv) {
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}
