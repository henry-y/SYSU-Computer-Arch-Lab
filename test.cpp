#include <iostream>
#include <immintrin.h>  // AVX2 header

int main() {
    // 初始化一个 128 位整数向量 a，包含 16 个字节
    __m128i a = _mm_set_epi8(0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08,
                             0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00);

    // 设置一个掩码 left_shuffle_mask，用于字节重排
    __m128i left_shuffle_mask = _mm_set_epi16(6, 5, 4, 3, 2, 1, 0, 0);

    // 使用 _mm_shuffle_epi8 对向量 a 进行字节重排
    __m128i result = _mm_shuffle_epi8(a, left_shuffle_mask);

    // 将结果转换为字节数组以方便输出
    alignas(16) uint8_t res[16];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(res), result);

    // 输出结果
    std::cout << "Original vector a: ";
    for (int i = 0; i < 16; ++i) {
        std::cout << std::hex << "0x" << (int)res[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Shuffled result: ";
    for (int i = 0; i < 8; ++i) {  // 只输出前 8 个字节，结果长度为 8 字节
        std::cout << "0x" << (int)res[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
