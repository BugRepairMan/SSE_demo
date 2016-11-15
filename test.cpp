#include <iostream>
#include <cstdint>
#include <chrono>
#include <x86intrin.h>
#include <omp.h>

using namespace std;

namespace Normal {
	class LCG {
		public:
			LCG(uint32_t seed) : mSeed(seed) {}
			float operator()() {
				mSeed = mSeed * 214013 + 2531011;
				union {
					uint32_t u;
					float f;
				}u = { (mSeed >> 9) | 0x3F800000 };
				return u.f - 1.0f;
			}
		private:
			uint32_t mSeed;
	};

	unsigned test(const unsigned simulate_total) 
	{
		unsigned inside_count = 0;

		LCG rng(0);
		for (unsigned i = 1; i < simulate_total; i++){
			float a = rng();
			float b = rng();

			if (a * a + b * b < 1.0f)
				inside_count++;

			a = 1.0f - a;
			b = 1.0f - b;

			if (a * a + b * b < 1.0f)
				inside_count++;
		}
		return inside_count;
	}
}

namespace SSE {
	static const __m128i cLCG1 = _mm_set1_epi32(214013);
	static const __m128i cLCG2 = _mm_set1_epi32(2531011);
	static const __m128i cLCGmask = _mm_set1_epi32(0x3F800000);
	static const __m128i cOnei = _mm_set1_epi32(1);
	static const __m128 cOne = _mm_set1_ps(1.0f);

	inline __m128i mul_uint32(__m128i a, __m128i b) {
		return _mm_mullo_epi32(a, b);
		//const __m128i tmp1 = _mm_mul_epu32(a, b);
		//const __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
		//return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));
	}

	class LCG_SSE {
		public:
			LCG_SSE(__m128i seed) : mSeed(seed) {}
			__m128 operator()() {
				mSeed = _mm_add_epi32(mul_uint32(mSeed, cLCG1), cLCG2);
				const __m128i u = _mm_or_si128(_mm_srli_epi32(mSeed, 9), cLCGmask);
				return _mm_sub_ps(_mm_castsi128_ps(u), cOne);
			}
		private:
			__m128i mSeed;
	};

	unsigned test(const unsigned simulate_total)
	{
		__m128i inside_count1 = _mm_setzero_si128();
		__m128i inside_count2 = _mm_setzero_si128();

		LCG_SSE rng1(_mm_setr_epi32(0, 1, 2, 3)), rng2(_mm_setr_epi32(4, 5, 6, 7));
		for (unsigned i = 1; i < simulate_total / 4; i++){
			const __m128 a = rng1();
			const __m128 b = rng2();

			const __m128 r1 = _mm_cmplt_ps(_mm_add_ps(_mm_mul_ps(a, a), _mm_mul_ps(b, b)), cOne);
			inside_count1 = _mm_add_epi32(inside_count1, _mm_and_si128(_mm_castps_si128(r1), cOnei));

			const __m128 c = _mm_sub_ps(cOne, a);
			const __m128 d = _mm_sub_ps(cOne, b);

			const __m128 r2 = _mm_cmplt_ps(_mm_add_ps(_mm_mul_ps(c, c), _mm_mul_ps(d, d)), cOne);
			inside_count2 = _mm_add_epi32(inside_count2, _mm_and_si128(_mm_castps_si128(r2), cOnei));
		}

		unsigned inside_count = 0;
		unsigned *c1 = (unsigned *)(&inside_count1);
		unsigned *c2 = (unsigned *)(&inside_count2);
		for (int i = 0; i < 4; i++)
			//inside_count += inside_count1.m128i_u32[i] + inside_count2.m128i_u32[i];
			inside_count += c1[i] + c2[i];

		return inside_count;
	}
}

namespace AVX2 {
	static const __m256i cLCG1 = _mm256_set1_epi32(214013);
	static const __m256i cLCG2 = _mm256_set1_epi32(2531011);
	static const __m256i cLCGmask = _mm256_set1_epi32(0x3F800000);
	static const __m256i cOnei = _mm256_set1_epi32(1);
	static const __m256 cOne = _mm256_set1_ps(1.0f);

	inline __m256i avx_mul_uint32(__m256i a, __m256i b) {
		return _mm256_mullo_epi32(a, b);
	}

	inline __m256 avx_cmplt_ps(__m256 a, __m256 b)
	{
		return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
	}

	class LCG_AVX {
		public:
			LCG_AVX(__m256i seed) : mSeed(seed) {}
			__m256 operator()() {
				mSeed = _mm256_add_epi32(avx_mul_uint32(mSeed, cLCG1), cLCG2);
				const __m256i u = _mm256_or_si256(_mm256_srli_epi32(mSeed, 9), cLCGmask);
				return _mm256_sub_ps(_mm256_castsi256_ps(u), cOne);
			}
		private:
			__m256i mSeed;
	};


	unsigned test(const unsigned simulate_total)
	{
		__m256i inside_count1 = _mm256_setzero_si256();
		__m256i inside_count2 = _mm256_setzero_si256();

		LCG_AVX rng1(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)), rng2(_mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15));
		for (unsigned i = 1; i < simulate_total / 8; i++){
			const __m256 a = rng1();
			const __m256 b = rng2();

			const __m256 r1 = avx_cmplt_ps(_mm256_add_ps(_mm256_mul_ps(a, a), _mm256_mul_ps(b, b)), cOne);
			inside_count1 = _mm256_add_epi32(inside_count1, _mm256_and_si256(_mm256_castps_si256(r1), cOnei));

			const __m256 c = _mm256_sub_ps(cOne, a);
			const __m256 d = _mm256_sub_ps(cOne, b);

			const __m256 r2 = avx_cmplt_ps(_mm256_add_ps(_mm256_mul_ps(c, c), _mm256_mul_ps(d, d)), cOne);
			inside_count2 = _mm256_add_epi32(inside_count2, _mm256_and_si256(_mm256_castps_si256(r2), cOnei));
		}

		unsigned inside_count = 0;
		unsigned *c1 = (unsigned *)(&inside_count1);
		unsigned *c2 = (unsigned *)(&inside_count2);
		for (int i = 0; i < 8; i++)
			inside_count += c1[i] + c2[i];

		return inside_count;
	}
}

namespace SSE_OpenMP{
	static const __m128i cLCG1 = _mm_set1_epi32(214013);
	static const __m128i cLCG2 = _mm_set1_epi32(2531011);
	static const __m128i cLCGmask = _mm_set1_epi32(0x3F800000);
	static const __m128i cOnei = _mm_set1_epi32(1);
	static const __m128 cOne = _mm_set1_ps(1.0f);

	inline __m128i mul_uint32(__m128i a, __m128i b) {
		return _mm_mullo_epi32(a, b);
		//const __m128i tmp1 = _mm_mul_epu32(a, b);
		//const __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
		//return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0)));
	}

	class LCG {
		public:
			LCG(__m128i seed) : mSeed(seed) {}
			__m128 operator()() {
				mSeed = _mm_add_epi32(mul_uint32(mSeed, cLCG1), cLCG2);
				const __m128i u = _mm_or_si128(_mm_srli_epi32(mSeed, 9), cLCGmask);
				return _mm_sub_ps(_mm_castsi128_ps(u), cOne);
			}
		private:
			__m128i mSeed;
	};

	unsigned test(const unsigned simulate_total)
	{
		unsigned inside_count = 0;

#pragma omp parallel num_threads(4) reduction(+ : inside_count)
		{
			__m128i inside_count1 = _mm_setzero_si128();
			__m128i inside_count2 = _mm_setzero_si128();

			int j = omp_get_thread_num() * 8;
			LCG rng1(_mm_setr_epi32(j + 0, j + 1, j + 2, j + 3)), rng2(_mm_setr_epi32(j + 4, j + 5, j + 6, j + 7));

			for (unsigned i = 1; i < simulate_total / 16; i++) {
				const __m128 a = rng1();
				const __m128 b = rng2();

				const __m128 r1 = _mm_cmplt_ps(_mm_add_ps(_mm_mul_ps(a, a), _mm_mul_ps(b, b)), cOne);
				inside_count1 = _mm_add_epi32(inside_count1, _mm_and_si128(_mm_castps_si128(r1), cOnei));

				const __m128 c = _mm_sub_ps(cOne, a);
				const __m128 d = _mm_sub_ps(cOne, b);

				const __m128 r2 = _mm_cmplt_ps(_mm_add_ps(_mm_mul_ps(c, c), _mm_mul_ps(d, d)), cOne);
				inside_count2 = _mm_add_epi32(inside_count2, _mm_and_si128(_mm_castps_si128(r2), cOnei));
			}

			unsigned *c1 = (unsigned *)(&inside_count1);
			unsigned *c2 = (unsigned *)(&inside_count2);
			for (int i = 0; i < 4; i++)
				inside_count += c1[i] + c2[i];
		}

		return inside_count;
	}
}

int main()
{
	const unsigned simulate_total = 250000000;
	unsigned inside_count = 0;

	// 1. Normal test:
	auto start = chrono::system_clock::now();

	inside_count = Normal::test(simulate_total);	
	
	auto end = chrono::system_clock::now();
	auto elapse =chrono::duration_cast<chrono::duration<double>> (end - start);
	cout << "Normal: Time = " << elapse.count() << " s\n";

	cout << inside_count / double(simulate_total) * 2 << "\n\n";

	// 2. SSE test:
	start = chrono::system_clock::now();
	
	inside_count = SSE::test(simulate_total);
	
	end = chrono::system_clock::now();
	elapse =chrono::duration_cast<chrono::duration<double>> (end - start);
	cout << "SSE: Time =  " << elapse.count() << " s\n";

	cout << inside_count / double(simulate_total) * 2 << "\n\n";

	// 3. AVX2 test:
	start = chrono::system_clock::now();
	
	inside_count = AVX2::test(simulate_total);
	
	end = chrono::system_clock::now();
	elapse =chrono::duration_cast<chrono::duration<double>> (end - start);
	cout << "AVX2: Time =  " << elapse.count() << " s\n";

	cout << inside_count / double(simulate_total) * 2 << "\n\n";

	// 4. SSE_OpenMP test:
	start = chrono::system_clock::now();
	
	inside_count = SSE_OpenMP::test(simulate_total);
	
	end = chrono::system_clock::now();
	elapse =chrono::duration_cast<chrono::duration<double>> (end - start);
	cout << "SSE_OpenMP: Time =  " << elapse.count() << " s\n";

	cout << inside_count / double(simulate_total) * 2 << "\n\n";

	return 0;
}
