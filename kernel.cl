#if 0
class LCG {
		public:
			LCG(uint32 seed) : mSeed(seed) {}
			float operator()() {
				mSeed = mSeed * 214013 + 2531011;
				union {
					uint32 u;
					float f;
				}u = { (mSeed >> 9) | 0x3F800000 };
				return u.f - 1.0f;
			}
		private:
			uint32 mSeed;
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
#endif

float rng(unsigned *seed) {
	unsigned mSeed = *seed;

	mSeed = mSeed * 214013 + 2531011;
	union {
		unsigned u;
		float f;
	}u = { (mSeed >> 9) | 0x3F800000 };

	*seed = mSeed;
	return u.f - 1.0f;
}

__kernel void
compute(__global unsigned int *count, unsigned int simulate_local){
	unsigned int index = get_global_id(0);
	unsigned int local_count = 0;

	float a,b;
	unsigned seed = 0;

	for(unsigned int i = 0; i < simulate_local; ++i) {
		a = rng(&seed);
		b = rng(&seed);

		if (a * a + b * b < 1.0f)
			local_count++;

		a = 1.0f - a;
		b = 1.0f - b;

		if (a * a + b * b < 1.0f)
			local_count++;
	}
	count[index] = local_count;
}
	
