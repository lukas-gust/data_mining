import numpy as np

class CountMinSketch:
    _mersenne_prime = (1 << 61) - 1
    _max_hash = (1 << 32) - 1

    def __init__(self, k, t, seed=None):
        self.k = k
        self.t = t

        if seed:
            self.generator = np.random.RandomState(seed)
        else:
            self.generator = np.random

        self.salts = self._generate_hash_funcs(self.t)

        self.counts = np.zeros(shape=(self.t, self.k), dtype=int)

    def _generate_hash_funcs(self, t):
        rands1 = self.generator.randint(1, self.k, t, dtype=np.uint64)
        rands2 = self.generator.randint(0, self.k, t, dtype=np.uint64)
        salts1 = np.array([salt for salt in rands1])
        salts2 = np.array([salt for salt in rands2])
        return [salts1, salts2]

    def _h_func(self, hashable):
        hv = np.uint64(hash(hashable))
        hashes = np.bitwise_and((self.salts[0] * hv + self.salts[1]) % self.k, np.uint64(self._mersenne_prime))
        return hashes

    def process_stream(self, stream):
        for item in stream:
            indices = self._h_func(item)
            for i, row in enumerate(self.counts):
                row[indices[i]] += 1

        return self.counts

    def count_min(self, item):
        indices = self._h_func(item)
        min_sketch = min([row[indices[i]] for i, row in enumerate(self.counts)])
        return min_sketch

if __name__ == '__main__':
    cms = CountMinSketch(10, 5)
    c = cms.process_stream('abacadaeafagahaiajakalamanaoapaqarasatauavawaxayaza')
    print(c)
    print(cms.count_min('a'))