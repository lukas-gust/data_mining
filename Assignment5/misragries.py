#!/usr/bin/python3
from collections import Counter

class MG():
    def __init__(self, k):
        self.k = k
        self.c = Counter()

    def process_stream(self, data_stream):
        for i in data_stream:
            if i in self.c:
                self.c.update(i)
            else:
                if len(self.c) < self.k-1:
                    self.c.update(i)
                else:
                    for a in self.c.copy():
                        self.c[a] -= 1
                        if self.c[a] == 0:
                            del self.c[a]

        assert len(self.c) <= self.k
        return self.c

if __name__ == '__main__':
    data = 'aaaaaabcdefasdfasghs'

    mg = MG(9)
    counts = mg.process_stream(data)
    print(counts)
