import random

def random_sample(num, all_samples, sample_method):
    if sample_method == 'average':
        return [int(all_samples / num) for i in range(num)]
    if sample_method == 'random':
        block = int(all_samples / 50)
        ns = [1 for i in range(num)]
        for i in range(block-num):
            index = random.randint(0, num-1)
            ns[index] += 1
        ns = [ns[i]*50 for i in range(num)]
        return ns

        sizes = []
        for i in range(num):
            size = random.randint(1, 2 * all_samples / num)
            sizes.append(size)
        for size in sizes:
            size = size / sum(sizes) * all_samples
        return sizes