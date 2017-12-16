import sys
import random

random.seed(1337)

# run "python make-rockyou-full.py > data/rockyou-full.txt" to create this file
# if it doesn't already exist
with open('../data/rockyou-full.txt', 'r') as f:

    # we can't line buffer because we need everything to randomize
    lines = f.readlines()

    # filter only passwords with 10 characters or fewer
    print('[info] filtering rockyou to include only 10 character or less passwords')
    lines = filter(lambda x: len(x) <= 10, lines)

    # randomize order
    print('[info] shuffling rockyou')
    random.shuffle(lines)

    split = int(len(lines) * 0.80)

    with open('../data/train.txt', 'w') as f:
        print('[info] saving 80% ({}) of dataset for training in ../data/train.txt'.format(split))
        f.write(''.join(lines[0:split]))

    with open('../data/test.txt', 'w') as f:
        print('[info] saving 20% ({}) of dataset for testing in ../data/test.txt'.format(len(lines) - split))
        f.write(''.join(lines[split:]))
