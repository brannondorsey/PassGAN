# https://stackoverflow.com/questions/886237/how-can-i-randomize-the-lines-in-a-file-using-standard-tools-on-red-hat-linux
import random

with open('../data/rockyou-full.txt', 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
