import os
import time
import pickle
import argparse

import tensorflow as tf
import numpy as np
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import utils
import models

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-dir', '-i',
                        required=True,
                        dest='input_dir',
                        help='Trained model directory. The --output-dir value used for training.')

    parser.add_argument('--checkpoint', '-c',
                        required=True,
                        dest='checkpoint',
                        help='Model checkpoint to use for sampling. Expects a .ckpt file.')

    parser.add_argument('--output', '-o',
                        default='samples.txt',
                        help='File path to save generated samples to (default: samples.txt)')

    parser.add_argument('--num-samples', '-n',
                        type=int,
                        default=1000000,
                        dest='num_samples',
                        help='The number of password samples to generate (default: 1000000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length. Use the same value that you did for training. (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error('"{}" folder doesn\'t exist'.format(args.input_dir))

    if not os.path.exists(args.checkpoint + '.meta'):
        parser.error('"{}.meta" file doesn\'t exist'.format(args.checkpoint))

    if not os.path.exists(os.path.join(args.input_dir, 'charmap.pickle')):
        parser.error('charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))

    if not os.path.exists(os.path.join(args.input_dir, 'inv_charmap.pickle')):
        parser.error('inv_charmap.pickle doesn\'t exist in {}, are you sure that directory is a trained model directory'.format(args.input_dir))

    return args

args = parse_args()

with open(os.path.join(args.input_dir, 'charmap.pickle'), 'rb') as f:
    charmap = pickle.load(f, encoding='latin1')

with open(os.path.join(args.input_dir, 'inv_charmap.pickle'), 'rb') as f:
    inv_charmap = pickle.load(f, encoding='latin1')

fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))

with tf.Session() as session:

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in xrange(len(samples)):
            decoded = []
            for j in xrange(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    def save(samples):
        with open(args.output, 'a') as f:
                for s in samples:
                    s = "".join(s).replace('`', '')
                    f.write(s + "\n")

    saver = tf.train.Saver()
    saver.restore(session, args.checkpoint)

    samples = []
    then = time.time()
    start = time.time()
    for i in xrange(int(args.num_samples / args.batch_size)):
        
        samples.extend(generate_samples())

        # append to output file every 1000 batches
        if i % 1000 == 0 and i > 0: 
            
            save(samples)
            samples = [] # flush

            print('wrote {} samples to {} in {:.2f} seconds. {} total.'.format(1000 * args.batch_size, args.output, time.time() - then, i * args.batch_size))
            then = time.time()
    
    save(samples)
    print('finished in {:.2f} seconds'.format(time.time() - start))
