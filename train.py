import os
import sys
import time
import pickle
import argparse
import numpy as np
import tensorflow.compat.v1 as tf  # Use TensorFlow 1.x compatibility mode
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior for 1.x compatibility

import utils
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot
import models

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    parser.add_argument('--iters', '-n',
                        type=int,
                        default=200000,
                        dest='iters',
                        help='The number of training iterations (default: 200000)')

    parser.add_argument('--batch-size', '-b',
                        type=int,
                        default=64,
                        dest='batch_size',
                        help='Batch size (default: 64).')

    parser.add_argument('--seq-length', '-l',
                        type=int,
                        default=10,
                        dest='seq_length',
                        help='The maximum password length (default: 10)')

    parser.add_argument('--layer-dim', '-d',
                        type=int,
                        default=128,
                        dest='layer_dim',
                        help='The hidden layer dimensionality for the generator and discriminator (default: 128)')

    parser.add_argument('--critic-iters', '-c',
                        type=int,
                        default=10,
                        dest='critic_iters',
                        help='The number of discriminator weight updates per generator update (default: 10)')

    parser.add_argument('--lambda', '-p',
                        type=int,
                        default=10,
                        dest='lamb',
                        help='The gradient penalty lambda hyperparameter (default: 10)')

    return parser.parse_args()

args = parse_args()

# Load dataset
lines, charmap, inv_charmap = utils.load_dataset(
    path=args.training_data,
    max_length=args.seq_length
)

# Ensure output directories exist
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

# Save character maps with pickle (use HIGHEST_PROTOCOL for efficiency)
with open(os.path.join(args.output_dir, 'charmap.pickle'), 'wb') as f:
    pickle.dump(charmap, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(args.output_dir, 'inv_charmap.pickle'), 'wb') as f:
    pickle.dump(inv_charmap, f, protocol=pickle.HIGHEST_PROTOCOL)

# If reading pickled files later, use 'latin1' to avoid decoding errors
with open(os.path.join(args.output_dir, 'charmap.pickle'), 'rb') as f:
    charmap = pickle.load(f, encoding='latin1')

with open(os.path.join(args.output_dir, 'inv_charmap.pickle'), 'rb') as f:
    inv_charmap = pickle.load(f, encoding='latin1')

# Define placeholders
real_inputs_discrete = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_length])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = models.Generator(args.batch_size, args.seq_length, args.layer_dim, len(charmap))
fake_inputs_discrete = tf.argmax(fake_inputs, axis=fake_inputs.get_shape().ndims-1)

disc_real = models.Discriminator(real_inputs, args.seq_length, args.layer_dim, len(charmap))
disc_fake = models.Discriminator(fake_inputs, args.seq_length, args.layer_dim, len(charmap))

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# Gradient penalty for WGAN
alpha = tf.random.uniform(
    shape=[args.batch_size, 1, 1],
    minval=0.0,
    maxval=1.0
)

differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha * differences)
gradients = tf.gradients(models.Discriminator(interpolates, args.seq_length, args.layer_dim, len(charmap)), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
disc_cost += args.lamb * gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - args.batch_size + 1, args.batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + args.batch_size]],
                dtype='int32'
            )

# Training loop
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = [
            [inv_charmap[s] for s in sample] for sample in samples
        ]
        return decoded_samples

    gen = inf_train_gen()

    for iteration in range(args.iters):
        start_time = time.time()

        # Train generator
        if iteration > 0:
            session.run(gen_train_op)

        # Train critic
        for _ in range(args.critic_iters):
            _data = next(gen)
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_inputs_discrete: _data})

        # Generate and save samples
        if iteration % 100 == 0:
            samples = []
            for _ in range(10):
                samples.extend(generate_samples())

            sample_file = os.path.join(args.output_dir, 'samples', f'samples_{iteration}.txt')
            with open(sample_file, 'w', encoding='latin1') as f:
                for sample in samples:
                    f.write("".join(sample) + "\n")

        # Save checkpoints
        if iteration % args.save_every == 0 and iteration > 0:
            saver = tf.train.Saver()
            saver.save(session, os.path.join(args.output_dir, 'checkpoints', f'checkpoint_{iteration}.ckpt'))

        # Update plots and tick
        if iteration % 100 == 0:
            lib.plot.flush()

        # Ensure tick is called every iteration
        lib.plot.tick()
