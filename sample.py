import tensorflow as tf
import numpy as np
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import utils.utils

OUTPUT=10e8
BATCH_SIZE=64
SEQ_LEN = 10 # Sequence length in characters
DIM = 128 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.

def make_noise(shape):
    return tf.random_normal(shape)

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)

def Generator(n_samples, prev_outputs=None):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    return output

lines, charmap, inv_charmap = utils.load_dataset(
    path='data/train.txt',
    max_length=SEQ_LEN
)

fake_inputs = Generator(BATCH_SIZE)

with tf.Session() as session:

    session.run(tf.initialize_all_variables())

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


    saver = tf.train.Saver()
    saver.restore(session, 'experiments/paper/checkpoints/195000.ckpt')

    samples = []
    for i in xrange(int(OUTPUT/BATCH_SIZE)):
        if i % 1000 == 0: print(i * BATCH_SIZE)
        samples.extend(generate_samples())

    with open('samples2.txt', 'w') as f:
        for s in samples:
            s = "".join(s).replace('`', '')
            f.write(s + "\n")
