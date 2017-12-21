from __future__ import print_function
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]

output_dir = '.'

def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush():
	prints = []

	for name, vals in _since_last_flush.items():
		prints.append("{}\t{}".format(name, np.mean(vals.values())))
		_since_beginning[name].update(vals)

		x_vals = np.sort(_since_beginning[name].keys())
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(os.path.join(output_dir, name.replace(' ', '_')+'.jpg'))

	print("iter {}\t{}".format(_iter[0], "\t".join(prints)))
	_since_last_flush.clear()

	with open(os.path.join(output_dir, 'log.pkl'), 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)