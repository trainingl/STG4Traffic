# import sys
# sys.path.append('../')

import numpy as np
import pandas as pd
import networkx as nx
import random
import warnings
warnings.filterwarnings("ignore")

#################################### Node2Vec #########################################
class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		G = self.G
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		walk = [start_node]
		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break
		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print ('Walk iteration:')
		for walk_iter in range(num_walks):
			print (str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self.G
		p = self.p
		q = self.q
		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		G = self.G
		is_directed = self.is_directed
		alias_nodes = {}
		for node in G.nodes():
			unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			alias_nodes[node] = alias_setup(normalized_probs)
		alias_edges = {}
		triads = {}
		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges
		return


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)
	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K * prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)
	return J, q


def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)
	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]
#################################### Node2Vec #########################################



#################################### GenerateSE #######################################
import numpy as np
import networkx as nx
from gensim.models import Word2Vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000
Adj_file = '../data/METR-LA/Adj(METR).txt'
SE_file = '../data/METR-LA/SE(METR).txt'

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())

    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, vector_size = dimensions, window = 10, min_count=0, sg=1,
        workers = 8, epochs = iter)
    model.wv.save_word2vec_format(output_file)
    return
    
# from lib.utils import *
# graph_pkl = "../data/METR-LA/processed/adj_mx.pkl"
# _, _, adj_mx = load_pickle(graph_pkl)
# num_nodes = adj_mx.shape[0]
# with open(Adj_file, 'w+', encoding = "utf-8") as f:
# 	for i in range(num_nodes):
# 		for j in range(num_nodes):
# 			f.write(str(i) + ' ' + str(j) + ' ' + str('%.6f' % adj_mx[i, j]) + '\n')
# nx_G = read_graph(Adj_file)
# G = Graph(nx_G, is_directed, p, q)
# G.preprocess_transition_probs()
# walks = G.simulate_walks(num_walks, walk_length)
# learn_embeddings(walks, dimensions, SE_file)  # 这里需要train一个词模型
#################################### GenerateSE #######################################


#######################################################################################
import argparse
from lib.data_loader import *
from lib.generate_data import *

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y


class DataLoaderTE(object):
    def __init__(self, xs, ys, te, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            t_padding = np.repeat(te[-1:], num_padding, axis=0)
            # ycl_padding = np.repeat(ycl[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            te = np.concatenate([te, t_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.te = te

    def shuffle(self):
        # Disrupt the original sample.
        permutation = np.random.permutation(self.size)
        xs, ys, te = self.xs[permutation], self.ys[permutation], self.te[permutation]
        self.xs = xs
        self.ys = ys
        self.te = te

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                te_i = self.te[start_ind: end_ind, ...]
                yield (x_i, y_i, te_i)
                self.current_ind += 1
        return _wrapper()


def loadData(args):
	data = {}
	df = pd.read_hdf(args.traffic_file)
	Traffic = df.values
	# train/val/test 
	num_step = df.shape[0]
	train_steps = round(args.train_ratio * num_step)
	test_steps = round(args.test_ratio * num_step)
	val_steps = num_step - train_steps - test_steps
	# X, Y
	train = Traffic[: train_steps]
	val = Traffic[train_steps : train_steps + val_steps]
	test = Traffic[-test_steps :]
    # X, Y 
	trainX, trainY = seq2instance(train, args.window, args.horizon)
	valX, valY = seq2instance(val, args.window, args.horizon)
	testX, testY = seq2instance(test, args.window, args.horizon)

	# normalization
	scaler = StandardScaler(mean=np.mean(trainX), std=np.std(trainX))
	trainX = scaler.inverse_transform(trainX)
	valX = scaler.inverse_transform(valX)
	testX = scaler.inverse_transform(testX)

	# spatial embedding 
	f = open(args.SE_file, mode = 'r')
	lines = f.readlines()
	temp = lines[0].split(' ')
	N, dims = int(temp[0]), int(temp[1])
	SE = np.zeros(shape = (N, dims), dtype = np.float32)
	for line in lines[1 :]:
		temp = line.split(' ')
		index = int(temp[0])
		SE[index] = temp[1 :]
	data['SE'] = SE
    # temporal embedding 
	time = pd.DatetimeIndex(df.index)
	dayofweek = np.reshape(np.array(time.weekday), (-1, 1))
	# timeofday = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
	timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // 300
	timeofday = np.reshape(timeofday, (-1, 1))
	Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
	train = Time[: train_steps]
	val = Time[train_steps : train_steps + val_steps]
	test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
	trainTE = seq2instance(train, args.window, args.horizon)
	trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
	# print(len(trainX), trainTE.shape, len(trainY))
	valTE = seq2instance(val, args.window, args.horizon)
	valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
	testTE = seq2instance(test, args.window, args.horizon)
	testTE = np.concatenate(testTE, axis = 1).astype(np.int32)
	
	data['train_loader'] = DataLoaderTE(trainX, trainY, trainTE, args.batch_size)
	data['val_loader'] = DataLoaderTE(valX, valY, valTE, args.batch_size)
	data['test_loader'] = DataLoaderTE(testX, testY, testTE, args.batch_size)
	return data, scaler
#######################################################################################

# if __name__ == "__main__":
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("--traffic_file", type=str, default="../data/METR-LA/metr-la.h5")
# 	parser.add_argument("--window", type=int, default=12)
# 	parser.add_argument("--horizon", type=int, default=12)
# 	parser.add_argument("--train_ratio", type=float, default=0.7)
# 	parser.add_argument("--test_ratio", type=float, default=0.2)
# 	parser.add_argument("--batch_size", type=int, default=64)
# 	parser.add_argument("--SE_file", type=str, default='../data/METR-LA/SE(METR).txt')
# 	args = parser.parse_args()
# 	data, scaler = loadData(args)
# 	for idx, (X, Y, TE) in enumerate(data['train_loader'].get_iterator()):
# 		print(X.shape, Y.shape, TE.shape)
# 		break