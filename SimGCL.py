import reckit
import torch
from torch import nn
import scipy.sparse as sp
import random
import numpy as np
import logging
import torch.optim as optim
import multiprocessing as mp
import os
import sys
from time import strftime
from time import localtime
import argparse
from RankingMetrics import *


class SimGCL(nn.Module):
	def __init__(self, data_train, data_test, num_users, num_items, num_tags, embedding_size, temperature,
	             layer, batch_size, eps, noise_type):
		super(SimGCL, self).__init__()
		self.train_data = data_train
		self.test_data = data_test
		self.num_users = num_users
		self.num_items = num_items
		self.num_tags = num_tags
		self.embedding_size = embedding_size
		self.layers = layer
		self.temperature = temperature
		self.batch_size = batch_size
		self.eps = eps
		self.noise_type = noise_type
		self.train_uit_dict = self.get_uit_dict(self.train_data)
		self.test_uit_dict = self.get_uit_dict(self.test_data)
		self.present_train_size = 0

		self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_size)
		self.embedding_ut = torch.nn.Embedding(num_embeddings=self.num_tags, embedding_dim=self.embedding_size)

		self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_size)
		self.embedding_it = torch.nn.Embedding(num_embeddings=self.num_tags, embedding_dim=self.embedding_size)

		nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
		nn.init.xavier_uniform_(self.embedding_ut.weight, gain=1)

		nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
		nn.init.xavier_uniform_(self.embedding_it.weight, gain=1)

		self.Graph_ut = self.getSparseGraph(self.num_users, self.num_tags, 0)
		self.Graph_it = self.getSparseGraph(self.num_items, self.num_tags, 1)

	def getSparseGraph(self, num1, num2, id):
		Graph = None
		device = 'cuda'
		Mat = sp.dok_matrix((num1, num2), dtype=np.float32)
		adj_mat = sp.dok_matrix((num1 + num2, num1 + num2), dtype=np.float32)
		adj_mat = adj_mat.tolil()
		R = Mat.tolil()
		for data in self.train_data:
			tags = data[2]
			index = data[id]
			R[index, tags] = 1
		adj_mat[:num1, num1:] = R
		adj_mat[num1:, :num1] = R.T
		adj_mat = adj_mat.todok()

		norm_adj = self.norm_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
		Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
		Graph = Graph.coalesce().to(device)

		return Graph

	def norm_adj_single(self, adj_mat):
		rowsum = np.array(adj_mat.sum(axis=1))
		d_inv = np.power(rowsum, -0.5).flatten()
		d_inv[np.isinf(d_inv)] = 0.
		d_mat = sp.diags(d_inv)
		norm_adj = d_mat.dot(adj_mat)
		norm_adj = norm_adj.dot(d_mat)
		norm_adj = norm_adj.tocsr()
		return norm_adj

	def _convert_sp_mat_to_sp_tensor(self, X):
		coo = X.tocoo().astype(np.float32)
		row = torch.Tensor(coo.row).long()
		col = torch.Tensor(coo.col).long()
		index = torch.stack([row, col])
		data = torch.FloatTensor(coo.data)
		return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

	def forward(self, users, items, pos_tags, neg_tags):
		# light
		all_users, all_tags_user = self.computer(self.Graph_ut, self.embedding_user, self.embedding_ut, self.num_users)
		all_items, all_tags_item = self.computer(self.Graph_it, self.embedding_item, self.embedding_it, self.num_items)

		users_emb = all_users[users]
		items_emb = all_items[items]

		pos_ut_emb = all_tags_user[pos_tags]
		pos_it_emb = all_tags_item[pos_tags]

		neg_ut_emb = all_tags_user[neg_tags]
		neg_it_emb = all_tags_item[neg_tags]

		pos_scores = torch.sum(torch.mul(users_emb, pos_ut_emb) + torch.mul(items_emb, pos_it_emb), dim=1)
		neg_scores = torch.sum(torch.mul(users_emb, neg_ut_emb) + torch.mul(items_emb, neg_it_emb), dim=1)

		# ut对比
		noise_users_1, noise_tu_emb_1 = self.noise_computer(self.Graph_ut, self.embedding_user, self.embedding_ut,
		                                                    self.num_users, self.noise_type[0])
		noise_users_2, noise_tu_emb_2 = self.noise_computer(self.Graph_ut, self.embedding_user, self.embedding_ut,
		                                                    self.num_users, self.noise_type[1])

		noise_users_emb_1 = nn.functional.normalize(noise_users_1, dim=1)
		noise_users_emb_2 = nn.functional.normalize(noise_users_2, dim=1)
		noise_tus_emb_1 = nn.functional.normalize(noise_tu_emb_1, dim=1)
		noise_tus_emb_2 = nn.functional.normalize(noise_tu_emb_2, dim=1)

		noise_user_emb_1 = noise_users_emb_1[users]
		noise_user_emb_2 = noise_users_emb_2[users]
		noise_tu_emb_1 = noise_tus_emb_1[pos_tags]
		noise_tu_emb_2 = noise_tus_emb_2[pos_tags]

		# ssl_user_loss
		pos_user = torch.exp(torch.sum(torch.mul(noise_user_emb_1, noise_user_emb_2), dim=1) / self.temperature)
		ttl_user = torch.matmul(noise_user_emb_1, noise_users_emb_2.T)
		ttl_user = torch.sum(torch.exp(ttl_user / self.temperature), dim=1)
		ssl_loss_user = -torch.sum(torch.log(pos_user / ttl_user))

		# ssl_tag_loss
		pos_tag_1 = torch.exp(torch.sum(torch.mul(noise_tu_emb_1, noise_tu_emb_2), dim=1) / self.temperature)
		ttl_tag_1 = torch.matmul(noise_tu_emb_1, noise_tus_emb_2.T)
		ttl_tag_1 = torch.sum(torch.exp(ttl_tag_1 / self.temperature), dim=1)
		ssl_loss_tag_1 = -torch.sum(torch.log(pos_tag_1 / ttl_tag_1))

		# it对比
		noise_items_1, noise_tis_emb_1 = self.noise_computer(self.Graph_it, self.embedding_item, self.embedding_it,
		                                                    self.num_items, self.noise_type[0])
		noise_items_2, noise_tis_emb_2 = self.noise_computer(self.Graph_it, self.embedding_item, self.embedding_it,
		                                                    self.num_items, self.noise_type[1])

		noise_items_emb_1 = nn.functional.normalize(noise_items_1, dim=1)
		noise_items_emb_2 = nn.functional.normalize(noise_items_2, dim=1)
		noise_tis_emb_1 = nn.functional.normalize(noise_tis_emb_1, dim=1)
		noise_tis_emb_2 = nn.functional.normalize(noise_tis_emb_2, dim=1)

		noise_item_emb_1 = noise_items_emb_1[items]
		noise_item_emb_2 = noise_items_emb_2[items]
		noise_ti_emb_1 = noise_tis_emb_1[pos_tags]
		noise_ti_emb_2 = noise_tis_emb_2[pos_tags]

		# ssl_item_loss
		pos_item = torch.exp(torch.sum(torch.mul(noise_item_emb_1, noise_item_emb_2), dim=1) / self.temperature)
		ttl_item = torch.matmul(noise_item_emb_1, noise_items_emb_2.T)
		ttl_item = torch.sum(torch.exp(ttl_item / self.temperature), dim=1)
		ssl_loss_item = -torch.sum(torch.log(pos_item / ttl_item))

		# ssl_tag_loss
		pos_tag_2 = torch.exp(torch.sum(torch.mul(noise_ti_emb_1, noise_ti_emb_2), dim=1) / self.temperature)
		ttl_tag_2 = torch.matmul(noise_ti_emb_1, noise_tis_emb_2.T)
		ttl_tag_2 = torch.sum(torch.exp(ttl_tag_2 / self.temperature), dim=1)
		ssl_loss_tag_2 = -torch.sum(torch.log(pos_tag_2 / ttl_tag_2))

		return pos_scores - neg_scores, ssl_loss_tag_1 + ssl_loss_tag_2 + ssl_loss_user + ssl_loss_item 

	def computer(self, Graph, emb1, emb2, num):
		emb_1 = emb1.weight
		tags_emb = emb2.weight

		all_emb = torch.cat([emb_1, tags_emb])
		embs = [all_emb]
		for layer in range(self.layers):
			all_emb = torch.sparse.mm(Graph, all_emb)
			embs.append(all_emb)
		embs = torch.stack(embs, dim=1)
		light_out = torch.mean(embs, dim=1)
		emb, tags = torch.split(light_out, [num, self.num_tags])
		return emb, tags

	def noise_computer(self, Graph, emb1, emb2, num, noise_type):
		emb_1 = emb1.weight
		tags_emb = emb2.weight

		all_emb = torch.cat([emb_1, tags_emb])
		embs = []
		for layer in range(self.layers):
			emb = torch.sparse.mm(Graph, all_emb)
			if noise_type == 'uniform_noise':
				random_noise = torch.rand(emb.shape).cuda()
			if noise_type == 'Gaussian_noise':
				random_noise = torch.randn(size=emb.shape).cuda()
			emb += torch.mul(torch.sign(emb), nn.functional.normalize(random_noise)) * self.eps
			embs.append(emb)
		embs = torch.stack(embs, dim=1)
		light_out = torch.mean(embs, dim=1)
		entity_emb, tag_emb = torch.split(light_out, [num, self.num_tags])
		return entity_emb, tag_emb

	def predict(self, key):
		user_id = torch.from_numpy(np.array(key[0])).long().to('cuda')
		item_id = torch.from_numpy(np.array(key[1])).long().to('cuda')
		user_emb = self.embedding_user(user_id)
		item_emb = self.embedding_item(item_id)
		pred = torch.matmul(user_emb, self.embedding_ut.weight.T) + torch.matmul(item_emb, self.embedding_it.weight.T)
		return pred


	def get_uit_dict(self, data):
		res = {}
		for i in data:
			if (i[0], i[1]) not in res.keys():
				res[(i[0], i[1])] = set()
			res[(i[0], i[1])].add(i[2])
		return res

	def get_train_batch(self):
		len_data = len(self.train_data)
		if self.present_train_size + self.batch_size > len_data - 1:
			res = self.train_data[self.present_train_size:len_data] + \
			      self.train_data[0:self.present_train_size + self.batch_size - len_data]
		else:
			res = self.train_data[self.present_train_size:self.present_train_size + self.batch_size]
		self.present_train_size += self.batch_size
		self.present_train_size %= len_data
		return res

	def get_feed_dict(self, datas):
		user_list = []
		item_list = []
		pos_tag_list = []
		neg_tag_list = []
		for data in datas:
			nt = random.sample(range(self.num_tags), 1)[0]
			while nt in self.train_uit_dict[(data[0], data[1])]:
				nt = random.sample(range(self.num_tags), 1)[0]
			user_list.append(data[0])
			item_list.append(data[1])
			pos_tag_list.append(data[2])
			neg_tag_list.append(nt)
		return user_list, item_list, pos_tag_list, neg_tag_list


def load_uit(path):
	num_users = -1  # u
	num_items = -1  # i
	num_tags = -1  # t
	data = []
	with open(path) as f:
		for line in f:
			line = [int(i) for i in line.split('\t')[:3]]  ##line [1464, 5461, 253]
			data.append(line)
			num_users = max(line[0], num_users)
			num_items = max(line[1], num_items)
			num_tags = max(line[2], num_tags)
	num_users, num_items, num_tags = num_users + 1, num_items + 1, num_tags + 1  # 从零开始计数
	return data, num_users, num_items, num_tags


def load_data(path):
	print('Loading train and test data...')
	sys.stdout.flush()  # 刷新缓冲区
	train_data, num_users, num_items, num_tags = load_uit(path + '.train')  # 训练集
	test_data, num_users2, num_items2, num_tags2 = load_uit(path + '.test')
	num_users = max(num_users, num_users2)  # 建立总集
	num_items = max(num_items, num_items2)
	num_tags = max(num_tags, num_tags2)
	print('Number of users: %d' % num_users)
	print('Number of items: %d' % num_items)
	print('Number of tags: %d' % num_tags)
	print('Number of train data: %d' % len(train_data))
	print('Number of test data: %d' % len(test_data))

	logging.info('Number of users: %d' % num_users)  # logging.info()打印一切信息，确定一切正常运行
	logging.info('Number of items: %d' % num_items)
	logging.info('Number of tags: %d' % num_tags)
	logging.info('Number of train data: %d' % len(train_data))
	logging.info('Number of test data: %d' % len(test_data))
	sys.stdout.flush()
	return train_data, test_data, num_users, num_items, num_tags,


def train(train_data, test_data, num_users, num_items, num_tags):
	device = torch.device('cuda')
	batch_total = int(len(train_data) / args.batch_size)
	simgcl = SimGCL(train_data, test_data, num_users, num_items, num_tags, args.embedding_size, args.temperature,
	                args.layer, args.batch_size, args.eps, args.noise_type).to(device)

	optimizer = optim.Adam(simgcl.parameters(), lr=args.lr, weight_decay=args.reg_rate)

	history_p_at_3 = []
	history_p_at_5 = []
	history_p_at_10 = []
	history_p_at_20 = []
	history_r_at_3 = []
	history_r_at_5 = []
	history_r_at_10 = []
	history_r_at_20 = []
	history_ndcg_at_3 = []
	history_ndcg_at_5 = []
	history_ndcg_at_10 = []
	history_ndcg_at_20 = []

	for epoch in range(args.epochs):
		simgcl.train()
		simgcl_loss = 0
		for k in range(1, batch_total + 1):
			user_list, item_list, pos_tag_list, neg_tag_list = simgcl.get_feed_dict(simgcl.get_train_batch())
			user_id = torch.from_numpy(np.array(user_list)).long().to(device)
			item_id = torch.from_numpy(np.array(item_list)).long().to(device)
			pos_tag_id = torch.from_numpy(np.array(pos_tag_list)).long().to(device)
			neg_tag_id = torch.from_numpy(np.array(neg_tag_list)).long().to(device)
			pred, ssl_loss = simgcl(user_id, item_id, pos_tag_id, neg_tag_id)

			batch_loss = -torch.log(torch.sigmoid(pred)).sum() + args.ssl_reg * ssl_loss
			simgcl_loss += batch_loss
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()

		if (epoch + 1) % args.verbose == 0:
			simgcl.eval()
			res = []
			with torch.no_grad():
				for key in simgcl.test_uit_dict.keys():
					pred = simgcl.predict(key)
					pred = pred.cpu().detach().numpy().tolist()
					rank_list = reckit.arg_top_k(pred, 20)
					test_list = simgcl.test_uit_dict[key]
					p_3, r_3, ndcg_3 = precision_recall_ndcg_at_k(3, rank_list[:3], test_list)
					p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, rank_list[:5], test_list)
					p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, rank_list[:10], test_list)
					p_20, r_20, ndcg_20 = precision_recall_ndcg_at_k(20, rank_list[:20], test_list)
					res.append([p_3, p_5, p_10, p_20, r_3, r_5, r_10, r_20, ndcg_3, ndcg_5, ndcg_10, ndcg_20])

				res = np.array(res)
				res = np.mean(res, axis=0)

				history_p_at_3.append(res[0])
				history_p_at_5.append(res[1])
				history_p_at_10.append(res[2])
				history_p_at_20.append(res[3])
				history_r_at_3.append(res[4])
				history_r_at_5.append(res[5])
				history_r_at_10.append(res[6])
				history_r_at_20.append(res[7])
				history_ndcg_at_3.append(res[8])
				history_ndcg_at_5.append(res[9])
				history_ndcg_at_10.append(res[10])
				history_ndcg_at_20.append(res[11])

				print(
					" %04d Loss: %.2f \t pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
					(
						epoch + 1, simgcl_loss, res[0], res[4], res[8], res[1], res[5], res[9], res[2], res[6], res[10],
						res[3],
						res[7], res[11]))
				logging.info(
					" %04d Loss: %.2f \t pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
					(
						epoch + 1, simgcl_loss, res[0], res[4], res[8], res[1], res[5], res[9], res[2], res[6], res[10],
						res[3],
						res[7], res[11]))



	best_pre5_index = np.argmax(history_p_at_5)
	best_pre3 = history_p_at_3[best_pre5_index]
	best_pre5 = history_p_at_5[best_pre5_index]
	best_pre10 = history_p_at_10[best_pre5_index]
	best_pre20 = history_p_at_20[best_pre5_index]
	best_rec3 = history_r_at_3[best_pre5_index]
	best_rec5 = history_r_at_5[best_pre5_index]
	best_rec10 = history_r_at_10[best_pre5_index]
	best_rec20 = history_r_at_20[best_pre5_index]
	best_ndcg3 = history_ndcg_at_3[best_pre5_index]
	best_ndcg5 = history_ndcg_at_5[best_pre5_index]
	best_ndcg10 = history_ndcg_at_10[best_pre5_index]
	best_ndcg20 = history_ndcg_at_20[best_pre5_index]
	print(
		"Best Epochs: pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
		(best_pre3, best_rec3, best_ndcg3, best_pre5, best_rec5, best_ndcg5, best_pre10, best_rec10, best_ndcg10,
		 best_pre20, best_rec20, best_ndcg20))
	logging.info(
		"Best Epochs: pre3: %.5f  rec3: %.5f  ndcg3: %.5f  pre5: %.5f  rec5: %.5f ndcg5: %.5f  pre10:  %.5f  rec10:  %.5f ndcg10:  %.5f  pre20:  %.5f  rec20:  %.5f ndcg20:  %.5f" % \
		(best_pre3, best_rec3, best_ndcg3, best_pre5, best_rec5, best_ndcg5, best_pre10, best_rec10, best_ndcg10,
		 best_pre20, best_rec20, best_ndcg20))
	out_max(best_pre3, best_rec3, best_ndcg3, best_pre5, best_rec5, best_ndcg5, best_pre10, best_rec10, best_ndcg10,
	        best_pre20, best_rec20, best_ndcg20)


def out_max(pre3, rec3, ndcg3, pre5, rec5, ndcg5, pre10, rec10, ndcg10, pre20, rec20, ndcg20):
	code_name = os.path.basename(__file__).split('.')[0]
	log_path_ = "log/%s/" % code_name
	if not os.path.exists(log_path_):
		os.makedirs(log_path_)
	csv_path = log_path_ + "%s.csv" % args.dataset
	log = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
		args.embedding_size, args.lr, args.reg_rate, args.ssl_reg, args.layer, args.temperature, args.eps,
		pre3, rec3, ndcg3,
		pre5, rec5, ndcg5,
		pre10, rec10, ndcg10,
		pre20, rec20, ndcg20)
	if not os.path.exists(csv_path):
		with open(csv_path, 'w') as f:
			f.write(
				"embedding_size,learning_rate,reg_rate,ssl_reg,layer,temperature,eps,pre3,recall3,ndcg3,pre5,recall5,ndcg5,pre10,recall10,ndcg10,pre20,recall20,ndcg20" + '\n')
			f.write(log + '\n')
			f.close()
	else:
		with open(csv_path, 'a+') as f:
			f.write(log + '\n')
			f.close()


def setup_seed(seed):
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed_all(seed)


def parse_args():
	parser = argparse.ArgumentParser(description="Go SimGCL")
	parser.add_argument('--dataset_path', nargs='?', default='./dataset/ml-10m/',
	                    help='Data path.')
	parser.add_argument('--dataset', nargs='?', default='ml-10m_5',
	                    help='Name of the dataset.')
	parser.add_argument('--batch_size', type=int, default=1024,
	                    help="the batch size for bpr loss training procedure")
	parser.add_argument('--embedding_size', type=int, default=64,
	                    help="the embedding size of lightGCN")
	parser.add_argument('--layer', type=int, default=1,
	                    help="the layer num of lightGCN")
	parser.add_argument('--lr', type=float, default=0.001,
	                    help="the learning rate")
	parser.add_argument('--temperature', type=float, default=0.03,
	                    help="the hyper-parameter")
	parser.add_argument('--reg_rate', type=float, default=0.001,
	                    help='Regularization coefficient for user and item embeddings.')
	parser.add_argument('--ssl_reg', type=float, default=0.001,
	                    help='Regularization coefficient for user and item embeddings.')
	parser.add_argument('--eps', type=float, default=0.1, help="the hyperspherical radius")
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--seed', type=int, default=100, help='random seed')
	parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
	parser.add_argument('--noise_type', type=list, default=['uniform_noise', 'uniform_noise'],
	                    help='type of noise. Like [uniform_noise, uniform_noise],[uniform_noise, Gaussian_noise],[Gaussian_noise, Gaussian_noise]')
	return parser.parse_args()


if __name__ == '__main__':
	code_name = os.path.basename(__file__).split('.')[0]
	args = parse_args()
	print(args)
	setup_seed(args.seed)
	log_path = "log/%s_%s/" % (code_name, strftime('%Y-%m-%d', localtime()))
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	log_path = log_path + "%s_embed_size%.4f_reg%.5f_ssl_reg%.5f_lr%0.5f_eps%.3f_temperature%.3f_layer%.4f_noise_type%s_%s_%s" % (
		args.dataset, args.embedding_size, args.reg_rate, args.ssl_reg, args.lr, args.eps, args.temperature, args.layer,
		args.noise_type[0], args.noise_type[1], strftime('%Y_%m_%d_%H', localtime()))
	logging.basicConfig(filename=log_path, level=logging.INFO)
	logging.info(args)

	data_train, data_test, num_users, num_items, num_tags = load_data(args.dataset_path + args.dataset)
	train(data_train, data_test, num_users, num_items, num_tags)
