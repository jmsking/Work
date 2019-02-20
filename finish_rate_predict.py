#! /usr/bin/python
# encoding:utf-8

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from preprocessing_module import PreprocModule
from gbdt_lr_model import GbdtLrModel
import sys
import platform
import pickle
import os
from dj_log import log

class FinishRatePredModel():
	'''
	保姆线索签单完成率预估模型
	@author MaySnow
	'''
	def __init__(self
		,save_standard_d_path = '/opt/test/chenjie/stand_d.bin'
		,save_standard_c_path = '/opt/test/chenjie/stand_c.bin'
		,save_gbdt_enc_path = '/opt/test/chenjie/gbdt_enc.bin'
		,save_gbdt_model_path = '/opt/test/chenjie/gbdt_model.bin'
		,save_lr_model_path='/opt/test/chenjie/lr_model.bin'
		,save_threshold_path='/opt/test/chenjie/threshold.bin'
		,K=10):
		'''
		初始化方法
		Args:
			save_standard_d_path: 离散特征指标标准化器
			save_standard_c_path: 连续特征指标标准化器
			save_gbdt_enc_path: GBDT OneHotEncoder存放路径
			save_gbdt_model_path: GBDT模型文件存放路径
			save_lr_model_path: LR模型文件存放路径
			save_threshold_path: 阈值存放路径
			K: K折交叉验证
		'''
		curr_sys = platform.system()
		if curr_sys == 'Windows':
			self._save_standard_d_path = 'F:/stand_d.bin'
			self._save_standard_c_path = 'F:/stand_c.bin'
			self._save_gbdt_enc_path = 'F:/gbdt_enc.bin'
			self._save_gbdt_model_path = 'F:/gbdt_model.bin'
			self._save_lr_model_path = 'F:/lr_model.bin'
			self._save_threshold_path = 'F:/threshold.bin'
			log.info('Current OS is Windows.')
		elif curr_sys == 'Linux':
			self._save_standard_d_path = save_standard_d_path
			self._save_standard_c_path = save_standard_c_path
			self._save_gbdt_enc_path = save_gbdt_enc_path
			self._save_gbdt_model_path = save_gbdt_model_path
			self._save_lr_model_path = save_lr_model_path
			self._save_threshold_path = save_threshold_path
			log.info('Current OS is Linux.')
		else:
			log.warning('Not suppose other OS except Windows and Linux.')
		self._K = K
		
	def _reTraining(self, hql = None, offline = False):
		'''
		重新训练模型
		Args:
			hql: 查询数据HQL
			offline: 是否以离线数据进行训练,默认为 False
		Returns:
			model: GbdtLrModel封装模型
			enc_d: 离散属性特征标准化方法
			enc_c: 连续属性特征标准化方法
			gbdt_enc: GBDT OneHotEncoder
			gbdt_model: GBDT模型
			lr_model: LR模型
			threshold: 阈值
		'''
		pre_module = PreprocModule(self._save_standard_d_path, self._save_standard_c_path)
		if offline:
			path = 'clue-adviser-index-10w.xlsx'
			sheet_name = '查询结果'
			X, y = pre_module.readxlsx(path, sheet_name.decode('utf-8'))
		else:
			if hql is None and len(sys.argv) < 3:
				log.error('Please give completed parameters.\n'
				'Example: python finish_rate_predict.py online "select * from <database>.<table_name> [where-clause]"')
				return
			#从命令行中获取
			_hql = ''
			if hql is None:
				_hql = sys.argv[2]
			else:
				_hql = hql
			log.info("Obtain online data from HQL -> %s", _hql)
			X, y = pre_module.readHive(_hql)
		p_rate = pre_module.posRateStatis(y) #得到正样例的比率
		X, y = np.array(X), np.array(y)
		while p_rate <= 0.2:
			# 需要进行样本不平衡处理
			log.info("Start imbalance process.")
			X, y = pre_module.imbalanceProcess(X, y)
			p_rate = pre_module.posRateStatis(y.tolist())
			log.info("p_rate is: %f", p_rate)
		model = GbdtLrModel(random_state = 55)
		max_auc = 0
		for nEstimator in range(80,101,10):
			log.debug("***Current n_estimators is: %d", nEstimator)
			_threshold = []
			_auc = []
			_precision = []
			_recall = []
			model.setNEstimators(nEstimator)
			for iter, [X_train_d, X_train_c, X_test_d, X_test_c, y_train, y_test, enc_d, enc_c] in enumerate(self._featuresSplit(pre_module, X, y)):
				log.debug("--------Times: [%d]----------", iter+1)
				gbdt_enc, gbdt_model, lr_model, threshold, auc, precision, recall \
					= model.buildModel(X_train_d, X_train_c, X_test_d, X_test_c, y_train, y_test)
				_threshold.append(threshold)
				_auc.append(auc)
				_precision.append(precision)
				_recall.append(recall)
			avg_threshold = sum(_threshold) / len(_threshold)
			avg_precision = sum(_precision) / len(_precision)
			avg_recall = sum(_recall) / len(_recall)
			avg_auc = sum(_auc) / len(_auc)
			if avg_auc > max_auc:
				opt_gbdt_enc = gbdt_enc
				opt_gbdt_model = gbdt_model
				opt_lr_model = lr_model
				opt_threshold = avg_threshold
				max_auc = avg_auc
			log.info("nEstimators[%d] -> avg_threshold: %f - avg_precision: %f - avg_recall: %f - avg_auc: %f", 
				nEstimator, avg_threshold, avg_precision, avg_recall, avg_auc)
		self._saveModel(opt_gbdt_enc, opt_gbdt_model, opt_lr_model, avg_threshold)
		return model, enc_d, enc_c, gbdt_enc, gbdt_model, lr_model, avg_threshold
	
	def _saveModel(self, gbdt_enc, gbdt_model, lr_model, threshold):
		'''
		保存模型
		Args:
			gbdt_enc: GBDT OneHotEncoder
			gbdt_model: GBDT模型
			lr_model: LR模型
			threshold: 阈值
		'''
		if self._save_gbdt_enc_path is not None:
			pickle.dump(gbdt_enc, open(self._save_gbdt_enc_path, 'wb'))
			log.info("Save GBDT encoder success. The path is %s", self._save_gbdt_enc_path)
		if self._save_gbdt_model_path is not None:
			pickle.dump(gbdt_model, open(self._save_gbdt_model_path,'wb'))
			log.info("Save GBDT model success. The path is %s", self._save_gbdt_model_path)
		if self._save_lr_model_path is not None:
			pickle.dump(lr_model, open(self._save_lr_model_path,'wb'))
			log.info("Save LR model success. The path is %s", self._save_lr_model_path)
		if self._save_threshold_path is not None:
			pickle.dump(threshold, open(self._save_threshold_path,'wb'))
			log.info("Save threshold success. The path is %s", self._save_threshold_path)
	
	def _featuresSplit(self, pre_module, X, y=None, enc_d=None, enc_c=None, phase='train'):
		'''
		特征指标分离,划分为连续指标及离散指标
		Args:
			pre_module: 预处理模块
			X: 数据特征集
			y: 数据特征标记
			enc_d: 离散属性特征标准化方法
			enc_c: 连续属性特征标准化方法
			phase: 划分属性的阶段, train 或者 predict
		Returns:
			训练阶段->
			X_train_d: 训练集分离后的离散指标
			X_train_c: 训练集分离后的连续指标
			X_test_d: 测试集分离后的离散指标
			X_test_c: 测试集分离后的连续指标
			y_train: 训练集标记
			y_test: 测试集标记
			enc_d: 离散指标标准化器
			enc_c: 连续指标标准化器
			预测阶段->
			X_d: 离散指标
			X_c: 连续指标
		'''
		X, d_cols, enc_d, enc_c = pre_module.standardization(X, enc_d, enc_c)
		#训练阶段
		if phase == 'train':
			if y is None:
				log.error("Labels can not be empty.")
				return
			#--CV--
			k_cv = StratifiedKFold(n_splits=self._K)
			for train, test in k_cv.split(X, y):
				X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
				X_train_d, X_train_c = np.array([item[:d_cols] for item in X_train]), np.array([item[d_cols:] for item in X_train]) #训练数据集离散指标与连续数值指标分离
				X_test_d, X_test_c = np.array([item[:d_cols] for item in X_test]), np.array([item[d_cols:] for item in X_test]) #测试数据集离散指标与连续数值指
				yield X_train_d, X_train_c, X_test_d, X_test_c, y_train, y_test, enc_d, enc_c
		#预测阶段
		elif phase == 'predict':
			X_d, X_c = np.array([item[:d_cols] for item in X]), np.array([item[d_cols:] for item in X])
			yield X_d, X_c
	
	def isPositive(self, X=None):
		'''
		判断输入数据是否为正样例
		Args:
			X: 待预测的数据特征样本 [rows, cols]
		Returns:
			is_positive: 1 -> 为正样例,0 -> 为负样例
		'''
		if X is None:
			log.error("Input samples can not be empty.")
			return
		
		is_exist_stand_d = os.path.exists(self._save_standard_d_path)
		is_exist_stand_c = os.path.exists(self._save_standard_c_path)
		is_exist_gbdt_enc = os.path.exists(self._save_gbdt_enc_path)
		is_exist_gbdt_model = os.path.exists(self._save_gbdt_model_path)
		is_exist_lr_model = os.path.exists(self._save_lr_model_path)
		is_exist_threshold = os.path.exists(self._save_threshold_path)
		
		model = GbdtLrModel()
		enc_d = None
		enc_c = None
		gbdt_enc = None
		gbdt_model = None
		lr_model = None
		threshold = None
		
		#未找到模型,重新开始训练模型
		if not is_exist_stand_d or not is_exist_stand_c or \
			not is_exist_gbdt_enc or \
			not is_exist_gbdt_model or \
			not is_exist_lr_model or \
			not is_exist_threshold:
			log.warning("Can not find the path of model file or threshold file.\n"
			"Program will start training model.")
			_hql = "select * from jz_mart_cs.s_union_index_bm where is_sign_order <> 0"
			model, enc_d, enc_c, gbdt_enc, gbdt_model, lr_model, threshold = self._reTraining(_hql)
		else:
			#加载模型
			enc_d = pickle.load(open(self._save_standard_d_path, 'rb'))
			enc_c = pickle.load(open(self._save_standard_c_path, 'rb'))
			gbdt_enc = pickle.load(open(self._save_gbdt_enc_path, 'rb'))
			gbdt_model = pickle.load(open(self._save_gbdt_model_path, 'rb'))
			lr_model = pickle.load(open(self._save_lr_model_path, 'rb'))
			threshold = pickle.load(open(self._save_threshold_path, 'rb'))
		pre_module = PreprocModule()
		for X_data_d, X_data_c in self._featuresSplit(pre_module, X, enc_d=enc_d, enc_c=enc_c, phase = 'predict'):
			X_ext = model.combineFeatures(gbdt_model, gbdt_enc, X_data_c, X_data_d)
			prediction = lr_model.predict_proba(X_ext)[:,1]
			is_positive = (prediction >= threshold)*2-1
		return is_positive
		
	
if __name__ == '__main__':
	if len(sys.argv) < 2:
		log.error("Arguments need greater than 1.\n"
		"Example: python finish_rate_predict.py <offline|online|predict>")
	model = FinishRatePredModel()
	if sys.argv[1] == 'offline':
		#本机训练
		log.info("Start offline training, please wait...")
		model._reTraining(offline=True)
	elif sys.argv[1] == 'online':
		log.info("Start online training, please wait...")
		#服务器上训练
		model._reTraining()
	elif sys.argv[1] == 'predict':
		#离线预测
		pre_module = PreprocModule()
		path = 'clue-adviser-index-predict.xlsx'
		sheet_name = '查询结果'
		X, y = pre_module.readxlsx(path, sheet_name.decode('utf-8'))
		is_positive = model.isPositive(X)
		meet = (is_positive == y)*1
		accuracy = float(sum(meet)) / len(meet)
		log.info("Accuracy is: %f", accuracy)
		recall_rate = recall_score(y, is_positive)
		log.info("Recall rate is: %f", recall_rate)
	else:
		log.error("Arguments error.\n"
		"Example: python finish_rate_predict.py <offline|online|predict>")