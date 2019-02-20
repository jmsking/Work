#! /usr/bin/python
# encoding:utf-8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.ranking import roc_auc_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.data import OneHotEncoder
from scipy.sparse.construct import hstack
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import pickle
import os
import sys
from dj_log import log

class GbdtLrModel():
	'''
	GBDT+LR模型
	'''
	def __init__(self, n_estimators=80, gbdt_learning_rate=1, max_depth=1, random_state=None
		,recall_rate=0.95):
		'''
		初始化方法
		Args:
			n_estimators: GBDT中估计器的个数
			gbdt_learning_rate: GBDT学习率
			max_depth: GBDT树的最大高度
			random_state: 如何设置该值,可以保证相同数据相同参数时训练的结果一致
			recall_rate: 模型召回率(查全率)
		'''
		self._n_estimators = n_estimators
		self._gbdt_learning_rate = gbdt_learning_rate
		self._max_depth = max_depth
		self._random_state = random_state
		self._recall_rate = recall_rate
	
	def setNEstimators(self, n_estimators):
		'''
		设置GBDT中估计器的个数(用于CV中超参数的设置)
		'''
		self._n_estimators = n_estimators
		
	def buildModel(self, X_train_d, X_train_c, X_test_d, X_test_c, y_train, y_test):
		'''
		开始构建模型
		Args:
			X_train_d: 离散特征训练数据
			X_train_c: 连续特征训练数据
			X_test_d: 离散特征测试数据
			X_test_c: 连续特征测试数据
			y_train: 训练数据标记 {-1, 1}
			y_test: 测试数据标记 {-1, 1}
		Returns:
			gbc_enc: GBDT OneHotEncoder
			gbc: GBDT模型
			comb_model: 训练得到的组合模型
			threshold: 正负样例阈值, Pred_Prob >= threshold 为正样例; Pred_Prob < threshold 为负样例
			comb_model_auc: 模型AUC
			precision: 模型精度
			recall: 模型召回率
		'''
		if self._random_state is not None:
			gbc = GradientBoostingClassifier(n_estimators=self._n_estimators, learning_rate=self._gbdt_learning_rate, max_depth=self._max_depth, random_state=self._random_state).fit(X_train_c, y_train)
		else:
			gbc = GradientBoostingClassifier(n_estimators=self._n_estimators, learning_rate=self._gbdt_learning_rate, max_depth=self._max_depth).fit(X_train_c, y_train)
		X_train_leaves = gbc.apply(X_train_c)[:,:,0]
		X_test_leaves = gbc.apply(X_test_c)[:,:,0]
		(X_train_rows, cols) = X_train_leaves.shape
		gbc_enc = OneHotEncoder().fit(np.concatenate([X_train_leaves,X_test_leaves], axis = 0))
		X_trans = gbc_enc.transform(np.concatenate([X_train_leaves,X_test_leaves], axis = 0))
		X_train_ext = hstack([X_trans[:X_train_rows,:], X_train_d])
		X_test_ext = hstack([X_trans[X_train_rows:,:], X_test_d])
		log.debug("Combine features done.")
		comb_model = LogisticRegression().fit(X_train_ext, y_train)
		log.debug("Training done.")
		comb_model_pred = comb_model.predict_proba(X_test_ext)[:,1]
		precision, recall, thresholds = precision_recall_curve(y_test, comb_model_pred)
		ap = average_precision_score(y_test, comb_model_pred)
		recall_meet = recall >= self._recall_rate
		recall_meet_min = len([item for item in recall_meet if item == True])
		threshold = thresholds[recall_meet_min-1]
		log.debug("threshold: %f - precision: %f - recall: %f", threshold, precision[recall_meet_min-1], recall[recall_meet_min-1])
		comb_model_auc = roc_auc_score(y_test, comb_model_pred)
		log.debug("AUC score is: %f", comb_model_auc)
		return gbc_enc, gbc, comb_model, threshold, comb_model_auc, precision[recall_meet_min-1], recall[recall_meet_min-1]
	
	def combineFeatures(self, gbdt_model, gbdt_enc, X_data_c=None, X_data_d=None):
		'''
		进行特征的组合
		Args:
			gbdt_model: GBDT模型
			gbdt_enc: GBDT叶子节点OneHotEncoder
			X_data_c: 待组合连续特征
			X_data_d: 待组合离散特征
		Returns:
			X_ext: 组合后的特征
		'''
		if X_data_c is None and X_data_d is None:
			log.error("Feature can not be None.")
			return
		X_ext = None
		if X_data_c is not None:
			X_leaves = gbdt_model.apply(X_data_c)[:,:,0]
			X_ext = gbdt_enc.transform(X_leaves)
		if X_data_d is not None:
			if X_ext is not None:
				X_ext = hstack([X_ext, X_data_d])
		return X_ext