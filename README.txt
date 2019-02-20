签单完成率预估模型，目前仅支持二分类问题，类别以-1和1进行表示
1：正样例，即已签单
-1：负样例，即未签单

文件列表
	日志模块：dj_log.py
	签单完成率预估模型主函数：finish_rate_predict.py
	模型核心模块：gbdt_lr_model.py
	预处理及读写模块：preprocessing_module.py
	日志配置文件：log_config.ini
	简介：README.txt

功能定义：目前支持三种模式：offline、online及predict模式
	1. offline 模式
		启动方式：python finish_rate_predict.py offline
		程序将以离线数据集进行模型训练及模型选择
	2. online 模式
		程序将从线上Hive库读取数据进行模型训练及模型选择
		PS：线上模式请确保能正常访问线上Hive库
	3. predict 模式
		启动方式：python finish_rate_predict.py predict
		程序基于接口测试数据集对模型进行功能测试
