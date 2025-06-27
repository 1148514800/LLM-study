# Rag-test

## 运行前准备：

1、安装依赖包：pip install -r requirements.txt；

2、运行ragas_demo.py，正确输出结果则依赖包安装成功

## 运行说明
1、按照“input.csv”准备输入文件，对应字段内容需自行从mysql或调接口获取；

字段  | 说明 | 来源
------------- | ------------- | -------------
question  | 问题 | 开源数据集=智能体输入
answer  | 实际回答 | 智能体输出
contexts  | 检索到的上下文 | 智能体输出
ground_truths  | 参考答案 | 开源数据集

2、运行ragas_csv.py，获取对应指标，其中chat模型使用豆包，embedding模型使用Maas，可替换为其他模型；
