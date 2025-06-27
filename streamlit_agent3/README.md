# 智能客服系統

## 运行环境
建议使用 Python>=3.10

可参考如下命令进行环境创建
```commandline
conda create -n agent python=3.10 -y
conda activate agent
```

## 安装依赖
```commandline
pip install -r requirements.txt
```

## 配置OpenAI 环境变量

### Windows 导入环境变量

注意：每次执行完，需要重启PyCharm才能生效

```powershell
setx OPENAI_BASE_URL "https://api.openai.com/v1"
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


## 运行项目

使用以下命令行运行webui
```bash
streamlit run rag.py
```

## 搜索提示词
如何查询账户余额？
