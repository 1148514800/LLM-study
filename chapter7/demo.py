from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat, TestLMChat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding, gteEmbedding

# 没有保存数据库
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获取data目录下的所有文件内容并分割
print("读取到的文档内容数量:", len(docs))

vector = VectorStore(docs)
embedding = gteEmbedding() # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # 将向量和文档内容保存到storage目录，下次再用可以直接加载本地数据库

question = 'git的原理是什么？'

content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
chat = TestLMChat(path='/home/hpclp/disk/q/models/Qwen2.5-1.5B-Instruct')
print(chat.chat(question, [], content))


