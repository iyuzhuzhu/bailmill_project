from pymongo import MongoClient

# 数据库连接信息
connection_url = "mongodb://localhost:27017/"
db_name = "bm"
collection_name = "bm1_ai"

# 创建MongoClient对象
client = MongoClient(connection_url)

# 选择数据库和集合
db = client[db_name]
collection = db[collection_name]

# 获取文档数量
doc_count = collection.count_documents({})

print(f"Collection {collection_name} 中的文档数量为: {doc_count}")