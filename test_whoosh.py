from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.qparser import QueryParser

# 定义模式
schema = Schema(title=TEXT(stored=True), content=TEXT, doc_url=ID)

# 创建索引目录
import os

if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

# 创建索引
index = create_in("indexdir", schema)

writer = index.writer()
# 假设我们有一些文档添加到索引
writer.add_document(title=u"1 1 4 5 5 5 5", content=u"244 575 577 767", doc_url=u"/a")
writer.add_document(title=u"1 1 4 5 5", content=u"244 575", doc_url=u"/a")
writer.add_document(title=u"1 4 4", content=u"244", doc_url=u"/a")
writer.add_document(title=u"2 2 2 7", content=u"11 12 13", doc_url=u"/b")

# 提交写入的内容
writer.commit()

searcher = index.searcher()

# 创建一个查询解析器
query_parser = QueryParser("content", index.schema)

# 解析查询字符串
query = query_parser.parse("13")

# 执行搜索
results = searcher.search(query)

# 遍历结果
for hit in results:
    print(hit['title'])  # 输出查找到的文档的标题

# 关闭搜索器
searcher.close()