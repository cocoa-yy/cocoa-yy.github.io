---
date: 2025-08-15
title: LangChain实践
description: 
mermaid: true
mathjax: true
category:
  - 智能体
tags:
  - langchain
  - 智能体
  - 大模型
  - 项目实践
ogImage: https://astro-yi.obs.cn-east-3.myhuaweicloud.com/avatar.png
---
# 教程资料
- [Chatbots](https://python.langchain.com/docs/tutorials/chatbot/): 构建一个集成记忆功能的聊天机器人
- [Agents](https://python.langchain.com/docs/tutorials/agents/): 构建一个与外部工具交互的智能体
- [Retrieval Augmented Generation (RAG) Part 1](https://python.langchain.com/docs/tutorials/rag/): 构建一个使用文档提供响应信息的应用程序（本博客实践内容）
- [Retrieval Augmented Generation (RAG) Part 2](https://python.langchain.com/docs/tutorials/qa_chat_history/): 构建一个包含对用户交互记忆和多步骤检索 RAG 应用程序
- [Question-Answering with SQL](https://python.langchain.com/docs/tutorials/sql_qa/): 构建一个执行 SQL 查询来提供响应信息的问答系统
- [Summarization](https://python.langchain.com/docs/tutorials/summarization/): 生成长文本的摘要
- [Question-Answering with Graph Databases](https://python.langchain.com/docs/tutorials/graph/): 构建一个查询图数据库来提供响应信息的问答系统

# 预备内容

- **LangChain 官网**：https://www.langchain.com/
- **Python 文档入口**：https://python.langchain.com/docs/get_started/introduction

# hello world 例子
## 环境设置
创建环境
`conda create --name langchain_env python=3.11`

激活环境
`conda activate langchain_env`

安装包
`conda install langchain -c conda-forge`
`pip install -qU "langchain[openai]"`

**注册使用 LangSmith**
LangSmith 用于追踪、调试和评估 LLM 应用    
- 官网：https://smith.langchain.com/ ，可使用谷歌登录
- 获取 API 密钥和项目名称：Settings-API Keys
`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`

**环境变量**
.env 文件
```python
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT="default"
OPENROUTER_API_KEY=""
```
*OpenRouter官网： https://openrouter.ai/

## 主代码
```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 从 .env 文件加载所有环境变量
load_dotenv()

# LangSmith的配置，确保开启
os.environ["LANGSMITH_TRACING"] = "true"

# 初始化 ChatOpenAI 模型
model = ChatOpenAI(
    model="deepseek/deepseek-chat-v3-0324:free",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY") # 使用 os.getenv() 更安全
)

from langchain_core.messages import HumanMessage, SystemMessage
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# 调用模型并获取返回结果
response = model.invoke(messages)

# 打印回复的文本内容
print(response.content)
```

## 提示模版类
```python
# 导入提示模板类
from langchain_core.prompts import ChatPromptTemplate

# 定义系统和用户的模板
system_template = "将以下内容从英语翻译成{language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# 使用模板生成提示词
prompt = prompt_template.invoke({"language": "意大利语", "text": "Hello, world!"})

# 将格式化后的提示词传递给模型(流式输出)
for chunk in model.stream(prompt):
    print(chunk.content, end="", flush=True)
```

# RAG 应用

**教程链接**：https://python.langchain.com/docs/tutorials/rag  
将自己的文档（PDF、文本文件等）变成一个可供 LLM 检索的知识库

## 概述（索引+检索生成）
![LangChain-RAG1](/pic/langchain实践/LangChain-RAG1.png)  
1. 加载：通过文档加载器加载数据
2. 拆分：将大的 `Documents` 分解成更小的块，对于索引数据和将数据传入模型都很有用，因为大块数据更难搜索，而且无法适应模型有限的上下文窗口
3. 存储：需要一个地方来存储和索引分割的内容，以便以后可以对其进行搜索，通常通过使用向量存储和嵌入模型来完成
![LangChain-RAG2](/pic/langchain实践/LangChain-RAG2.png)  
1. 检索：给定用户输入，使用检索器从存储中检索相关的片段
2. 生成：ChatModel / LLM 使用包含问题和检索数据的提示产生一个答案  

一旦索引了数据，将使用 **LangGraph** 作为编排框架来实施检索和生成步骤

## 完整代码
`rag.py`
```python
import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
  
llm = ChatOpenAI(
    # model="deepseek/deepseek-chat-v3-0324:free",
    model="google/gemini-2.5-flash-lite",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)
  
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    openai_api_base="https://api.siliconflow.cn/v1",
    openai_api_key=os.getenv("SILICONFLOW_API_KEY")
)

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

# 创建应用
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
  
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)
  
# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")
  
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
```

`.env`
```python
LANGSMITH_API_KEY=""
LANGSMITH_PROJECT=""
OPENROUTER_API_KEY=""
LANGSMITH_TRACING=""
SILICONFLOW_API_KEY=""
```


## 代码分析
### 索引-加载文档
参考文档：https://python.langchain.com/docs/how_to/#document-loaders

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# 使用 BeautifulSoup ，从完整 HTML 中解析出 标题、正文
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

# WebBaseLoader使用 urllib 从网页 URL 加载 HTML
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
```

```
Total characters: 43131
```

```python
print(docs[0].page_content[:500])
```

```


      LLM Powered Autonomous Agents
    
Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng


Building agents with LLM (large language model) as its core controller is a cool concept. Several proof-of-concepts demos, such as AutoGPT, GPT-Engineer and BabyAGI, serve as inspiring examples. The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.
Agent System Overview#
In
```

### 索引-分割文档
使用一个递归字符文本分割器（RecursiveCharacterTextSplitter），递归地使用常见的分隔符（如换行符）来分割文档，直到每个块达到合适的尺寸
不同的文档类型会有不同的分割器，详见https://python.langchain.com/docs/how_to/#text-splitters
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个文本块（chunk）的最大字符数
    chunk_overlap=200,  # 相邻文本块之间重叠的字符数，保留文本块之间的上下文连续性
    add_start_index=True,  # 在每个文档块的元数据中添加一个字段，追踪原始位置
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")
```

```
Split blog post into 66 sub-documents.
```

### 索引-存储文档
现在需要索引上文的 66 个文本块，以便在运行时进行搜索。方法是嵌入每个文档分片的内容，并将这些嵌入插入到向量存储中。给定一个输入查询，可以使用向量搜索来检索相关文档。

- 使用在教程开始时选择的向量存储和嵌入模型，通过单一命令嵌入并存储所有的文档分片
```python
document_ids = vector_store.add_documents(documents=all_splits)
print(document_ids[:3])
```

```
['07c18af6-ad58-479a-bfb1-d508033f9c64', '9000bf8e-1993-446f-8d4d-f4e507ba4b8f', 'ba3b5d14-bed9-4f5f-88be-44c88aedc2e6']
```

嵌入模型：https://python.langchain.com/docs/how_to/#embedding-models
向量存储：https://python.langchain.com/docs/how_to/#vector-stores

### 检索生成
创建一个简单的应用程序，它接收用户的问题，搜索与该问题相关的文档，将检索到的文档和初始问题传递给模型，并返回一个答案。  

- **LangChain Hub** 是一个用于存储和共享 **LangChain** 组件（如提示、链等）的中心化仓库，通过 `langchain.hub.pull()` 方法，可以获取社区共享组件。
- `prompt = hub.pull("rlm/rag-prompt")` 从 Hub 拉取一个名为 `"rlm/rag-prompt"` 的特定提示，这个提示是为 **RAG** (Retrieval-Augmented Generation) 任务设计的，旨在将检索到的上下文与用户问题结合起来，生成更准确的答案。
```python
from langchain import hub

# 对于某些非美国地区用户，可能需要将其替换为其他特定的区域性 URL，以便正确连接并拉取提示
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)
```

```
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: (question goes here) 
Context: (context goes here) 
Answer:
```

自定义提示词
```python
from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
```
### LangGraph 整合
使用 LangGraph 将检索和生成步骤整合到一个单一的应用程序中

- 需要定义三件事：应用程序的状态、节点（步骤）、控制流（步骤的顺序）

**状态**：控制输入到应用程序的数据、在步骤之间传输的数据以及应用程序输出的数据，类型为 `TypedDict` 或者 Pydantic BaseModel
对于一个简单的 RAG 应用程序，可以只跟踪输入问题、检索到的上下文和生成的答案：
```python
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
```

**节点（应用步骤）**
一个简单的检索和生成例子
```python
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}
```

- 检索步骤：使用输入问题运行相似性搜索
- 生成步骤：将检索到的上下文和原始问题格式化为聊天模型的提示

**控制流**
将应用程序编译成一个单一的 `graph` 对象
```python
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

LangGraph 提供了内置工具，用于可视化应用程序控制流
```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

LangGraph 支持多种调用模式，包括同步、异步和流式传输
- Invoke 调用
```python
result = graph.invoke({"question": "What is Task Decomposition?"})

print(f"Context: {result['context']}\n\n")
print(f"Answer: {result['answer']}")
```

- Stream steps 流水线步骤
```python
for step in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="updates"
):
    print(f"{step}\n\n----------------\n")
```

- Stream [tokens](https://python.langchain.com/docs/concepts/tokens/) 流令牌
```python
for message, metadata in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="messages"
):
    print(message.content, end="|")
```

- 异步调用
```python
result = await graph.ainvoke(...)
async for step in graph.astream(...):
```

### 查询分析
- 除了语义搜索，还可以构建结构化过滤器（例如“查找自 2020 年以来的文档”）
- 模型可以重写用户查询，用户的查询可能是多方面的或包含无关的语言，将其转化为更有效的搜索查询

**静态元数据**
```python
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

all_splits[0].metadata
```

```
{'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/',
 'start_index': 8,
 'section': 'beginning'}
```

**更新向量存储中的文档**
使用简单的 InMemoryVectorStore，因为会用到它的一些特定功能（元数据过滤）
```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)
```

**为搜索查询定义一个模式**
使用结构化输出，这里将查询定义为一个包含字符串查询和文档部分（“开头”、“中间”或“结尾”）的对象
```python
from typing import Literal
from typing_extensions import Annotated

class Search(TypedDict):
    """Search query."""
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]
```

**在 LangGraph 应用程序中添加一个步骤，用于从用户的原始输入生成查询**
```python
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()
```

**完整代码**
```python
from typing import Literal

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import Annotated, List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


# Update metadata (illustration purposes)
total_documents = len(all_splits)
third = total_documents // 3

for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"


# Index chunks
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)


# Define schema for search
class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()
```

**可以通过特别询问帖子末尾的上下文来测试我们的实现**
```python
for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")
```

参考资料：https://python.langchain.com/docs/how_to/#query-analysis
