# ChatAgent

[English](./README.md)

基于Python的大语言模型智能体框架。通过ChatAgent部署的在线智能体已为OpenRL团队内部提供了超过百万次稳定的API调用。

## Features

- [x] 支持多模态大语言模型
- [x] 支持OpenAI API
- [x] 支持阿里云千问、智谱GLM、微软Azure等API调用
- [x] 支持不同智能体并行与串行调用
- [x] 支持添加api_key实现访问控制
- [x] 支持设置最大并发数，即模型同时处理请求最大数量
- [x] 支持自定义复杂的智能体交互策略

## Installation

```bash
pip install ChatAgent
```

## Usage

我们在`examples`目录下提供了一些示例，可以直接运行查看效果。

### 1. Qwen/ZhiPu API 转 OpenAI API 示例

只需要十多行代码，就可以将Qwen/ZhiPu API转换为OpenAI API。具体代码和测试用例请参考[examples/qwen2openai](./examples/qwen2openai)和[examples/glm2openai](./examples/glm2openai)。
```python
import os
from ChatAgent import serve
from ChatAgent.chat_models.base_chat_model import BaseChatModel
from ChatAgent.agents.dashscope_chat_agent import DashScopeChatAgent
from ChatAgent.protocol.openai_api_protocol import MultimodalityChatCompletionRequest
class QwenMax(BaseChatModel):
    def init_agent(self):
        self.agent = DashScopeChatAgent(model_name='qwen-max',api_key=os.getenv("QWEN_API_KEY"))
    def create_chat_completion(self, request):
        return self.agent.act(request)
@serve.create_chat_completion()
async def implement_completions(request: MultimodalityChatCompletionRequest):
    return QwenMax().create_chat_completion(request)
serve.run(host="0.0.0.0", port=6367)
```

### 2. 多个智能体做ensemble

我们在[examples/multiagent_ensemble](./examples/multiagent_ensemble)中提供了一个多个智能体做ensemble回答用户问题的示例，可以直接运行查看效果。

### 3. 基于RAG查询结果进行智能体问答

我们在[examples/rag](./examples/rag)中提供了一个基于RAG查询结果进行智能体问答的示例，可以直接运行查看效果。

## Citation

如果您使用了ChatAgent，请引用我们：
```bibtex
@misc{ChatAgent2024,
    title={ChatAgent},
    author={Shiyu Huang},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/OpenRL-Lab/ChatAgent}},
    year={2024},
}

