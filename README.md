# ChatAgent

[中文说明](./README_CN.md)

A Python-based large language model agent framework. 
The online agents deployed through ChatAgent have provided over a million stable API calls for the internal OpenRL team.

## Features

- [x] Supports multimodal large language models
- [x] Supports OpenAI API
- [x] Supports API calls to Qwen on Alibaba Cloud, Zhipu AI's GLM, Microsoft Azure, etc.
- [x] Supports parallel and sequential calls of different agents
- [x] Supports adding an api key for access control
- [x] Supports setting a maximum number of concurrent requests, i.e., the maximum number of requests a model can handle at the same time
- [x] Supports customizing complex agent interaction strategies

## Installation

```bash
pip install ChatAgent
```

## Usage

We provide some examples in the `examples` directory, which you can run them directly to explore ChatAgent's abilities.

### 1. Example for Qwen/ZhiPu API to OpenAI API

With just over a dozen lines of code, you can convert the Qwen/ZhiPu API to the OpenAI API. 
For specific code and test cases, please refer to [examples/qwen2openai](./examples/qwen2openai) and [examples/glm2openai](./examples/glm2openai).
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

### 2. Ensemble with Multiple Agents

We provide an example in [examples/multiagent_ensemble](./examples/multiagent_ensemble) where multiple agents perform ensemble to answer user questions.

### 3. Agent Q&A Based on RAG Query Results

We provide an example in [examples/rag](./examples/rag) of agent Q&A based on RAG query results.

## Citation

If you use ChatAgent, please cite us:
```bibtex
@misc{ChatAgent2024,
    title={ChatAgent},
    author={Shiyu Huang},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/OpenRL-Lab/ChatAgent}},
    year={2024},
}
```