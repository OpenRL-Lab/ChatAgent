Welcome to ChatAgent!
====================

[`Github <https://github.com/OpenRL-Lab/ChatAgent>`_]

``ChatAgent`` is a Python-based agent framework for large language models.
The online agents deployed through ChatAgent have provided over a million stable API calls for the internal OpenRL team.

Installation
-------

.. code-block:: bash

    pip install ChatAgent-py

Features
-------

* Supports multimodal large language models
* Supports OpenAI API
* Supports API calls to Qwen on Alibaba Cloud, Zhipu AI's GLM, Microsoft Azure, etc.
* Supports parallel and sequential calls of different agents
* Supports adding an api key for access control
* Supports setting a maximum number of concurrent requests, i.e., the maximum number of requests a model can handle at the same time
* Supports customizing complex agent interaction strategies

Usage
-------

We provide some examples in the ``examples`` directory, which you can run them directly to explore ChatAgent's abilities.

1. Example for Qwen/ZhiPu API to OpenAI API

With just over a dozen lines of code, you can convert the Qwen/ZhiPu API to the OpenAI API.
For specific code and test cases, please refer to `examples/qwen2openai <https://github.com/OpenRL-Lab/ChatAgent/tree/main/examples/qwen2openai>`_ and `examples/glm2openai <https://github.com/OpenRL-Lab/ChatAgent/tree/main/examples/glm2openai>`_ .

.. code-block:: python

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


2. Ensemble with Multiple Agents

We provide an example in `examples/multiagent_ensemble <https://github.com/OpenRL-Lab/ChatAgent/tree/main/examples/multiagent_ensemble>`_ where multiple agents perform ensemble to answer user questions.

3. Agent Q&A Based on RAG Query Results

We provide an example in `examples/rag <https://github.com/OpenRL-Lab/ChatAgent/tree/main/examples/rag>`_ of agent Q&A based on RAG query results.


Citing ChatAgent
-----------------

If our work has been helpful to you, please feel free to cite us:

.. code-block:: bibtex

    @misc{ChatAgent2024,
        title={ChatAgent},
        author={Shiyu Huang},
        publisher = {GitHub},
        howpublished = {\url{https://github.com/OpenRL-Lab/ChatAgent}},
        year={2024},
    }