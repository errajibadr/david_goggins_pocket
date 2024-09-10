import os
from anthropic import Anthropic
from groq import Groq

from langchain.schema.runnable import RunnableLambda
from langchain.schema import BaseMessage
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from langchain.tools import BaseTool

import instructor


class LLMSettings(BaseSettings):
    temperature: float = 0.0
    max_tokens: int = 2000
    max_retries: int = 3
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class OpenAISettings(LLMSettings):
    model: str = "gpt-4o"
    api_key: str = os.environ["OPENAI_API_KEY"]


class AnthropicSettings(LLMSettings):
    model: str = "claude-3-5-sonnet-20240620"
    api_key: str = os.environ["ANTHROPIC_API_KEY"]


class LlamaSettings(LLMSettings):
    # Run it in local or in Groq
    local: bool = os.environ.get("LLAMA_LOCAL", False)

    local_url: str = "http://localhost:11434/v1"
    local_model: str = "llama3.1-8b-8192"

    # Groq Settings
    model: str = "llama-3.1-8b-instant"
    api_key: str = os.environ["GROQ_API_KEY"]


class AppSettings(BaseSettings):
    app_name: str = "Gen Ai"
    openai_settings: OpenAISettings = OpenAISettings()
    anthropic_settings: AnthropicSettings = AnthropicSettings()
    llama_settings: LlamaSettings = LlamaSettings()


class LLMClient:
    def __init__(self, provider: str = "openai", with_langsmith: bool = True):
        self.provider = provider
        self.tracable = with_langsmith
        self.settings = getattr(AppSettings(), f"{provider}_settings")
        self.client = self.__initializor__()

    def __initializor__(self):
        if self.provider == "openai":
            openai = OpenAI(api_key=self.settings.api_key)
            client = instructor.from_openai(openai, mode=instructor.Mode.TOOLS)
        elif self.provider == "anthropic":
            client = instructor.from_anthropic(Anthropic(api_key=self.settings.api_key))
        elif self.provider == "llama":
            if not self.settings.local:
                client = instructor.from_groq(
                    Groq(api_key=self.settings.api_key), mode=instructor.Mode.JSON
                )
            else:
                client = instructor.from_openai(
                    OpenAI(
                        base_url=self.settings.local_url, api_key=self.settings.api_key
                    ),
                    mode=instructor.Mode.TOOLS,
                )
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

        if self.tracable:
            client = wrap_openai(client, completions_name=self.provider)
        return client

    def call_llm(
        self, response_model: BaseModel, messages: list[dict[str, str]], **kwargs
    ):
        completion_params = {
            "model": kwargs.get("model", self.settings.model),
            "messages": messages,
            "response_model": response_model,
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "top_p": kwargs.get("top_p", self.settings.top_p),
        }
        return self.client.chat.completions.create(**completion_params)

    # def prompt_format(self, prompt: BaseModel):
    #     return RunnableLambda(lambda x: prompt.format(**x))

    def call_chain_llm(
        self, response_model: BaseModel, messages: list[BaseMessage], **kwargs
    ):
        messages = [
            {
                "role": message.dict().get("type")
                if message.dict().get("type") != "human"
                else "user",
                "content": message.dict().get("content"),
            }
            for message in messages
        ]
        return self.call_llm(response_model=response_model, messages=messages, **kwargs)

    def rlambda(self, response_model: BaseModel, **kwargs) -> RunnableLambda:
        return RunnableLambda(
            lambda x: self.call_chain_llm(
                response_model=response_model, messages=x.to_messages(), **kwargs
            )
        )

    def bind_tools(self, tools: list[BaseTool]):
        return self.client.bind_tools(tools)


if __name__ == "__main__":
    llm_client = LLMClient(provider="llama", with_langsmith=True)

    class ResponseModel(BaseModel):
        response: str = Field(description="The response from the LLM")
        reasoning: str | None = Field(
            default=None, description="The reasoning behind the response"
        )

    ResponseModel(response=2)
    # print(
    #     llm_client.call_llm(
    #         response_model=ResponseModel,
    #         messages=[{"role": "user", "content": "talk to me like David Goggins"}],
    #     )
    # )

    # from langchain.prompts import ChatPromptTemplate

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a helpful assistant"),
    #         ("user", "{input}"),
    #     ]
    # )

    # chain = prompt | llm_client.rlambda(ResponseModel)
    # print(chain.invoke({"input": "Hello, how are you today?"}))

    # val = chain.invoke({"input": "Hello, how are you today?"})

    # val_2 = prompt.invoke({"input": "Hello, how are you today?"})

    # val_2.to_messages()[0].dict
