from typing import Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

from instructor import Instructor, from_openai, Mode


class AnswerUser(BaseModel):
    type: Literal["answer_user"] = "answer_user"
    answer: str = Field(description="The answer to the user question")
    reasoning: str | None = Field(
        default=None, description="The reasoning behind the answer"
    )


class SearchInRag(BaseModel):
    """Rag tool use to search in the RAG about data of our beautiful restaurant Districto Frances"""

    type: Literal["search_in_rag"] = "search_in_rag"
    question: str = Field(description="The question to search in the RAG")


class ResponseModel(BaseModel):
    tool_use: AnswerUser | SearchInRag = Field(
        description="The tool to use from the RAG model"
    )


class RAGState(BaseModel):
    messages: list[dict[str, str]]

    def chat_history(self) -> str:
        """return the 20 last messages.
        Returns:
            str: list of formatted messages in a markdown style
            user : <user_message>
            assistant : <assistant_message>
        """
        return "\n".join([f"{msg['role']} : {msg['content']}" for msg in self.messages])


class Assistant:
    _client: Instructor
    _rag_state: RAGState

    def __init__(self, runnable, rag_state: RAGState):
        self._client = from_openai(runnable, mode=Mode.TOOLS)
        self._rag_state = rag_state

    def __call__(self, message: str, *args, **kwargs) -> str:
        resp = self._client.chat.completions.create(
            response_model=ResponseModel,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": self._rag_state.chat_history()},
                {"role": "user", "content": message},
            ],
        )

        if resp.tool_use.type == "search_in_rag":
            print(f"Searching in the RAG for {resp.tool_use.question}")
            # TODO : search in the RAG
            pass
        elif resp.tool_use.type == "answer_user":
            print(f" reasoning : {resp.tool_use.reasoning}")

            self._rag_state.messages.append({"role": "user", "content": message})
            self._rag_state.messages.append(
                {"role": "assistant", "content": resp.tool_use.answer}
            )

            return resp.tool_use.answer


if __name__ == "__main__":
    load_dotenv()

    rag_state = RAGState(messages=[])
    assistant = Assistant(OpenAI(), rag_state)

    print(assistant("do you serve in Paris?"))
