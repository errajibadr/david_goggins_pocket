import unittest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field
from ai_agent.infrastructure.llm_client import LLMClient


class ResponseModel(BaseModel):
    response: str = Field(description="The response from the LLM")
    reasoning: str | None = Field(
        default=None, description="The reasoning behind the response"
    )


class TestLLMClient(unittest.TestCase):
    @patch("ai_agent.infrastructure.llm_client.OpenAI")
    @patch("ai_agent.infrastructure.llm_client.Anthropic")
    @patch("ai_agent.infrastructure.llm_client.Groq")
    def setUp(self, mock_groq, mock_anthropic, mock_openai):
        self.mock_openai = mock_openai
        self.mock_anthropic = mock_anthropic
        self.mock_groq = mock_groq

        # Create a mock for instructor.from_openai, etc.
        self.mock_instructor = MagicMock()
        with patch(
            "ai_agent.infrastructure.llm_client.instructor", self.mock_instructor
        ):
            self.client = LLMClient(provider="openai", with_langsmith=False)

    def test_initialization(self):
        self.assertEqual(self.client.provider, "openai")
        self.assertFalse(self.client.tracable)
        self.assertIsNotNone(self.client.settings)
        self.assertIsNotNone(self.client.client)

    @patch("ai_agent.infrastructure.llm_client.instructor")
    def test_initialization_anthropic(self, mock_instructor):
        client = LLMClient(provider="anthropic", with_langsmith=False)
        self.assertEqual(client.provider, "anthropic")
        mock_instructor.from_anthropic.assert_called_once()

    @patch("ai_agent.infrastructure.llm_client.instructor")
    def test_initialization_llama(self, mock_instructor):
        client = LLMClient(provider="llama", with_langsmith=False)
        self.assertEqual(client.provider, "llama")
        mock_instructor.from_groq.assert_called_once()

    def test_call_llm(self):
        mock_response = MagicMock()
        self.client.client.chat.completions.create.return_value = mock_response

        response_model = ResponseModel
        messages = [{"role": "user", "content": "Hello"}]

        result = self.client.call_llm(response_model, messages)

        self.assertEqual(result, mock_response)
        self.client.client.chat.completions.create.assert_called_once()

    def test_call_chain_llm(self):
        mock_response = MagicMock()
        self.client.call_llm = MagicMock(return_value=mock_response)

        class TestMessage(BaseModel):
            type: str
            content: str

        messages = [
            TestMessage(type="system", content="You are an assistant"),
            TestMessage(type="human", content="Hello"),
        ]

        result = self.client.call_chain_llm(ResponseModel, messages)

        self.assertEqual(result, mock_response)
        self.client.call_llm.assert_called_once()

    def test_rlambda(self):
        mock_response = MagicMock()
        self.client.call_chain_llm = MagicMock(return_value=mock_response)

        rlambda = self.client.rlambda(ResponseModel)
        self.assertIsNotNone(rlambda)

        # Test the lambda function
        mock_messages = MagicMock()
        mock_messages.to_messages.return_value = []
        result = rlambda.invoke(mock_messages)

        self.assertEqual(result, mock_response)
        self.client.call_chain_llm.assert_called_once_with(
            response_model=ResponseModel, messages=[]
        )


if __name__ == "__main__":
    unittest.main()
