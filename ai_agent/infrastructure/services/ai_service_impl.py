from ai_agent.application.interfaces.ai_service import AIService
from ai_agent.domain.models import Task, QuickStartAction
from ai_agent.infrastructure.llm_client import LLMClient
from ai_agent.agents.prompts import RECOMMENDATIONS_SYSTEM_PROMPT
from langchain_core.prompts import ChatPromptTemplate


class AIServiceImpl(AIService):
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_quick_start_action(self, task: Task) -> QuickStartAction:
        messages = ChatPromptTemplate.from_messages(
            [
                ("system", RECOMMENDATIONS_SYSTEM_PROMPT),
                ("user", "{task}"),
            ]
        )

        chain = messages | self.llm_client.rlambda(
            QuickStartAction, temperature=0, top_p=0.5
        )

        return chain.invoke({"task": task.model_dump()})
