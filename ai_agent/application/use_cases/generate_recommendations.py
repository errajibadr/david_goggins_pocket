from ai_agent.domain.models import Task, QuickStartAction
from ai_agent.application.interfaces.ai_service import AIService


class GenerateRecommendations:
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service

    def get_recommendations(self, task: Task) -> QuickStartAction:
        return self.ai_service.generate_quick_start_action(task)


if __name__ == "__main__":
    from ai_agent.infrastructure.services.ai_service_impl import AIServiceImpl
    from ai_agent.infrastructure.llm_client import LLMClient

    sample_task = Task(
        id="2",
        title="prepare for my sales meeting",
        description="i need to prepare my chatbot sales pitch",
        creation_date="2024-03-15T10:00:00",
        status="Not Started",
        priority="High",
    )

    # Initialize the AIService implementation"
    llm_client = LLMClient(provider="llama")
    ai_service = AIServiceImpl(llm_client)

    # Initialize the GenerateRecommendations class
    recommender = GenerateRecommendations(ai_service)

    # Generate recommendations for the sample task
    recommendations = recommender.get_recommendations(sample_task)

    # Print the results
    print("Generated Quick Start Action:")
    print(f"Priority: {recommendations.priority}")
    print("Actions:")
    for action in recommendations.actions:
        print(f"- {action}")
    print(f"Motivation Quote: {recommendations.motivation_quote}")
