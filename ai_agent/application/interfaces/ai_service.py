from abc import ABC, abstractmethod
from ai_agent.domain.models import Task, QuickStartAction


class AIService(ABC):
    @abstractmethod
    def generate_quick_start_action(self, task: Task) -> QuickStartAction:
        pass
