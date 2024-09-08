from abc import ABC, abstractmethod
from typing import List
from ai_agent.domain.models import Task


class TaskRepository(ABC):
    @abstractmethod
    def get_tasks(self) -> List[Task]:
        pass
