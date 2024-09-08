from abc import ABC, abstractmethod
from typing import List
from .models import Task, QuickStartAction, DailyPlan


class TaskRepository(ABC):
    @abstractmethod
    def get_tasks(self) -> List[Task]:
        pass


class AIService(ABC):
    @abstractmethod
    def generate_recommendations(self, task: Task) -> QuickStartAction:
        pass

    @abstractmethod
    def create_daily_plan(self, tasks: List[Task]) -> DailyPlan:
        pass
