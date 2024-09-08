from ai_agent.application.interfaces.task_repository import TaskRepository
from ai_agent.domain.models import Task
from typing import List

from ai_agent.infrastructure.notion_local_client import NotionClient


class NotionTaskRepository(TaskRepository):
    def __init__(self, db_id: str = None):
        self.db_id = db_id
        self.client = NotionClient(db_id)

    def get_tasks(self, db_id: str = None) -> List[Task]:
        return self.client.get_backlog_from_notion()
