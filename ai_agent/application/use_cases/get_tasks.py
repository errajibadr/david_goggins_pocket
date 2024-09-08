from ai_agent.domain.interfaces import TaskRepository


class GetTasks:
    def __init__(self, task_repository: TaskRepository):
        self.task_repository = task_repository

    def execute(self):
        return self.task_repository.get_tasks()
