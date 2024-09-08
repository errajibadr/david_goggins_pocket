from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, date


class Task(BaseModel):
    id: str = Field(description="The ID of the task")
    title: str = Field(description="The title of the task")
    description: str = Field(description="The description of the task")
    creation_date: datetime = Field(description="The creation date of the task")
    start_date: Optional[date] = Field(None, description="The start date of the task")
    end_date: Optional[date] = Field(None, description="The end date of the task")
    status: str = Field(description="The status of the task")
    priority: str = Field(
        description="The priority of the task for 0-5, 99 is undefined priority"
    )
    complexity: Optional[int] = Field(
        None, description="Task complexity on a scale of 1-5"
    )

    @staticmethod
    def from_dict(notion_data: dict) -> "Task":
        # Parse creation date
        creation_time = datetime.fromisoformat(
            notion_data["Date de création"]["created_time"].replace("Z", "+00:00")
        )
        creation_date = creation_time.date()

        return Task(
            id=notion_data["id"],
            title=notion_data["Name"]["title"][0]["plain_text"],
            description="\n".join(notion_data["content"]),
            creation_date=datetime.fromisoformat(
                notion_data["Date de création"]["created_time"].replace("Z", "+00:00")
            ),
            start_date=(
                date.fromisoformat(notion_data["Échéance"]["date"]["start"])
                if notion_data.get("Échéance") and notion_data["Échéance"]["date"]
                else None
            ),
            end_date=(
                date.fromisoformat(notion_data["Échéance"]["date"]["end"])
                if notion_data.get("Échéance")
                and notion_data["Échéance"]["date"]
                and notion_data["Échéance"]["date"].get("end")
                else None
            ),
            status=notion_data["État"]["status"]["name"],
            priority=(
                notion_data["Priorité"]["select"]["name"]
                if notion_data["Priorité"]["select"]
                else "99"
            ),
        )

    def to_str(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Description: {self.description}\n"
            f"Creation Date: {self.creation_date}\n"
            f"Start Date: {self.start_date}\n"
            f"End Date: {self.end_date}\n"
            f"Status: {self.status}\n"
            f"Priority: {self.priority}"
        )


class Step(BaseModel):
    action: str = Field(description="The step To take")
    actionable_examples: List[str] = Field(
        description="actionable and concrete examples of the step"
    )


class QuickStartAction(BaseModel):
    priority: int = Field(description="The priority of the task")
    actions: List[Step] = Field(
        description="The list of actions to take. each one with 2-3 examples of what to do"
    )
    motivation_quote: str = Field(
        description="A DAVID GOGGINS-style motivational quote"
    )


class DailyPlan(BaseModel):
    tasks: List[Task]
    recommendations: List[str]
    energy_level: int = Field(description="User's energy level on a scale of 1-10")
    schedule_conflicts: List[str] = Field(
        description="Any scheduling conflicts for tasks"
    )
