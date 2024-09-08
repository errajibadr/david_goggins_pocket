from datetime import datetime, date, timezone

from ai_agent.domain.models import Task


def test_task_creation():
    task = Task(
        id="1",
        title="Test Task",
        description="This is a test task",
        creation_date=datetime(2023, 1, 1),
        start_date=date(2023, 1, 2),
        end_date=date(2023, 1, 3),
        status="In Progress",
        priority="1",
    )

    assert task.id == "1"
    assert task.title == "Test Task"
    assert task.description == "This is a test task"
    assert task.creation_date == datetime(2023, 1, 1)
    assert task.start_date == date(2023, 1, 2)
    assert task.end_date == date(2023, 1, 3)
    assert task.status == "In Progress"
    assert task.priority == "1"


def test_task_from_dict_with_no_due_date():
    notion_data = {
        "id": "task_id",
        "Date de création": {
            "id": "'Y6%3C",
            "type": "created_time",
            "created_time": "2024-08-14T11:02:00.000Z",
        },
        "Bouton terminé": {"id": "QAkI", "type": "button", "button": {}},
        "État": {
            "id": "%5EOE%40",
            "type": "status",
            "status": {"id": "1", "name": "À faire", "color": "red"},
        },
        "Échéance": {"id": "_mgZ", "type": "date", "date": None},
        "Name": {
            "id": "title",
            "type": "title",
            "title": [
                {
                    "type": "text",
                    "text": {
                        "content": "rechercher : cibler des gens avec mon profil et le type de services qu'ils proposent",
                        "link": None,
                    },
                    "annotations": {
                        "bold": False,
                        "italic": False,
                        "strikethrough": False,
                        "underline": False,
                        "code": False,
                        "color": "default",
                    },
                    "plain_text": "rechercher : cibler des gens avec mon profil et le type de services qu'ils proposent",
                    "href": None,
                }
            ],
        },
        "Priorité": {"select": None},
        "content": ["This is the task content"],
    }

    task = Task.from_dict(notion_data)

    assert task.id == "task_id"
    assert (
        task.title
        == "rechercher : cibler des gens avec mon profil et le type de services qu'ils proposent"
    )
    assert task.description == "This is the task content"
    assert task.creation_date == datetime(2024, 8, 14, 11, 2, tzinfo=timezone.utc)
    assert task.start_date is None
    assert task.end_date is None
    assert task.status == "À faire"
    assert task.priority == "99"


def test_task_from_dict():
    notion_data = {
        "id": "2",
        "Name": {"title": [{"plain_text": "Notion Task"}]},
        "content": ["This is a task from Notion"],
        "Date de création": {"created_time": "2023-01-01T00:00:00Z"},
        "Échéance": {"date": {"start": "2023-01-02", "end": "2023-01-03"}},
        "État": {"status": {"name": "In Progress"}},
        "Priorité": {"select": {"name": "10"}},
    }

    task = Task.from_dict(notion_data)

    assert task.id == "2"
    assert task.title == "Notion Task"
    assert task.description == "This is a task from Notion"
    assert task.creation_date == datetime(2023, 1, 1, tzinfo=timezone.utc)
    assert task.start_date == date(2023, 1, 2)
    assert task.end_date == date(2023, 1, 3)
    assert task.status == "In Progress"
    assert task.priority == "10"


def test_task_to_str():
    task = Task(
        id="3",
        title="String Representation Task",
        description="Testing to_str method",
        creation_date=datetime(2023, 1, 1),
        start_date=date(2023, 1, 2),
        end_date=date(2023, 1, 3),
        status="Completed",
        priority="5",
    )

    expected_str = (
        "Title: String Representation Task\n"
        "Description: Testing to_str method\n"
        "Creation Date: 2023-01-01 00:00:00\n"
        "Start Date: 2023-01-02\n"
        "End Date: 2023-01-03\n"
        "Status: Completed\n"
        "Priority: 5"
    )

    assert task.to_str() == expected_str
