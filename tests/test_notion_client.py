import pytest
from unittest.mock import patch
from ai_agent.infrastructure.notion_local_client import NotionClient
from ai_agent.domain.models import Task


@pytest.fixture
def mock_notion_client(monkeypatch):
    monkeypatch.setenv("NOTION_TOKEN", "fake_token")
    with patch("ai_agent.infrastructure.notion_local_client.Client"):
        yield NotionClient()


def test_get_backlog_from_notion(mock_notion_client):
    # Mock the database query response
    mock_notion_client.client.databases.query.return_value = {
        "results": [
            {
                "id": "page1",
                "properties": {
                    "Name": {"title": [{"plain_text": "Task 1"}]},
                    "État": {"status": {"name": "À faire"}},
                    "Priorité": {"select": {"name": "5"}},
                    "Date de création": {"created_time": "2023-03-01T12:00:00Z"},
                    "Échéance": {"date": {"start": "2023-03-10", "end": None}},
                },
            }
        ]
    }

    # Mock the page content response
    mock_notion_client.client.blocks.children.list.return_value = {
        "results": [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Task 1 description"}]},
            }
        ]
    }

    tasks = mock_notion_client.get_backlog_from_notion()

    assert len(tasks) == 1
    assert isinstance(tasks[0], Task)
    assert tasks[0].title == "Task 1"
    assert tasks[0].status == "À faire"
    assert tasks[0].priority == "5"
    assert tasks[0].description == "Task 1 description"


def test_get_page_content_from_notion(mock_notion_client):
    mock_notion_client.client.blocks.children.list.return_value = {
        "results": [
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Paragraph 1"}]},
            },
            {
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Paragraph 2"}]},
            },
            {"type": "heading", "heading": {"rich_text": [{"plain_text": "Heading"}]}},
        ]
    }

    content = mock_notion_client.get_page_content_from_notion("fake_page_id")

    assert content == ["Paragraph 1", "Paragraph 2"]
    assert len(content) == 2
