import json
from dotenv import load_dotenv
import os
from typing import List, Dict

from notion_client import Client
from ai_agent.domain.models import Task


__database_id__ = "7ca341349d914a7dadbd513be2067820"
__page_id__ = "fake_page_id"


class NotionClient:
    def __init__(self, db_id: str = None):
        self.client = Client(auth=os.environ["NOTION_TOKEN"])
        self.database_id = db_id or __database_id__

    def get_backlog_from_notion(self) -> List[Task]:
        filter = {
            "or": [
                {"property": "État", "status": {"equals": "À faire"}},
                {"property": "État", "status": {"equals": "En cours"}},
            ]
        }

        response = self.client.databases.query(
            database_id=self.database_id, filter=filter
        )
        results = response["results"]

        tasks = []
        for id, task in zip(range(len(results)), results):
            content = self.get_page_content_from_notion(task["id"])
            task_data = {"id": str(id), "content": content, **task["properties"]}
            tasks.append(Task.from_dict(task_data))

        return tasks

    def get_page_content_from_notion(self, page_id: str) -> List[str]:
        content = self.client.blocks.children.list(page_id)

        paragraphs = [
            block["paragraph"]["rich_text"][0]["plain_text"]
            for block in content["results"]
            if block["type"] == "paragraph" and block["paragraph"]["rich_text"]
        ]
        return paragraphs


if __name__ == "__main__":
    load_dotenv()
    client = NotionClient()
    print(client.get_backlog_from_notion())
