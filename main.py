import asyncio
from datetime import date, datetime
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import OpenAI as LangchainOpenAI

from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI

# from openai import OpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain.schema import BaseMessage
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from pydantic import BaseModel, Field
import instructor
import os
from notion_client import Client


load_dotenv()

# Initialize Notion client
notion = Client(auth=os.environ["NOTION_TOKEN"])

# Patch OpenAI with instructor
# openai = instructor.from_openai(OpenAI())

openai = LangchainOpenAI()


# Wrap the OpenAI client with LangSmith
client = wrap_openai(OpenAI())

# Patch the client with instructor
client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)


# Define data models
class Task(BaseModel):
    title: str
    description: str
    creation_date: datetime
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: str
    priority: str

    @staticmethod
    def from_dict(notion_data: dict) -> "Task":
        title = notion_data["Name"]["title"][0]["plain_text"].strip()
        description = "\n".join(notion_data["content"])
        status = notion_data["État"]["status"]["name"]
        priority = (
            notion_data["Priorité"]["select"]["name"]
            if notion_data["Priorité"]["select"]
            else "No Priority"
        )

        # Parse creation date
        creation_time = datetime.fromisoformat(
            notion_data["Date de création"]["created_time"].replace("Z", "+00:00")
        )
        creation_date = creation_time.date()

        # Parse 'Échéance' if it exists
        start_date = None
        end_date = None
        if notion_data.get("Échéance") and notion_data["Échéance"]["date"]:
            echeance = notion_data["Échéance"]["date"]
            start_date = date.fromisoformat(echeance["start"])
            if echeance.get("end"):
                end_date = date.fromisoformat(echeance["end"])

        return Task(
            title=title,
            description=description,
            creation_date=creation_date,
            start_date=start_date,
            end_date=end_date,
            status=status,
            priority=priority,
        )

    def to_str(self) -> str:
        return f"Title: {self.title}\nDescription: {self.description}\nCreation Date: {self.creation_date}\nStart Date: {self.start_date}\nEnd Date: {self.end_date}\nStatus: {self.status}\nPriority: {self.priority}"


class QuickStartAction(BaseModel):
    priority: int = Field(description="The priority of the task")
    actions: list[str] = Field(description="The list of action to take.")


class DailyPlan(BaseModel):
    tasks: list[Task]
    recommendations: list[str]


# Define the state for the agent
class AgentState(BaseModel):
    messages: list[BaseMessage]
    tasks: list[Task] = []
    daily_plan: DailyPlan | None = None
    messages: list[str] = []


def get_backlog_from_notion(database_id):
    """{'Date de création': {'id': "'Y6%3C", 'type': 'created_time', 'created_time': '2024-08-14T11:02:00.000Z'}, 'Bouton terminé': {'id': 'QAkI', 'type': 'button', 'button': {}}, 'État': {'id': '%5EOE%40', 'type': 'status', 'status': {'id': '1', 'name': 'À faire', 'color': 'red'}}, 'Échéance': {'id': '_mgZ', 'type': 'date', 'date': None}, 'Name': {'id': 'title', 'type': 'title', 'title': [{'type': 'text', 'text': {'content': 'rechercher : cibler des gens avec mon profil et le type de services qu’ils proposent', 'link': None}, 'annotations': {'bold': False, 'italic': False, 'strikethrough': False, 'underline': False, 'code': False, 'color': 'default'}, 'plain_text': 'rechercher : cibler des gens avec mon profil et le type de services qu’ils proposent', 'href': None}]}}

    Returns:
        _type_: _description_
    """
    filter = {
        "or": [
            {"property": "État", "status": {"equals": "À faire"}},
            {"property": "État", "status": {"equals": "En cours"}},
        ]
    }

    # Effectuer la requête avec le filtre
    response = notion.databases.query(database_id=database_id, filter=filter)

    results = response["results"]

    tasks = [
        {"content": get_page_cotent_from_notion(task["id"]), **task["properties"]}
        for task in results
    ]

    # page = get_page_cotent_from_notion(tasks[0]["id"])

    # Parcourir les résultats filtrés
    return tasks


def get_page_cotent_from_notion(page_id):
    content = notion.blocks.children.list(page_id)

    paragraphs = [
        block["paragraph"]["rich_text"][0]["plain_text"]
        for block in content["results"]
        if block["type"] == "paragraph" and block["paragraph"]["rich_text"]
    ]

    return paragraphs


# Function to fetch today's tasks from Notion


# Function to generate recommendations for each task
def generate_recommendations(task: Task) -> list[str]:
    from langchain.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """
    You are a helpful assistant doing task management. 
    You are given a task and you have to generate a quick action steps list and assign a daily priority.
    The quick action steps list should be a list of 3-5 action steps to accomplish the task.
    The actions steps should be in the following format:
    - Action step 1
    - Action step 2
    - Action step 3
    - Action step 4
    - Action step 5
    The action steps should be clear and precise on what to do to start or accomplish the task.
    your goal is to make starting the task as easy as possible for the user.
    
    Moreover you have to assign a priority to the task based on the due date and priority.
    
    the task : 
    {task}
        """
    )

    system_prompt = """
    You are a helpful assistant doing task management. 
    You are given a task and you have to generate a quick action steps list and assign a daily priority.
    The quick action steps list should be a list of 3-5 action steps to accomplish the task.
    The actions steps should be in the following format:
    - Action step 1
    - Action step 2
    - Action step 3
    - Action step 4
    - Action step 5
    The action steps should be clear and precise on what to do to start or accomplish the task.
    your goal is to make starting the task as easy as possible for the user.
    
    Moreover you have to assign a priority to the task based on the due date and priority.
    """

    chain = prompt | openai

    # result = chain.invoke(
    #     input={
    #         "task": task.to_str(),
    #     },
    #     config={
    #         "temperature": 0.2,
    #         "response_model": QuickStartAction,
    #     },
    # )

    format_prompt = RunnableLambda(lambda x: prompt.format_prompt(**x))

    invoke_model = RunnableLambda(
        lambda x: client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": msg.content} for msg in x.to_messages()
            ],
            max_retries=2,
            response_model=QuickStartAction,
        )
    )
    chain = format_prompt | invoke_model
    result = chain.invoke(input={"task": task.to_str()})
    print(result)

    # result_2 = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {
    #             "role": "user",
    #             "content": f"Can you help me with this task:\n{task.to_str()}",
    #         },
    #     ],
    #     max_retries=2,
    #     response_model=QuickStartAction,
    # )

    # print(result_2)
    # recommendations = result.split("\n")
    # return [rec.strip() for rec in recommendations if rec.strip()]


# Define graph nodes
def process_tasks(agent_state: AgentState) -> AgentState:
    tasks = agent_state.tasks
    recommendations = []
    for task in tasks:
        recommendations.extend(generate_recommendations(task))
    daily_plan = DailyPlan(tasks=tasks, recommendations=recommendations)
    return AgentState(tasks=tasks, daily_plan=daily_plan, messages=agent_state.messages)


# Create the graph
workflow = StateGraph(AgentState)
workflow.add_node("process_tasks", process_tasks)

# Set the entrypoint
workflow.add_edge(START, "process_tasks")
workflow.add_edge("process_tasks", END)

app = workflow.compile()


# Run the workflow
def main():
    # todays_tasks = get_todays_tasks()
    # result = app.invoke({"tasks": todays_tasks, "messages": []})
    # print("Daily Plan:")
    # print(f"Tasks: {result['daily_plan'].tasks}")
    # print(f"Recommendations: {result['daily_plan'].recommendations}")
    page_id = "33b30a0981e341c7b499fc4158b2f494"

    # print(get_page_cotent_from_notion(page_id=page_id))

    page_2 = "https://www.notion.so/7ca341349d914a7dadbd513be2067820?v=4c3e3c25788549d095a3e082b583b84b&pvs=4"
    db_id = "7ca341349d914a7dadbd513be2067820"

    # database_id = db_id
    # response = get_backlog_from_notion(database_id=database_id)

    # database_id = db_id
    # # response = notion.databases.query(database_id=database_id)

    # tasks = [Task.from_dict(notion_data=page) for page in response]
    # # Parcourez les résultats

    # for task in tasks:
    #     print("--------------------------------")
    #     print(task)

    test_task = Task(
        title="rechercher : cibler des gens avec mon profil et le type de services qu'ils proposent",
        description="",
        creation_date=datetime(2024, 8, 14),
        start_date=None,
        end_date=None,
        status="À faire",
        priority="2",
    )

    generate_recommendations(test_task)
    i = 1


if __name__ == "__main__":
    main()
