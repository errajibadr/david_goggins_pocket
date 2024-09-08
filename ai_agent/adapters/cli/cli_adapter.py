import click
from ai_agent.application.use_cases.get_tasks import GetTasks
from ai_agent.application.use_cases.generate_recommendations import (
    GenerateRecommendations,
)
from ai_agent.infrastructure.repositories.notion_task_repository import (
    NotionTaskRepository,
)


@click.group()
@click.pass_context
def cli(ctx):
    """AI Agent CLI for task management and recommendations."""
    ctx.ensure_object(dict)
    ctx.obj["get_tasks_use_case"] = GetTasks(NotionTaskRepository())
    ctx.obj["generate_recommendations_use_case"] = None  # Assuming this exists


@cli.command()
@click.pass_context
def get_tasks(ctx):
    """Fetch and display tasks"""
    tasks = ctx.obj["get_tasks_use_case"].execute()
    for task in tasks:
        click.echo(f"ID: {task.id}, Task: {task.title.strip()}")


@cli.command()
@click.option(
    "--task-id",
    prompt="Enter task ID",
    help="ID of the task to generate recommendations for",
)
@click.pass_context
def recommend(ctx, task_id: str):
    """Generate recommendations for a specific task"""
    if ctx.obj["generate_recommendations_use_case"] is None:
        click.echo("Recommendation generation is not available.")
        return
    recommendations = ctx.obj["generate_recommendations_use_case"].execute(task_id)
    for rec in recommendations:
        click.echo(f"Recommendation: {rec}")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
