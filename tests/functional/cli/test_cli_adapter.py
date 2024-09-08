import pytest
from unittest.mock import patch, MagicMock


from ai_agent.domain.models import Task
from ai_agent.application.use_cases.get_tasks import GetTasks
from ai_agent.application.use_cases.generate_recommendations import (
    GenerateRecommendations,
)


# @pytest.fixture
# def mock_get_tasks():
#     get_tasks = MagicMock(spec=GetTasks)
#     get_tasks.execute.return_value = [
#         Task(
#             id="1",
#             title="Test Task 1",
#             description="Description 1",
#             creation_date="2023-01-01",
#             status="TODO",
#             priority="HIGH",
#         ),
#         Task(
#             id="2",
#             title="Test Task 2",
#             description="Description 2",
#             creation_date="2023-01-02",
#             status="IN_PROGRESS",
#             priority="MEDIUM",
#         ),
#     ]
#     return get_tasks


# @pytest.fixture
# def mock_generate_recommendations():
#     generate_recommendations = MagicMock(spec=GenerateRecommendations)
#     generate_recommendations.execute.return_value = [
#         "Start with a small step",
#         "Break the task into smaller parts",
#         "Set a timer for 25 minutes and focus",
#     ]
#     return generate_recommendations


# @pytest.fixture
# def cli_adapter(mock_get_tasks, mock_generate_recommendations):
#     return CLIAdapter(mock_get_tasks, mock_generate_recommendations)


# def test_get_tasks(cli_adapter, capsys):
#     with patch("builtins.input", side_effect=["get_tasks", "quit"]):
#         cli_adapter.run()

#     captured = capsys.readouterr()
#     assert "Task: Test Task 1" in captured.out
#     assert "Task: Test Task 2" in captured.out


# def test_recommend(cli_adapter, capsys):
#     with patch("builtins.input", side_effect=["recommend", "1", "quit"]):
#         cli_adapter.run()

#     captured = capsys.readouterr()
#     assert "Recommendation: Start with a small step" in captured.out
#     assert "Recommendation: Break the task into smaller parts" in captured.out
#     assert "Recommendation: Set a timer for 25 minutes and focus" in captured.out


# def test_invalid_command(cli_adapter, capsys):
#     with patch("builtins.input", side_effect=["invalid_command", "quit"]):
#         cli_adapter.run()

#     captured = capsys.readouterr()
#     assert "Invalid command" in captured.out


# def test_quit(cli_adapter):
#     with patch("builtins.input", return_value="quit"):
#         cli_adapter.run()
#     # If the function returns without error, the test passes
