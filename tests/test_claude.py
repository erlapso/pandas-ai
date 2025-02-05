from io import BytesIO
from unittest.mock import MagicMock, patch
from extensions.llms.bedrock.pandasai_bedrock.claude import BedrockClaude
from pandasai.core.prompts.base import BasePrompt
from pandasai.agent.state import AgentState
import json


def test_bedrock_claude_call_method():
    """
    Test the call method of BedrockClaude.
    
    This test verifies:
    1. The correct formatting of the request body
    2. The proper invocation of the Bedrock client
    3. The correct parsing and return of the response
    
    It mocks the necessary components, including the Bedrock client,
    BasePrompt, and AgentState, to isolate the BedrockClaude.call method.
    """
    # Mock the boto3 client
    mock_client = MagicMock()
    
    # Mock the response from the Bedrock client
    mock_response = {
        "body": BytesIO(json.dumps({"content": [{"text": "Mocked response"}]}).encode())
    }
    mock_client.invoke_model.return_value = mock_response

    # Initialize BedrockClaude with the mock client
    claude = BedrockClaude(
        bedrock_runtime_client=mock_client,
        model="anthropic.claude-3-sonnet-20240229-v1:0"
    )

    # Create a mock prompt
    mock_prompt = MagicMock(spec=BasePrompt)
    mock_prompt.to_string.return_value = "Test prompt"

    # Create a mock context with a mock memory
    mock_memory = MagicMock()
    mock_memory.agent_description = "Test agent"
    mock_memory.all.return_value = [
        {"is_user": True, "message": "User message"},
        {"is_user": False, "message": "Assistant message"}
    ]
    mock_context = MagicMock(spec=AgentState)
    mock_context.memory = mock_memory

    # Call the method
    result = claude.call(mock_prompt, mock_context)

    # Assert that the client was called with the correct parameters
    mock_client.invoke_model.assert_called_once()
    call_args = mock_client.invoke_model.call_args[1]
    assert call_args["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Parse the body to check its contents
    body = json.loads(call_args["body"])
    assert body["system"] == "Test agent"
    assert len(body["messages"]) == 3  # System message, user message, assistant message
    assert body["messages"][-1]["role"] == "user"
    assert body["messages"][-1]["content"][-1]["text"] == "Test prompt"

    # Assert that the result is correct
    assert result == "Mocked response"
