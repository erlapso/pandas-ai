from pandasai.helpers.memory import Memory
import pytest

class DummyObject:
    def __str__(self):
        return "dummy"
def test_to_json_empty_memory():
    memory = Memory()
    assert memory.to_json() == []
def test_to_json_with_messages():
    memory = Memory()
    # Add test messages
    memory.add("Hello", is_user=True)
    memory.add("Hi there!", is_user=False)
    memory.add("How are you?", is_user=True)
    expected_json = [
        {"role": "user", "message": "Hello"},
        {"role": "assistant", "message": "Hi there!"},
        {"role": "user", "message": "How are you?"},
    ]
    assert memory.to_json() == expected_json
def test_to_json_message_order():
    memory = Memory()
    # Add messages in specific order
    messages = [("Message 1", True), ("Message 2", False), ("Message 3", True)]
    for msg, is_user in messages:
        memory.add(msg, is_user=is_user)
    result = memory.to_json()
    # Verify order is preserved
    assert len(result) == 3
    assert result[0]["message"] == "Message 1"
    assert result[1]["message"] == "Message 2"
    assert result[2]["message"] == "Message 3"
def test_to_openai_messages_empty():
    memory = Memory()
    assert memory.to_openai_messages() == []
def test_to_openai_messages_with_agent_description():
    memory = Memory(agent_description="I am a helpful assistant")
    memory.add("Hello", is_user=True)
    memory.add("Hi there!", is_user=False)
    expected_messages = [
        {"role": "system", "content": "I am a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_to_openai_messages_without_agent_description():
    memory = Memory()
    memory.add("Hello", is_user=True)
    memory.add("Hi there!", is_user=False)
    memory.add("How are you?", is_user=True)
    expected_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_clear_and_truncation_and_last_message():
    """
    Test clear functionality, message truncation for long assistant messages,
    retrieval of last message, previous conversation, and overall conversation formatting.
    """
    # Create a Memory instance with a memory size of 3 to hold three messages.
    memory = Memory(memory_size=3)
    # Add a user message (should not be truncated).
    memory.add("User message", is_user=True)
    # Add an assistant message longer than 100 characters to trigger truncation.
    long_message = "A" * 150
    memory.add(long_message, is_user=False)
    # Add another user message.
    memory.add("Another user message", is_user=True)
    # Retrieve messages using get_messages (should include exactly 3 messages).
    messages = memory.get_messages()
    
    # Check that the assistant message was correctly truncated.
    truncated_assistant = long_message[:100] + " ..."
    expected_assistant_msg = "### ANSWER\n " + truncated_assistant
    # messages[0] is the first user message, messages[1] the assistant, messages[2] the final user message.
    assert messages[1] == expected_assistant_msg
    # Test that get_last_message returns the last message (user message).
    expected_last_msg = "### QUERY\n " + "Another user message"
    assert memory.get_last_message() == expected_last_msg
    # Test that get_previous_conversation returns the conversation excluding the last message.
    expected_previous = "### QUERY\n " + "User message" + "\n" + expected_assistant_msg
    assert memory.get_previous_conversation() == expected_previous
    # Test that get_conversation returns all messages joined by newline.
    expected_conversation = "\n".join(messages)
    assert memory.get_conversation() == expected_conversation
    # Verify the count before and after clearing the memory.
    assert memory.count() == 3
    memory.clear()
    assert memory.count() == 0
    assert memory.get_conversation() == ""
def test_property_size_and_last_and_get_messages_with_limit():
    """
    Test the Memory property 'size', the last() method, and get_messages() when an explicit limit is provided.
    """
    # Create Memory with a memory_size of 5 (this is not the count of messages, but a parameter used as default limit)
    memory = Memory(memory_size=5)
    
    # Add three messages.
    memory.add("Message 1", is_user=True)   # user message
    memory.add("Message 2", is_user=False)  # assistant message
    memory.add("Message 3", is_user=True)   # user message
    
    # Test that the size property returns the provided memory_size
    assert memory.size == 5
    
    # Test that last() returns the last message in the list as a dictionary.
    expected_last_message = {"message": "Message 3", "is_user": True}
    assert memory.last() == expected_last_message
    # Test that get_messages with an explicit limit returns the last two messages.
    # The formatting for user messages: "### QUERY\n " + message.
    # The formatting for assistant messages: "### ANSWER\n " + (truncated message if longer than 100 characters).
    expected_messages = [
        "### ANSWER\n Message 2",
        "### QUERY\n Message 3",
    ]
    # Using limit=2 to get the last two added messages.
    assert memory.get_messages(limit=2) == expected_messages
def test_non_string_messages():
    """
    Test that non-string messages (e.g. integers) are handled correctly in formatted output and JSON conversion.
    """
    memory = Memory()
    # Add a non-string message for the assistant role.
    memory.add(987654321, is_user=False)
    # Add a non-string message for the user role.
    memory.add(12345, is_user=True)
    
    # Test get_messages with a limit so that both messages are returned.
    # The expected formatting for assistant messages: "### ANSWER\n " + message (or its truncated version if needed)
    # and for user messages: "### QUERY\n " + message.
    expected_assistant = "### ANSWER\n 987654321"
    expected_user = "### QUERY\n 12345"
    messages = memory.get_messages(limit=2)
    assert messages == [expected_assistant, expected_user]
    
    # Verify that to_json returns the messages with the non-string values intact.
    expected_json = [
        {"role": "assistant", "message": 987654321},
        {"role": "user", "message": 12345},
    ]
    assert memory.to_json() == expected_json
def test_truncate_exact_length():
    """
    Test that an assistant message exactly 100 characters long is not truncated.
    """
    memory = Memory()
    # Create a message with exactly 100 characters.
    exact_message = "A" * 100
    memory.add(exact_message, is_user=False)
    
    # Retrieve the last message using get_messages with limit=1.
    result = memory.get_messages(limit=1)
    expected = "### ANSWER\n " + exact_message  # since message is exactly 100 characters, it should not be truncated.
    
    assert result == [expected]
def test_memory_size_zero_returns_all_messages():
    """
    Test that when Memory is instantiated with a memory_size of 0,
    the get_messages method returns all the messages in the conversation.
    This is because slicing with -0 (i.e. list[-0:]) returns the entire list.
    """
    memory = Memory(memory_size=0)
    memory.add("Message 1", is_user=True)
    memory.add("Message 2", is_user=False)
    result = memory.get_messages()
    expected = [
        "### QUERY\n Message 1",
        "### ANSWER\n Message 2"
    ]
    assert result == expected
def test_last_empty_raises_index_error():
    """
    Test that calling last() on an empty Memory instance raises an IndexError.
    This verifies the behavior of the last() method when no messages have been added.
    """
    memory = Memory()
    with pytest.raises(IndexError):
        _ = memory.last()
def test_to_openai_messages_long_message_no_truncation():
    """
    Test that when a long assistant message is added, get_messages() returns a truncated version
    while to_openai_messages() returns the full untruncated message.
    """
    memory = Memory(memory_size=2)
    long_message = "B" * 150  # Create a message longer than 100 characters.
    memory.add("User query", is_user=True)
    memory.add(long_message, is_user=False)
    
    # Verify that get_messages returns the truncated version for the assistant message.
    messages = memory.get_messages()
    expected_truncated = "### ANSWER\n " + long_message[:100] + " ..."
    assert messages[1] == expected_truncated
    
    # Verify that to_openai_messages returns the full assistant message without truncation.
    expected_openai = [
        {"role": "user", "content": "User query"},
        {"role": "assistant", "content": long_message},
    ]
    assert memory.to_openai_messages() == expected_openai
def test_to_openai_messages_with_empty_agent_description():
    """
    Test that when Memory is instantiated with an empty agent description,
    the to_openai_messages method does not include a system message.
    This ensures that an empty string for agent_description is treated as falsy.
    """
    memory = Memory(agent_description="")
    memory.add("User saying hi", is_user=True)
    memory.add("Assistant replying", is_user=False)
    expected_messages = [
        {"role": "user", "content": "User saying hi"},
        {"role": "assistant", "content": "Assistant replying"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_get_messages_limit_exceeds_count():
    """
    Test get_messages when the provided limit exceeds the number of messages.
    Ensures that only available messages are returned without errors.
    """
    memory = Memory()
    # Add fewer messages than the requested limit
    memory.add("Test query", is_user=True)
    memory.add("Test answer", is_user=False)
    
    # Request a limit greater than the count of messages
    messages = memory.get_messages(limit=10)
    
    # Expected formatting: 
    # For user messages: "### QUERY\n " + message
    # For assistant messages: "### ANSWER\n " + message 
    expected = [
        "### QUERY\n Test query",
        "### ANSWER\n Test answer"
    ]
    assert messages == expected
def test_update_agent_description():
    """
    Test updating the agent description of a Memory instance after initialization.
    This verifies that changing the agent_description attribute causes the system message
    to be included in the output of to_openai_messages, while preserving previously added messages.
    """
    # Create a Memory instance without an initial agent description.
    memory = Memory()
    
    # Add messages before setting a system prompt.
    memory.add("Initial query", is_user=True)
    memory.add("Initial answer", is_user=False)
    
    # With no agent description set, to_openai_messages should not include a system message.
    expected_without_system = [
        {"role": "user", "content": "Initial query"},
        {"role": "assistant", "content": "Initial answer"}
    ]
    assert memory.to_openai_messages() == expected_without_system
    
    # Now update the agent description.
    memory.agent_description = "Updated system prompt"
    
    # Expected messages now should include the system message at the beginning.
    expected_with_system = [
        {"role": "system", "content": "Updated system prompt"},
        {"role": "user", "content": "Initial query"},
        {"role": "assistant", "content": "Initial answer"}
    ]
    assert memory.to_openai_messages() == expected_with_system
def test_get_messages_negative_limit():
    """
    Test get_messages with a negative limit value. In this scenario, a negative limit is passed,
    which causes Python's list slicing to behave as per its standard rules.
    For example, with three messages added and limit = -2, the slicing evaluates to self._messages[2:],
    returning only the last message.
    """
    memory = Memory()
    memory.add("Msg1", is_user=True)
    memory.add("Msg2", is_user=False)
    memory.add("Msg3", is_user=True)
    
    # For limit = -2, slicing becomes self._messages[2:], which should return only the third message.
    result = memory.get_messages(limit=-2)
    expected = ["### QUERY\n Msg3"]
    assert result == expected
def test_negative_memory_size_behavior():
    """
    Test that when Memory is instantiated with a negative memory_size,
    get_messages uses the negative value as the limit. In this case, the slicing
    returns only the messages from a specific index onward.
    """
    # Instantiate Memory with a negative memory_size.
    memory = Memory(memory_size=-2)
    
    # Add 5 user messages.
    messages_list = ["Msg1", "Msg2", "Msg3", "Msg4", "Msg5"]
    for msg in messages_list:
        memory.add(msg, is_user=True)
    
    # Here, memory_size is -2 so the default limit becomes -2.
    # Slicing: self._messages[-(-2):] = self._messages[2:], i.e. return messages from index 2 onward.
    expected = [
        "### QUERY\n Msg3",
        "### QUERY\n Msg4",
        "### QUERY\n Msg5",
    ]
    
    result = memory.get_messages()
    assert result == expected
def test_get_messages_with_zero_limit():
    """
    Test get_messages behavior when an explicit limit of 0 is provided.
    Since -0 is equivalent to 0 in Python slicing, the entire list of messages should be returned.
    """
    memory = Memory(memory_size=3)
    memory.add("Alpha", is_user=True)
    memory.add("Beta", is_user=False)
    memory.add("Gamma", is_user=True)
    
    expected = [
        "### QUERY\n Alpha",
        "### ANSWER\n Beta",
        "### QUERY\n Gamma"
    ]
    assert memory.get_messages(limit=0) == expected
def test_to_openai_messages_with_whitespace_agent_description():
    """
    Test that when Memory is instantiated with a whitespace agent_description,
    the to_openai_messages method includes the system message with the whitespace value.
    """
    memory = Memory(agent_description="   ")
    memory.add("User asks", is_user=True)
    memory.add("Assistant replies", is_user=False)
    expected_messages = [
        {"role": "system", "content": "   "},
        {"role": "user", "content": "User asks"},
        {"role": "assistant", "content": "Assistant replies"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_non_string_agent_description():
    """
    Test that agent_description accepts a non-string value (e.g. an integer)
    and the to_openai_messages method includes a system message with the non-string value unmodified.
    """
    memory = Memory(agent_description=1234)
    memory.add("User test", is_user=True)
    memory.add("Assistant test", is_user=False)
    expected_messages = [
        {"role": "system", "content": 1234},
        {"role": "user", "content": "User test"},
        {"role": "assistant", "content": "Assistant test"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_default_memory_size_behavior():
    """
    Test that when Memory is initialized with the default memory_size (1),
    get_messages returns only the last message even if more messages are added.
    """
    # Default memory_size is 1
    memory = Memory()
    memory.add("First message", is_user=True)
    memory.add("Second message", is_user=False)
    memory.add("Third message", is_user=True)
    
    # Since memory_size is 1, only the last message should be returned.
    expected = ["### QUERY\n Third message"]
    result = memory.get_messages()
    
    assert result == expected
def test_all_method_returns_correct_messages():
    """
    Test that the all() method returns the raw list of messages added in the correct order.
    """
    memory = Memory()
    messages_to_add = [
        ("Test query", True),
        ("Test answer", False),
        ("Another query", True)
    ]
    for msg, is_user in messages_to_add:
        memory.add(msg, is_user=is_user)
    
    expected = [
        {"message": "Test query", "is_user": True},
        {"message": "Test answer", "is_user": False},
        {"message": "Another query", "is_user": True},
    ]
    
    assert memory.all() == expected
def test_none_message_handling():
    """
    Test that when None is added as a message for both user and assistant,
    the get_messages and to_json methods handle it correctly by converting
    None to its string representation where necessary and preserving None in JSON output.
    """
    memory = Memory()
    # Add None as message for a user and an assistant.
    memory.add(None, is_user=True)
    memory.add(None, is_user=False)
    
    # Test get_messages. For the assistant, _truncate is applied which converts None to "None" at formatting.
    expected_messages = [
        "### QUERY\n None",
        "### ANSWER\n None"
    ]
    messages = memory.get_messages(limit=2)
    assert messages == expected_messages
    
    # Test to_json conversion; it should preserve the None values.
    expected_json = [
        {"role": "user", "message": None},
        {"role": "assistant", "message": None},
    ]
    assert memory.to_json() == expected_json
def test_get_last_message_empty():
    """
    Test that get_last_message returns an empty string when the Memory instance has no messages.
    """
    memory = Memory()
    assert memory.get_last_message() == ""
def test_to_openai_messages_only_system_message():
    """
    Test that when Memory is instantiated with a non-empty agent_description and no conversation messages,
    to_openai_messages returns a list containing only the system message.
    """
    memory = Memory(agent_description="Only system message test")
    # No messages are added to memory.
    expected = [{"role": "system", "content": "Only system message test"}]
    assert memory.to_openai_messages() == expected
def test_clear_then_add_messages():
    """
    Test that clearing the memory resets conversation history and allows new messages
    to be added correctly afterward.
    """
    memory = Memory(memory_size=5)
    # Add some initial messages.
    memory.add("Initial user message", is_user=True)
    memory.add("Initial assistant message", is_user=False)
    # Verify that memory is not empty.
    assert memory.count() == 2
    assert memory.get_conversation() != ""
    
    # Clear the memory.
    memory.clear()
    # Ensure conversation history is cleared.
    assert memory.count() == 0
    assert memory.get_conversation() == ""
    
    # Add new messages after clearing.
    memory.add("New user message", is_user=True)
    memory.add("New assistant message", is_user=False)
    expected_messages = [
        "### QUERY\n New user message",
        "### ANSWER\n New assistant message"
    ]
    # Verify that the new conversation is as expected.
    assert memory.get_conversation() == "\n".join(expected_messages)
def test_clear_preserves_agent_description():
    """
    Test that clearing the memory resets the conversation messages
    but preserves the agent description. After clearing, to_openai_messages
    should return only the system message if agent_description is set.
    """
    memory = Memory(memory_size=3, agent_description="My agent")
    memory.add("User msg", is_user=True)
    memory.add("Assistant msg", is_user=False)
    # Verify that messages are added.
    assert memory.count() == 2
    # Clear the memory messages.
    memory.clear()
    # After clearing, messages should be empty.
    assert memory.count() == 0
    # The agent_description should still be present, so to_openai_messages returns just the system message.
    expected = [{"role": "system", "content": "My agent"}]
    assert memory.to_openai_messages() == expected
def test_user_message_no_truncation_for_long_message():
    """
    Test that a long user message is not truncated by the get_messages method.
    User messages should always display in full, even if they exceed the 100 character limit.
    """
    memory = Memory()
    long_user_message = "X" * 150  # Create a long message of 150 characters.
    memory.add(long_user_message, is_user=True)
    
    # Since the message is from the user, no truncation should be applied.
    expected_formatted = "### QUERY\n " + long_user_message
    
    # Using limit=1 to ensure only the last message (which is our long user message) is returned.
    result = memory.get_messages(limit=1)
    assert result == [expected_formatted]
def test_non_string_complex_types():
    """
    Test that complex non-string types (e.g., a dict for an assistant message and a list for a user message)
    are handled correctly. Verifies that get_messages returns their string representations while to_json preserves the original types.
    """
    memory = Memory()
    complex_assistant = {'key': 'value'}
    complex_user = ['list', 'item']
    memory.add(complex_assistant, is_user=False)
    memory.add(complex_user, is_user=True)
    
    expected_assistant_msg = "### ANSWER\n " + str(complex_assistant)
    expected_user_msg = "### QUERY\n " + str(complex_user)
    messages = memory.get_messages(limit=2)
    assert messages == [expected_assistant_msg, expected_user_msg]
    
    expected_json = [
        {"role": "assistant", "message": complex_assistant},
        {"role": "user", "message": complex_user},
    ]
    assert memory.to_json() == expected_json
def test_dynamic_memory_size_update():
    """
    Test that updating the _memory_size attribute after adding messages changes the default limit
    for get_messages, ensuring that the conversation retrieval reflects the new memory size.
    """
    memory = Memory(memory_size=3)
    messages_list = [
        ("msg1", True),
        ("msg2", False),
        ("msg3", True),
        ("msg4", False),
        ("msg5", True)
    ]
    for msg, is_user in messages_list:
        memory.add(msg, is_user)
    
    # Initially, memory_size is 3 so get_messages() returns the last 3 messages.
    expected_default = [
        "### QUERY\n msg3",
        "### ANSWER\n msg4",
        "### QUERY\n msg5"
    ]
    assert memory.get_messages() == expected_default
    
    # Update the _memory_size attribute to a new value and verify the change.
    memory._memory_size = 2
    expected_updated = [
        "### ANSWER\n msg4",
        "### QUERY\n msg5"
    ]
    assert memory.get_messages() == expected_updated
    assert memory.get_conversation() == "\n".join(expected_updated)
def test_get_previous_conversation_single_message():
    """
    Test that get_previous_conversation returns an empty string when there is only one message
    in the conversation.
    """
    memory = Memory()
    memory.add("Only one message", is_user=True)
    # With only one message added, there is no previous conversation.
    assert memory.get_previous_conversation() == ""
def test_to_openai_messages_with_false_agent_description():
    """
    Test that when Memory is instantiated with agent_description set to False,
    the to_openai_messages method does not include a system message.
    This verifies that falsy agent_description values (other than empty strings already tested)
    prevent the system message from being added.
    """
    memory = Memory(agent_description=False)
    memory.add("Test user message", is_user=True)
    memory.add("Test assistant message", is_user=False)
    
    # Since agent_description is False (which is falsy),
    # the system message should not be included.
    expected_messages = [
        {"role": "user", "content": "Test user message"},
        {"role": "assistant", "content": "Test assistant message"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_get_messages_negative_limit_excessive():
    """
    Test that get_messages returns an empty list when a negative limit exceeds
    the number of messages in the conversation.
    For example, with one message and limit = -10, slicing (self._messages[10:])
    should yield an empty list.
    """
    memory = Memory()
    memory.add("Only message", is_user=True)
    
    # With limit = -10, slicing becomes self._messages[10:], which should be empty.
    result = memory.get_messages(limit=-10)
    assert result == []
def test_all_list_reference_modification():
    """
    Test that the all() method returns the internal list of messages.
    Any modifications to the returned list should reflect in the Memory instance.
    """
    memory = Memory()
    # Add one initial message.
    memory.add("Initial message", is_user=True)
    
    # Retrieve the internal messages list.
    all_messages = memory.all()
    
    # Modify the returned list externally.
    all_messages.append({"message": "Injected message", "is_user": False})
    
    # Since all_messages is the internal list, the count should now be 2.
    assert memory.count() == 2
    
    # Verify that the newly injected message is present along with the original one.
    expected = [
        {"message": "Initial message", "is_user": True},
        {"message": "Injected message", "is_user": False}
    ]
    assert memory.all() == expected
def test_to_openai_messages_with_zero_agent_description():
    """
    Test that when Memory is instantiated with agent_description set to 0,
    the to_openai_messages method does not include a system message since 0 is falsy.
    """
    memory = Memory(agent_description=0)
    memory.add("User message", is_user=True)
    memory.add("Assistant message", is_user=False)
    expected = [
         {"role": "user", "content": "User message"},
         {"role": "assistant", "content": "Assistant message"},
    ]
    assert memory.to_openai_messages() == expected
def test_get_messages_invalid_limit_type():
    """
    Test that get_messages raises a TypeError when an invalid limit type (non-integer)
    is provided. This verifies that the function does not accept non-numeric limits,
    as slicing with a non-integer value is unsupported.
    """
    memory = Memory()
    memory.add("Test message", is_user=True)
    with pytest.raises(TypeError):
        # Passing a string as limit should cause a TypeError when used in slicing.
        memory.get_messages(limit="invalid")
def test_invalid_memory_size_type():
    """
    Test that initializing Memory with a non-integer memory_size (e.g., a string)
    results in a TypeError when get_messages is called due to invalid slicing.
    """
    memory = Memory(memory_size="five")
    memory.add("Test message", is_user=True)
    with pytest.raises(TypeError):
        memory.get_messages()
def test_get_conversation_empty_with_agent_description():
    """
    Test that get_conversation returns an empty string when no conversation messages
    have been added, even if an agent_description is provided.
    This ensures that the system message (agent_description) is not included in
    the conversation output.
    """
    memory = Memory(agent_description="Test system prompt")
    # Even though agent_description is provided, get_conversation should only join conversation messages.
    assert memory.get_conversation() == ""
def test_message_with_newlines_formatting():
    """
    Test that messages containing newline characters are handled correctly in the formatted output.
    Verifies that the get_messages method preserves newline characters within the message content.
    """
    memory = Memory()
    user_msg = "Hello\nWorld"
    assistant_msg = "Response\nline2"
    memory.add(user_msg, is_user=True)
    memory.add(assistant_msg, is_user=False)
    
    expected = [
        "### QUERY\n " + user_msg,
        "### ANSWER\n " + assistant_msg
    ]
    
    result = memory.get_messages(limit=2)
    assert result == expected
def test_get_messages_float_limit_raises_type_error():
    """
    Test that get_messages raises a TypeError when a float is passed as the limit.
    This ensures that only integer limits are accepted for slicing.
    """
    memory = Memory()
    memory.add("Test message", is_user=True)
    with pytest.raises(TypeError):
        # Passing a float as limit should cause a TypeError when used in slicing.
        memory.get_messages(limit=1.5)
def test_assistant_message_borderline_truncation():
    """
    Test that an assistant message exactly 101 characters long is truncated to the first 100 characters 
    followed by " ...". This verifies the _truncate method's behavior for messages just over the limit.
    """
    memory = Memory()
    # Create a message with exactly 101 characters.
    message_101 = "B" * 101
    memory.add(message_101, is_user=False)
    # Retrieve the message using get_messages with limit=1.
    result = memory.get_messages(limit=1)
    # Expected truncated message: first 100 characters + " ..." with proper formatting.
    expected_truncated = "B" * 100 + " ..."
    expected_output = "### ANSWER\n " + expected_truncated
    assert result == [expected_output]
def test_custom_object_message_formatting():
    """
    Test that messages of a custom object type with a __str__ method are handled correctly.
    - The get_messages method returns their string representations.
    - The to_json method preserves the original object types.
    """
    memory = Memory(memory_size=2)
    dummy_obj = DummyObject()
    # Add the dummy object as a user message and as an assistant message
    memory.add(dummy_obj, is_user=True)
    memory.add(dummy_obj, is_user=False)
    
    # For a user message, get_messages should call str(dummy_obj)
    expected_messages = [
        "### QUERY\n " + str(dummy_obj),
        "### ANSWER\n " + str(dummy_obj)
    ]
    messages = memory.get_messages(limit=2)
    assert messages == expected_messages
    
    # The to_json method should preserve the original object references in the messages.
    expected_json = [
        {"role": "user", "message": dummy_obj},
        {"role": "assistant", "message": dummy_obj},
    ]
    assert memory.to_json() == expected_json
def test_get_conversation_ignores_system_message():
    """
    Test that get_conversation returns only the actual conversation messages and
    does not include the system message provided by agent_description.
    This verifies that get_conversation is solely focused on the user and assistant messages.
    """
    memory = Memory(memory_size=3, agent_description="Test System")
    memory.add("User message one", is_user=True)
    memory.add("Assistant message one", is_user=False)
    memory.add("User message two", is_user=True)
    
    expected_messages = [
        "### QUERY\n User message one",
        "### ANSWER\n Assistant message one",
        "### QUERY\n User message two"
    ]
    expected_conversation = "\n".join(expected_messages)
    
    # get_conversation should only join the added conversation messages, ignoring the agent_description.
    assert memory.get_conversation() == expected_conversation
def test_get_conversation_with_explicit_limit():
    """
    Test that get_conversation returns the conversation string with the specified limit,
    overriding the default memory_size. This ensures that the explicit limit parameter is
    used correctly in joining the last messages.
    """
    # Initialize Memory with a default memory_size of 5.
    memory = Memory(memory_size=5)
    # Add a series of messages.
    messages = [
        ("Msg1", True),
        ("Msg2", False),
        ("Msg3", True),
        ("Msg4", False)
    ]
    for msg, is_user in messages:
        memory.add(msg, is_user)
    
    # When an explicit limit of 2 is provided, the last two messages should be returned.
    # The formatted values:
    # For "Msg3" (user): "### QUERY\n Msg3"
    # For "Msg4" (assistant): "### ANSWER\n Msg4" (no truncation needed as length is short)
    expected_conversation = "\n".join([
        "### QUERY\n Msg3",
        "### ANSWER\n Msg4"
    ])
    
    # Verify that get_conversation with an explicit limit of 2 returns the expected string.
    assert memory.get_conversation(limit=2) == expected_conversation
def test_update_agent_description_to_none():
    """
    Test that updating agent_description from a non-empty value to None
    removes the system message from the output of to_openai_messages.
    Initially, the system message is included when agent_description is truthy,
    and once updated to None, only conversation messages are returned.
    """
    # Initialize Memory with an agent description and add messages.
    memory = Memory(agent_description="Initial system prompt")
    memory.add("User message", is_user=True)
    memory.add("Assistant message", is_user=False)
    
    # Verify that the system message is included.
    expected_with_system = [
        {"role": "system", "content": "Initial system prompt"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]
    assert memory.to_openai_messages() == expected_with_system
    # Update agent_description to None.
    memory.agent_description = None
    # The system message should be removed from the OpenAI messages.
    expected_without_system = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]
    assert memory.to_openai_messages() == expected_without_system