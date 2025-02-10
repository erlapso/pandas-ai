from pandasai.helpers.memory import Memory
import pytest

# No additional imports needed; using the existing:
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
def test_memory_full_functionality():
    """
    Test several functions of the Memory class:
    - Adding messages and verifying count and order.
    - Truncation of long assistant messages.
    - Retrieval of messages via get_messages, get_conversation, get_previous_conversation, and get_last_message.
    - Clearing memory.
    """
    # Create a Memory instance with memory_size=3 and an agent description.
    memory = Memory(memory_size=3, agent_description="Test agent")
    
    # Add messages to memory.
    memory.add("User message", True)
    long_answer = "A" * 150  # Create a long message (150 characters) for testing truncation.
    memory.add(long_answer, False)
    memory.add("Another user query", True)
    
    # Check that the count method returns the correct number of messages.
    assert memory.count() == 3
    
    # Check that the last method returns the correct (last) message entry.
    last_entry = memory.last()
    assert last_entry == {"message": "Another user query", "is_user": True}
    
    # Test get_last_message for proper formatting and content.
    last_message_text = memory.get_last_message()
    # The last message is a user message so it should be prefixed with "### QUERY" and contain the query.
    assert "### QUERY" in last_message_text
    assert "Another user query" in last_message_text
    
    # Test get_messages with an explicit limit.
    messages_limited = memory.get_messages(limit=2)
    # For the assistant message, due to its length it should be truncated.
    truncated_long_answer = "A" * 100 + " ..."
    expected_limited = [
        f"### ANSWER\n {truncated_long_answer}",
        "### QUERY\n Another user query"
    ]
    assert messages_limited == expected_limited
    
    # Test get_conversation default behavior (using memory_size = 3).
    conversation = memory.get_conversation()
    expected_conversation = "\n".join(memory.get_messages())
    assert conversation == expected_conversation
    
    # Test get_previous_conversation which should exclude the last message when more than one exists.
    previous_conversation = memory.get_previous_conversation()
    messages_all = memory.get_messages()
    expected_previous = "" if len(messages_all) <= 1 else "\n".join(messages_all[:-1])
    assert previous_conversation == expected_previous
    
    # Test clear method to ensure that it empties the memory.
    memory.clear()
    assert memory.count() == 0
    assert memory.all() == []
def test_truncate_edge_case():
    """
    Test that an assistant message with exactly 100 characters is not truncated.
    """
    memory = Memory(memory_size=1)
    exact_message = "A" * 100  # Create exactly 100-character message.
    memory.add(exact_message, is_user=False)
    result = memory.get_messages()[0]
    expected = f"### ANSWER\n {exact_message}"
    assert result == expected
def test_empty_memory_conversation_methods():
    """
    Test conversation retrieval methods for an empty Memory instance.
    This verifies that get_conversation, get_previous_conversation, get_last_message,
    and get_messages return empty outputs when no messages have been added.
    """
    memory = Memory()
    assert memory.get_conversation() == ""
    assert memory.get_previous_conversation() == ""
    assert memory.get_last_message() == ""
    assert memory.get_messages() == []
def test_memory_size_property_and_default_limit():
    """
    Test that the Memory.size property returns the memory_size provided at initialization
    and that get_messages() without a limit argument defaults to using self._memory_size.
    This also verifies that even if more messages are added than the memory_size,
    only the most recent memory_size number of messages are returned by get_messages().
    """
    memory = Memory(memory_size=2)
    memory.add("User message 1", True)
    memory.add("Assistant reply 1", False)
    memory.add("User message 2", True)
    
    assert memory.size == 2
    
    messages = memory.get_messages()
    expected_messages = [
        "### ANSWER\n Assistant reply 1",
        "### QUERY\n User message 2",
    ]
    assert messages == expected_messages
    assert memory.get_conversation() == "\n".join(expected_messages)
def test_last_on_empty_memory():
    """
    Test that calling last() on an empty Memory instance raises an IndexError.
    """
    memory = Memory()
    with pytest.raises(IndexError):
        memory.last()
def test_get_previous_conversation_single_message():
    """
    Test that get_previous_conversation returns an empty string when exactly one message is present.
    """
    memory = Memory(memory_size=5)
    memory.add("Single user message", is_user=True)
    assert memory.get_previous_conversation() == ""
def test_get_messages_limit_over_count():
    """
    Test that get_messages with a limit greater than the number of stored messages returns all messages.
    """
    memory = Memory(memory_size=5)
    memory.add("Test query", is_user=True)
    memory.add("Test answer", is_user=False)
    messages = memory.get_messages(limit=10)
    expected_messages = [
        "### QUERY\n Test query",
        "### ANSWER\n Test answer",
    ]
    assert messages == expected_messages
def test_truncate_non_string_message_raises_error():
    """
    Test that _truncate raises a TypeError when given a non-string message that exceeds max_length.
    This simulates a scenario where the message is not a string and verifies that the slicing operation
    (which expects a string) fails as designed.
    """
    memory = Memory()
    large_int = int("1" * 150)
    with pytest.raises(TypeError):
        memory._truncate(large_int, max_length=100)
def test_to_openai_messages_empty_agent_description():
    """
    Test that to_openai_messages does not include a system message when the agent_description is an empty string.
    This verifies that an empty agent description does not result in a system message,
    keeping the message list limited to the user and assistant messages.
    """
    memory = Memory(agent_description="")
    memory.add("User message", is_user=True)
    memory.add("Assistant reply", is_user=False)
    expected_messages = [
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant reply"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_get_messages_with_zero_limit():
    """
    Test that get_messages returns all messages when limit is set to 0.
    Even though 0 might be seen as "no messages", slicing with -0 in Python
    returns the entire list of messages.
    """
    memory = Memory(memory_size=3)
    memory.add("First user message", True)
    memory.add("First assistant reply", False)
    memory.add("Second user message", True)
    expected = [
        "### QUERY\n First user message",
        "### ANSWER\n First assistant reply",
        "### QUERY\n Second user message",
    ]
    assert memory.get_messages(limit=0) == expected
def test_to_openai_messages_empty_with_agent_description():
    """
    Test that to_openai_messages returns only the system message when no other messages are added
    but when an agent_description is provided.
    """
    memory = Memory(agent_description="I am your friendly assistant")
    expected_messages = [{"role": "system", "content": "I am your friendly assistant"}]
    assert memory.to_openai_messages() == expected_messages
def test_clear_does_not_affect_agent_description():
    """
    Test that the clear() method removes all conversation messages but does not affect the agent_description.
    """
    memory = Memory(agent_description="Test agent description")
    memory.add("Message 1", True)
    memory.add("Message 2", False)
    
    memory.clear()
    assert memory.all() == []
    assert memory.agent_description == "Test agent description"
def test_to_json_immutable():
    """
    Test that modifications to the JSON output from to_json() do not affect the internal memory state.
    This verifies that a new list is created on each call to to_json().
    """
    memory = Memory()
    memory.add("Immutable test", is_user=True)
    original_json = memory.to_json()
    
    # Modify the returned JSON list
    original_json.append({"role": "assistant", "message": "Modified!"})
    
    # The internal state should remain unchanged.
    expected = [{"role": "user", "message": "Immutable test"}]
    assert memory.to_json() == expected
def test_empty_message_additions():
    """
    Test that adding empty strings for both user and assistant messages is handled properly.
    Verifies that the get_messages() method returns the correct formatting with prefixes even when the message text is empty.
    """
    memory = Memory(memory_size=5)
    memory.add("", True)
    memory.add("", False)
    expected = [
        "### QUERY\n ",
        "### ANSWER\n ",
    ]
    assert memory.get_messages() == expected
def test_get_messages_with_negative_limit():
    """
    Test that get_messages with a negative limit returns the slice of messages
    starting with the absolute index. For example, if limit=-2 is provided,
    then the last 2 messages (self._messages[2:]) are returned.
    """
    # Create a Memory instance with any memory_size (will not be used because limit is provided)
    memory = Memory(memory_size=5)
    
    # Add 4 messages with alternating user and assistant roles.
    memory.add("User message 1", True)          # index 0
    memory.add("Assistant reply 1", False)        # index 1
    memory.add("User message 2", True)            # index 2
    memory.add("Assistant reply 2", False)        # index 3
    
    # When a negative limit -2 is provided, negative slicing works as:
    # self._messages[-(-2):] == self._messages[2:], which should correspond to the last two messages.
    expected = [
         "### QUERY\n User message 2",
         "### ANSWER\n Assistant reply 2",
    ]
    result = memory.get_messages(limit=-2)
    assert result == expected
def test_memory_with_zero_memory_size():
    """
    Test that if Memory is initialized with memory_size=0,
    then get_messages() returns all added messages (since using -0 slicing yields the entire list).
    """
    memory = Memory(memory_size=0)
    memory.add("User message", is_user=True)
    memory.add("Assistant reply", is_user=False)
    expected_messages = [
        "### QUERY\n User message",
        "### ANSWER\n Assistant reply"
    ]
    # Because get_messages() uses -0 slicing when memory_size is 0,
    # the entire list of messages is returned.
    assert memory.get_messages() == expected_messages
def test_non_string_user_message():
    """
    Test that non-string user messages are properly handled by get_messages()
    by converting them to their string representation.
    """
    memory = Memory()
    # Add an integer as a user message
    memory.add(12345, is_user=True)
    # The expected output should cast the integer to a string in the message output.
    expected = "### QUERY\n 12345"
    assert memory.get_messages() == [expected]
def test_non_string_assistant_message():
    """
    Test that non-string assistant messages are properly handled by get_messages().
    Specifically, it verifies that when an assistant message is a non-string value (e.g., an integer)
    and its string representation length is within the max_length, the message is returned correctly.
    """
    memory = Memory()
    # Add a non-string assistant message (integer), which should be converted to string in output.
    memory.add(9, is_user=False)
    expected = "### ANSWER\n 9"
    assert memory.get_messages() == [expected]
def test_get_messages_negative_limit_exceeding_count():
    """
    Test that get_messages with a negative limit whose absolute value exceeds 
    the number of stored messages returns an empty list.
    """
    memory = Memory()
    memory.add("Message 1", True)
    memory.add("Message 2", False)
    # When limit is -5, slicing the messages list (which contains 2 messages) as:
    # self._messages[-(-5):] -> self._messages[5:] gives an empty list.
    result = memory.get_messages(limit=-5)
    expected = []
    assert result == expected
def test_non_string_assistant_message_with_list_truncation():
    """
    Test that a non-string assistant message provided as a list that, when converted to a string,
    exceeds the maximum length is properly truncated. The expected output is the string representation
    of the first max_length items of the list followed by an ellipsis.
    """
    # Create a long list with 150 integers.
    long_list = list(range(150))
    memory = Memory()
    # Add the long list as an assistant message.
    memory.add(long_list, is_user=False)
    
    # Getting messages will call _truncate() on the assistant message.
    # Compute the expected truncated part:
    # Slicing the list returns the first 100 elements.
    truncated_part = long_list[:100]
    # When converted to string inside f-string, Python will call str() on the list slice.
    expected_message = f"### ANSWER\n {str(truncated_part)} ..."
    
    # Retrieve messages from Memory.
    result = memory.get_messages()
    assert result == [expected_message]
def test_negative_memory_size_behavior():
    """
    Test that Memory initialized with a negative memory_size uses it directly for slicing.
    For example, if memory_size is -2 and 4 messages are added, get_messages() should return
    messages starting from index 2 (i.e. messages at indexes 2 and 3).
    """
    memory = Memory(memory_size=-2)
    memory.add("User 1", True)          # index 0
    memory.add("Assistant 1", False)     # index 1
    memory.add("User 2", True)           # index 2
    memory.add("Assistant 2", False)     # index 3
    # Since memory_size is -2, get_messages() uses limit = -2, which means:
    # self._messages[-(-2):] is self._messages[2:], returning the last 2 messages.
    expected = [
        "### QUERY\n User 2",
        "### ANSWER\n Assistant 2",
    ]
    assert memory.get_messages() == expected
def test_changing_agent_description():
    """
    Test that updating the agent_description property after adding messages
    influences the output of to_openai_messages. Initially, no system message is present,
    but after updating agent_description, the system message should be prepended.
    """
    memory = Memory()  # Initially, agent_description is None
    memory.add("Test query", is_user=True)
    memory.add("Test answer", is_user=False)
    
    # Verify that initially, no system message is present in the OpenAI messages.
    result_initial = memory.to_openai_messages()
    expected_initial = [
        {"role": "user", "content": "Test query"},
        {"role": "assistant", "content": "Test answer"},
    ]
    assert result_initial == expected_initial
    
    # Now update the agent_description property.
    memory.agent_description = "Updated system info"
    
    # Now, to_openai_messages() should prepend the system message.
    result_updated = memory.to_openai_messages()
    expected_updated = [
        {"role": "system", "content": "Updated system info"},
        {"role": "user", "content": "Test query"},
        {"role": "assistant", "content": "Test answer"},
    ]
    assert result_updated == expected_updated
def test_all_returns_mutable_reference():
    """
    Test that the list returned by all() is a reference to the internal memory.
    Modifying the returned list should affect the memory's internal state.
    """
    # Create a Memory instance and add a message.
    memory = Memory()
    memory.add("Hello", is_user=True)
    
    # Retrieve the internal list using the all() method.
    messages_ref = memory.all()
    
    # Modify the retrieved list by appending a new message.
    messages_ref.append({"message": "Injected message", "is_user": False})
    
    # The change should reflect in the internal count.
    assert memory.count() == 2
    
    # Additionally, verify that the last message is the one that was injected.
    last_message = memory.last()
    assert last_message == {"message": "Injected message", "is_user": False}
def test_to_openai_messages_non_string_agent_description():
    """
    Test that if agent_description is non-string, it is included as is in the system message
    of the output from to_openai_messages.
    """
    memory = Memory(agent_description=12345)
    memory.add("Hello", is_user=True)
    memory.add("Hi there!", is_user=False)
    expected_messages = [
        {"role": "system", "content": 12345},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_get_conversation_with_negative_limit():
    """
    Test that get_conversation correctly returns a newline-joined string of the messages
    when a negative limit is provided. Negative limits translate to slicing starting from
    the given absolute index.
    """
    memory = Memory(memory_size=5)
    # Add four messages in alternating roles.
    memory.add("User message 1", is_user=True)          # index 0
    memory.add("Assistant reply 1", is_user=False)       # index 1
    memory.add("User message 2", is_user=True)           # index 2
    memory.add("Assistant reply 2", is_user=False)       # index 3
    
    # With limit = -2, get_messages() should return the last two messages (index 2 and 3)
    expected_messages = [
        "### QUERY\n User message 2",
        "### ANSWER\n Assistant reply 2",
    ]
    # get_conversation should join these messages separated by newline characters.
    expected_conversation = "\n".join(expected_messages)
    result_conversation = memory.get_conversation(limit=-2)
    assert result_conversation == expected_conversation
def test_update_agent_description_to_empty():
    """
    Test that updating the agent_description to an empty string after messages have been added
    removes the system message from the output of to_openai_messages.
    """
    # Initialize with a non-empty agent description and add messages.
    memory = Memory(agent_description="Initial description")
    memory.add("Test query", is_user=True)
    memory.add("Test answer", is_user=False)
    
    # Verify that initially the system message is included.
    expected_initial = [
        {"role": "system", "content": "Initial description"},
        {"role": "user", "content": "Test query"},
        {"role": "assistant", "content": "Test answer"},
    ]
    assert memory.to_openai_messages() == expected_initial
    
    # Update the agent_description to an empty string.
    memory.agent_description = ""
    
    # Now, the system message should no longer be part of the output.
    expected_updated = [
        {"role": "user", "content": "Test query"},
        {"role": "assistant", "content": "Test answer"},
    ]
    assert memory.to_openai_messages() == expected_updated
def test_truncate_message_length_101():
    """
    Test that an assistant message with exactly 101 characters is properly truncated.
    The expected behavior is that the message is truncated to the first 100 characters followed by an ellipsis.
    """
    memory = Memory()
    # Create an assistant message with 101 "A" characters.
    long_message = "A" * 101
    memory.add(long_message, is_user=False)
    
    # Expected truncated message: first 100 characters + " ..."
    expected_truncated = "A" * 100 + " ..."
    expected_output = f"### ANSWER\n {expected_truncated}"
    
    # get_messages() should return the truncated version.
    assert memory.get_messages() == [expected_output]
def test_update_memory_size_affects_get_messages():
    """
    Test that updating the memory_size attribute after adding messages
    changes the default number of messages returned by get_messages() when no limit is provided.
    """
    # Create a Memory instance with memory_size=3
    memory = Memory(memory_size=3)
    # Add 4 messages
    memory.add("Msg 1", True)   # index 0: user
    memory.add("Msg 2", False)  # index 1: assistant
    memory.add("Msg 3", True)   # index 2: user
    memory.add("Msg 4", False)  # index 3: assistant
    # With memory_size=3, get_messages() returns the last 3 messages (indices 1,2,3)
    expected_initial = [
        "### ANSWER\n Msg 2",
        "### QUERY\n Msg 3",
        "### ANSWER\n Msg 4",
    ]
    assert memory.get_messages() == expected_initial
    # Now update memory_size to 2 so that only the last 2 messages are returned by default.
    memory._memory_size = 2
    expected_updated = [
        "### QUERY\n Msg 3",
        "### ANSWER\n Msg 4",
    ]
    assert memory.get_messages() == expected_updated
def test_get_messages_with_non_integer_limit():
    """
    Test that providing a non-integer limit (e.g., a float) to get_messages() raises a TypeError,
    since list slicing requires an integer index.
    """
    memory = Memory(memory_size=3)
    memory.add("Test message", True)
    with pytest.raises(TypeError):
        memory.get_messages(limit=2.5)
def test_update_agent_description_to_none():
    """
    Test that updating the agent_description to None after messages have been added
    removes the system message from the output of to_openai_messages.
    """
    # Initialize with a non-None agent_description and add messages.
    memory = Memory(agent_description="Initial system message")
    memory.add("User question", is_user=True)
    memory.add("Assistant answer", is_user=False)
    
    # Verify that initially the system message is included.
    expected_initial = [
        {"role": "system", "content": "Initial system message"},
        {"role": "user", "content": "User question"},
        {"role": "assistant", "content": "Assistant answer"},
    ]
    assert memory.to_openai_messages() == expected_initial
    
    # Now update the agent_description to None and verify the system message is removed.
    memory.agent_description = None
    expected_updated = [
        {"role": "user", "content": "User question"},
        {"role": "assistant", "content": "Assistant answer"},
    ]
    assert memory.to_openai_messages() == expected_updated
def test_get_messages_negative_limit_exact_count():
    """
    Test that get_messages with a negative limit equal to the number of stored messages returns an empty list.
    This verifies that slicing in get_messages behaves as expected when the absolute value of the negative limit
    matches the count of messages.
    """
    # Create a Memory instance (the memory_size is not used since an explicit limit is provided).
    memory = Memory(memory_size=10)
    # Add two messages.
    memory.add("User message", True)
    memory.add("Assistant reply", False)
    # There are exactly 2 messages; a limit of -2 leads to slicing: self._messages[-(-2):] == self._messages[2:]
    # which will return an empty list since there are no messages starting from index 2.
    result = memory.get_messages(limit=-2)
    assert result == []
def test_assistant_complex_non_string_message():
    """
    Test that a complex non-string assistant message (a dictionary) is correctly handled:
    - get_messages() should output the string conversion of the message (using no truncation when short),
    - to_openai_messages() should include the original dictionary type.
    """
    memory = Memory()
    complex_msg = {"key": "value"}
    memory.add(complex_msg, is_user=False)
    
    # For get_messages, message is for an assistant so _truncate is applied.
    # Its string representation is expected to be within max_length, hence no truncation occurs.
    expected_get = f"### ANSWER\n {str(complex_msg)}"
    assert memory.get_messages() == [expected_get]
    
    # For to_openai_messages, the assistant message should be included as is.
    expected_openai = [{"role": "assistant", "content": complex_msg}]
    assert memory.to_openai_messages() == expected_openai
def test_multiline_messages_preserved():
    """
    Test that multi-line messages are preserved correctly in the output of get_messages.
    Verifies that both user and assistant messages containing newline characters are returned with the proper prefixes,
    and that no additional modifications occur to the newline characters.
    """
    memory = Memory(memory_size=5)
    user_message = "User line1\nUser line2"
    assistant_message = "Assistant line1\nAssistant line2"
    
    memory.add(user_message, is_user=True)
    memory.add(assistant_message, is_user=False)
    
    expected_user = f"### QUERY\n {user_message}"
    expected_assistant = f"### ANSWER\n {assistant_message}"
    expected_messages = [expected_user, expected_assistant]
    
    assert memory.get_messages() == expected_messages
def test_truncate_with_tuple_message():
    """
    Test that an assistant message provided as a tuple (a non-string iterable) that, when converted to a string,
    exceeds the max_length, undergoes proper truncation.
    
    The expected behavior is that the message is truncated to the first 100 elements (using slicing)
    followed by an ellipsis, and returned with the "### ANSWER" prefix.
    """
    # Create a tuple with 150 integers.
    long_tuple = tuple(range(150))
    memory = Memory()
    memory.add(long_tuple, is_user=False)
    
    # When converting a tuple to str, its length might exceed 100 characters.
    # According to _truncate, the returned string should be:
    # f"{long_tuple[:100]} ..." if len(str(long_tuple)) > 100 else long_tuple
    expected_truncated = f"{long_tuple[:100]} ..."
    expected_message = f"### ANSWER\n {expected_truncated}"
    
    result = memory.get_messages()
    assert result == [expected_message]
def test_to_openai_messages_agent_description_whitespace():
    """
    Test that to_openai_messages includes a system message when agent_description consists of whitespace.
    Even though whitespace may be considered 'blank', it is not an empty string so the system message
    should still be prepended to the OpenAI messages.
    """
    memory = Memory(agent_description="   ")
    memory.add("User input", is_user=True)
    memory.add("Assistant response", is_user=False)
    expected = [
        {"role": "system", "content": "   "},
        {"role": "user", "content": "User input"},
        {"role": "assistant", "content": "Assistant response"},
    ]
    assert memory.to_openai_messages() == expected