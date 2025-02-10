from pandasai.helpers.memory import Memory
import pytest

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
def test_getters_and_clear():
    """
    Test various getters of the Memory class, including get_messages, get_conversation,
    get_previous_conversation, get_last_message, and the clear functionality.
    This test also verifies that long assistant messages are truncated correctly.
    """
    # Initialize with larger memory_size to capture all messages by default.
    memory = Memory(memory_size=10)
    
    # Add messages: two queries and two answers (the second answer is long to trigger truncation)
    memory.add("Query 1", is_user=True)
    memory.add("Answer 1", is_user=False)
    memory.add("Query 2", is_user=True)
    
    long_answer = "L" * 120  # 120-character long string
    expected_truncated = long_answer[:100] + " ..."
    memory.add(long_answer, is_user=False)
    
    # Test get_messages returns all the messages with appropriate formatting.
    messages = memory.get_messages()
    assert len(messages) == 4
    assert messages[0].startswith("### QUERY")
    assert "Query 1" in messages[0]
    
    assert messages[1].startswith("### ANSWER")
    assert "Answer 1" in messages[1]
    
    assert messages[2].startswith("### QUERY")
    assert "Query 2" in messages[2]
    
    assert messages[3].startswith("### ANSWER")
    assert expected_truncated in messages[3]
    
    # Test get_conversation returns a single string containing the conversation.
    conversation = memory.get_conversation()
    assert expected_truncated in conversation  # ensures truncated answer is in the conversation
    
    # Test get_previous_conversation returns the conversation without the last message.
    previous_convo = memory.get_previous_conversation()
    assert expected_truncated not in previous_convo
    assert "Answer 1" in previous_convo  # earlier assistant answer should be present
    
    # Test get_last_message returns the last message (the truncated long answer).
    last_message = memory.get_last_message()
    assert expected_truncated in last_message
    
    # Test clear method empties the conversation.
    memory.clear()
    assert memory.all() == []
    assert memory.get_conversation() == ""
    assert memory.get_last_message() == ""
    assert memory.get_previous_conversation() == ""
def test_last_message_empty_and_size_property():
    """
    Test that the Memory.size property returns the initialized memory size and that calling the 'last' method
    on an empty Memory instance raises an IndexError.
    """
    # Initialize Memory with a specific size
    memory = Memory(memory_size=5)
    
    # Verify the size property returns the correct value.
    assert memory.size == 5
    
    # Ensure that the "last" method raises IndexError when the memory is empty.
    with pytest.raises(IndexError):
        _ = memory.last()
def test_count_and_custom_limit():
    """
    Test that Memory.count() returns the correct number of messages and that get_messages
    with a custom limit returns only the last 'n' messages in the correct formatting.
    """
    memory = Memory(memory_size=10)
    
    # Add 4 messages: alternating between user queries and assistant answers.
    memory.add("User message 1", is_user=True)
    memory.add("Assistant answer 1", is_user=False)
    memory.add("User message 2", is_user=True)
    memory.add("Assistant answer 2", is_user=False)
    
    # Verify that count() returns 4.
    assert memory.count() == 4
    
    # Verify that get_messages() with a custom limit returns only the last 2 messages.
    custom_messages = memory.get_messages(limit=2)
    expected_messages = [
        "### QUERY\n User message 2",
        "### ANSWER\n Assistant answer 2"
    ]
    assert custom_messages == expected_messages
def test_memory_size_limit_enforced():
    """
    Test that if the Memory is initialized with a limited memory_size, 
    the get_messages and get_conversation methods only return the most recent messages up to that limit,
    even if more messages have been added.
    """
    # Initialize Memory with a memory_size of 3.
    memory = Memory(memory_size=3)
    
    # Add 5 messages. The first two messages will be dropped when retrieving conversation due 
    # to the limited memory_size.
    memory.add("User 1", is_user=True)
    memory.add("Assistant 1", is_user=False)
    memory.add("User 2", is_user=True)
    memory.add("Assistant 2", is_user=False)
    memory.add("User 3", is_user=True)
    
    # With memory_size=3, only the last three messages should be returned.
    expected_messages = [
        "### QUERY\n User 2",
        "### ANSWER\n Assistant 2",
        "### QUERY\n User 3"
    ]
    
    # Verify that get_messages returns only the most recent messages up to memory_size.
    assert memory.get_messages() == expected_messages
    
    # Verify that get_conversation properly joins these messages into one string.
    expected_conversation = "\n".join(expected_messages)
    assert memory.get_conversation() == expected_conversation
def test_truncate_with_non_string():
    """
    Test the _truncate method with non-string message types.
    
    This test verifies two scenarios:
    1. When a non-string message's string representation does not exceed the max_length,
       the original message is returned unchanged.
    2. When a non-string message's string representation exceeds max_length,
       a TypeError is raised because slicing is attempted on a non-subscriptable type.
    """
    memory = Memory()
    # Scenario 1: Non-string message that is not longer than max_length.
    # Here, the integer 12 has a string representation "12" of length 2,
    # which is not greater than max_length=2. So, it should return 12 unmodified.
    result = memory._truncate(12, max_length=2)
    assert result == 12
    # Scenario 2: Non-string message that exceeds the max_length.
    # The integer 12345, when converted to string ("12345"), has length 5 which is greater than max_length=3.
    # The method then attempts to slice the integer (12345[:3]) causing a TypeError.
    with pytest.raises(TypeError):
        memory._truncate(12345, max_length=3)
def test_last_method_and_truncate_edge_case():
    """
    Test that the last() method returns the last raw message correctly and that
    the _truncate method does not modify messages that are exactly of max_length.
    """
    memory = Memory(memory_size=5)
    
    # Add two messages and verify last() returns the most recent one.
    memory.add("First message", is_user=True)
    memory.add("Last message", is_user=False)
    last_message = memory.last()
    assert last_message == {"message": "Last message", "is_user": False}
    
    # Test _truncate with a string that is exactly the max_length.
    exact_length_message = "a" * 10  # 10 characters long
    result = memory._truncate(exact_length_message, max_length=10)
    # Since the message length is exactly max_length, it should be returned unchanged.
    assert result == exact_length_message
def test_non_string_message_handling():
    """
    Test handling of non-string messages in Memory.
    The test adds an integer as a user message and an integer as an assistant message.
    It verifies:
      - to_json returns the messages with the original types.
      - get_messages formats the assistant message correctly (using _truncate, converting non-string to str).
      - to_openai_messages retains the original message values.
    """
    memory = Memory(memory_size=10, agent_description="Agent description")
    
    # Adding a non-string user message (an integer)
    memory.add(123, is_user=True)
    # Adding a non-string assistant message (a float)
    memory.add(45.67, is_user=False)
    
    # Verify to_json returns the messages with the original types.
    expected_json = [
        {"role": "user", "message": 123},
        {"role": "assistant", "message": 45.67}
    ]
    assert memory.to_json() == expected_json
    
    # Verify get_messages returns strings formatted correctly.
    messages = memory.get_messages()
    # For user message, it should include "123" as string.
    assert messages[0] == "### QUERY\n 123"
    # For assistant message, since the float's string representation is short, it should be returned unchanged.
    # The string representation of 45.67 is "45.67" and should appear in the message.
    assert messages[1] == "### ANSWER\n 45.67"
    
    # Verify to_openai_messages includes the agent description and retains the original non-string types.
    expected_openai_messages = [
        {"role": "system", "content": "Agent description"},
        {"role": "user", "content": 123},
        {"role": "assistant", "content": 45.67}
    ]
    assert memory.to_openai_messages() == expected_openai_messages
def test_get_messages_custom_limit_exceeds_count():
    """
    Test that get_messages returns all available messages even when the custom limit
    parameter exceeds the number of messages in the memory.
    """
    memory = Memory(memory_size=5)
    memory.add("Test Query", is_user=True)
    memory.add("Test Answer", is_user=False)
    
    # Even though the custom limit is larger than the total messages, it should return only the available messages.
    messages = memory.get_messages(limit=10)
    expected_messages = [
        "### QUERY\n Test Query",
        "### ANSWER\n Test Answer"
    ]
    assert messages == expected_messages
def test_to_openai_messages_with_empty_agent_description():
    """
    Test that to_openai_messages does not include a system message 
    when the agent_description is an empty string.
    This ensures that an empty agent description is treated as Falsey
    and does not add a system message.
    """
    memory = Memory(agent_description="")
    memory.add("Test Query", is_user=True)
    memory.add("Test Answer", is_user=False)
    expected_openai_messages = [
        {"role": "user", "content": "Test Query"},
        {"role": "assistant", "content": "Test Answer"}
    ]
    assert memory.to_openai_messages() == expected_openai_messages
def test_truncate_with_list():
    """
    Test the _truncate method with a list input.
    This test ensures that when a sliceable but non-string message (like a list)
    is provided, the method returns the correct truncated result.
    """
    memory = Memory()
    list_message = [1, 2, 3, 4, 5]
    # Using a max_length of 3 to trigger truncation.
    max_length = 3
    expected = f"{list_message[:max_length]} ..."
    result = memory._truncate(list_message, max_length=max_length)
    assert result == expected
def test_get_messages_limit_zero_returns_all():
    """
    Test that supplying a custom limit of 0 to get_messages returns all messages.
    Due to Python's slicing behavior, using -0 (which is equivalent to 0) 
    returns the entire list instead of no messages.
    """
    memory = Memory(memory_size=3)
    memory.add("Test Query", is_user=True)
    memory.add("Test Answer", is_user=False)
    
    expected_messages = [
        "### QUERY\n Test Query",
        "### ANSWER\n Test Answer",
    ]
    
    # Even though limit is 0, because -0 is 0, slicing returns the full list.
    assert memory.get_messages(limit=0) == expected_messages
    
    expected_conversation = "\n".join(expected_messages)
    # Similarly, passing limit=0 to get_conversation returns the full conversation.
    assert memory.get_conversation(limit=0) == expected_conversation
def test_clear_preserves_agent_description():
    """
    Test that clearing the Memory removes all conversation messages
    while leaving the agent_description intact (so that to_openai_messages
    will still output the system message even if there are no user or assistant messages).
    """
    memory = Memory(agent_description="Persistent Agent")
    memory.add("Message before clear", is_user=True)
    memory.add("Assistant reply", is_user=False)
    # Ensure messages exist before clearing.
    assert len(memory.all()) == 2
    # Clear conversation messages
    memory.clear()
    # After clearing, conversation messages should be empty.
    assert memory.all() == []
    
    # Since agent_description is still set, to_openai_messages should return a system message.
    expected = [{"role": "system", "content": "Persistent Agent"}]
    assert memory.to_openai_messages() == expected
def test_none_messages():
    """
    Test that adding None as a message for both user and assistant is handled correctly.
    Verifies that:
      - to_json retains the None type.
      - get_messages formats the None value as "None" in the output.
      - to_openai_messages retains the None value.
    Note: Memory is instantiated with memory_size=2 to capture both messages.
    """
    memory = Memory(memory_size=2)
    memory.add(None, is_user=True)
    memory.add(None, is_user=False)
    
    # Verify to_json returns the messages with None intact.
    expected_json = [
        {"role": "user", "message": None},
        {"role": "assistant", "message": None}
    ]
    assert memory.to_json() == expected_json
    
    # Verify get_messages returns strings with "None" as the content.
    expected_get_messages = [
        "### QUERY\n None",
        "### ANSWER\n None"
    ]
    assert memory.get_messages() == expected_get_messages
    
    # Verify to_openai_messages returns the messages with None intact (no system message by default).
    expected_openai = [
        {"role": "user", "content": None},
        {"role": "assistant", "content": None}
    ]
    assert memory.to_openai_messages() == expected_openai
def test_get_messages_negative_limit():
    """
    Test that get_messages with a negative limit returns messages according to Python's slicing rules.
    In this scenario, passing a negative limit results in slicing of the messages list in a non-intuitive way.
    For example, if there are 5 messages and limit is -3, then self._messages[-(-3):] becomes self._messages[3:],
    which should return the 4th and 5th messages.
    """
    memory = Memory(memory_size=5)
    # Add 5 messages in alternating roles.
    memory.add("Msg1", is_user=True)   # index 0 -> QUERY
    memory.add("Msg2", is_user=False)  # index 1 -> ANSWER
    memory.add("Msg3", is_user=True)   # index 2 -> QUERY
    memory.add("Msg4", is_user=False)  # index 3 -> ANSWER
    memory.add("Msg5", is_user=True)   # index 4 -> QUERY
    # When limit is negative (-3), then according to the code,
    # limit becomes -3 and slicing self._messages[-(-3):] equals self._messages[3:]
    # This should return the messages at indices 3 and 4.
    expected = [
        "### ANSWER\n Msg4",
        "### QUERY\n Msg5"
    ]
    messages = memory.get_messages(limit=-3)
    assert messages == expected
def test_memory_size_zero_behavior():
    """
    Test that initializing Memory with memory_size=0 returns all messages
    in get_messages and get_conversation due to Python's slicing behavior with -0.
    """
    memory = Memory(memory_size=0)
    memory.add("Zero Query", is_user=True)
    memory.add("Zero Answer", is_user=False)
    expected_messages = [
        "### QUERY\n Zero Query",
        "### ANSWER\n Zero Answer"
    ]
    # Even though memory_size is 0, slicing with -0 returns the full list
    assert memory.get_messages() == expected_messages
    assert memory.get_conversation() == "\n".join(expected_messages)
def test_truncate_bytes_input():
    """
    Test _truncate with a bytes input that exceeds max_length.
    Verifies that the method returns the correctly formatted truncated representation
    of the bytes object as a string.
    """
    memory = Memory()
    byte_message = b'abcdefghijKLMNOP'
    max_length = 10
    # The expected truncated value uses the first 'max_length' bytes followed by " ..."
    expected = f"{byte_message[:max_length]} ..."
    result = memory._truncate(byte_message, max_length=max_length)
    assert result == expected
def test_truncate_with_zero_max_length():
    """
    Test _truncate with max_length set to 0.
    For a non-empty message, it should return " ..." (i.e., empty slice plus ellipsis),
    while for an empty message, it should return the unchanged empty string.
    """
    memory = Memory()
    
    # Non-empty message should be truncated to just " ..."
    non_empty_message = "hello"
    expected_non_empty = " ..."
    assert memory._truncate(non_empty_message, max_length=0) == expected_non_empty
    
    # Empty message should return an empty string, as its length is 0 (not greater than max_length)
    empty_message = ""
    assert memory._truncate(empty_message, max_length=0) == ""
def test_negative_memory_size_default():
    """
    Test that initializing Memory with a negative memory_size results in get_messages() slicing the
    internal messages list starting from the absolute value of memory_size.
    
    For example, if memory_size is -2 and 4 messages are added, then get_messages() will perform
    self._messages[-(-2):] which is equivalent to self._messages[2:], returning the messages at indices 2 and 3.
    """
    memory = Memory(memory_size=-2)
    memory.add("Msg1", is_user=True)
    memory.add("Msg2", is_user=False)
    memory.add("Msg3", is_user=True)
    memory.add("Msg4", is_user=False)
    
    # With memory_size = -2, get_messages returns messages starting from index 2 (i.e., the last 2 messages).
    expected_messages = [
        "### QUERY\n Msg3",
        "### ANSWER\n Msg4"
    ]
    assert memory.get_messages() == expected_messages
def test_to_openai_messages_with_whitespace_agent_description():
    """
    Test that when Memory is initialized with an agent_description that consists
    of whitespace (a truthy value), the system message is included in the output
    of to_openai_messages.
    """
    memory = Memory(agent_description=" ")
    memory.add("User test", is_user=True)
    memory.add("Assistant test", is_user=False)
    expected_messages = [
        {"role": "system", "content": " "},
        {"role": "user", "content": "User test"},
        {"role": "assistant", "content": "Assistant test"},
    ]
    assert memory.to_openai_messages() == expected_messages
def test_non_integer_memory_size():
    """
    Test that initializing Memory with a non-integer memory_size results in a TypeError when calling get_messages.
    This is because the memory_size is used as a slice index, and a non-integer will cause a TypeError.
    """
    # Initialize Memory with a non-integer memory_size.
    memory = Memory(memory_size="not int")
    memory.add("Test message", is_user=True)
    # Expect a TypeError when get_messages attempts to slice with a non-integer index.
    with pytest.raises(TypeError):
        _ = memory.get_messages()
def test_to_openai_messages_with_non_string_agent_description():
    """
    Test that to_openai_messages includes a system message when agent_description is 
    a non-string truthy value (e.g., an integer) and that the system message is included in the expected order.
    """
    memory = Memory(agent_description=42)
    memory.add("Query non string agent", is_user=True)
    memory.add("Answer non string agent", is_user=False)
    expected_messages = [
        {"role": "system", "content": 42},
        {"role": "user", "content": "Query non string agent"},
        {"role": "assistant", "content": "Answer non string agent"}
    ]
    assert memory.to_openai_messages() == expected_messages
def test_add_with_non_boolean_is_user():
    """
    Test that adding messages with non-boolean values for is_user (using truthy/falsey values)
    is handled correctly. A truthy non-boolean (like a non-empty string) should be treated as a user query,
    and a falsey non-boolean (like 0) should be treated as an assistant answer.
    """
    memory = Memory(memory_size=10)
    
    # Use a non-boolean truthy value for is_user: a non-empty string "yes" (evaluates True)
    memory.add("Non bool True", is_user="yes")
    # Use a non-boolean falsey value for is_user: integer 0 (evaluates False)
    memory.add("Non bool False", is_user=0)
    
    expected_messages = [
        "### QUERY\n Non bool True",
        "### ANSWER\n Non bool False"
    ]
    
    # Asserting that get_messages formats the messages based on truthiness of is_user
    assert memory.get_messages() == expected_messages
def test_readding_after_clear():
    """
    Test that after clearing the Memory, new messages can be added and retrieved correctly.
    Also verifies that the agent_description is preserved across clear calls.
    """
    # Initialize Memory with an agent_description and a specific memory_size.
    memory = Memory(memory_size=5, agent_description="Persistent Agent")
    
    # Add initial messages.
    memory.add("Initial Query", is_user=True)
    memory.add("Initial Answer", is_user=False)
    
    # Verify the conversation with the initial messages.
    initial_conversation = memory.get_conversation()
    assert "Initial Query" in initial_conversation
    assert "Initial Answer" in initial_conversation
    
    # Clear all messages.
    memory.clear()
    assert memory.all() == []
    # Also, clearing should not remove the agent_description.
    expected_system_message = [{"role": "system", "content": "Persistent Agent"}]
    assert memory.to_openai_messages() == expected_system_message
    
    # Re-add new messages after clear.
    memory.add("New Query", is_user=True)
    memory.add("New Answer", is_user=False)
    
    # Get the new conversation and verify that only the new messages are present.
    new_messages = memory.get_messages()
    assert len(new_messages) == 2
    assert "New Query" in new_messages[0]
    assert "New Answer" in new_messages[1]
    
    # Check the full openai messages including the preserved system message.
    expected_openai = [
        {"role": "system", "content": "Persistent Agent"},
        {"role": "user", "content": "New Query"},
        {"role": "assistant", "content": "New Answer"}
    ]
    assert memory.to_openai_messages() == expected_openai
def test_get_messages_negative_limit_exceeding_length():
    """
    Test that get_messages returns an empty list when a negative limit
    is provided whose absolute value exceeds the number of messages.
    For example, if there are 3 messages and limit is -10,
    slicing with self._messages[-(-10):] will return [].
    """
    memory = Memory(memory_size=10)
    # Add fewer messages than the absolute value of the negative limit.
    memory.add("Msg1", is_user=True)
    memory.add("Msg2", is_user=False)
    memory.add("Msg3", is_user=True)
    # Using a custom limit of -10 should result in an empty list.
    messages = memory.get_messages(limit=-10)
    assert messages == []
def test_all_returns_mutable_list():
    """
    This test verifies that the list returned by Memory.all() is the actual internal list.
    Specifically, modifications to the returned list (e.g., appending a new message) should directly affect 
    the Memory object's internal state.
    """
    memory = Memory(memory_size=5)
    memory.add("Test1", is_user=True)
    memory.add("Test2", is_user=False)
    
    # Retrieve the internal messages list.
    all_messages = memory.all()
    
    # Modify the returned list by appending a new message.
    all_messages.append({"message": "Injected", "is_user": True})
    
    # The count should now reflect the appended message.
    assert memory.count() == 3
    # The last message should be the injected one.
    assert memory.last() == {"message": "Injected", "is_user": True}
def test_add_with_list_message():
    """
    Test adding list-type messages for both user and assistant.
    This verifies that:
      - to_json returns the original list values.
      - get_messages converts the lists to their string representation with proper formatting.
      - to_openai_messages retains the original list types.
    """
    memory = Memory(memory_size=10, agent_description="Test Agent")
    
    user_list = [1, 2, 3]
    assistant_list = ["a", "b", "c"]
    
    memory.add(user_list, is_user=True)
    memory.add(assistant_list, is_user=False)
    
    # Verify to_json returns messages with the original list types
    expected_json = [
        {"role": "user", "message": user_list},
        {"role": "assistant", "message": assistant_list},
    ]
    assert memory.to_json() == expected_json
    
    # Verify get_messages returns formatted strings with lists converted to strings
    # For a user message, it's directly the list converted to string; for assistant, _truncate returns the message if it's under max_length.
    expected_get_messages = [
        f"### QUERY\n {user_list}",
        f"### ANSWER\n {assistant_list}"
    ]
    assert memory.get_messages() == expected_get_messages
    # Verify to_openai_messages includes the system message from agent_description and retains the original list values.
    expected_openai = [
        {"role": "system", "content": "Test Agent"},
        {"role": "user", "content": user_list},
        {"role": "assistant", "content": assistant_list},
    ]
    assert memory.to_openai_messages() == expected_openai
def test_multiple_clear_calls():
    """
    Test that calling clear multiple times does not affect the agent_description and that
    the conversation is correctly cleared each time. After clearing repeatedly, new messages
    can be added and retrieved correctly, while the agent_description remains preserved.
    """
    memory = Memory(memory_size=5, agent_description="Persistent Agent")
    # Add initial messages.
    memory.add("Message 1", is_user=True)
    memory.add("Message 2", is_user=False)
    
    # Verify that messages exist.
    assert memory.count() == 2
    assert "Message 1" in memory.get_conversation()
    assert "Message 2" in memory.get_conversation()
    
    # First clear.
    memory.clear()
    assert memory.all() == []
    assert memory.get_conversation() == ""
    
    # Second clear (should be benign).
    memory.clear()
    assert memory.all() == []
    assert memory.get_conversation() == ""
    
    # Re-add new messages after multiple clears.
    memory.add("Message 3", is_user=True)
    memory.add("Message 4", is_user=False)
    
    # Check that conversation shows only the new messages.
    expected_conversation = "### QUERY\n Message 3\n### ANSWER\n Message 4"
    assert memory.get_conversation() == expected_conversation
    
    # Verify that to_openai_messages still includes the system message with the agent_description.
    expected_openai = [
        {"role": "system", "content": "Persistent Agent"},
        {"role": "user", "content": "Message 3"},
        {"role": "assistant", "content": "Message 4"},
    ]
    assert memory.to_openai_messages() == expected_openai
def test_modify_agent_description():
    """
    Test modifying the agent_description after the Memory instance has been initialized and messages added.
    This test ensures that updating the agent_description attribute changes the system message
    in the output of to_openai_messages, while preserving the rest of the conversation.
    """
    # Initialize Memory with an initial agent description and add some messages.
    memory = Memory(agent_description="Initial Agent")
    memory.add("Hello", is_user=True)
    memory.add("Hi", is_user=False)
    # Get openai messages and verify the initial system message.
    openai_messages_initial = memory.to_openai_messages()
    assert openai_messages_initial[0] == {"role": "system", "content": "Initial Agent"}
    expected_initial = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    assert openai_messages_initial[1:] == expected_initial
    # Modify the agent_description after messages have been added.
    memory.agent_description = "Updated Agent"
    # Verify that to_openai_messages reflects the updated agent description.
    openai_messages_updated = memory.to_openai_messages()
    assert openai_messages_updated[0] == {"role": "system", "content": "Updated Agent"}
    # Ensure that the conversation messages remain unchanged.
    expected_updated = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]
    assert openai_messages_updated[1:] == expected_updated
def test_memory_size_field_remains_after_clear():
    """
    Test that the memory_size property remains unchanged after calling clear()
    on the Memory instance. This ensures that even when the conversation messages
    are cleared, the internal memory_size setting (and thus slicing behavior) is preserved.
    """
    memory = Memory(memory_size=5, agent_description="Testing Agent")
    # Initially memory.size should reflect the given memory_size.
    assert memory.size == 5
    # Add some messages.
    memory.add("Some Query", is_user=True)
    memory.add("Some Answer", is_user=False)
    # Ensure the memory_size property remains unchanged after additions.
    assert memory.size == 5
    # Clear the conversation.
    memory.clear()
    # After clearing, memory.size should still return the same memory_size.
    assert memory.size == 5
def test_get_previous_conversation_single_message():
    """
    Test that get_previous_conversation returns an empty string when there is only one message.
    Since get_previous_conversation is designed to exclude the last message, with only one message 
    present it should return an empty string.
    """
    memory = Memory(memory_size=5)
    memory.add("Only message", is_user=True)
    assert memory.get_previous_conversation() == ""