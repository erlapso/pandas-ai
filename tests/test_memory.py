from pandasai.helpers.memory import Memory

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
def test_truncate_method():
                """
                Test the _truncate method of the Memory class.
                This test checks if the method correctly truncates long messages
                and leaves short messages unchanged.
                """
                memory = Memory()
                
                # Test with a short message (less than 100 characters)
                short_message = "This is a short message."
                assert memory._truncate(short_message) == short_message
                
                # Test with a long message (more than 100 characters)
                long_message = "This is a very long message that exceeds the default max length of 100 characters. It should be truncated by the _truncate method."
                truncated_message = memory._truncate(long_message)
                assert len(truncated_message) == 104  # 100 characters + " ..."
                assert truncated_message.endswith(" ...")
                assert truncated_message.startswith(long_message[:100])
                
                # Test with a custom max_length
                custom_length = 50
                custom_truncated = memory._truncate(long_message, max_length=custom_length)
                assert len(custom_truncated) == 54  # 50 characters + " ..."
                assert custom_truncated.endswith(" ...")
                assert custom_truncated.startswith(long_message[:custom_length])
def test_clear_memory():
        """
        Test the clear() method of the Memory class.
        This test verifies that the clear() method successfully removes all messages from the memory.
        """
        # Create a Memory instance
        memory = Memory()
        
        # Add some messages
        memory.add("Hello", is_user=True)
        memory.add("Hi there!", is_user=False)
        memory.add("How are you?", is_user=True)
        
        # Verify that messages were added
        assert memory.count() == 3
        
        # Clear the memory
        memory.clear()
        
        # Verify that the memory is empty
        assert memory.count() == 0
        assert memory.all() == []
        assert memory.get_messages() == []