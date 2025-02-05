from pandasai.core.prompts.base import BasePrompt


def test_base_prompt_str_method():
    """
    Test that the __str__ method of BasePrompt returns the rendered template.
    This test verifies that the __str__ method correctly renders the template with provided variables.
    """
    class TestPrompt(BasePrompt):
        template = "Hello {{ name }}!"

    prompt = TestPrompt(name="World")

    assert str(prompt) == "Hello World!"
    assert prompt.render() == "Hello World!"


def test_base_prompt_render_method():
    """
    Test that the render method of BasePrompt correctly handles different types of variables.
    This test covers various scenarios including string, numeric, and list variables.
    """
    class TestPrompt(BasePrompt):
        template = "Name: {{ name }}, Age: {{ age }}, Hobbies: {{ hobbies|join(', ') }}"

    prompt = TestPrompt(name="Alice", age=30, hobbies=["reading", "swimming", "coding"])
    rendered = prompt.render()
    assert rendered == "Name: Alice, Age: 30, Hobbies: reading, swimming, coding"
    
    prompt2 = TestPrompt(name="Bob", age=25.5, hobbies=["gaming"])
    rendered2 = prompt2.render()
    assert rendered2 == "Name: Bob, Age: 25.5, Hobbies: gaming"

    prompt3 = TestPrompt(name="Charlie", age=40, hobbies=[])
    rendered3 = prompt3.render()
    assert rendered3 == "Name: Charlie, Age: 40, Hobbies: "
