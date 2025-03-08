from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()
import json

# Initialize the Groq client
client = Groq()
# Specify the model to be used (we recommend Llama 3.3 70B)
MODEL = 'llama-3.3-70b-versatile'

def calculate(expression):
    """Evaluate a mathematical expression"""
    try:
        # Attempt to evaluate the math expression
        result = eval(expression)
        return json.dumps({"result": result})
    except:
        # Return an error message if the math expression is invalid
        return json.dumps({"error": "Invalid expression"})
    
# imports calculate function from step 1
def run_conversation(user_prompt):
    # Initialize the conversation with system and user messages
    messages=[
        {
            "role": "system",
            "content": "You are a calculator assistant. Use the calculate function to perform mathematical operations and provide the results."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    # Define the available tools (i.e. functions) for our model to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
    ]
    # Make the initial API call to Groq
    response = client.chat.completions.create(
        model=MODEL, # LLM to use
        messages=messages, # Conversation history
        stream=False,
        tools=tools, # Available tools (i.e. functions) for our LLM to use
        tool_choice="auto", # Let our LLM decide when to use tools
        max_completion_tokens=4096 # Maximum number of tokens to allow in our response
    )
    # Extract the response and any tool call responses
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    print(response_message.content)
    print(tool_calls)

    if tool_calls:
        # Define the available tools that can be called by the LLM
        available_functions = {
            "calculate": calculate,
        }
        # Add the LLM's response to the conversation
        messages.append(response_message)

        # Process each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            # Call the tool and get the response
            function_response = function_to_call(
                expression=function_args.get("expression")
            )
            # Add the tool response to the conversation
            messages.append(
                {
                    "tool_call_id": tool_call.id, 
                    "role": "tool", # Indicates this message is from tool use
                    "name": function_name,
                    "content": function_response,
                }
            )
        # Make a second API call with the updated conversation
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        # Return the final response
        return second_response.choices[0].message.content
# Example usage
user_prompt = "What is 25 * 15 + 10?"
print(run_conversation(user_prompt))