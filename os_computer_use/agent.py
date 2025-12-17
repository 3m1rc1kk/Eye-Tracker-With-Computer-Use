import json


def extract_message_values(obj):
    values = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key != "role":
                values.extend(extract_message_values(value))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(extract_message_values(item))
    else:
        values.append(str(obj))
    return values


def format_message(obj):
    role = obj.get("role", "")
    values = extract_message_values(obj)
    return f"{role.upper()}: {' '.join(value for value in values if value)}"


class QwenAgent:
    functions = []

    def __init__(self, qwen):
        self.qwen = qwen
        self.messages = []
        self.function_map = {}

    def call_function(self, function_call):
        def function_response(message):
            return {"role": "function", "name": func_name, "content": message}

        func_name = function_call["name"].lower()
        func_impl = self.function_map.get(func_name)
        if func_impl:
            try:
                func_args = json.loads(function_call["arguments"])
                result = func_impl(**func_args) if func_args else func_impl()
                return function_response(result)
            except Exception as e:
                return function_response(
                    [{"text": f"Error executing function: {str(e)}"}]
                )
        else:
            return function_response([{"text": "Function not implemented."}])

    def execute_function_calls(self, responses):
        called_function = False
        for rsp in responses:
            if rsp.get("function_call", None):
                called_function = True
                func_rsp = self.call_function(rsp["function_call"])
                self.messages.append(func_rsp)
                print(format_message(func_rsp))
        return called_function

    def run(self):
        responses = []
        should_continue = True
        n = 1

        while should_continue:
            response_stream = self.qwen.chat(
                messages=self.messages, functions=self.functions
            )
            responses = list(response_stream)[-1]
            for response in responses:
                print(format_message(response))
            self.messages.extend(responses)
            should_continue = self.execute_function_calls(responses)
            n = n + 1
