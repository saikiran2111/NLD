import json
from openai import OpenAI
import os
import openai
from dotenv import load_dotenv
load_dotenv()



client = OpenAI(api_key = "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")




def extract_final_output(json_data):
    prompt = f'''
    You will be given the JSON formatted workflow data, json_data: ```{json_data}``` 
    Using the input data, prepare a workflow. If there are any common elements between activities, do not repeat them.
    For example, if the file path is required by multiple activities, specify it only once and reference it in other steps as 'The 'INPUT' specified in the 'ACTIVITY' activity will be used.'.
    Generate the final workflow in JSON format using the following schema:

    [
        {{
            "name": "Activity Name",
            "description": "Activity Description",
            "input": {{
                "Parameter1": "Parameter1 Description",
                "Parameter2": "Parameter2 Description",
                ...
            }}
        }},
        ...
    ]
    '''


    def process_file(prompt):
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.6,
        )
        return response

    full_response = process_file(prompt)
    # print(full_response)
    input_token = full_response.usage.prompt_tokens
    output_token = full_response.usage.completion_tokens

    content = full_response.choices[0].message.content


    workflow_json = content.replace("\\'", "'").replace("\'", "'").split('```json\n')[1].split('\n```')[0]


    workflow = json.loads(workflow_json)

    final_response = [{'FunctionName': step['name'], 'Description': step['description'], 'Input': step['input']} for step in workflow]


    
    
    return final_response,input_token,output_token






# def load_json(file_path):
#         with open(file_path, 'r') as json_file:
#             data = json.load(json_file)
#         return data


# if __name__ == "__main__":
#     file_path = "output.json"
#     json_data = load_json(file_path)
#     print(extract_final_output(json_data))

    
