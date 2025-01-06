from openai import OpenAI
import configparser
import json
from pydantic import BaseModel,Field,model_validator
from langchain.output_parsers import PydanticOutputParser
import os


config = configparser.ConfigParser()
config.read('config.ini')
os.environ["OPENAI_API_KEY"] = config['general']['api_key']

client = OpenAI()





class LineList(BaseModel):
    lines: list[str] = Field(description="Lines of text")

    # Using a model_validator to process raw input if necessary
    @model_validator(mode="after")
    def process_lines(cls, values):
        if isinstance(values, str):  # Support initializing with raw text
            lines = values.strip().split("\n")
            return {"lines": lines}
        return values
    

class queryExpansion():

    def __init__(self):
        pass
    
    def expand(self,query):

        System_prompt = """ You are a linguiest. Your will be given a query, your job is to see if the query can be split into 
                            individual queries. If not then provide 5 different versions of the query. """
        
        completion = client.beta.chat.completions.parse(
                    model = 'gpt-4o-mini',
                    temperature=0,
                    max_tokens=800,
                    messages=[
                        {"role": "system", "content": System_prompt},
                        {"role": "user", "content": query},
                    ],
                    response_format=LineList,
                    )
        queries = completion.choices[0].message.parsed
        return queries.lines


if __name__ == "__main__":
    qe = queryExpansion()
    queries = qe.expand("what is CEO name? what does he earn and is it good as per market standards?")
    # print('done!')
    # print(type(queries))
    # print(queries)
    # print(len(queries))
    for query in queries:
        print(query)
    
    
    