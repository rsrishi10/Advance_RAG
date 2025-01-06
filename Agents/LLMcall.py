from tkinter import filedialog as fd
from langchain_openai import ChatOpenAI
from datetime import datetime
import configparser
from Agents.prompts import QAprompt
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate


class responseFromLLM(BaseModel):
        """a string of text"""

        line: str = Field(description="Relevance answeres for given context")

config = configparser.ConfigParser()
config.read('config.ini')

# Load the .env file
# load_dotenv()
OPAIKey = config['general']['api_key']


# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini", 
                    streaming=True,
                    max_tokens= 500,
                    temperature=0.0,
                    openai_api_key=OPAIKey)

# LLM with tool and validation
llm_with_tool = model.with_structured_output(responseFromLLM)

def get_answere(context ,question,model=model):

    

    # Prompt
    prompt = PromptTemplate(
        template=QAprompt,
        input_variables=["context", "question"],
    )

    # Chain
    
      
      
      
    




    chain = prompt | llm_with_tool
    result = chain.invoke({"question": question, "context": context})

    return result.line


if __name__ == "__main__":
    from RagTool.rag import retrival
    import time
    st = time.perf_counter()
    cn = str(input('Enter Collection name!'))
    rg = retrival('cde',r"PDF\handbook.pdf")

    question ='whats this song?'
    context = rg.mul_query_collection(question,query_expansion=False)
    en = time.perf_counter()
    print(en-st)
    # print(context)
    print(get_answere(context,question))
    print(en-st)