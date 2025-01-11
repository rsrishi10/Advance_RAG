import chromadb.utils.embedding_functions as embedding_functions
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict,Tuple, Any
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
import configparser
from RagTool.QueryExpansion import queryExpansion
from sentence_transformers import CrossEncoder
from langchain_community.document_transformers import LongContextReorder
from concurrent.futures import ThreadPoolExecutor
from RagTool.crossEncoder import cross_encoder

config = configparser.ConfigParser()
config.read('config.ini')

# Load the .env file
# load_dotenv()
api_key = config['general']['api_key']
chunk_size = int(config['general']['chunk_size'])
chroma_client = chromadb.Client(settings=Settings(allow_reset=True))

chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True))

reordering = LongContextReorder()


class create_collection():
    def __init__(self,name):
        ...
        
class retrival():
    def __init__(self, 
                 Collection_name:str, 
                 pdf_path:str,
                 chunk_size:int = chunk_size,
                 chroma_client=chroma_client):
        
        
        self.embeddings = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-large"
            )
        # self.vectors = self.embeddings.embed_documents(doc_strings)
        self.chunk_size = chunk_size
        # db = Chroma.from_documents(new_docs, embeddings)
        # retriever = db.as_retriever(search_kwargs={"k": 6})
        

        self.client = chroma_client
        existing_collections = self.client.list_collections()

        collection_exists = any(coll.name == Collection_name for coll in existing_collections)

        if collection_exists:
          self.collection =  self.client.get_or_create_collection(
            name=Collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function = self.embeddings  
            )
        #   print('in '*80)
        else:
            print(f"Creating collection named {Collection_name}! ")
            self.collection =  self.client.get_or_create_collection(
            name=Collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function = self.embeddings  
            )
            # print('out '*80)
            self.add_to_collection(pdf_path)
            print('Done!')

    def get_chunks(self,pdf_dir:str)-> List[Dict[str, Any]]:

        # loader = DirectoryLoader(pdf_dir, loader_cls=TextLoader) #glob="./*.txt"
        # docs = loader.load()

        loader = PyPDFLoader(pdf_dir)
        docs = loader.load()
        overlap = self.chunk_size*0.1
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                       chunk_overlap=overlap
                                                       )
        splits = text_splitter.split_documents(docs)
        metadata = []
        text = []
        for doc in splits:
            metadata.append(doc.metadata)
            text.append(doc.page_content)

        result_list = [{'text': txt, 'metadata': meta} for txt, meta in zip(text, metadata)]

        return result_list
    
    def add_to_collection(self,pdf_dir:str)->None:

        chunks = self.get_chunks(pdf_dir)
        page_content_list = []
        metadata = []
        ids = []

        # Extract page_content and append to the list
        for i, doc in enumerate(chunks):
            ids.append(str(i + 1))
            page_content_list.append(doc['text'])
            metadata.append(doc['metadata'])

        self.collection.add(
            documents=page_content_list,
            metadatas=metadata,
            ids=ids
        )


    # def lost_in_middle(self,reranked_docs,flag = False):
    #     if flag:
    #         reordered_docs = reordering.transform_documents(reranked_docs)
    #         return reordered_docs


    def mul_query_collection(self,user_question:str):
        qe = queryExpansion()
        queries = qe.expand(user_question)
        print(f'Expanded queries are: {queries}')

        with ThreadPoolExecutor() as executor:
            listofdocsdict = list(executor.map(self.query_collection, queries))
        # import pickle
        # with open('pickle','wb') as f:
            # pickle.dump(listofdocsdict,f)
        unique_ids = set()
        unique_docs = []

        for docsdict in listofdocsdict:
            for id,doc1 in zip(docsdict["ids"][0],docsdict["documents"][0]):
                if id not in unique_ids:
                    unique_docs.append(doc1)
                    unique_ids.add(id)

        unique_docs = list(unique_docs)

        unique_docs = cross_encoder(user_question,unique_docs)
        # print('reranked'*90)
        return unique_docs

    def query_collection(self,user_question:str,query_expansion:bool=True):
        

        docs = self.collection.query(
            query_texts=[user_question],
            n_results= 5
        )
        if not query_expansion:
            docs = docs["documents"][0]

        # docs = self.rerank(user_question,unique_contents)
        return docs




if __name__ == "__main__":
    import time
    
    rg = retrival('abc')
    
    rg.add_to_collection(r"PDF\handbook.pdf")
    # existing_collections = rg.client.list_collections()
    # print(existing_collections)
    st = time.perf_counter()
    context = rg.mul_query_collection('whats the CEO name? what is his earnings and is it according to market standards?')
    # context = rg.query_collection('whats the CEO name?',query_expansion=False)
    en = time.perf_counter()

    print(en-st)

    print(type(context))
    print(len(context))
    # for key in context.keys():
    #     print(key)
    # print(context)

    