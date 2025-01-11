from sentence_transformers import CrossEncoder
from typing import List,Tuple
import configparser
config = configparser.ConfigParser()
config.read('config.ini')
max_length = int(config['general']['chunk_size'])

model = CrossEncoder('CrossEncoder', max_length=max_length)
# model.save_pretrained('CrossEncoder')

def cross_encoder(query:str,unique_contents:List,top_k:int=5):
    pairs = []
    for doc in unique_contents:
        pairs.append([query, doc])
    scores = model.predict(pairs)

    scored_docs = zip(scores, unique_contents)
    sorted_docs = sorted(scored_docs, reverse=True)
    reranked_docs = [doc for _, doc in sorted_docs][0:top_k]
    reranked_docs
    return reranked_docs

if __name__ == "__main__":
    print(cross_encoder('para',['Paragraph65','Paragraph2','Paragraph3']))