import cohere
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import pandas as pd
import math
import time
import random

class CohereEmbeddings(Embeddings):
    def __init__(self, api_key):
        self.client = cohere.Client(api_key=api_key)

    def embed_documents(self, texts):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.embed(
                    texts=texts,
                    model="embed-multilingual-v3.0",
                    input_type="classification"
                )
                return response.embeddings
            except Exception as e:
                if "429" in str(e):
                    wait_time = 60
                    print(f"Rate limit hit. Waiting {wait_time:.2f} seconds (Attempt {attempt + 1})...")
                    time.sleep(wait_time)
                else:
                    raise

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def create_and_save_video_vectorstore(vectorstore_path, batch_size=50):
    embedding_function = CohereEmbeddings(api_key="d")
    
    video_csv_file = 'group_video.csv'
    video_data = pd.read_csv(video_csv_file)
    
    documents = [
        f"{row['oper_nm']} > {row['aggrp_nm']} > {row['trng_nm']}\n제목: {row['vdo_ttl_nm']}\n동영상링크: http://openapi.kspo.or.kr/web/video/{row['file_nm']}\n이미지링크: {row['img_file_url']}/{row['img_file_nm']}"
        for _, row in video_data.iterrows()
    ]
    
    num_batches = math.ceil(len(documents) / batch_size)
    final_vectorstore = None
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = documents[start_idx:end_idx]
        print(f"Processing batch {batch_idx + 1}/{num_batches}...")
        
        try:
            vectorstore = FAISS.from_texts(
                texts=batch, 
                embedding=embedding_function
            )
            
            if final_vectorstore is None:
                final_vectorstore = vectorstore
            else:
                final_vectorstore.merge_from(vectorstore)
        
        except Exception as e:
            print(f"Batch processing error: {e}")
            continue
    
    if final_vectorstore:
        final_vectorstore.save_local(vectorstore_path)
        print(f"벡터스토어가 {vectorstore_path}에 저장되었습니다.")
    else:
        print("벡터스토어 생성에 실패했습니다.")

# 실행
vectorstore_file_path = "video_vectorstore"
create_and_save_video_vectorstore(vectorstore_file_path, batch_size=50)