import os
from langchain_mistralai import MistralAIEmbeddings
from api_keys import MISTRAL_API_KEY

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

embedding_model = MistralAIEmbeddings(model="mistral-embed")
