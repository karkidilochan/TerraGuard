import os
from dotenv import load_dotenv

load_dotenv()

MISTRALAI_API_KEY = os.getenv("MISTRALAI_API_KEY")
