import os
from dotenv import load_dotenv

# Force load .env
load_dotenv(dotenv_path=".env", override=True)

groq_api_key = os.getenv("GROQ_API_KEY")
db_uri = os.getenv("DB_URI")

print("GROQ API Key:", groq_api_key if groq_api_key else "Not found")
print("DB URI:", db_uri if db_uri else "Not found")
