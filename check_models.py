from google import genai
from dotenv import load_dotenv
import os

load_dotenv("/u/arkimjh/code/ECCV-jh/.env")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: API Key not found")
    exit(1)

client = genai.Client(api_key=api_key)

model_name = "gemini-3-flash-preview"
print(f"Testing generation with '{model_name}'...")
try:
    resp = client.models.generate_content(model=model_name, contents="Hello")
    print(f"Success: {resp.text}")
except Exception as e:
    print(f"Failed: {e}")

model_name_prefixed = "models/gemini-3-flash-preview"
print(f"\nTesting generation with '{model_name_prefixed}'...")
try:
    resp = client.models.generate_content(model=model_name_prefixed, contents="Hello")
    print(f"Success: {resp.text}")
except Exception as e:
    print(f"Failed: {e}")
