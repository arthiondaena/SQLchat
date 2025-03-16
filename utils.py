import requests
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import ListSQLDatabaseTool, InfoSQLDatabaseTool
import re

def get_all_groq_model(api_key:str=None) -> list:
    if api_key is None:
        raise ValueError("API key is required")
    url = "https://api.groq.com/openai/v1/models"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    data = response.json()['data']
    model_ids = [model['id'] for model in data]

    return model_ids

def validate_api_key(api_key:str) -> bool:
    if len(api_key) == 0:
        return False
    try:
        get_all_groq_model(api_key=api_key)
        return True
    except Exception as e:
        return False

def validate_uri(uri:str) -> bool:
    try:
        SQLDatabase.from_uri(uri)
        return True
    except Exception as e:
        return False

def get_info(uri:str) -> dict[str, str] | None:
    db = SQLDatabase.from_uri(uri)
    dialect = db.dialect
    # List all the tables accessible to the user.
    access_tables = ListSQLDatabaseTool(db=db).invoke("")
    # List the table schemas of all the accessible tables.
    tables_schemas = InfoSQLDatabaseTool(db=db).invoke(access_tables)
    return {'sql_dialect': dialect, 'tables': access_tables, 'tables_schema': tables_schemas}

def extract_code_blocks(text):
    pattern = r"```(?:\w+)?\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

if __name__ == "__main__":
    models = (get_all_groq_model("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"))
    vision_audio = [model for model in models if 'vision' in model or 'whisper' in model]
    models = [model for model in models if model not in vision_audio]
    print(models)