import traceback
import uvicorn
from rag_recommender.api.main import app

def run_server():
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        print("Error running server:")
        traceback.print_exc()

if __name__ == "__main__":
    run_server()
