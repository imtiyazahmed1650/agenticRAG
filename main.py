from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}


def main():
    print("Hello from Agentic RAG/ Agentbot") 