import os
import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return "this is root"


@app.get("/test")
async def testing():
    return "test page"


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
