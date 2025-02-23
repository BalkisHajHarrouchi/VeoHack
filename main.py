from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import subprocess

app = FastAPI()

# Serve static files (CSS, JS, Images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=".")

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/app1")
async def launch_app1():
    subprocess.Popen(["streamlit", "run", "app1.py"])
    return Response(status_code=204)  # No Content

@app.get("/app2")
async def launch_app2():
    subprocess.Popen(["streamlit", "run", "app2.py", "--server.port", "8502"])
    return Response(status_code=204)  # No Content

@app.get("/app3")
async def launch_app3():
    subprocess.Popen(["streamlit", "run", "app3.py", "--server.port", "8509"])
    return Response(status_code=204)  # No Content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
