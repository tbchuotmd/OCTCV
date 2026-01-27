import uvicorn
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--port',default=8888,type=int)
    args = parser.parse_args()
    port = args.port

    uvicorn.run("app.app:app", 
                host="0.0.0.0", 
                port=port, 
                reload=True)