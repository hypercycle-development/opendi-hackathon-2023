"""
OpenDI HyperCycle Hackathon 2023
Challenge 2: Incremental Learning
"""
from pyhypercycle_aim import SimpleServer, JSONResponseCORS, aim_uri
from fastapi import FastAPI, Request
from pydantic import BaseModel
from incremental_model import AutoEncoderTrainer
import os

PORT = os.environ.get("PORT", 4001)

class IncrementalExample(SimpleServer):
    manifest = {"name": "IncrementalExample",
                "short_name": "inc-example",
                "version": "0.1",
                "license": "MIT",
                "author": "HyperCycle"
               }
    def __init__(self):
        pass

    @aim_uri(uri="/get_data", methods=["GET"],
             endpoint_manifest = {
                 "input_query": "",
                 "input_headers": "",
                 "output": {},
                 "documentation": "",
                 "example_calls": [{
                     "body": "",
                     "method": "GET",
                     "query": "",
                     "headers": "",
                     "output": {"epochs":165,"total_height":76365.65916019333,
                                "total_weight":69078.21611527249,"samples":493,
                                "average_weight":140.1180854265162,
                                "average_height":154.89991716063557}
                 }]
             })
    def get_data(self, request):
        client_id = request.query_params.get("client_id")
        params, epochs = AutoEncoderTrainer.get_data()
        data = {"params": params, "epochs": epochs}
        
        return JSONResponseCORS(data)

    @aim_uri(uri="/post_data", methods=["POST"],
             endpoint_manifest = {
                 "input_query": "",
                 "input_headers": "",
                 "output": {},
                 "documentation": "",
                 "example_calls": [{
                     "body": "",
                     "method": "POST",
                     "query": "",
                     "headers": "",
                     "output": {"updated": True}
                 }]
             })
    async def post_data(self, request):
        data = await request.json()
        result = AutoEncoderTrainer.post_data(data['params'], data['segment'], data['loss'], data['epochs'])
        return JSONResponseCORS({"updated": result})

def main():
    #example usage:
    app = IncrementalExample()
    app.run(uvicorn_kwargs={"port": PORT, "host": "0.0.0.0"})
    
if __name__=='__main__':
    main()


