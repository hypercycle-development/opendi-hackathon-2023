"""
OpenDI HyperCycle Hackathon 2023
Challenge 1: Federated Learning
"""
from pyhypercycle_aim import SimpleServer, JSONResponseCORS, aim_uri
from fastapi import FastAPI, Request
from pydantic import BaseModel
from federated_model import ExampleModel
import os

PORT = os.environ.get("PORT", 4000)

class FederatedExample(SimpleServer):
    manifest = {"name": "FederatedExample",
                "short_name": "fed-example",
                "version": "0.1",
                "license": "MIT",
                "author": "HyperCycle"
               }
    def __init__(self):
        pass

    @aim_uri(uri="/current_model", methods=["GET"],
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
    def get_current_model(self, request):
        data = ExampleModel.get_model_parameters()
        return JSONResponseCORS(data)

    @aim_uri(uri="/post_data/{epoch}", methods=["POST"],
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
    async def read_item(self, request):
        epoch = request.path_params['epoch']
        data = await request.json()
        result = ExampleModel.update_model(data, epoch)
        return JSONResponseCORS({"updated": result})

def main():
    #example usage:
    app = FederatedExample()
    app.run(uvicorn_kwargs={"port": PORT, "host": "0.0.0.0"})
    
if __name__=='__main__':
    main()


