import requests
import time
import sys
import random

server_location = "http://localhost:4000"

height = int(sys.argv[1])#165
weight = int(sys.argv[2])#74


def get_data():
    def noise():
        return 10*random.random()-5
    return {"height": height+noise(), "weight": weight+noise()}

def main():
    epoch = 0
    while True:
        data = requests.get(server_location+"/current_model").json()
        
        if data['epochs'] > epoch:
            epoch = data['epochs']
            #train the model on our own data
            new_data = get_data()
            #upload our updated weights
            print("Posted:", new_data, "Real Weight:", weight, "Real Height:", height)
            requests.post(server_location+"/post_data/"+str(epoch), json=new_data)
        time.sleep(10)

if __name__=='__main__':
    main()
