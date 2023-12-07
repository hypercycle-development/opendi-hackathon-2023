import requests
import time
import sys
import random
import pickle
from incremental_model import AutoEncoderTrainer, AE, reset_seeds

server_location = "http://localhost:4001"

def get_server_data():
    data = requests.get(server_location+"/get_data").json()
    return data['params'], data['epochs']

def post_server_data(params, segment, loss, epochs):
    data = {"params": params, "segment": segment, "loss": loss, "epochs": epochs}
    requests.post(server_location+"/post_data", json=data)



def main():
    epoch = 0
    segment = int(sys.argv[1])
    client_id = sys.argv[2]

    while True:
        reset_seeds()
        model = AE()
        serialized_data, epochs = get_server_data()
        state_dict = pickle.loads(serialized_data)
        model.load_state_dict(state_dict)
        loss = AutoEncoderTrainer.train(model, 1, 0)
        post_server_data(pickle.dumps(model.state_dict()), segment, loss, epochs)
        print("Client: 1 Epochs:", epochs, "Loss:", loss, "Segment: 0")
        time.sleep(1)

if __name__=='__main__':
    main()

