OpenD/I HyperCycle Challenge 1: Federated Learning Example


This is a basic example of federated learning. This consists of a server, along with a
 clients that have private data (their height and weight) to train the model, which will return the average weight and height of the clients.

Before sending out their weights and heights, the clients add some random noise to their data. The server then aggregates together the noised data, which will converge to the real, un-noised data over time.

In a more thorough example, the data received from the server would be the model weights for whatever model is being trained, and the post sent to the server would be the updated weights from training the model over some number of epochs on the client's data. The server would then update the model weights from this user.



To run this example, you can install the `requirements.txt`

  `pip install -r requirements.txt`

Run the server code:

  `python federated_server.py`


Then run some clients:

  `python federated_client.py 150 65`

  `python federated_client.py 160 75`


Over time, the server will display the average weight and height of 155 and 70.
