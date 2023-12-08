OpenD/I HyperCycle Challenge 2: Decentralized Incremental Learning Example


This is a basic example of incremental learning in a decentralized system to train a simple autoencoder ( https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/ ). This consists of a server, which holds the model, and clients that can fetch the model and then train it on segements of the training data (in this case, MNIST).

During each round, a client will select a segment of the data to work on (in this case, either even or odd entries of the MNIST data), fetch the latest model weights from the server, and then train the model on their segment of data.

After training the data, the client the submits their chaged model weights and their calculated loss to the server. If the calculated loss agrees with another client's submitted loss for this epoch and segment of training data, then the server will store the calculated new model weights and after receiving all segments for this epoch, average the model weights together to generate the model weights for the next epoch. 

Two things to note: the training step of this algorithm needs to be deterministic in order to ensure that two different clients will produce the same model weight output. This is why the `torch.seed(0)` and `random.seed(0)` lines exist inside the `incremental_model.py` file, though these only work for making CPU computation deterministic and more care would be needed to make GPU training deterministic as well (see tensorflow/pytorch documentation for more information). Secondly, the above consensus rule for selecting the result of training on the segment of data is not secure, since a malicious actor could replay their own bad weights and loss calculation to the server again to interfere with the training process. Issues like this should be addressed by your submission.

To run this example, you can install the `requirements.txt`

  `pip install -r requirements.txt`

Run the server code:

  `python incremental_server.py`


Then run some clients:

  `python incremental_client.py 0 id0`

  `python incremental_client.py 1 id1`

  `python incremental_client.py 0 id2`

  `python incremental_client.py 1 id3`


Over time, the number of epochs will increase and the calculated loss will decrease.
