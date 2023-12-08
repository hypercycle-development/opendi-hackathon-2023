# opendi-hackathon-2023
Resources for the HyperCycle challenges at OpenD/I Hackathon 2023

Examples for challenges 1, 2, & 3 are written here.

# Challenge 1: Federated Learning

Federated learning concerns creating a machine learning system wherein training is distributed amongst several clients, with each client training the model on their own data. In this case, the data is not shared amongst other clients, or the server, but is still used during the training process.

In the example given, a server runs the code located in `federated_server.py` and waits for new data. A client running `federated_client.py` posts their data (height and weight) to the server, but adds a random offset to the data. Over time, as more clients post their data to the server, this noise will cancel out and will converge to the real average height and weight of the population.

The model in this case is just an average of the inputs. In a more elaborate example, each client could train the model on their own local data (for example, training a cat detecting model on pictures of their own cats), and upload the results of their training without revealing that data to the server.

Methods used to protect client privacy is the main goal, with a secondary goal of trying to deal with malicious actors submitting false data to the server. 

# Challenge 2: Decentralized Incremental Learning

Incremental learning concerns training a model overtime step-by-step instead of all at once. In this challenge, the goal is to create a decentralized incremental learning algorithm. In contrast to challenge 1, the decentralized requirement is the focus, where it is assumed that some actors in the learning model may post invalid or incorrect results to the server in order to undermine it.

In the example, a simple autoencoder is given, trained on the MNIST dataset. Each client gets the current model state (from the /get_data endpoint) and then trains it on a segment of the dataset (either even or odd entries). When the server receives two client requests reporting the same loss for the given segment training on this epoch (acting as a consensus), the server records the result. Once all segments are computed for this epoch, the resulting weights are averaged together to give the model weights for the next epoch.

In a more elaborate example, better protections would be given against malicious actors than just a single consensus (in this example, a malicious actor could just post the same incorrect value twice). This could involve reputation mechanisms, or forms of model verification on the server end

# Challenge 3: LLM Genetic Algorithm

Genetic algorithms can be used to solve optimization problems on large and complex spaces of possible solutions by mimicing aspects of natural evolutionary processes. In this challenge, the goal is to create a genetic algorithm to reverse engineer an effective prompt to an LLM that will cause the LLM to generate output similar to a supplied target output. Here the focus is on how to adequately represent the solution space (the set of prompts), how to measure the fitness of solutions (including how good a prompt is at generating output similar to the target), and how to optimize the performance of the genetic algorithm.

In the example, the genetic algorithm itself uses binary genotypes (binary lists). The fitness of a genotype is not computed directly, but by first computing its phenotype, where the genotpye is interpreted as a one-hot encoding of a set of words, here a randomly generated set of adjectives. The fitness of a genotype is then computed based on the similarity of the sentiment of the corresponding phenotype to the sentiment of the desired output. Note that in this simple example, there is no LLM call, so the output is the same as the prompt.

In a more elaborate example, you will first need to determine the kind of output that you want to engineer a prompt for (this should be a solution to some simple practical problem), then determine how to encode the prompts so that the genetic algorithm can be applied and how to calculate the fitness of a prompt, based on some measure of how well the LLM output generated from the prompt solves the simple practical problem. This will require making calls to the LLM to evaluate the fitness, so you will need to be mindful of how to optimize this computation to make it feasible to run the genetic algorithm.

# HyperCycle Node requirements

While nodes in the hypercycle system may be high-powered, nodes for these challenge (ie: the clients) should be assumed to be running on commodity hardware (eg: 16gb of ram, 4 core cpu, some GPU), though there could be a many of them (eg: 1000's). 


The server code itself should adhear to the HyperCycle AIM (AI Machine) specs. For easiest integerations, the examples given use a python library to create the server code to receive requests, that will add in extra metadata to make the server follow the required specs.

For similiplicy, you can also use the same library used (pyhypercycle_aim) or implement the required spec yourself (for example, in the case of writing a server in a different language). The most important thing to add is a `/manigest.json` endpoint, which will describe how to interact with your server. An example is given below:
```
GET /manifest.json
{
    "name": "IncrementalExample",
    "short_name": "inc-example",
    "version": "0.1",
    "license": "MIT",
    "author": "HyperCycle",
    "endpoints": [
        {
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "Returns the parameters of this model, along with the number of epochs computed so far.",
            "example_calls": [
                {
                    "body": "",
                    "method": "GET",
                    "query": "",
                    "headers": "",
                    "output": {
                        "epochs": 3,
                        "params": "gASVgQAAAAAAAACMC2Nv..."
                    }
                }
            ],
            "uri": "/get_data",
            "input_methods": [
                "GET"
            ]
        },
        {
            "input_body": {
                "params": "<Base64 encoded of pickled state_dict>",
                "segment": "<Int>",
                "loss": "<Float>",
                "epochs": "<Int>"
            }, 
            "input_query": "",
            "input_headers": "",
            "output": {},
            "documentation": "",
            "example_calls": [
                {
                    "body": {
                        "params": "gASVgQAAAAAAAACMC2Nv...",
                        "segement": 1,
                        "loss": 0.6894218985755133,
                        "epochs": 3
                    },
                    "method": "POST",
                    "query": "",
                    "headers": "",
                    "output": {
                        "updated": true
                    }
                }
            ],
            "uri": "/post_data",
            "input_methods": [
                "POST"
            ]
        }
    ]
}
```

A Dockerfile for launching your server can also be added. Examples are given in the challenge folders, and in practice are used to create containers for the AI machines that can be deployed easily across different nodes.









