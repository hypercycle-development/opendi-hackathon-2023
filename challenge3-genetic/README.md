OpenD/I HyperCycle Challenge 3: Genetic Algorithm for Automated Prompt Engineering Example

Genetic algorithms can be used to solve optimization problems on large and complex spaces of possible solutions by mimicing aspects of natural evolutionary processes. In this challenge, the goal is to create a genetic algorithm to reverse engineer an effective prompt to an LLM that will cause the LLM to generate output similar to a supplied target output. Here the focus is on how to adequately represent the solution space (the set of prompts), how to measure the fitness of solutions (including how good a prompt is at generating output similar to the target), and how to optimize the performance of the genetic algorithm.

In the example, the genetic algorithm itself uses binary genotypes (binary lists). The fitness of a genotype is not computed directly, but by first computing its phenotype, where the genotpye is interpreted as a one-hot encoding of a set of words, here a randomly generated set of adjectives. The fitness of a genotype is then computed based on the similarity of the sentiment of the corresponding phenotype to the sentiment of the desired output. Note that in this simple example, there is no LLM call, so the output is the same as the prompt.

In a more elaborate example, you will first need to determine the kind of output that you want to engineer a prompt for (this should be a solution to some simple practical problem), then determine how to encode the prompts so that the genetic algorithm can be applied and how to calculate the fitness of a prompt, based on some measure of how well the LLM output generated from the prompt solves the simple practical problem. This will require making calls to the LLM to evaluate the fitness, so you will need to be mindful of how to optimize this computation to make it feasible to run the genetic algorithm.

To run this example, you can install the requirements.txt

`pip install -r requirements.txt`

Run the server code:

`python genetic_server.py`

Then run the test code

`python test_genetic_server.py`

Edit line 7 of `test_genetic_server.py` to change the target output, which is assumed here to be a string containing several adjectives.
