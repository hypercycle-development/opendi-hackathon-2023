"""
OpenDI HyperCycle Hackathon 2023
Challenge 3: Genetic Algorithm for Automated Prompt Engineering
"""
import os

from pyhypercycle_aim import SimpleServer, JSONResponseCORS, aim_uri

from genetic_algorithm import GeneticAlgorithm
from prompt_objective import PromptObjective

PORT = os.environ.get("PORT", 4002)


class GeneticExample(SimpleServer):
    manifest = {"name": "GeneticExample",
                "short_name": "gen-example",
                "version": "0.1",
                "license": "MIT",
                "author": "HyperCycle"
                }

    def __init__(self):
        pass

    @aim_uri(uri="/prompt", methods=["POST"],
             endpoint_manifest={
                 "input_query": "",
                 "input_headers": "",
                 "output": {},
                 "documentation": "Returns the prompt and the score based on the desired output",
                 "example_calls": [{
                     "body": {"target_output": "simple, lively, strong"},
                     "method": "POST",
                     "query": "",
                     "headers": "",
                     "output": {"prompt": "simple, lively, strong", "score": 0.004}
                 }]
             })
    async def prompt(self, request):
        # define the total iterations
        n_iter = 100
        # bits
        n_bits = 20
        # define the population size
        n_pop = 100
        # crossover rate
        r_cross = 0.9
        # mutation rate
        r_mut = 1.0 / float(n_bits)

        request_json = await request.json()
        target_output = request_json['target_output']
        pa = PromptObjective(n_bits, target_output)

        # perform the genetic algorithm search
        best_genotype, score = GeneticAlgorithm.genetic_algorithm(pa.objective,
                                                                  n_bits,
                                                                  n_iter,
                                                                  n_pop,
                                                                  r_cross,
                                                                  r_mut)
        best = pa.phenotype(best_genotype)
        return JSONResponseCORS({"prompt": best, "score": score})


def main():
    # example usage:
    app = GeneticExample()
    app.run(uvicorn_kwargs={"port": PORT, "host": "0.0.0.0"})


if __name__ == '__main__':
    main()
