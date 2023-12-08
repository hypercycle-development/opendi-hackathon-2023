#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import wraps

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wonderwords import RandomWord


class PromptObjective:

    def __init__(self, n_bits, target):
        self.word_list = []
        r = RandomWord()
        for i in range(n_bits):
            self.word_list.append(r.word(include_parts_of_speech=["adjectives"]))
        self.objective = self.best_prompt(target)
        # Initialize VADER so we can use it later
        self.sentimentAnalyzer = SentimentIntensityAnalyzer()
        self.scores = self.sentimentAnalyzer.polarity_scores(target)

    # objective function
    def best_prompt(self, target_output):
        @wraps(target_output)
        def prompt_loss(genotype):
            # decode the binary vector into a list of keywords
            keywords = self._decode(genotype)
            # get the phenotype (prompt) corresponding to the input genotype
            prompt = self._get_prompt(keywords)
            # here you define the loss function for the algorithm, which must
            # assess how far the output generate by the prompt is to the target,
            # but can include other factors as well, such as prompt size
            loss = self._get_error(prompt, target_output)
            return loss

        return prompt_loss

    def _decode(self, x):
        keywords = []
        for i, item in enumerate(x):
            if item:
                keywords.append(self.word_list[i])
        return keywords

    def _get_prompt(self, keywords):
        return ', '.join(keywords)

    def phenotype(self, genotype):
        return self._get_prompt(self._decode(genotype))

    def _get_error(self, prompt, target_output):
        output = self._evaluate_prompt(prompt)
        error = self._measure_difference(output, target_output)
        return error

    def _evaluate_prompt(self, prompt):
        # here you would call the LLM to generate its output based on the 
        # input prompt and return the output
        return prompt

    def _measure_difference(self, output, target_output):
        # to measure the quality of the generated output you need some way to
        # measure how close it is to the target; this computation goes here
        # this example just counts the number of spaces
        scores = self.sentimentAnalyzer.polarity_scores(output)
        diff = abs(scores['neg'] - self.scores['neg']) + \
               abs(scores['neu'] - self.scores['neu']) + \
               abs(scores['pos'] - self.scores['pos'])
        return diff
