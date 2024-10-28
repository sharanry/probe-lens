from typing import Callable

from probe_lens.experiments.experiments import ProbeExperiment

LETTERS = "abcdefghijklmnopqrstuvwxyz"
WORDS_DATASET = "https://www.mit.edu/~ecprice/wordlist.10000"


def default_spelling_prompt_generator(word: str):
    return f"The word '{word}' is spelled:"


def first_letter_index(word: str):
    return LETTERS.index(word.strip().lower()[0])


class FirstLetterSpelling(ProbeExperiment):
    def __init__(
        self,
        words: list[str],
        prompt_fn: Callable[[str], str] = default_spelling_prompt_generator,
        class_fn: Callable[[str], int] = first_letter_index,
    ):
        super().__init__("First Letter Spelling Experiment")
        self.words = words
        self.prompt_fn = prompt_fn
        self.class_fn = class_fn
        self.generate_data()

    def generate_data(self):
        self.classes = [self.class_fn(word) for word in self.words]
        self.prompts = [self.prompt_fn(word) for word in self.words]
        self.data = list(zip(self.prompts, self.classes))

    def get_data(self) -> list[tuple[str, int]]:
        return self.data
