from probe_lens.experiments.spelling import LETTERS, FirstLetterSpelling


def test_first_letter_spelling():
    words = ["example", "words", "to", "spell"]
    spelling_task = FirstLetterSpelling(words)
    data = spelling_task.data
    classes = [c for _, c in data]
    assert classes == [LETTERS.index(word.lower()[0]) for word in words]
