from collections import defaultdict

class SimpleBPE:

    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab = {}

    def train(self, text):

        words = [list(word) for word in text.split()]

        vocab = set()

        for word in words:
            vocab.update(word)

        # init vocab
        self.token_to_id = {t: i for i, t in enumerate(sorted(vocab))}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab = self.token_to_id

        # NOTE: skipping full merge logic here for clarity

    def encode(self, text):

        # VERY IMPORTANT: must return integers
        tokens = []

        for word in text.split():

            for ch in word:

                tokens.append(self.token_to_id.get(ch, 0))

        return tokens

    def decode(self, tokens):

        return "".join([
            self.id_to_token.get(t, "") for t in tokens
        ])
