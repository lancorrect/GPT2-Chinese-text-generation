import collections

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab

class WordPiece:
    def __init__(self, vocab, unk_token='[UNK]', max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        text = text.lower()
        text = text.strip().split()
        output_tokens = []
        for token in text:
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            start = 0
            is_bad = False
            sub_tokens = []
            while start<len(chars):
                end = len(chars)
                cur_substr = None
                while start<end:
                    sub_str = "".join(chars[start:end])
                    if start > 0:
                        sub_str = "##" + sub_str
                    if sub_str in self.vocab:
                        cur_substr = sub_str
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break
                else:
                    sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        
        return output_tokens

if __name__ == '__main__':
    vocab_file = '../cache/vocab.txt'
    vocab = load_vocab(vocab_file)
    tokenizer = WordPiece(vocab)

    # text = "You can get around that behavior by passing it when instantiating this tokenizer or when you call it on some text"
    text = 'passÅing'
    output = tokenizer.tokenize(text)
    print(output)