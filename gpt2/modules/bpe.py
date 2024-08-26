"""
bpe is short for Byte Pair Encoder. It translates arbitrary utf-8 strings into
sequences of integers, where each integer represents small chunks of commonly
occuring characters. This implementation is based on openai's gpt2 encoder.py:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
but was mildly modified because the original implementation is a bit confusing.
I also tried to add as many comments as possible, my own understanding of what's
going on.
"""

from functools import lru_cache
import json
from typing import List, Optional, Union
import regex as re

@lru_cache()
def bytes_to_unicode():
    """
    Every possible byte (really an integer 0..255) gets mapped by OpenAI to a unicode
    character that represents it visually. Some bytes have their appearance preserved
    because they don't cause any trouble. These are defined in list bs. For example:
    chr(33) returns "!", so in the returned dictionary we simply have d[33] -> "!".
    However, chr(0), for example, is '\x00', which looks ugly. So OpenAI maps these
    bytes, into new characters in a range where chr() returns a single nice character.
    So in the final dictionary we have d[0] -> 'Ā' instead, which is just chr(0 + 2**8).
    In particular, the space character is 32, which we can see by ord(' '). Instead,
    this function will shift space (32) by 256 to 288, so d[32] -> 'Ġ'.
    So this is just a simple one-to-one mapping of bytes 0..255 into unicode characters
    that "look nice", either in their original form, or a funny shifted character
    like 'Ā', or 'Ġ', etc.
    """
    # the 188 integers that render fine in their original form and need no shifting
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]  # all integers b in bs will simply map to chr(b) in the output dict
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    n = 0
    for b in range(2**8):
        if b not in bs:
            # if this byte is "ugly" then map it to the next available "nice" character
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    return d


def get_pairs(word):
    """
    Return all bigrams as a set of tuples, of consecutive elements in the iterable word.
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPETokenizer:

    def __init__(self, encoder_path, bpe_path):
        with open(bpe_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        # light postprocessing: strip the version on first line and the last line is a blank
        bpe_merges = [
            tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]
        ]
        # byte encoder/decoder
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # bpe token encoder/decoder
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        # bpe merge list that defines the bpe "tree", of tuples (a,b) that are to merge to token ab
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # the splitting pattern used for pre-tokenization
        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions <-- original openai comment
        """
        ok so what is this regex looking for, exactly?
        python re reference: https://docs.python.org/3/library/re.html
        - the vertical bars | is OR, so re.findall will chunkate text as the pieces match, from left to right
        - '\'s' would split up things like Andrej's -> (Andrej, 's)
        - ' ?\p{L}': optional space followed by 1+ unicode code points in the category "letter"
        - ' ?\p{N}': optional space followed by 1+ unicode code points in the category "number"
        - ' ?[^\s\p{L}\p{N}]+': optional space, then 1+ things that are NOT a whitespace, letter or number
        - '\s+(?!\S)': 1+ whitespace characters (e.g. space or tab or etc) UNLESS they are followed by non-whitespace
                       so this will consume whitespace characters in a sequence but exclude the last whitespace in
                       that sequence. that last whitespace has the opportunity to then match the optional ' ?' in
                       earlier patterns.
        - '\s+': 1+ whitespace characters, intended probably to catch a full trailing sequence of whitespaces at end of string
        So TLDR:
        - we are special casing a few common apostrophe constructs ('s, 't, 're, ...) and making those into separate tokens
        - we then separate out strings into consecutive chunks of 1) letters, 2) numbers, 3) non-letter-numbers, 4) whitespaces
        """
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.cache = {}
        self.special_tokens = {"<|endoftext|>"}

    def add_special_tokens(self, new_tokens: List[str]):
        start_idx = len(self.encoder)

        for i, token in enumerate(new_tokens):
            if token in self.encoder:
                raise ValueError(f"Token '{token}' already exists in the encoder.")

            self.encoder[token] = start_idx + i
            self.decoder[start_idx + i] = token

            # no need to update BPE ranks for special tokens as they are not merged
            self.cache[token] = token
        self.special_tokens.update(new_tokens)

    def get_vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        """
        this function uses self.bpe_ranks to iteratively merge all the possible bpe tokens
        up the tree. token is a string of one individual 'word' (after regex tokenization)
        and after byte encoding, e.g. 'Ġthere'.
        """
        # token is a string of one individual 'word', after byte encoding, e.g. 'Ġthere'

        # memoization, for efficiency
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)  # individual characters that make up the token, in a tuple
        pairs = get_pairs(word)  # get all bigrams

        if not pairs:
            return token

        while True:

            # find the next lowest rank bigram that can be merged
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break  # no more bigrams are eligible to be merged
            first, second = bigram

            # we will now replace all occurences of (first, second) in the list of current
            # words into one merged token first_second, in the output list new_words
            new_word = []
            i = 0
            while i < len(word):

                # find the next occurence of first in the sequence of current words
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                # if this occurence is also followed by second, then merge them into one
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # all occurences of (first, second) have been merged to first_second
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        # concat all words into a string, and use ' ' as the separator. Note that
        # by now all characters have been byte encoded, guaranteeing that ' ' is
        # not used in the actual data and is a 'special' delimiter character
        word = " ".join(word)

        # cache the result and return
        self.cache[token] = word
        return word

    def token_to_id(self, token: str) -> int:
        return self.encoder.get(token, 0)

    def encode(
        self,
        texts: Union[str, List[str]],
    ):
        """strings go in, lists of integers comes out"""
        if not isinstance(texts, list):
            texts = [texts]
        indices = []
        for text in texts:
            bpe_idx = []
            # pre-tokenize the input text into string tokens (words, roughly speaking)
            tokens = re.findall(self.pat, text)
            # process each token into BPE integers
            for token in tokens:
                # encode the token as a bytes (b'') object
                token_bytes = token.encode("utf-8")
                # translate all bytes to their unicode string representation and flatten
                token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
                # perform all the applicable bpe merges according to self.bpe_ranks
                token_merged = self.bpe(token_translated).split(" ")
                # translate all bpe tokens to integers
                token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
                # extend our running list of all output integers
                bpe_idx.extend(token_ix)
            indices.append(bpe_idx)
            
        return indices

    def decode(
        self, bpe_idx: Union[List[List[int]], List[int]], skip_special_tokens=True
    ):
        """lists of integers come in, a list of string comes out"""
        if not isinstance(bpe_idx[0], list):
            bpe_idx = [bpe_idx]

        texts = []
        for idx in bpe_idx:
            # inverse map the integers to get the tokens
            tokens_merged = [self.decoder[token] for token in idx]
            # inverse the byte encoder, e.g. recovering 'Ġ' -> ' ', and get the bytes
            tokens_flat = "".join(tokens_merged)
            tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
            # recover the full utf-8 string
            text = tokens_bytes.decode("utf-8", errors="replace")
            if skip_special_tokens:
                text = re.sub("|".join(self.special_tokens), " ", text)
            texts.append(text)
        return texts
