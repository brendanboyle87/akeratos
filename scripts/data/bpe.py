"""SmolLM2's byte-level BPE, loaded from a local file, and the merge-prefix switch.

It never downloads: `BPE.from_file` and `BPE.from_store` read bytes from disk, check their SHA-256 when one
is given, and build the tokenizer from those same bytes with `tokenizers.Tokenizer.from_buffer`.

Ids: 0 is `<|endoftext|>`, the separator that opens each document; 1-16 are reserved, never emitted, and
masked out of the softmax; content ids are 17..V-1, stored as uint16 after `layout.content_ids`'s range check.

It imports the standard library, NumPy and, when it builds a tokenizer, `tokenizers`; never MLX.
"""

import hashlib
import json

import numpy as np

from layout import FIRST_ID, RESERVED, SEP_ID, content_ids

SMOLLM2 = {"repo_id": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
           "vocab": 49_152}
GOLDEN = (("Hello world", [19556, 905]),)        # SmolLM2's file only, so the tests' fixture never checks it ❓
NAMES = {0: "<|endoftext|>", 1: "<|im_start|>", 2: "<|im_end|>"}
PREFIXES = (32_768, 16_384, 8_192)
MAX_BYTES = 32 << 20                              # tokenizer.json is at most 32 MiB
ROUND_TRIP = ("def f(x):\n\treturn x ** 2  # done\r\n", "naïve café, Zürich; Ελληνικά; 日本語のテキスト",
              "é and é differ", "emoji 🙂👩‍💻🏳️‍🌈", "nul\x00inside", "digits 1234567890 3.14159 2026-10-14",
              "    leading spaces and trailing   ", "<|im_start|>user\nhi<|im_end|>")


def bytes_to_unicode():
    """GPT-2's table: each byte to the printable character byte-level BPE vocabularies use for it."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_CHAR = bytes_to_unicode()
CHAR_BYTE = {c: b for b, c in BYTE_CHAR.items()}


class TokenizerError(ValueError):
    pass


class BPE:
    def __init__(self, data, sha256=None, threads=None):
        if len(data) > MAX_BYTES:
            raise TokenizerError(f"tokenizer.json is {len(data)} bytes, above the {MAX_BYTES}-byte cap")
        self.data = bytes(data)
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        if sha256 is not None and sha256 != self.sha256:
            raise TokenizerError(f"tokenizer.json SHA-256 {self.sha256} differs from the recorded {sha256}")
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_buffer(self.data)
        # tokenizer.json does not carry this, and `from_buffer` leaves it False (tokenizers 0.23.3): with it,
        # special-token text inside a document is encoded as plain bytes.
        self.tok.encode_special_tokens = True
        self.spec = json.loads(self.data)
        self.vocab = self.tok.get_vocab_size(with_added_tokens=True)
        self._bytes = None

    @classmethod
    def from_file(cls, path, sha256=None):
        with open(path, "rb") as f:
            data = f.read(MAX_BYTES + 1)
        return cls(data, sha256)

    @classmethod
    def from_store(cls, store):
        """The copy in `<store>/tokenizer/` that the store was encoded with (the merge-prefix cut when the
        store has one), checked against the root manifest."""
        import os
        with open(os.path.join(store, "manifest.json")) as f:
            t = json.load(f)["tokenizer"]
        files = {e["path"]: e for e in t["files"]}
        name = "tokenizer.json" if t.get("merge_prefix") is None else f"tokenizer.prefix-{t['merge_prefix']}.json"
        return cls.from_file(os.path.join(store, "tokenizer", name), files[name]["sha256"])

    def encode(self, texts):
        """Ids for each text, with no special tokens added, as int64 arrays."""
        return [np.asarray(e.ids, np.int64) for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def raw(self, i):
        """The raw bytes of token i (b'' for the special ids), through GPT-2's byte-to-unicode table."""
        if i <= RESERVED[1]:
            return b""
        t = self.tok.id_to_token(int(i))
        try:
            return bytes(CHAR_BYTE[c] for c in t)
        except KeyError:
            raise TokenizerError(f"token {i} ({t!r}) is not written in the byte-level alphabet") from None

    def token_bytes(self):
        """uint16 [V]: each id's raw byte length, 0 for ids 0-16: the store's `token_bytes.u16`.

        A vocabulary not written in the byte-level alphabet fails the check that the lengths sum to each
        document's UTF-8 length.
        """
        if self._bytes is None:
            self._bytes = np.array([len(self.raw(i)) for i in range(self.vocab)], np.uint16)
        return self._bytes

    def document(self, ids):
        """A stored document: [0] + its ids as '<u2', after the range check."""
        return np.concatenate([np.zeros(1, np.uint16), content_ids(ids, self.vocab)])

    def check(self, golden=False):
        """Check the tokenizer; returns the record of what was checked and raises on a failure.

        The record names each check by the number beside it below; (1) is the SHA-256 check on loading.
        Every check but (2) and (3) holds for any byte-level BPE, so the tests run them on their fixture;
        `golden` adds (2) and (3), which hold only for SmolLM2's file.
        """
        rec = {"sha256": self.sha256, "vocab": self.vocab, "normalizer": self.spec.get("normalizer"),
               "pre_tokenizer": self.spec.get("pre_tokenizer"), "post_processor": self.spec.get("post_processor")}
        added = self.tok.get_added_tokens_decoder()
        plain = sorted(i for i, t in added.items() if not t.special)
        listed = sorted(e["id"] for e in self.spec.get("added_tokens", []) if not e.get("special"))
        if plain or listed:
            raise TokenizerError(f"non-special added tokens {plain or listed}: encode_special_tokens does not stop "
                                 f"them matching")
        for i, name in NAMES.items():                                            # (4) the special ids
            if self.tok.id_to_token(i) != name:
                raise TokenizerError(f"id {i} is {self.tok.id_to_token(i)!r}, expected {name!r}")
        if any(i not in added or not added[i].special for i in range(SEP_ID, RESERVED[1] + 1)):
            raise TokenizerError("ids 0-16 are not all special added tokens")
        probe = "a<|endoftext|>b"                                                 # (5) special text as bytes
        ids = self.encode([probe])[0]
        if (ids.size and int(ids.min()) < FIRST_ID) or self.decode(ids) != probe:
            raise TokenizerError(f"special-token text is not encoded as bytes: {ids.tolist()}")
        tb = self.token_bytes()                                                   # (6) round trips, byte counts
        for text, ids in zip(ROUND_TRIP, self.encode(ROUND_TRIP)):
            content_ids(ids, self.vocab)
            raw = b"".join(self.raw(i) for i in ids)
            if self.decode(ids) != text or raw != text.encode() or int(tb[ids].sum()) != len(text.encode()):
                raise TokenizerError(f"round trip or byte check failed on {text!r}")
        # (7) no BOS or EOS post-processor
        with_sp = [e.ids for e in self.tok.encode_batch(list(ROUND_TRIP), add_special_tokens=True)]
        if with_sp != [i.tolist() for i in self.encode(ROUND_TRIP)]:
            raise TokenizerError("encode adds tokens when add_special_tokens is True: a BOS/EOS "
                                 "post-processor")
        if golden:
            v = self.tok.get_vocab(with_added_tokens=True)                        # (2) the vocabulary
            if self.vocab != SMOLLM2["vocab"] or max(v.values()) != SMOLLM2["vocab"] - 1:
                raise TokenizerError(f"vocabulary {self.vocab}, largest id {max(v.values())}")
            for text, want in GOLDEN:                                             # (3) the golden ids
                got = self.encode([text])[0].tolist()
                if got != want:
                    raise TokenizerError(f"{text!r} encodes to {got}, expected {want}")
        rec["checks"] = ["P-45", "P-49 (1)", "(4)", "(5)", "(6)", "(7)"] + (["(2)", "(3)"] if golden else [])
        return rec


def truncate(data, size):
    """The merge-prefix switch: a tokenizer.json (bytes) cut to `size` ids. Returns (bytes, info).

    It keeps the special tokens, the 256 byte tokens and the tokens made by the first merges, in merge order,
    until `size` tokens are kept, drops every later merge, and renumbers the kept ids densely in their original
    order. `size` counts ids, not merges. When the ids already run specials, bytes, then merges in order, the
    renumbering is the identity, which `info["identity"]` reports.
    """
    spec = json.loads(data)
    model = spec["model"]
    if model.get("type") != "BPE":
        raise TokenizerError(f"model type {model.get('type')}, expected BPE")
    vocab = model["vocab"]
    merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model["merges"]]
    keep = {e["id"] for e in spec.get("added_tokens", []) if e.get("special")}
    keep |= {vocab[c] for c in BYTE_CHAR.values()}
    if len(keep) > size:
        raise TokenizerError(f"{len(keep)} special and byte tokens do not fit in {size}")
    kept_merges = 0
    for a, b in merges:
        if len(keep) == size:
            break
        if a not in vocab or b not in vocab or vocab[a] not in keep or vocab[b] not in keep:
            raise TokenizerError(f"merge {a!r} {b!r} uses a token outside the kept vocabulary")
        keep.add(vocab[a + b])
        kept_merges += 1
    if len(keep) < size:
        raise TokenizerError(f"only {len(keep)} ids can be kept, fewer than {size}")
    old = np.array(sorted(keep), np.int64)                     # old[new] = old id
    new_of = {int(o): n for n, o in enumerate(old)}
    model["vocab"] = {t: new_of[i] for t, i in vocab.items() if i in new_of}
    model["merges"] = [[a, b] for a, b in merges[:kept_merges]]
    for e in spec.get("added_tokens", []):
        if e["id"] not in new_of:
            raise TokenizerError(f"added token {e['content']!r} (id {e['id']}) falls outside the kept ids")
        e["id"] = new_of[e["id"]]
    out = json.dumps(spec, ensure_ascii=False).encode()
    return out, {"size": size, "merges_kept": kept_merges, "identity": bool(np.array_equal(old, np.arange(size))),
                 "old_ids": old}
