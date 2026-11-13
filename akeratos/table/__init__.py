"""The n-gram memory: a large learned table, its rows found by hashing bigrams and trigrams.

`config` holds the size, format and switches; `hasher` the host keys; `shards` the sole-owner
store with its write-back tripwire; `table` the working set, the replay row-Adam and the
streamed checkpoints; `accum` the gradient sum over micro-batches; `rate` the rate rule, the
decay and the residual share rho. The model reads the working set through `table_term` in
`akeratos/model.py`; the table never imports the model. The plain-words tour is
`docs/learn/n-gram-memory.md`.
"""
