"""Data loading: the store's read side, packing, the chat format and decontamination.

`store` reads the store's splits, `packer` packs their documents into training rows, `chat`
holds the post-training chat format and `decontam` checks eval and SFT items against the store.
No module here imports MLX.
"""
