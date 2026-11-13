"""Typed seams over partly typed packages.

`mlx_seam.py` wraps each MLX call whose stubs strict pyright cannot fully type in an annotated
function, so the rest of the package calls typed code.
"""
