# gexLineage

**gexLineage** is a program and set of supporting libraries for modeling
gene expression with phylogenetic lineage structure.

## Requirements

- CMake 3.x or later
- C/C++ compiler (GCC, Clang, etc.)
- [PHAST](https://github.com/CshlSiepelLab/phast) (Phylogenetic Analysis with Space/Time models)

## Building from Source

If PHAST is installed in a standard location, CMake will usually find it
automatically. Otherwise, you can specify its install prefix explicitly:

```bash
cmake -S . -B build \
   -DCMAKE_BUILD_TYPE=Release \
   -DPHAST_ROOT=/path/to/phast
cmake --build build
cmake --install build
```

Here, `PHAST_ROOT` should point to the installation prefix of PHAST (e.g.,
`/opt/homebrew/opt/phast`).

## Usage

Filtering, PCA initialization, and factor-model fitting are exposed as separate
programs. Run each with `--help` for its detailed options:

```bash
gexFilter --help
gexPca --help
gexFactor --help
```

- `gexFilter` preprocesses the expression matrix and retains genes with
  phylogenetic signal.
- `gexPca` finds tree-aware principal components that can initialize the
  factor model (optional).
- `gexFactor` fits the latent Brownian factor model and writes the inferred
  factors, gene loadings, and reconstructed expression.

## License

Both PHAST and gexLineage are distributed under the BSD 3-Clause License, a
permissive academic license that allows redistribution and modification
provided that attribution is retained. See [LICENSE](LICENSE) for details.
