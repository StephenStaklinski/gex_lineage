# Single-cell phylogenetic factor analysis (scPFA)

## Requirements

- CMake 3.x or later
- C/C++ compiler (GCC, Clang, etc.)
- [PHAST](https://github.com/CshlSiepelLab/phast) (Phylogenetic Analysis with Space/Time models)

## Building from Source

Specify the PHAST installation prefix explicitly:

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
- `gexPca` finds tree-aware principal components. This could optionally be used to initialize the factor
  analysis model, though I usually prefer the default random initialization. It can also be used independently 
  for its own tree-aware PCA analysis to compare and contrast with the factor analysis result.
- `gexFactor` fits the latent Brownian factor model and writes the inferred
  factors, gene loadings, and reconstructed expression.

## License

Both PHAST and gexLineage are distributed under the BSD 3-Clause License, a
permissive academic license that allows redistribution and modification
provided that attribution is retained. See [LICENSE](LICENSE) for details.
