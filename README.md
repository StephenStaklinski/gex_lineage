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

## Multithreading Support
Uses OpenMP by default if found by CMake. OpenMP is not required. 

Note for MacOS: if OpenMP is not found by default, try explicitly
specifying the LLVM compiler during configuration, e.g., 
```sh
cmake -S . -B build -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang
```

The number of threads can be controlled with the -j option.

## Usage

Filtering, PCA initialization, and factor-model fitting are exposed as separate
programs. Run each with `--help` for its detailed options:

```bash
gexFilter --help
gexPca --help
gexFactor --help
gexLatentFlow --help
```

`gexPca` writes maxPhyloPCA results, including a
`.pca.eigenvectors.tsv` loading matrix. Pass that matrix to
`gexFactor --pca`; when `--pca` is omitted, `gexFactor` initializes its
loadings randomly using the configured seed.

### Regenerating latent-flow output

`gexLatentFlow` reconstructs internal latent-factor states from an existing
fit without rerunning model optimization. Given the tree file originally used
for fitting and the fit prefix, run:

```bash
gexLatentFlow --trees trees.nex --fit-prefix results/sample.fit
```

This reads `results/sample.fit.F.tsv` and
`results/sample.fit.summary.tsv`, then writes or replaces only
`results/sample.fit.latent_flow.tsv`. To preserve an existing table, provide a
different `--outprefix`. Factor scores and Brownian variances can instead be
supplied explicitly; see `gexLatentFlow --help`.

## Support

For questions or bug reports, please use the project issue tracker.

## License

Both PHAST and gexLineage are distributed under the BSD 3-Clause License, a
permissive academic license that allows redistribution and modification
provided that attribution is retained. See [LICENSE](LICENSE) for details.
