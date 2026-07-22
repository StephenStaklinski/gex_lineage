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

Filtering and factor-model fitting are exposed as separate programs. Run each
with `--help` for its detailed options:

```bash
gexFilter --help
gexFactor --help
```

## Support

For questions or bug reports, please use the project issue tracker.

## License

Both PHAST and gexLineage are distributed under the BSD 3-Clause License, a
permissive academic license that allows redistribution and modification
provided that attribution is retained. See [LICENSE](LICENSE) for details.
