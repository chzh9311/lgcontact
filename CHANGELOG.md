<a name="unreleased"></a>
## [Unreleased]


<a name="v1.0.0"></a>
## [v1.0.0] - 2026-01-15
### Feat
- Version 1 of the diffusion training.
- Update the gridae to VAE structure.
- Enable checkpointing according to validation performance.
- Add an extra linear layer in encoding & decoding.
- Enable baseline training&testing.

### Fix
- Fix the problem of unaligned hand & object.
- Update data loading scripts to allow preprocessing of grid data.
- Fix the dimension mismatches for gridae test.
- Fix bugs; Added random seeds for reproducibility.

### Misc
- Update the arrangement of visualization images.

### Optim
- Use hdf5 for fast IO instead of npz.


<a name="v0.2.0"></a>
## [v0.2.0] - 2026-01-04
### Feat
- New baseline training code.
- Enable end-to-end training with autoencoder initialization.
- Add evaluation code.
- Enable e2e training of the generator.
- Add support for generator training.
- Implement conditional vqvae.

### Misc
- Rename the attribute model to ae for better readability.
- rename model config names.

### Refactor
- Reorganized multiple file structures.


<a name="v0.1.1"></a>
## [v0.1.1] - 2025-12-25
### Feat
- New structure, losses & visualization for unconditional LGVQVAE training & testing.

### Refactor
- Move visualization code to vis.py.


<a name="v0.1.0"></a>
## v0.1.0 - 2025-12-23
### Feat
- Built the structure of end-to-end model.
- Enable dumping local grids first.

### Misc
- Update gitignore.


[Unreleased]: /compare/v1.0.0...HEAD
[v1.0.0]: /compare/v0.2.0...v1.0.0
[v0.2.0]: /compare/v0.1.1...v0.2.0
[v0.1.1]: /compare/v0.1.0...v0.1.1
