<a name="unreleased"></a>
## [Unreleased]

### Bug Fixes
- Fix missing attributes regarding msdf_path.

### Features
- Enable mdm-style diffusion training & inference.
- Separate the UNet model with diffusion scheduler.
- Enable sampling test.
- rework the visualization and logging of the model.
- Updated data loading and processing scripts accordingly.
- Implement scripts to preprocess the grid data.


<a name="v1.0.0"></a>
## [v1.0.0] - 2026-01-15
### Bug Fixes
- Fix the problem of unaligned hand & object.
- Update data loading scripts to allow preprocessing of grid data.
- Fix the dimension mismatches for gridae test.
- Fix bugs; Added random seeds for reproducibility.

### Features
- Version 1 of the diffusion training.
- Update the gridae to VAE structure.
- Enable checkpointing according to validation performance.
- Add an extra linear layer in encoding & decoding.
- Enable baseline training&testing.


<a name="v0.2.0"></a>
## [v0.2.0] - 2026-01-04
### Code Refactoring
- Reorganized multiple file structures.

### Features
- New baseline training code.
- Enable end-to-end training with autoencoder initialization.
- Add evaluation code.
- Enable e2e training of the generator.
- Add support for generator training.
- Implement conditional vqvae.


<a name="v0.1.1"></a>
## [v0.1.1] - 2025-12-25
### Code Refactoring
- Move visualization code to vis.py.

### Features
- New structure, losses & visualization for unconditional LGVQVAE training & testing.


<a name="v0.1.0"></a>
## v0.1.0 - 2025-12-23
### Features
- Built the structure of end-to-end model.
- Enable dumping local grids first.


[Unreleased]: /compare/v1.0.0...HEAD
[v1.0.0]: /compare/v0.2.0...v1.0.0
[v0.2.0]: /compare/v0.1.1...v0.2.0
[v0.1.1]: /compare/v0.1.0...v0.1.1
