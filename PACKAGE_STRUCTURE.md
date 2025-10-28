# WICMAD.jl Package Structure

This document describes the structure of the WICMAD.jl package after conversion from a standalone script to a proper Julia package.

## Directory Structure

```
WICMAD_jl/
├── src/                          # Package source code
│   ├── WICMAD.jl                 # Main package module
│   ├── utils.jl                  # Utility functions
│   ├── kernels.jl                # Kernel definitions
│   ├── wavelets.jl               # Wavelet operations
│   ├── icm_cache.jl              # Infinite cluster model caching
│   ├── init.jl                   # Initialization functions
│   ├── mh_updates.jl             # Metropolis-Hastings updates
│   ├── wavelet_block.jl          # Wavelet block operations
│   ├── postprocessing.jl         # Post-processing utilities
│   ├── plotting.jl               # Plotting functions
│   └── common.jl                 # Common utilities
├── examples/                      # Example scripts and data
│   ├── arrowhead_example.jl      # ArrowHead dataset example
│   ├── simulation_study_example.jl # Comprehensive simulation study
│   ├── simple_example.jl         # Minimal working example
│   ├── ArrowHead/                 # ArrowHead dataset files
│   ├── plots/                     # Generated plots
│   └── simstudy_results/          # Simulation study results
├── test/                         # Test suite
│   └── runtests.jl               # Main test file
├── docs/                         # Documentation (empty for now)
├── Project.toml                  # Package metadata and dependencies
├── Manifest.toml                  # Exact dependency versions
└── README.md                     # Package documentation
```

## Key Changes Made

### 1. Package Structure
- **Created proper Julia package layout** with `src/`, `examples/`, `test/`, and `docs/` directories
- **Moved core functionality** to `src/WICMAD.jl` as the main module
- **Organized examples** in dedicated `examples/` directory
- **Added test suite** in `test/runtests.jl`

### 2. Module Organization
- **Main module**: `src/WICMAD.jl` contains the primary `wicmad()` function and exports
- **Submodules**: All supporting functionality remains in separate files
- **Clean exports**: Only essential functions are exported (`wicmad`, `adj_rand_index`, `choose2`, `init_diagnostics`)

### 3. Example Scripts
- **ArrowHead Example**: `examples/arrowhead_example.jl` - Demonstrates real dataset analysis
- **Simulation Study**: `examples/simulation_study_example.jl` - Comprehensive multi-dataset evaluation
- **Simple Example**: `examples/simple_example.jl` - Minimal working example for beginners

### 4. Package Metadata
- **Updated Project.toml** with proper package name, UUID, version, and author information
- **Maintained all dependencies** from the original project
- **Added compatibility constraints** for Julia 1.11+

### 5. Documentation
- **Comprehensive README.md** with installation, usage, and API documentation
- **Code examples** showing typical usage patterns
- **Performance tips** and best practices
- **SLURM integration** documentation for HPC usage

## Usage Patterns

### As a Package
```julia
using Pkg
Pkg.develop(path="/path/to/WICMAD.jl")
using WICMAD

# Use the package
result = wicmad(Y, t; wf="la8", n_iter=1000)
```

### Running Examples
```bash
# Simple example
julia examples/simple_example.jl

# ArrowHead dataset
julia examples/arrowhead_example.jl

# Comprehensive simulation study
julia examples/simulation_study_example.jl
```

### Running Tests
```bash
julia --project=. test/runtests.jl
```

## Migration Notes

### From Original Scripts
- **simulation_study.jl** → **examples/simulation_study_example.jl** (updated for package usage)
- **run_arrowhead.jl** → **examples/arrowhead_example.jl** (updated for package usage)
- **All core functionality** → **src/WICMAD.jl** (maintained with same API)

### SLURM Compatibility
- **Maintained SLURM support** in all example scripts
- **Worker setup** remains the same with proper project path handling
- **Parallel processing** works identically to original implementation

### Data Handling
- **ArrowHead dataset** moved to `examples/ArrowHead/`
- **Generated plots** moved to `examples/plots/`
- **Results** moved to `examples/simstudy_results/`

## Benefits of Package Structure

1. **Reusability**: Easy to import and use in other projects
2. **Maintainability**: Clear separation of concerns and modular design
3. **Testability**: Comprehensive test suite ensures reliability
4. **Documentation**: Professional documentation and examples
5. **Distribution**: Can be easily shared and installed by others
6. **Development**: Proper development workflow with Pkg.jl

## Next Steps

1. **Generate UUID**: Replace placeholder UUID in Project.toml with a real one
2. **Update Author**: Replace placeholder author information
3. **Add CI/CD**: Set up GitHub Actions for automated testing
4. **Documentation**: Add more detailed API documentation
5. **Performance**: Profile and optimize critical functions
6. **Extensions**: Add more wavelet families or clustering methods

The package is now ready for distribution and can be easily installed and used by others in the Julia community.
