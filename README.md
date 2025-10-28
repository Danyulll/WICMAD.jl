# WICMAD.jl

**W**avelet-based **I**nfinite **C**luster **M**odel for **A**rbitrary **D**ata

A Julia package for functional clustering using wavelet-based infinite cluster models with semi-supervised learning capabilities.

## Overview

WICMAD.jl implements a sophisticated Bayesian nonparametric approach to functional clustering that combines:

- **Wavelet decompositions** for functional data representation
- **Infinite cluster models** for automatic cluster number determination
- **Semi-supervised learning** with revealed cluster assignments
- **Multiple wavelet families** for robust performance across different data types

## Features

- 🎯 **Automatic cluster detection** - No need to specify the number of clusters
- 🌊 **Wavelet-based representation** - Handles both smooth and irregular functional data
- 🎓 **Semi-supervised learning** - Incorporate partial label information
- 🔄 **Multiple wavelet families** - Haar, Daubechies, Coiflets, Least Asymmetric, Best Localized, Fejer-Korovkin
- 📊 **Comprehensive diagnostics** - Track convergence and cluster evolution
- ⚡ **Parallel processing** - SLURM-compatible for high-performance computing
- 📈 **Rich visualization** - Built-in plotting capabilities for results analysis

## Installation

### From Source

```julia
using Pkg
Pkg.develop(url="https://github.com/yourusername/WICMAD.jl.git")
```

### Local Development

```julia
using Pkg
Pkg.develop(path="/path/to/WICMAD.jl")
```

## Quick Start

```julia
using WICMAD
using Random

# Generate some synthetic functional data
Random.seed!(42)
n_samples = 50
P = 64
t = collect(range(0, 1, length=P))

# Create two classes of functional data
Y = []
labels = []

# Class 1: Sine waves
for i in 1:25
    y = sin.(2π * t) .+ 0.1 * randn(P)
    push!(Y, y)
    push!(labels, 1)
end

# Class 2: Cosine waves  
for i in 1:25
    y = cos.(2π * t) .+ 0.1 * randn(P)
    push!(Y, y)
    push!(labels, 2)
end

# Run WICMAD clustering
result = wicmad(Y, t; n_iter=1000, burn=500, wf="la8")

# Extract final cluster assignments
final_clusters = result.Z[end, :]
println("Final clusters: ", length(unique(final_clusters)))
```

## Examples

The package includes comprehensive examples in the `examples/` directory:

### ArrowHead Dataset Example

```bash
julia examples/arrowhead_example.jl
```

This example demonstrates:
- Loading real functional data from UCR Time Series Archive
- Semi-supervised learning setup
- Comparison across multiple wavelet families
- Performance evaluation using Adjusted Rand Index

### Simulation Study Example

```bash
julia examples/simulation_study_example.jl
```

This example shows:
- Comprehensive evaluation across multiple datasets
- Parallel processing with SLURM
- Statistical performance analysis
- Best wavelet selection per dataset

## API Reference

### Main Function

```julia
wicmad(Y, t; kwargs...)
```

**Parameters:**

- `Y::Vector` - Vector of functional observations (matrices)
- `t` - Time grid for functional data

**Key Arguments:**

- `n_iter::Int = 6000` - Number of MCMC iterations
- `burn::Int = 3000` - Burn-in period
- `thin::Int = 5` - Thinning interval
- `wf::String = "la8"` - Wavelet family (see supported wavelets below)
- `revealed_idx::Vector{Int} = Int[]` - Indices of revealed cluster assignments
- `K_init::Int = 5` - Initial number of clusters
- `warmup_iters::Int = 100` - Warmup period for semi-supervised learning
- `diagnostics::Bool = true` - Enable diagnostic tracking

**Returns:**

A named tuple containing:
- `Z` - Cluster assignments for each iteration
- `alpha` - Concentration parameter samples
- `kern` - Kernel type samples
- `params` - Cluster parameter samples
- `K_occ` - Number of occupied clusters per iteration
- `loglik` - Log-likelihood samples
- `diagnostics` - Diagnostic information (if enabled)

### Supported Wavelets

The package supports the official wavelet families from Julia's Wavelets.jl package:

- **Haar**: `"haar"`
- **Daubechies**: `"db1"` through `"db20"` (where `"db1"` is equivalent to `"haar"`)
- **Coiflets**: `"coif2"`, `"coif4"`, `"coif6"`, `"coif8"`
- **Symlets**: `"sym4"`, `"sym5"`, `"sym6"`, `"sym7"`, `"sym8"`, `"sym9"`, `"sym10"`
- **Battle**: `"batt2"`, `"batt4"`, `"batt6"`
- **Beylkin**: `"beyl"`
- **Vaidyanathan**: `"vaid"`

### Automatic Wavelet Selection

WICMAD.jl includes an automatic wavelet selection feature that uses MCMC-based scoring to choose the most appropriate wavelet for your data:

```julia
# Enable automatic wavelet selection (default when revealed_idx is provided)
result = wicmad(Y, t; 
                revealed_idx=[1, 2, 3, 4, 5],  # Indices of known normal curves
                wf_candidates=["haar", "db4", "db8", "coif4", "sym8"])  # Optional custom candidates
```

**How it works:**
1. **MCMC Scoring**: Runs 3000 iterations (1000 burn-in) of the wavelet spike-slab model for each candidate
2. **Multi-Metric Evaluation**: Combines time-domain MSE, wavelet-domain MSE, and sparsity metrics
3. **Automatic Selection**: Chooses the wavelet with the best weighted score
4. **Analysis Continuation**: Uses selected wavelet for the main WICMAD analysis

**Benefits:**
- **Model-Consistent**: Uses the actual wavelet spike-slab model for selection
- **Data-Driven**: Based on comprehensive MCMC evaluation, not just visual inspection
- **Reproducible**: Deterministic selection process with detailed logging
- **Efficient**: No manual intervention required

### Utility Functions

```julia
adj_rand_index(z1, z2)  # Calculate Adjusted Rand Index
choose2(n)              # Binomial coefficient C(n,2)
init_diagnostics(diagnostics, keep, n_iter)  # Initialize diagnostic tracking
```

## Parallel Processing

WICMAD.jl supports parallel processing for large-scale analyses:

### SLURM Integration

```bash
#!/bin/bash
#SBATCH --job-name=wicmad
#SBATCH --ntasks=8
#SBATCH --time=24:00:00

julia examples/simulation_study_example.jl
```

The package automatically detects SLURM environment variables and launches workers accordingly.

### Local Parallel Processing

```julia
using Distributed
addprocs(4)  # Add 4 worker processes

# Your WICMAD analysis will automatically use available workers
```

## Performance Tips

1. **Wavelet Selection**: Start with `"la8"` (Least Asymmetric) for most applications
2. **Iterations**: Use at least 2000 iterations for reliable results
3. **Burn-in**: Set burn-in to 30-50% of total iterations
4. **Semi-supervised**: Reveal 10-20% of labels for best performance
5. **Parallel Processing**: Use multiple cores for large datasets

## Citation

If you use WICMAD.jl in your research, please cite:

```bibtex
@software{wicmad2024,
  title={WICMAD.jl: Wavelet-based Infinite Cluster Model for Arbitrary Data},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/WICMAD.jl}
}
```

## Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the Julia ecosystem
- Inspired by Bayesian nonparametric clustering methods
- Wavelet implementations from the Wavelets.jl package
- Parallel processing support via Distributed.jl and ClusterManagers.jl

## Support

For questions, issues, or feature requests, please:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/yourusername/WICMAD.jl/issues)
3. Create a new issue with detailed information

## Changelog

### v0.1.0
- Initial release
- Core WICMAD algorithm implementation
- Multiple wavelet family support
- Semi-supervised learning capabilities
- SLURM parallel processing support
- Comprehensive examples and documentation