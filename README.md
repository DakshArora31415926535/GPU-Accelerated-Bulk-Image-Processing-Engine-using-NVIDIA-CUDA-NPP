# GPU-Accelerated Multi-Batch Image Processing Pipeline (CUDA + NPP)

## Overview
This project implements a GPU-accelerated image transformation pipeline powered by NVIDIA’s NPP framework. The system has been enhanced to support automated multi-batch execution, enabling processing of multiple grayscale images through a list-driven workflow. The design reflects practical high-throughput environments such as dataset preparation, archival imaging systems, and cloud-scale media pipelines where efficiency and automation are critical.

## Key Features
- Automated multi-batch processing through `--list images.txt`
- GPU-based spatial interpolation and image re-sampling using NPP primitives
- Minimal CPU involvement with parallelized GPU computation
- Demonstration dataset included for validation purposes
- Structured for scalability and extensibility

## Repository Structure
```
/src          → CUDA/NPP source code (main processing pipeline)
/data         → Input & sample output dataset
                 ├── lena1.pgm
                 └── lena_resized.pgm  (sample processed output)
/Makefile     → build instructions
/README.md    → documentation
images.txt    → file list for batch processing
```

## Running the Program

### Single-Image Example
```bash
./resizeNPP --input data/lena1.pgm --scale 0.5
```

### Multi-Batch Example
Add paths to `images.txt`:
```
data/lena1.pgm
```

Run batch mode:
```bash
./resizeNPP --list images.txt --scale 0.5
```

## Dataset Notes
The repository includes `lena1.pgm` and a generated example output `lena_resized.pgm` as proof-of-concept representatives. Any additional `.pgm` grayscale images may be placed in the `/data` directory and referenced through `images.txt`.

## Intended Use Cases
- Dataset preprocessing for ML and CV pipelines
- Digital asset management systems
- Bulk archival imaging workflows
- Automated image transformation backends

## Summary
This implementation demonstrates a scalable GPU-centric batch processing architecture using CUDA and NPP to efficiently handle high-volume grayscale image workloads, providing a foundation for advanced parallel imaging pipelines and enterprise-level performance optimization.
