# Monero GPU Cracker

**Monero GPU Cracker v0.1** is a standalone tool designed to crack Monero wallet passwords using GPU acceleration via CUDA. It takes a hash extracted by `monero2john.py` and tests it against a provided wordlist. This project is heavily derived from the Monero module in [John the Ripper Bleeding Jumbo](https://github.com/openwall/john), with significant enhancements including a CUDA-ported slow hash for GPU acceleration, while retaining CPU-based finalization and validation.

## Features
- **CUDA-Accelerated Slow Hash**: Leverages NVIDIA GPU power to compute the CryptoNight slow hash, significantly speeding up password cracking.
- **Wordlist-Based Cracking**: Tests passwords from a user-supplied wordlist against a Monero wallet hash.
- **Session Save/Resume**: Supports saving progress to resume later with `-s` and `-r` options.
- **Tunable Performance**: Adjustable batch size (`-b`) and threads per block (`-t`), with defaults optimized for NVIDIA GeForce RTX 4090.
- **Benchmark Mode**: Test performance with dummy passwords using `-B [<num>]` (defaults to 20,000).
- **Detailed Stats**: Verbosity levels (0-3) provide GPU memory usage, hash rate, temperature, and more.

## Notes
- **Tested Hardware**: This tool has been tested exclusively on an NVIDIA GeForce RTX 4090. Performance may vary on other GPUs.
- **Batch Size**: Defaults to an automatic setting based on available GPU memory (80% of free memory), but can be tuned with `-b <batch_size>` for optimization.
- **Threads**: Set to 32 by default, which was optimal for the RTX 4090. Adjust with `-t <threads>` if needed for other GPUs.

## Requirements
- **NVIDIA GPU**: CUDA-capable GPU required (e.g., RTX 4090).
- **CUDA Toolkit**: Installed with `nvcc` (e.g., version 11.x or later).
- **NVIDIA Driver**: Compatible with your CUDA Toolkit version.
- **NVML Library**: For GPU monitoring (`libnvidia-ml`).
- **OpenMP**: For CPU parallelization (`-fopenmp` support in compiler).
- **Operating System**: Tested on Linux (e.g., Ubuntu); may work on other UNIX-like systems with adjustments.

## Installation

### Cloning the Repository
```bash
git clone https://github.com/cynicalpeace/monero_gpu_cracker.git
cd monero_cracker
```

### Compilation
The project requires compiling CUDA source files separately before linking:

#### Compile CUDA Object Files:
```bash
nvcc -rdc=true -c aes_cuda.cu -o aes_cuda.o
nvcc -rdc=true -c cn_cuda.cu -o cn_cuda.o
nvcc -rdc=true -c keccak_cuda.cu -o keccak_cuda.o
```

#### Link and Build the Executable:
```bash
nvcc -rdc=true -arch=sm_89 -o monero_cracker main.c slow_hash_plug.c keccak_plug.c chacha_plug.c blake256_plug.c groestl_plug.c jh_plug.c skein.c oaes_lib_plug.c KeccakSponge.c KeccakF-1600-opt64.c common.c memory.c misc.c aes_cuda.o keccak_cuda.o cn_cuda.o -Xcompiler -fopenmp -I. -I./mbedtls -lcudart -L/usr/local/cuda/lib64 -L/usr/lib/nvidia -lnvidia-ml

./monero_cracker
```

### Notes
- Ensure the CUDA Toolkit and NVIDIA driver paths match your system (e.g., adjust `-L/usr/local/cuda/lib64` if necessary).

## Usage
```
Monero GPU Cracker v0.1 by cynicalpeace
GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
Usage: ./monero_cracker <hash_file> --wordlist <wordlist_file> [options]
Benchmark Mode: ./monero_cracker -B [<num_passwords>] [options]

Options:
  -t, --threads <threads>    Set threads per block (default: 32)
  -b, --batch <batch_size>   Set batch size (default: auto)
  -B, --benchmark [<num>]    Run benchmark with <num> passwords (default: 20000)
  -s, --session <name>       Save progress under <name>
  -r, --resume <name>        Resume from session <name>
  -c, --checkpoint <seconds> Checkpoint save interval (default: 60s)
  -v, --verbose <level>      Verbosity (0-3, default: 2)
  -w, --wordlist <file>      Wordlist file
  -h, --help                 Show this help

Note: <hash_file> should contain only the hash from monero2john.py.
             e.g., '$monero$0*886500ad343766b8850dee...'
```

### Examples:
```bash
./monero_cracker hash.txt --wordlist words.txt -t 128 -b 5000
./monero_cracker -B 10000 -t 96 -b 5000
./monero_cracker hash.txt --wordlist words.txt -r mysession
```

### Sample Output
```bash
$ ./monero_cracker --benchmark --batch 10000 --threads 32 --verbose 0
Monero GPU Cracker v0.1 by cynicalpeace
GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
Using GPU: NVIDIA GeForce RTX 4090 with 24195 MB total memory
Total GPU memory: 24195 MB, Free: 23007 MB, Required: 20000 MB
Running benchmark: 20000 passwords, 32 threads, 10000 batch size
Note: Threads and batch size can be set with -t and -b

--- Result ---
Benchmark stopped at the last full batch
Passwords tried: 20000 / 20000
GPU time: 10.316 seconds
CPU time: 0.004 seconds
Total time: 10.367 seconds
Average Throughput: 1938.66 hashes/second
Program completed
```

## License
This project is licensed under the MIT License. See the license header in `main.c` for details.

Original code from John the Ripper Bleeding Jumbo is used under its relaxed BSD-style license or public domain terms where applicable (see John the Ripper repo).

## Author
- **cynicalpeace** - 2025
- GitHub: [cynicalpeace](https://github.com/cynicalpeace)


## Contributing

Feel free to submit issues or pull requests on GitHub. If testing on other GPUs, please report performance metrics (e.g., hash rate, optimal threads/batch size) to improve compatibility.

## Troubleshooting

- **NVML Errors**: Ensure `libnvidia-ml` is installed (`sudo apt-get install libnvidia-ml-dev` on Ubuntu).
- **CUDA Errors**: Verify CUDA Toolkit installation and driver compatibility.
- **Missing Files**: Ensure all `.c`, `.cu`, and `.h` files from the repository are present in the build directory.

## To Do

- Add support for additional GPU architectures.
- Implement support for accepting multiple hashes from a single file for concurrent cracking.
- Profile kernel for memory optimization.
