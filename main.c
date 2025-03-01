/*
 * Monero GPU Cracker v0.1
 * Copyright (c) 2025 cynicalpeace
 * GitHub: https://github.com/cynicalpeace/monero_gpu_cracker
 * 
 * This program is a standalone Monero wallet password cracker that uses CUDA
 * to accelerate the slow hash computation. It takes a hash from monero2john.py
 * and tests it against a wordlist. Derived from the Monero module in John the
 * Ripper Bleeding Jumbo (https://github.com/openwall/john), with significant
 * modifications including CUDA porting of the slow hash, while retaining CPU-
 * based finalization and validation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Original John the Ripper code is licensed under a relaxed BSD-style license
 * or public domain where applicable (see https://github.com/openwall/john).
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <omp.h>
 #include <time.h>
 #include <getopt.h>
 #include <nvml.h>
 #include <cuda_runtime.h>
 #include <unistd.h>
 #include <ctype.h>  // For isdigit
 #include "cn_cuda.h"
 #include "chacha.h"
 #include "common.h"
 #include "memory.h"
 
 #define FORMAT_TAG "$monero$0*"
 #define FORMAT_TAG_LEN (sizeof(FORMAT_TAG) - 1)
 #define IVLEN 8
 #define BINARY_SIZE 32
 #define SALT_SIZE (IVLEN + 2)
 #define MEMORY_PER_PASSWORD (1 << 21) + 200
 #define MIN_BATCH_SIZE 10
 #define MAX_BATCH_SIZE 10000
 #define DEFAULT_BENCHMARK_PASSWORDS 20000
 #define DEFAULT_CHECKPOINT_INTERVAL 60
 #define RATE_WINDOW_SIZE 5  // Number of batches to average hash rate over
 
 struct custom_salt {
     unsigned char iv[IVLEN];
     unsigned char *ct;
     unsigned int len;
 };
 
 // Global variables
 int verbosity = 2;
 int benchmark_mode = 0;
 long long benchmark_passwords = 0;
 char *session_name = NULL;
 char *checkpoint_file = NULL;
 int checkpoint_interval = DEFAULT_CHECKPOINT_INTERVAL;
 size_t passwords_tested = 0;
 size_t initial_passwords_tested = 0;
 size_t total_candidates = 0;
 double gpu_time = 0.0, cpu_time = 0.0;
 size_t total_passwords_processed = 0;  // Tracks total passwords processed
 double total_gpu_time = 0.0;           // Tracks total GPU time
 struct timespec start_time, end_time, last_checkpoint_time;
 FILE *wordlist_file = NULL;
 long long file_position = 0;
 size_t batch_size = 0;
 size_t actual_batch_size = 0;
 nvmlDevice_t nvml_device;
 char *generated_hash = NULL;
 double batch_rates[RATE_WINDOW_SIZE] = {0};
 int batch_count = 0;
 char *found_password = NULL;           // To store the found password
 
 // Function prototypes
 void print_help();
 void print_stats(int level, const char *first_pwd, const char *last_pwd);
 void generate_random_hash();
 void save_checkpoint();
 void load_checkpoint();
 
 // Save progress to checkpoint file
 void save_checkpoint() {
     if (!session_name || !checkpoint_file) return;
     FILE *checkpoint_fp = fopen(checkpoint_file, "w");
     if (checkpoint_fp) {
         fprintf(checkpoint_fp, "%zu %lld\n", passwords_tested, file_position);
         fclose(checkpoint_fp);
     } else {
         fprintf(stderr, "Warning: Could not save checkpoint to %s\n", checkpoint_file);
     }
 }
 
 // Load progress from checkpoint file
 void load_checkpoint() {
     if (!session_name || !checkpoint_file) return;
     FILE *checkpoint_fp = fopen(checkpoint_file, "r");
     if (checkpoint_fp) {
         if (fscanf(checkpoint_fp, "%zu %lld", &passwords_tested, &file_position) == 2) {
             initial_passwords_tested = passwords_tested;
             fseek(wordlist_file, file_position, SEEK_SET);
             printf("Continuing from line %zu (passwords tested: %zu)\n", passwords_tested, passwords_tested);
         } else {
             fprintf(stderr, "Warning: Invalid checkpoint file format in %s\n", checkpoint_file);
         }
         fclose(checkpoint_fp);
     } else {
         fprintf(stderr, "Warning: Could not open checkpoint file %s\n", checkpoint_file);
     }
 }
 
 // Print stats based on verbosity level after each batch
 void print_stats(int level, const char *first_pwd, const char *last_pwd) {
     if (level <= 0) return;
     double avg_rate = 0.0;
     int valid_batches = batch_count < RATE_WINDOW_SIZE ? batch_count : RATE_WINDOW_SIZE;
     if (valid_batches > 0) {
         for (int i = 0; i < valid_batches; i++) {
             avg_rate += batch_rates[i];
         }
         avg_rate /= valid_batches;
     }
 
     if (level >= 1) {
         printf("\n--- Stats ---\n");
         printf("Passwords tried: %zu / %zu\n", passwords_tested, total_candidates);
         printf("Current batch passwords: %s -> %s\n", first_pwd ? first_pwd : "N/A", last_pwd ? last_pwd : "N/A");
         printf("Hash rate: %.2f hashes/second\n", avg_rate);
     }
     if (level >= 2) {
         nvmlUtilization_t utilization;
         unsigned int temperature;
         nvmlDeviceGetUtilizationRates(nvml_device, &utilization);
         nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temperature);
         printf("Current batch size: %zu\n", batch_size);
         printf("GPU Usage: %u%%\n", utilization.gpu);
         printf("GPU Temperature: %u°C\n", temperature);
     }
     if (level >= 3) {
         printf("Total batches processed: %d\n", batch_count);
         printf("Recent batch rates: ");
         for (int i = 0; i < valid_batches; i++) {
             printf("%.2f ", batch_rates[i]);
         }
         printf("\n");
     }
     fflush(stdout);
 }
 
 // Generate a random hash for benchmarking
 void generate_random_hash() {
     generated_hash = malloc(2048);
     sprintf(generated_hash, "%s", FORMAT_TAG);
     for (int i = 0; i < 256; i++) {
         sprintf(generated_hash + strlen(generated_hash), "%02x", rand() % 256);
     }
 }
 
 // Display usage information
 void print_help() {
     printf("Monero GPU Cracker v0.1 by cynicalpeace\n");
     printf("GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
     printf("Usage: ./monero_cracker <hash_file> --wordlist <wordlist_file> [options]\n");
     printf("Benchmark Mode: ./monero_cracker -B [<num_passwords>] [options]\n");
     printf("\nOptions:\n");
     printf("  -t, --threads <threads>    Set threads per block (default: 32)\n");
     printf("  -b, --batch <batch_size>   Set batch size (default: auto)\n");
     printf("  -B, --benchmark [<num>]    Run benchmark with <num> passwords (default: %d)\n", DEFAULT_BENCHMARK_PASSWORDS);
     printf("  -s, --session <name>       Save progress under <name>\n");
     printf("  -r, --resume <name>        Resume from session <name>\n");
     printf("  -c, --checkpoint <seconds> Checkpoint save interval (default: 60s)\n");
     printf("  -v, --verbose <level>      Verbosity (0-3, default: 2)\n");
     printf("  -w, --wordlist <file>      Wordlist file\n");
     printf("  -h, --help                 Show this help\n");
     printf("\nNote: <hash_file> should contain only the hash from monero2john.py.\n\n");
     printf("            e.g., '$monero$0*886500ad343766b8850dee...'\n\n");
     printf("Examples:\n");
     printf("  ./monero_cracker hash.txt --wordlist words.txt -t 128 -b 5000\n");
     printf("  ./monero_cracker -B 50000 -t 32 -b 5000\n");
     printf("  ./monero_cracker hash.txt --wordlist words.txt -r mysession\n");
 }
 
 int main(int argc, char *argv[]) {
     FILE *hash_file = NULL;
     char *hash_fname = NULL, *wordlist_fname = NULL;
     char line[2048], *raw_hash = NULL;
     struct custom_salt cur_salt;
     unsigned char *salt_bytes = NULL;
     char **saved_key = NULL;
     unsigned char **plaintext = NULL;
     int found = 0;
     int threads_per_block = 32;
     char *batch_size_str = "auto";
     int batch_size_mode = 0;
     size_t fixed_batch_size = 0;
     int i;
     size_t salt_len;
 
     // Define long options
     static struct option long_options[] = {
         {"threads", required_argument, 0, 't'},
         {"batch", required_argument, 0, 'b'},
         {"benchmark", optional_argument, 0, 'B'},
         {"session", required_argument, 0, 's'},
         {"resume", required_argument, 0, 'r'},
         {"checkpoint", required_argument, 0, 'c'},
         {"verbose", required_argument, 0, 'v'},
         {"wordlist", required_argument, 0, 'w'},
         {"help", no_argument, 0, 'h'},
         {0, 0, 0, 0}
     };
 
     // Parse command-line arguments
     int opt;
     opterr = 0;
     while ((opt = getopt_long(argc, argv, "t:b:B::s:r:c:v:w:h", long_options, NULL)) != -1) {
         switch (opt) {
             case 't':
                 threads_per_block = atoi(optarg);
                 if (threads_per_block <= 0) {
                     fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                     fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                     fprintf(stderr, "Error: Threads must be positive\n");
                     return 1;
                 }
                 break;
             case 'b':
                 batch_size_str = optarg;
                 if (strcmp(batch_size_str, "auto") != 0) {
                     fixed_batch_size = atoi(batch_size_str);
                     if (fixed_batch_size <= 0) {
                         fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                         fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                         fprintf(stderr, "Error: Batch size must be positive or 'auto'\n");
                         return 1;
                     }
                     batch_size_mode = 1;
                 }
                 break;
             case 'B':
                 benchmark_mode = 1;
                 if (optarg) {
                     benchmark_passwords = atoll(optarg);
                     if (benchmark_passwords <= 0) {
                         fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                         fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                         fprintf(stderr, "Error: Benchmark passwords must be positive\n");
                         return 1;
                     }
                 } else if (optind < argc && isdigit(argv[optind][0])) {
                     benchmark_passwords = atoll(argv[optind]);
                     optind++;
                     if (benchmark_passwords <= 0) {
                         fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                         fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                         fprintf(stderr, "Error: Benchmark passwords must be positive\n");
                         return 1;
                     }
                 } else {
                     benchmark_passwords = DEFAULT_BENCHMARK_PASSWORDS;
                 }
                 break;
             case 's':
                 session_name = optarg;
                 checkpoint_file = malloc(strlen(session_name) + 9);
                 sprintf(checkpoint_file, "%s.restore", session_name);
                 break;
             case 'r':
                 session_name = optarg;
                 checkpoint_file = malloc(strlen(session_name) + 9);
                 sprintf(checkpoint_file, "%s.restore", session_name);
                 break;
             case 'c':
                 checkpoint_interval = atoi(optarg);
                 if (checkpoint_interval <= 0) {
                     fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                     fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                     fprintf(stderr, "Error: Checkpoint interval must be positive\n");
                     return 1;
                 }
                 break;
             case 'v':
                 verbosity = atoi(optarg);
                 if (verbosity < 0 || verbosity > 3) {
                     fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                     fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                     fprintf(stderr, "Error: Verbosity must be 0-3\n");
                     return 1;
                 }
                 break;
             case 'w':
                 wordlist_fname = optarg;
                 break;
             case 'h':
                 print_help();
                 return 0;
             case '?':
                 fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                 fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                 fprintf(stderr, "Error: Invalid option or missing argument\n\n");
                 fprintf(stderr, "\ttype ./monero_cracker --help for usage instructions\n\n");
                 return 1;
             default:
                 fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
                 fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
                 fprintf(stderr, "Error: Unexpected error parsing options\n");
                 return 1;
         }
     }
 
     // Validate arguments based on mode
     if (!benchmark_mode) {
         if (optind >= argc || !wordlist_fname) {
             fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
             fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
             fprintf(stderr, "Error: Hash file and --wordlist required in non-benchmark mode\n\n");
             fprintf(stderr, "\ttype ./monero_cracker --help for usage instructions\n\n");
             return 1;
         }
         hash_fname = argv[optind];
     } else if (optind < argc) {
         printf("Warning: Additional positional arguments ignored in benchmark mode\n");
     }
 
     // Initialize NVML
     if (nvmlInit() != NVML_SUCCESS || nvmlDeviceGetHandleByIndex(0, &nvml_device) != NVML_SUCCESS) {
         fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
         fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
         fprintf(stderr, "Error: NVML initialization failed\n");
         return 1;
     }
 
     // Print program information
     printf("Monero GPU Cracker v0.1 by cynicalpeace\n");
     printf("GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
     int device;
     cudaGetDevice(&device);
     struct cudaDeviceProp props;
     cudaGetDeviceProperties(&props, device);
     printf("Using GPU: %s with %zu MB total memory\n", props.name, props.totalGlobalMem / 1024 / 1024);
     nvmlMemory_t memory_info;
     nvmlDeviceGetMemoryInfo(nvml_device, &memory_info);
 
     // Handle hash
     if (benchmark_mode) {
         generate_random_hash();
         raw_hash = generated_hash;
     } else {
         hash_file = fopen(hash_fname, "r");
         if (!hash_file) {
             fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
             fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
             fprintf(stderr, "Error: Error opening hash file: %s\n", hash_fname);
             nvmlShutdown();
             return 1;
         }
         if (fgets(line, sizeof(line), hash_file)) {
             raw_hash = strdup(line);
             raw_hash[strcspn(raw_hash, "\n")] = 0;
         }
         fclose(hash_file);
         if (!raw_hash || strncmp(raw_hash, FORMAT_TAG, FORMAT_TAG_LEN) != 0) {
             fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
             fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
             fprintf(stderr, "Error: Invalid hash format\n");
             free(raw_hash);
             nvmlShutdown();
             return 1;
         }
     }
 
     // Parse hash
     char *p = raw_hash + FORMAT_TAG_LEN;
     salt_len = strlen(p) / 2;
     salt_bytes = mem_alloc_tiny(salt_len, MEM_ALIGN_WORD);
     for (i = 0; i < salt_len; i++) {
         sscanf(p + 2 * i, "%2hhx", &salt_bytes[i]);
     }
     memcpy(cur_salt.iv, salt_bytes, IVLEN);
     cur_salt.ct = salt_bytes + IVLEN + 2;
     cur_salt.len = salt_len - IVLEN - 2;
 
     // Open wordlist or prepare benchmark
     if (!benchmark_mode) {
         wordlist_file = fopen(wordlist_fname, "r");
         if (!wordlist_file) {
             fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
             fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
             fprintf(stderr, "Error: Error opening wordlist: %s\n", wordlist_fname);
             free(raw_hash);
             mem_free(salt_bytes);
             nvmlShutdown();
             return 1;
         }
         total_candidates = 0;
         while (fgets(line, sizeof(line), wordlist_file)) total_candidates++;
         rewind(wordlist_file);
         if (session_name && checkpoint_file) load_checkpoint();
 
         size_t free_mem, total_mem;
         cudaMemGetInfo(&free_mem, &total_mem);
         if (batch_size_mode == 0) {
             size_t available_mem = free_mem * 0.8;
             batch_size = available_mem / MEMORY_PER_PASSWORD;
             batch_size = batch_size < MIN_BATCH_SIZE ? MIN_BATCH_SIZE : (batch_size > MAX_BATCH_SIZE ? MAX_BATCH_SIZE : batch_size);
             batch_size = batch_size > total_candidates ? total_candidates : batch_size;
         } else {
             batch_size = fixed_batch_size;
             if (batch_size > total_candidates) batch_size = total_candidates;
         }
         printf("Total GPU memory: %zu MB, Free: %zu MB, Required: %zu MB\n", 
                total_mem / (1024 * 1024), free_mem / (1024 * 1024), (batch_size * MEMORY_PER_PASSWORD) / (1024 * 1024));
         printf("Starting key search: %lld passwords, %d threads, %zu batch size\n",
                (long long)total_candidates, threads_per_block, batch_size);
         printf("Note: Threads and batch size can be set with -t and -b\n");
         printf("Attempting hash: %.20s...\n", raw_hash);
     } else {
         total_candidates = benchmark_passwords;
     }
 
     // Start timing
     clock_gettime(CLOCK_MONOTONIC, &start_time);
     last_checkpoint_time = start_time;
 
     // Process in batches
     size_t memory_per_password = MEMORY_PER_PASSWORD;
     size_t min_batch_size = MIN_BATCH_SIZE;
     size_t max_batch_size = MAX_BATCH_SIZE;
 
     if (benchmark_mode) {
         size_t free_mem, total_mem;
         cudaMemGetInfo(&free_mem, &total_mem);
         if (batch_size_mode == 0) {
             size_t available_mem = free_mem * 0.8;
             batch_size = available_mem / memory_per_password;
             batch_size = batch_size < min_batch_size ? min_batch_size : (batch_size > max_batch_size ? max_batch_size : batch_size);
             batch_size = batch_size > benchmark_passwords ? benchmark_passwords : batch_size;
         } else {
             batch_size = fixed_batch_size;
             if (batch_size > benchmark_passwords) batch_size = benchmark_passwords;
         }
         printf("Total GPU memory: %zu MB, Free: %zu MB, Required: %zu MB\n", 
                total_mem / (1024 * 1024), free_mem / (1024 * 1024), (batch_size * MEMORY_PER_PASSWORD) / (1024 * 1024));
         printf("Running benchmark: %lld passwords, %d threads, %zu batch size\n",
                (long long)total_candidates, threads_per_block, batch_size);
         printf("Note: Threads and batch size can be set with -t and -b\n");
     }
 
     while ((benchmark_mode ? passwords_tested < benchmark_passwords : passwords_tested < total_candidates) && !found) {
         if (batch_size_mode == 0 && !benchmark_mode) {
             size_t free_mem, total_mem;
             cudaMemGetInfo(&free_mem, &total_mem);
             size_t available_mem = free_mem * 0.8;
             batch_size = available_mem / memory_per_password;
             batch_size = batch_size < min_batch_size ? min_batch_size : (batch_size > max_batch_size ? max_batch_size : batch_size);
             batch_size = batch_size > (total_candidates - passwords_tested) ? (total_candidates - passwords_tested) : batch_size;
         } else if (!benchmark_mode) {
             batch_size = fixed_batch_size;
             if (batch_size > total_candidates - passwords_tested) batch_size = total_candidates - passwords_tested;
         }
 
         // In benchmark mode, stop at the last full batch
         if (benchmark_mode && (passwords_tested + batch_size > benchmark_passwords)) {
             actual_batch_size = (benchmark_passwords - passwords_tested) >= batch_size ? batch_size : 0;
         } else {
             actual_batch_size = batch_size;
         }
 
         if (actual_batch_size == 0) break; // Exit if no full batch remains
 
         // Allocate batch buffers
         saved_key = mem_alloc(actual_batch_size * sizeof(char *));
         plaintext = mem_alloc(actual_batch_size * sizeof(unsigned char *));
         unsigned char *hashes = mem_alloc(actual_batch_size * BINARY_SIZE);
 
         // Read a batch
         size_t batch_index = 0;
         while (batch_index < actual_batch_size && passwords_tested < (benchmark_mode ? benchmark_passwords : total_candidates)) {
             if (!benchmark_mode) {
                 if (fgets(line, sizeof(line), wordlist_file)) {
                     line[strcspn(line, "\n")] = 0;
                     saved_key[batch_index] = mem_alloc_tiny(strlen(line) + 1, MEM_ALIGN_WORD);
                     strcpy(saved_key[batch_index], line);
                 } else break;
             } else {
                 saved_key[batch_index] = mem_alloc_tiny(16, MEM_ALIGN_WORD);
                 sprintf(saved_key[batch_index], "dummy_pass_%zu", passwords_tested);
             }
             plaintext[batch_index] = mem_alloc_tiny(cur_salt.len, MEM_ALIGN_WORD);
             batch_index++;
             passwords_tested++;
             if (!benchmark_mode) file_position = ftell(wordlist_file);
         }
         actual_batch_size = batch_index;
 
         struct timespec gpu_start, gpu_end_batch, cpu_end_batch;
         clock_gettime(CLOCK_MONOTONIC, &gpu_start);
 
         // Compute hashes on GPU
         int ret = compute_cn_slow_hash_cuda((const char **)saved_key, actual_batch_size, hashes, threads_per_block);
         if (ret == -1) {
             fprintf(stderr, "Monero GPU Cracker v0.1 by cynicalpeace\n");
             fprintf(stderr, "GitHub: https://github.com/cynicalpeace/monero_gpu_cracker\n");
             fprintf(stderr, "Error: Memory allocation failed. Exiting.\n");
             for (size_t j = 0; j < actual_batch_size; j++) {
                 mem_free(saved_key[j]);
                 mem_free(plaintext[j]);
             }
             mem_free(saved_key);
             mem_free(plaintext);
             mem_free(hashes);
             exit(1);
         }
 
         clock_gettime(CLOCK_MONOTONIC, &gpu_end_batch);
 
         // Parallel decryption
         #pragma omp parallel for
         for (i = 0; i < actual_batch_size; i++) {
             if (found) continue;
             struct chacha_ctx ctx;
             chacha_keysetup(&ctx, hashes + i * BINARY_SIZE, 256);
             chacha_ivsetup(&ctx, cur_salt.iv, NULL, IVLEN);
             chacha_encrypt_bytes(&ctx, cur_salt.ct, plaintext[i], cur_salt.len, 20);
             if (memmem(plaintext[i], cur_salt.len, "key_data", 8) || memmem(plaintext[i], cur_salt.len, "m_creation_timestamp", 20)) {
                 #pragma omp critical
                 if (!found) {
                     found_password = strdup(saved_key[i]);  // Store the found password
                     found = 1;
                 }
             }
         }
 
         clock_gettime(CLOCK_MONOTONIC, &cpu_end_batch);
 
         // Accumulate times and update batch rate
         double batch_gpu_time = (gpu_end_batch.tv_sec - gpu_start.tv_sec) + (gpu_end_batch.tv_nsec - gpu_start.tv_nsec) / 1e9;
         total_gpu_time += batch_gpu_time;
         total_passwords_processed += actual_batch_size;
         gpu_time += batch_gpu_time;
         cpu_time += (cpu_end_batch.tv_sec - gpu_end_batch.tv_sec) + (cpu_end_batch.tv_nsec - gpu_end_batch.tv_nsec) / 1e9;
 
         // Update moving average of batch rates
         double batch_rate = batch_gpu_time > 0 ? (double)actual_batch_size / batch_gpu_time : 0;
         batch_rates[batch_count % RATE_WINDOW_SIZE] = batch_rate;
         batch_count++;
 
         // Print stats after each batch
         if (verbosity >= 2 && actual_batch_size > 0) {
             const char *first_pwd = saved_key[0];
             const char *last_pwd = saved_key[actual_batch_size - 1];
             print_stats(verbosity, first_pwd, last_pwd);
         }
 
         // Checkpointing
         if (session_name && !benchmark_mode) {
             struct timespec now;
             clock_gettime(CLOCK_MONOTONIC, &now);
             double elapsed = (now.tv_sec - last_checkpoint_time.tv_sec) + (now.tv_nsec - last_checkpoint_time.tv_nsec) / 1e9;
             if (elapsed >= checkpoint_interval) {
                 save_checkpoint();
                 last_checkpoint_time = now;
             }
         }
 
         // Clean up
         for (i = 0; i < actual_batch_size; i++) {
             if (saved_key[i]) mem_free(saved_key[i]);
             if (plaintext[i]) mem_free(plaintext[i]);
         }
         mem_free(saved_key);
         mem_free(plaintext);
         mem_free(hashes);
     }
 
     if (!benchmark_mode && wordlist_file) fclose(wordlist_file);
 
     clock_gettime(CLOCK_MONOTONIC, &end_time);
 
     if (!found && !benchmark_mode) printf("No matching password found after testing %zu candidates\n", passwords_tested);
 
     double total_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
     double effective_hash_rate = total_gpu_time > 0 ? (double)total_passwords_processed / total_gpu_time : 0;
     printf("\n--- Result ---\n");
     if (benchmark_mode) printf("Benchmark stopped at the last full batch\n");
     printf("Passwords tried: %zu / %lld\n", total_passwords_processed, benchmark_mode ? benchmark_passwords : (long long)total_candidates);
     printf("GPU time: %.3f seconds\n", total_gpu_time);
     printf("CPU time: %.3f seconds\n", cpu_time);
     printf("Total time: %.3f seconds\n", total_time);
     printf("Average Throughput: %.2f hashes/second\n", effective_hash_rate);
     if (found_password) {
         printf("\033[31mFound password: %s\033[0m\n", found_password);
         free(found_password);  // Free the stored password memory
     }
 
     // Cleanup
     if (!benchmark_mode && raw_hash) free(raw_hash);
     if (generated_hash) free(generated_hash);
     mem_free(salt_bytes);
     nvmlShutdown();
 
     printf("Program completed\n");
     exit(0);
 }