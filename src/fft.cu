// File: AudioFilter.cu
// Compile with:
//   nvcc -std=c++17 AudioFilter.cu -I/path/to/AudioFile -o bin/AudioFilter

#define DR_WAV_IMPLEMENTATION
#include "AudioFile.h"
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <filesystem>
#include <cufft.h> // For the Fast Fourier Transform library
#include <cmath>   // For sqrt() to calculate magnitude
#include <fstream> // To write results to a CSV file
namespace fs = std::filesystem;

size_t workSize;

// CUDA error-check helper
inline void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg
                  << " (" << cudaGetErrorString(err) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}
// cuFFT error-check helper
inline void checkCufft(cufftResult err, const char *msg) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << "cuFFT Error: " << msg << " (" << err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

// Process a single WAV file: load, perform FFT, find peak frequency
void processFile(const fs::path &inPath, std::ofstream &resultsFile) {
    std::cout << "Processing: " << inPath.filename().string() << "\n";
    AudioFile<float> audioFile;

    // 1. Load the audio file
    if (!audioFile.load(inPath.string())) {
        std::cerr << "  ERROR: Failed to load " << inPath.filename().string() << "\n";
        return;
    }

    // For simplicity, we only process the first audio channel
    const int numSamples = audioFile.getNumSamplesPerChannel();
    const float sampleRate = audioFile.getSampleRate();
    std::vector<float> h_in = audioFile.samples[0];

    // --- GPU Processing Starts Here ---

    // 2. Allocate GPU memory and copy input data
    float* d_in = nullptr;
    checkCuda(cudaMalloc(&d_in, numSamples * sizeof(float)), "cudaMalloc d_in");
    checkCuda(cudaMemcpy(d_in, h_in.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice), "H2D memcpy");

    // The output of a real-to-complex FFT has (N/2 + 1) complex elements
    const int fft_output_size = (numSamples / 2) + 1;
    cufftComplex* d_out = nullptr;
    checkCuda(cudaMalloc(&d_out, fft_output_size * sizeof(cufftComplex)), "cudaMalloc d_out");

    // 3. Create and execute the cuFFT plan
    // 3. Create and execute the cuFFT plan
    cufftHandle plan;
    checkCufft(cufftCreate(&plan), "cufftCreate");
    checkCufft(cufftMakePlan1d(plan, numSamples, CUFFT_R2C, 1 ,&workSize), "cufftMakePlan1d");

    checkCufft(cufftExecR2C(plan, d_in, d_out), "cufftExecR2C");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after kernel");
    // 4. Copy complex results from GPU back to host
    std::vector<cufftComplex> h_out(fft_output_size);
    checkCuda(cudaMemcpy(h_out.data(), d_out, fft_output_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "D2H memcpy");

    // --- CPU Post-Processing ---

    // 5. Find the peak frequency on the CPU
    float maxMagnitude = 0.0f;
    int peakIndex = 0;
    // Start at i=1 to ignore the DC component (0 Hz)
    for (int i = 1; i < fft_output_size; ++i) {
        float magnitude = sqrtf(h_out[i].x * h_out[i].x + h_out[i].y * h_out[i].y);
        if (magnitude > maxMagnitude) {
            maxMagnitude = magnitude;
            peakIndex = i;
        }
    }
    float peakFrequency = static_cast<float>(peakIndex) * sampleRate / numSamples;

    // 6. Write result to the CSV file
    resultsFile << inPath.filename().string() << "," << peakFrequency << "\n";
    std::cout << "  > Peak Frequency: " << peakFrequency << " Hz\n";

    // 7. Clean up all resources
    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
}
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <input_dir> <output_dir>\n";
        return 1;
    }

    fs::path inDir  = argv[1];
    fs::path outDir = argv[2];
    fs::create_directories(outDir);

    // Create a CSV file for the output results
    std::ofstream resultsFile(outDir / "frequency_results.csv");
    resultsFile << "filename,peak_frequency_hz\n"; // Write CSV header

    for (auto &entry : fs::directory_iterator(inDir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".wav") {
            continue;
        }
        // No need to create an output path for each file anymore
        processFile(entry.path(), resultsFile); 
    }

    std::cout << "Done. Results saved to " 
              << (outDir / "frequency_results.csv") << "\n";
    resultsFile.close();
    return 0;
}