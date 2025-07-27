// File: fft_async.cu
// Compile with:
//   nvcc -std=c++17 fft_async.cu -I/path/to/AudioFile -lcufft -o bin/fft_async

#define DR_WAV_IMPLEMENTATION
#include "AudioFile.h"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <numeric>

#include <cuda_runtime.h>
#include <cufft.h>

namespace fs = std::filesystem;

// --- Helper Functions ---
inline void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCufft(cufftResult err, const char *msg) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << "cuFFT Error: " << msg << " (" << err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

// --- Data Structure for each stream's resources ---
struct StreamData {
    cudaStream_t stream;
    cufftHandle plan;
    fs::path filePath;
    
    // Pinned host memory for async copies
    float* h_in_pinned;
    cufftComplex* h_out_pinned;

    // Device memory
    float* d_in;
    cufftComplex* d_out;

    // Audio metadata
    int numSamples;
    float sampleRate;
    bool inUse = false;
};

// --- CPU task to analyze results after GPU is done ---
// --- New Helper function to convert frequency to a musical note ---
std::string frequencyToNote(float freq) {
    if (freq <= 0) return "N/A";
    const char* noteNames[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    
    // Calculate the MIDI note number
    int midiNote = round(12 * log2(freq / 440.0f) + 69);
    
    // Determine the note name and octave
    std::string noteName = noteNames[midiNote % 12];
    int octave = (midiNote / 12) - 1;
    
    return noteName + std::to_string(octave);
}
// --- CPU task to analyze results after GPU is done ---
void postProcess(StreamData& data, std::ofstream& resultsFile, int transformSize) {
    const int TOP_N = 3; // Find the top 3 frequencies
    const int fft_output_size = (data.numSamples / 2) + 1;
    
    // Store all frequency magnitudes and indices
    std::vector<std::pair<float, int>> magnitudes;
    magnitudes.reserve(fft_output_size);

    for (int i = 1; i < fft_output_size; ++i) { // Start at 1 to ignore DC offset
        float mag = sqrtf(data.h_out_pinned[i].x * data.h_out_pinned[i].x + data.h_out_pinned[i].y * data.h_out_pinned[i].y);
        magnitudes.push_back({mag, i});
    }

    // Sort to find the frequencies with the highest magnitude
    std::sort(magnitudes.begin(), magnitudes.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    std::cout << "  > Finished " << data.filePath.filename().string() << ":\n";
    resultsFile << data.filePath.filename().string();

    // Get the top N results
    for (int i = 0; i < TOP_N; ++i) {
        if (i < magnitudes.size()) {
            int peakIndex = magnitudes[i].second;
            float peakFrequency = static_cast<float>(peakIndex) * data.sampleRate / transformSize;
            std::string note = frequencyToNote(peakFrequency);

            std::cout << "    - Freq " << i + 1 << ": " << peakFrequency << " Hz (" << note << ")\n";
            resultsFile << "," << peakFrequency << "," << note;
        } else {
            resultsFile << ",,"; // Add empty columns if fewer than N peaks are found
        }
    }
    resultsFile << "\n";
    data.inUse = false; // Mark stream as free
}
// --- Main Application ---
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>\n";
        return 1;
    }

    fs::path inDir = argv[1];
    fs::path outDir = argv[2];
    fs::create_directories(outDir);

    // --- Collect all WAV files ---
    std::vector<fs::path> filesToProcess;
    for (const auto& entry : fs::directory_iterator(inDir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".wav") {
            filesToProcess.push_back(entry.path());
        }
    }
    if (filesToProcess.empty()) {
        std::cout << "No .wav files found in input directory.\n";
        return 0;
    }
    int totalFiles = filesToProcess.size();

    // --- Setup Asynchronous Pipeline ---
    const int NUM_STREAMS = 4;
    std::vector<StreamData> streamData(NUM_STREAMS);

    const int MAX_SAMPLES = 44100 * 60; // Max 60 seconds at 44.1kHz
    const int MAX_FFT_OUTPUTS = MAX_SAMPLES / 2 + 1;

    // --- Initialization Loop (with corrected plan creation) ---
    for (int i = 0; i < NUM_STREAMS; ++i) {
        checkCuda(cudaStreamCreate(&streamData[i].stream), "Stream Create");
        checkCufft(cufftCreate(&streamData[i].plan), "cufftCreate");
        
        // Configure the plan ONCE for the maximum size
        size_t workSize;
        checkCufft(cufftMakePlan1d(streamData[i].plan, MAX_SAMPLES, CUFFT_R2C, 1, &workSize), "Make plan for max size");
        
        checkCuda(cudaMallocHost(&streamData[i].h_in_pinned, MAX_SAMPLES * sizeof(float)), "MallocHost h_in");
        checkCuda(cudaMallocHost(&streamData[i].h_out_pinned, MAX_FFT_OUTPUTS * sizeof(cufftComplex)), "MallocHost h_out");
        
        checkCuda(cudaMalloc(&streamData[i].d_in, MAX_SAMPLES * sizeof(float)), "Malloc d_in");
        checkCuda(cudaMalloc(&streamData[i].d_out, MAX_FFT_OUTPUTS * sizeof(cufftComplex)), "Malloc d_out");
    }

    std::ofstream resultsFile(outDir / "frequency_results.csv");
    resultsFile << "filename,freq1_hz,note1,freq2_hz,note2,freq3_hz,note3\n";

    int fileIndex = 0;
    int processedCount = 0;
    std::cout << "Starting processing for " << totalFiles << " files with " << NUM_STREAMS << " streams...\n";

    // --- Main Processing Loop ---
    while (processedCount < totalFiles) {
        for (int i = 0; i < NUM_STREAMS; ++i) {
            // Check if a previously launched stream is done
            if (streamData[i].inUse && cudaStreamQuery(streamData[i].stream) == cudaSuccess) {
                postProcess(streamData[i], resultsFile, MAX_SAMPLES); // Pass MAX_SAMPLES
                processedCount++;
            }

            // If the stream is free and there are files left, launch a new one
            if (!streamData[i].inUse && fileIndex < totalFiles) {
                // Load audio file (CPU task)
                AudioFile<float> audioFile;
                if (!audioFile.load(filesToProcess[fileIndex])) {
                    std::cerr << "ERROR: Failed to load " << filesToProcess[fileIndex].filename().string() << ", skipping.\n";
                    fileIndex++;
                    processedCount++;
                    continue;
                }
                
                int currentSamples = audioFile.getNumSamplesPerChannel();
                if (currentSamples > MAX_SAMPLES) {
                    std::cerr << "ERROR: File " << filesToProcess[fileIndex].filename().string() << " is too large, skipping.\n";
                    fileIndex++;
                    processedCount++;
                    continue;
                }

                // Prepare data for this stream
                auto& currentStream = streamData[i];
                currentStream.inUse = true;
                currentStream.filePath = filesToProcess[fileIndex];
                currentStream.numSamples = currentSamples;
                currentStream.sampleRate = audioFile.getSampleRate();
                std::copy(audioFile.samples[0].begin(), audioFile.samples[0].end(), currentStream.h_in_pinned);
                
                // --- Launch Async GPU Pipeline (plan is already made) ---
                checkCufft(cufftSetStream(currentStream.plan, currentStream.stream), "cufftSetStream");
                
                checkCuda(cudaMemcpyAsync(currentStream.d_in, currentStream.h_in_pinned, currentStream.numSamples * sizeof(float), cudaMemcpyHostToDevice, currentStream.stream), "MemcpyAsync H2D");
                checkCufft(cufftExecR2C(currentStream.plan, currentStream.d_in, currentStream.d_out), "cufftExecR2C");
                checkCuda(cudaMemcpyAsync(currentStream.h_out_pinned, currentStream.d_out, (currentStream.numSamples / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost, currentStream.stream), "MemcpyAsync D2H");
                
                fileIndex++;
            }
        }
    }
    
    // --- Final Synchronization and Cleanup ---
    checkCuda(cudaDeviceSynchronize(), "Final Device Sync");
    
    // Inside the final cleanup loop at the end of main
    for (int i = 0; i < NUM_STREAMS; ++i) {
        if (streamData[i].inUse) {
            postProcess(streamData[i], resultsFile, MAX_SAMPLES); // Pass MAX_SAMPLES
        }
    }
        
    resultsFile.close();

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cufftDestroy(streamData[i].plan);
        cudaStreamDestroy(streamData[i].stream);
        cudaFreeHost(streamData[i].h_in_pinned);
        cudaFreeHost(streamData[i].h_out_pinned);
        cudaFree(streamData[i].d_in);
        cudaFree(streamData[i].d_out);
    }
    
    std::cout << "Done. All files processed.\n";
    return 0;
}