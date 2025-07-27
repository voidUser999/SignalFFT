# CUDA Audio FFT Analyzer

This is a high-performance audio analysis tool built with C++ and CUDA. It processes a directory of `.wav` files to determine their dominant frequencies and corresponding musical notes, leveraging GPU acceleration for high throughput.

This project was developed as the final capstone for the GPU Specialization course.

---

## Features

- **Batch Processing:** Efficiently processes an entire directory of `.wav` audio files.
- **GPU Acceleration:** Utilizes the NVIDIA cuFFT library to perform Fast Fourier Transforms on the GPU.
- **Asynchronous Pipeline:** Implements an advanced processing pipeline using multiple CUDA streams to overlap file I/O, memory transfers, and GPU computation, maximizing throughput.
- **Advanced Analysis:** Identifies the Top 3 dominant frequencies for each audio file.
- **Musical Note Identification:** Converts detected frequencies into their corresponding musical notes (e.g., 440 Hz -> A4).
- **Organized Output:** Saves all results into a single, clean `.csv` file for easy analysis.

---

## Dependencies

To build and run this project, you will need:

- A C++17 compliant compiler (e.g., g++).
- The NVIDIA CUDA Toolkit (v11.0 or newer recommended).
- The `AudioFile.h` single-header library for `.wav` file handling.

---

## Setup and Build

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
2.  **Get `AudioFile.h`:**

    - Download the single-header `AudioFile.h` file from the official repository: [https://github.com/adamstark/AudioFile](https://github.com/adamstark/AudioFile)
    - Create a `lib` directory in the project root.
    - Place the `AudioFile.h` file inside the `lib` directory. The included `Makefile` is already configured to look here.

3.  **Build the Project:**
    The project uses a `Makefile` for easy compilation. Simply run the `make` command from the project's root directory.
    ```bash
    make
    ```
    This will compile the source code and create an executable at `bin/fft_async`.

---

## How to Run

The program requires an input directory (containing your `.wav` files) and an output directory (where the results CSV will be saved).

```bash
make clean
make
make run
```

## Output Format

| filename        | freq1_hz | note1 | freq2_hz | note2 | freq3_hz | note3 |
| :-------------- | :------- | :---- | :------- | :---- | :------- | :---- |
| gtr-nylon22.wav | 116.675  | A#2   | 116.683  | A#2   | 116.667  | A#2   |
| JazzTrio.wav    | 466.75   | A#4   | 466.767  | A#4   | 466.733  | A#4   |
| violin.wav      | 495.383  | B4    | 495.4    | B4    | 495.367  | B4    |
| gongs.wav       | 123.828  | B2    | 123.846  | B2    | 123.81   | B2    |
| shakuhachi.wav  | 586.104  | D5    | 586.213  | D5    | 778.304  | G5    |
