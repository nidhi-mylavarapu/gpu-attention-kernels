#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

// VERY simple .npy loader (float32 only, C-contiguous)
inline std::vector<float> load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open " + path);

    // skip header (basic version)
    char header[128];
    f.read(header, 128);

    std::vector<float> data;
    float x;
    while (f.read(reinterpret_cast<char*>(&x), sizeof(float))) {
        data.push_back(x);
    }
    return data;
}
