#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "png.h"
#include <vector>
#include <assert.h>
#include <iostream>
#include <memory>
#include "utils/image.h"
#include "utils/dct.h"
#include <string>
#include <chrono>

// Mini funciones definidas para que el código no sea un puñado de líneas idénticas.
#define TIMER_START std::chrono::steady_clock::time_point _t0 = std::chrono::steady_clock::now();
#define TIMER_MS(label) std::cout<<"["<< (label)<<"] "<< std::chrono::duration_cast<std::chrono::microseconds>( \
    std::chrono::steady_clock::now()-_t0).count()<<" us\n"; _t0 = std::chrono::steady_clock::now();

Image<float> get_srm_3x3() {
    Image<float> kernel(3, 3, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -4); kernel.set(1, 2, 0, 2);
    kernel.set(2, 0, 0, -1); kernel.set(2, 1, 0, 2); kernel.set(2, 2, 0, -1);
    return kernel;
}

Image<float> get_srm_5x5() {
    Image<float> kernel(5, 5, 1);
    kernel.set(0, 0, 0, -1); kernel.set(0, 1, 0, 2); kernel.set(0, 2, 0, -2); kernel.set(0, 3, 0, 2); kernel.set(0, 4, 0, -1);
    kernel.set(1, 0, 0, 2); kernel.set(1, 1, 0, -6); kernel.set(1, 2, 0, 8); kernel.set(1, 3, 0, -6); kernel.set(1, 4, 0, 2);
    kernel.set(2, 0, 0, -2); kernel.set(2, 1, 0, 8); kernel.set(2, 2, 0, -12); kernel.set(2, 3, 0, 8); kernel.set(2, 4, 0, -2);
    kernel.set(3, 0, 0, 2); kernel.set(3, 1, 0, -6); kernel.set(3, 2, 0, 8); kernel.set(3, 3, 0, -6); kernel.set(3, 4, 0, 2);
    kernel.set(4, 0, 0, -1); kernel.set(4, 1, 0, 2); kernel.set(4, 2, 0, -2); kernel.set(4, 3, 0, 2); kernel.set(4, 4, 0, -1);
    return kernel;
}

Image<float> get_srm_kernel(int size) {
    assert(size == 3 || size == 5);
    switch(size){
        case 3: return get_srm_3x3();
        case 5: return get_srm_5x5();
    }
    return get_srm_3x3();
}

Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size) {
    auto begin = std::chrono::steady_clock::now();
    std::cout<<"\nComputing SRM "<<kernel_size<<"x"<<kernel_size<<"..."<<std::endl;
    TIMER_START

    Image<float> srm = image.to_grayscale().convert<float>();
    TIMER_MS("to_grayscale + convert<float>")

    srm = srm.convolution(get_srm_kernel(kernel_size));
    TIMER_MS("convolution")

    srm = srm.abs();
    TIMER_MS("abs")

    srm = srm.normalized();
    TIMER_MS("normalized")

    srm = srm * 255;
    TIMER_MS("srm * 255")

    Image<unsigned char> result = srm.convert<unsigned char>();
    TIMER_MS("convert<unsigned char>")

    auto end = std::chrono::steady_clock::now();
    std::cout<<"SRM elapsed time: "<<std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<<"us"<<std::endl;
    return result;
}

Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    auto begin = std::chrono::steady_clock::now();
    std::cout<<"\nComputing"; 
    if (invert) std::cout<<" inverse";
    else std::cout<<" direct";
    std::cout<<" DCT "<<block_size<<"x"<<block_size<<"..."<<std::endl;
    TIMER_START

    Image<float> grayscale = image.convert<float>().to_grayscale();
    TIMER_MS("convert<float> + to_grayscale")

    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);
    TIMER_MS("get_blocks")

    auto t_bucle = std::chrono::steady_clock::now();
    for(int i = 0; i < (int)blocks.size(); i++){
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
            for(int k = 0; k < blocks[i].size/2; k++)
                for(int l = 0; l < blocks[i].size/2; l++)
                    dctBlock[k][l] = 0.0;
            dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        } else {
            dct::assign(dctBlock, blocks[i], 0);
        }
        dct::delete_matrix(dctBlock);
    }
    TIMER_MS("bucle DCT")

    Image<unsigned char> result = grayscale.convert<unsigned char>();
    TIMER_MS("convert<unsigned char>")

    auto end = std::chrono::steady_clock::now();
    std::cout<<"DCT elapsed time: "<<std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<<"us"<<std::endl;
    return result;
}


Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality) {
    std::cout<<"\nComputing ELA..."<<std::endl;
    auto begin = std::chrono::steady_clock::now();
    TIMER_START

    Image<unsigned char> grayscale = image.to_grayscale();
    TIMER_MS("to_grayscale")

    save_to_file("_temp.jpg", grayscale, quality);
    TIMER_MS("save_to_file (I/O escritura)")

    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();
    TIMER_MS("load_from_file (I/O lectura)")

    compressed = compressed + (grayscale.convert<float>() * (-1));
    TIMER_MS("resta (+ convert<float>)")

    compressed = compressed.abs();
    TIMER_MS("abs")

    compressed = compressed.normalized() * 255;
    TIMER_MS("normalized * 255")

    Image<unsigned char> result = compressed.convert<unsigned char>();
    TIMER_MS("convert<uchar>")

    auto end = std::chrono::steady_clock::now();
    std::cout<<"ELA elapsed time: "<<std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()<<"us"<<std::endl;
    return result;
}


int main(int argc, char **argv) {
    if(argc == 1) {
        std::cerr << "Uso: ./detect <imagen>\n";
        exit(1);
    }
    auto t_global = std::chrono::steady_clock::now();
    int block_size = 8;
    std::cout<<"\nCargando imagen...\n";
    TIMER_START
    Image<unsigned char> image = load_from_file(argv[1]);
    TIMER_MS("load_from_file")
    std::cout<<"Dimensiones: "<<image.width<<"x"<<image.height<<"\ncanales: "<<image.channels<<"\n";
    std::cout<<"Num bloques DCT ("<<block_size<<"x"<<block_size<<"): "<< (image.width/block_size) * (image.height/block_size)<<"\n";
    save_to_file("srm_kernel_3x3.png", compute_srm(image, 3));
    save_to_file("srm_kernel_5x5.png", compute_srm(image, 5));
    save_to_file("ela.png", compute_ela(image, 90));
    save_to_file("dct_invert.png", compute_dct(image, block_size, true));
    save_to_file("dct_direct.png", compute_dct(image, block_size, false));
    long global_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_global).count();
    std::cout<<"--------------------------------------------------------------\n";
    std::cout<<"TIEMPO TOTAL SECUENCIAL: "<<global_us<<" us\n";
    std::cout<<"--------------------------------------------------------------\n";

    return 0;
}