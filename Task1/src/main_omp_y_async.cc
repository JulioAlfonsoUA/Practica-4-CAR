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
#include <future>
#include <omp.h>
#include <mutex>

static std::mutex mtx; 

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
        case 3:
            return get_srm_3x3();
        case 5:
            return get_srm_5x5();
    }
    return get_srm_3x3();
}


Image<unsigned char> compute_srm(const Image<unsigned char> &image, int kernel_size) {
    double begin = omp_get_wtime();
    {
        std::lock_guard<std::mutex> lk(mtx);
        std::cout<<"Computing SRM "<<kernel_size<<"x"<<kernel_size<<"..."<<std::endl;  
    }
    Image<float> srm = image.to_grayscale().convert<float>();
    srm = srm.convolution(get_srm_kernel(kernel_size));
    srm = srm.abs().normalized();
    srm = srm * 255;
    Image<unsigned char> result = srm.convert<unsigned char>();

    double end = (omp_get_wtime() - begin) * 1000.0;
    {
        std::lock_guard<std::mutex> lk(mtx);
        std::cout<<"SRM elapsed time: "<<end<<"ms"<<std::endl;
    }
    return result;
}

Image<unsigned char> compute_dct(const Image<unsigned char> &image, int block_size, bool invert) {
    double begin = omp_get_wtime();
    {
        std::lock_guard<std::mutex> lk(mtx);
        std::cout<<"Computing";
        if (invert) std::cout<<" inverse";
        else std::cout<<" direct";
        std::cout<<" DCT "<<block_size<<"x"<<block_size<<"..."<<std::endl;
    }

    Image<float> grayscale = image.convert<float>().to_grayscale();
    std::vector<Block<float>> blocks = grayscale.get_blocks(block_size);

    #pragma omp parallel for schedule(static) shared(blocks) firstprivate(block_size, invert)
    for (int i = 0; i < (int)blocks.size(); i++) {
        float **dctBlock = dct::create_matrix(block_size, block_size);
        dct::direct(dctBlock, blocks[i], 0);
        if (invert) {
          for(int k=0;k<blocks[i].size/2;k++)
            for(int l=0;l<blocks[i].size/2;l++)
              dctBlock[k][l] = 0.0;
          dct::inverse(blocks[i], dctBlock, 0, 0.0, 255.);
        }else dct::assign(dctBlock, blocks[i], 0);
        dct::delete_matrix(dctBlock);
    }
    Image<unsigned char> result = grayscale.convert<unsigned char>();
    double end = (omp_get_wtime() - begin) * 1000.0;
    {
        std::lock_guard<std::mutex> lk(mtx);
        std::cout<<"DCT elapsed time: "<<end<<"ms"<<std::endl;
    }
    return result;
}

Image<unsigned char> compute_ela(const Image<unsigned char> &image, int quality) {
    double begin = omp_get_wtime();
    {
        std::lock_guard<std::mutex> lk(mtx);
        std::cout << "Computing ELA...\n";
    }
    Image<unsigned char> grayscale = image.to_grayscale();
    save_to_file("_temp.jpg", grayscale, quality);
    Image<float> compressed = load_from_file("_temp.jpg").convert<float>();
    compressed = compressed + (grayscale.convert<float>()*(-1));
    compressed = compressed.abs().normalized() * 255;
    Image<unsigned char> result = compressed.convert<unsigned char>();
    double end = (omp_get_wtime() - begin) * 1000.0;
    {
        std::lock_guard<std::mutex> lk(mtx);
        std::cout<<"ELA elapsed: "<<end<<" ms\n";
    }
    return result;
}

int main(int argc, char **argv) {
    if (argc == 1) {
        std::cerr<<"Image filename missing from arguments. Usage ./dct <filename>"<<std::endl;
        exit(1);
    }
    int block_size = 8;
    Image<unsigned char> image = load_from_file(argv[1]);
    double begin = omp_get_wtime();
    auto f_srm3 = std::async(std::launch::async, [&]() {
        save_to_file("srm_kernel_3x3.png", compute_srm(image, 3));
    });
    auto f_srm5 = std::async(std::launch::async, [&]() {
        save_to_file("srm_kernel_5x5.png", compute_srm(image, 5));
    });
    auto f_ela = std::async(std::launch::async, [&]() {
        save_to_file("ela.png", compute_ela(image, 90));
    });
    auto f_dct_inv = std::async(std::launch::async, [&]() {
        save_to_file("dct_invert.png", compute_dct(image, block_size, true));
    });
    auto f_dct_dir = std::async(std::launch::async, [&]() {
        save_to_file("dct_direct.png", compute_dct(image, block_size, false));
    });
    f_srm3.get();
    f_srm5.get();
    f_ela.get();
    f_dct_inv.get();
    f_dct_dir.get();
    double end = (omp_get_wtime() - begin) * 1000.0;
    std::cout<<"\nTotal elapsed (parallel): "<<end<<" ms"<<std::endl;
    std::cout<<"OpenMP threads available: "<<omp_get_max_threads()<<std::endl;
    return 0;
}
