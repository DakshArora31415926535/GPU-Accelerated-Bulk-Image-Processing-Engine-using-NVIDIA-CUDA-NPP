/* GPU Image Bulk Processing using NVIDIA NPP
 * Modified to support multi-image batch execution via --list
 */

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <nppi_geometry_transforms.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

void processImage(const std::string &sFilename, float scale) {
    std::cout << "\nProcessing: " << sFilename << std::endl;

    std::ifstream infile(sFilename.c_str());
    if (!infile.good()) {
        std::cout << "❌ Unable to open file: " << sFilename << std::endl;
        return;
    }
    infile.close();

    // Load image from CPU and transfer to GPU
    npp::ImageCPU_8u_C1 oHostSrc;
    npp::loadImage(sFilename, oHostSrc);
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    NppiSize srcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiSize dstSize = {(int)(scale * srcSize.width), (int)(scale * srcSize.height)};
    npp::ImageNPP_8u_C1 oDeviceDst(dstSize.width, dstSize.height);

    // NPP scratch buffer allocation
    int nBufferSize = 0;
    Npp8u *pScratchBufferNPP = nullptr;
    nppiFilterCannyBorderGetBufferSize(dstSize, &nBufferSize);
    cudaMalloc((void **)&pScratchBufferNPP, nBufferSize);

    NppiRect srcRoi = {0, 0, srcSize.width, srcSize.height};
    NppiRect dstRoi = {0, 0, dstSize.width, dstSize.height};

    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = nullptr;

    // GPU processing operation
    nppiResize_8u_C1R_Ctx(
        oDeviceSrc.data(), oDeviceSrc.pitch(), srcSize, srcRoi,
        oDeviceDst.data(), oDeviceDst.pitch(), dstSize, dstRoi,
        NPPI_INTER_LANCZOS, nppStreamCtx);

    cudaFree(pScratchBufferNPP);

    // Copy back and save output
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    std::string sResultFilename = sFilename;
    auto dot = sResultFilename.rfind('.');
    if (dot != std::string::npos)
        sResultFilename = sResultFilename.substr(0, dot);
    sResultFilename += "_resized.pgm";

    saveImage(sResultFilename, oHostDst);
    std::cout << "✔ Output saved to: " << sResultFilename << std::endl;
}

int main(int argc, char *argv[]) {
    printf("%s Starting...\n\n", argv[0]);
    try {
        cudaDeviceInit(argc, (const char **)argv);
        printfNPPinfo(argc, argv);

        float scale = 0.5;
        if (checkCmdLineFlag(argc, (const char **)argv, "scale")) {
            scale = getCmdLineArgumentFloat(argc, (const char **)argv, "scale");
        }
        std::cout << "Scale factor: " << scale << std::endl;

        // Multi-batch image list processing
        if (checkCmdLineFlag(argc, (const char **)argv, "list")) {
            char *listFile = NULL;
            getCmdLineArgumentString(argc, (const char **)argv, "list", &listFile);

            std::ifstream fileList(listFile);
            if (!fileList.good()) {
                std::cerr << "❌ Unable to open list file!" << std::endl;
                return EXIT_FAILURE;
            }

            std::cout << "📂 Batch processing mode enabled\n";
            std::string line;
            while (std::getline(fileList, line)) {
                if (!line.empty())
                    processImage(line, scale);
            }

            std::cout << "\n🔥 Batch processing complete.\n";
            return EXIT_SUCCESS;
        }

        // Single image mode
        char *filePath = NULL;
        std::string sFilename;
        if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        } else {
            filePath = sdkFindFilePath("lena1.pgm", argv[0]);
        }

        sFilename = filePath ? filePath : "lena1.pgm";
        processImage(sFilename, scale);

        return EXIT_SUCCESS;
    }
    catch (...) {
        std::cerr << "Unhandled exception. Aborting." << std::endl;
        return EXIT_FAILURE;
    }
}
