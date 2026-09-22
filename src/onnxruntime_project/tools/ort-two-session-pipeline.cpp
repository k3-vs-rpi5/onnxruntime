// Two ONNX Runtime sessions in one process, to answer the pipeline question perf_test cannot.
//
// `onnxruntime_perf_test` runs one model per process, and two EP *processes* do not share this board:
// measured with spacemit-tcm-smi, all eight AI-core TCM blocks go to whichever process starts first
// and the second runs with none. Every "overlapped" pipeline number taken that way is therefore a
// contention measurement rather than a pipeline shape.
//
// This harness puts both models in one process - the shape the vendor documents for sharing compute
// (`SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD`, "all models share the same compute resources") - and runs
// three configurations:
//
//   serial    stage A then stage B, N times each       -> frame = T_A + T_B
//   overlap   one thread per session, started together -> frame = max(T_A, T_B) if it pipelines
//   both with --global-pool 1 and split thread counts, as the vendor's FAQ describes
//
// The SpacemiT provider is loaded the same way the vendor's own media backend loads it: dlopen
// libspacemit_ep.so and call OrtSessionOptionsSpaceMITEnvInit, because it is not registered with the
// runtime as an ordinary provider (a Python `providers=[...]` list silently falls back to the CPU EP).
//
// Build (cross, workspace toolchain):
//   riscv64-unknown-linux-gnu-g++ -O2 -std=c++17 ort-two-session-pipeline.cpp -o pipeline \
//       -I<onnxruntime headers> -L<onnxruntime lib> -lonnxruntime -ldl -lpthread
//
// Usage:
//   ./pipeline <modelA.onnx> <modelB.onnx> <mode:serial|overlap> <iterations> <intra-threads> [global-pool 0|1]

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using OrtSpacemitEnvInit = OrtStatus *(*)(OrtSessionOptions *, const char *const *, const char *const *,
                                          size_t);

static Ort::Env *g_env = nullptr;
static OrtSpacemitEnvInit g_spacemit_init = nullptr;

static void check(OrtStatus *status, const char *what) {
    if (status != nullptr) {
        std::cerr << what << ": " << Ort::GetApi().GetErrorMessage(status) << "\n";
        std::exit(1);
    }
}

// One model, its inputs allocated to the declared shapes (all inputs of these models are float).
struct Stage {
    std::string name;
    Ort::Session session{nullptr};
    std::vector<Ort::Value> inputs;
    // The input buffers must be owned here: CreateTensor wraps the pointer, it does not copy, so a
    // buffer local to the constructor is freed before the first Run (that dangling pointer is what
    // segfaulted the second build of this harness).
    std::vector<std::vector<float>> buffers;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    explicit Stage(const std::string &path, const std::vector<std::string> &provider_options,
                   const std::string &label, int intra_threads)
        : name(label) {
        std::vector<const char *> keys, values;
        for (const auto &option : provider_options) {
            auto split = option.find('|');
            keys.push_back(option.substr(0, split).c_str());
            values.push_back(option.substr(split + 1).c_str());
        }
        Ort::SessionOptions options;
        check(g_spacemit_init(options, keys.data(), values.data(), keys.size()),
              "SpacemiT EP init");
        // The provider's own core check reads the process's thread shape, and the vendor's media engine
        // sets the ORT pool explicitly right after the provider init (`smt-vision-wrapper.cpp`). Leaving
        // it at the runtime default makes the session fail with
        // "Not enough available AI cores for the thread pool: required N, available 0" - which is what
        // the first build of this harness hit.
        options.SetIntraOpNumThreads(intra_threads);
        session = Ort::Session(*g_env, path.c_str(), options);

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        buffers.resize(session.GetInputCount());
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_names.emplace_back(input_name.get());
            // Hold the TypeInfo: the tensor-shape view references its parent, and reading it from a
            // temporary returns rank 0 (which is how the first build of this harness failed).
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto info = type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();
            std::cerr << "  [" << label << "] input " << input_names.back() << " rank="
                      << info.GetDimensionsCount() << " shape=[";
            for (size_t d = 0; d < shape.size(); ++d) std::cerr << (d ? "," : "") << shape[d];
            std::cerr << "]\n";
            size_t count = 1;
            for (auto &dimension : shape) {
                if (dimension <= 0) dimension = 1;  // these models are static; guard anyway
                count *= static_cast<size_t>(dimension);
            }
            if (shape.empty()) {
                std::cerr << "  [" << label << "] FATAL: no tensor shape for input "
                          << input_names.back() << "\n";
                std::exit(1);
            }
            buffers[i].assign(count, 0.5f);
            inputs.push_back(Ort::Value::CreateTensor<float>(memory, buffers[i].data(), count,
                                                             shape.data(), shape.size()));
        }
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(output_name.get());
        }
    }

    std::vector<const char *> input_ptrs() const {
        std::vector<const char *> out;
        for (const auto &n : input_names) out.push_back(n.c_str());
        return out;
    }
    std::vector<const char *> output_ptrs() const {
        std::vector<const char *> out;
        for (const auto &n : output_names) out.push_back(n.c_str());
        return out;
    }
    // The input buffers must outlive the Values; the harness owns them for the process lifetime.
    void run() {
        auto ins = input_ptrs();
        auto outs = output_ptrs();
        session.Run(Ort::RunOptions{nullptr}, ins.data(), inputs.data(), inputs.size(), outs.data(),
                    outs.size());
    }
};

static double time_stage(Stage &stage, int iterations) {
    for (int i = 0; i < 3; ++i) stage.run();  // warm-up
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) stage.run();
    auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / iterations;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "usage: pipeline <modelA> <modelB> <serial|overlap> <iterations> <intra> "
                     "[global-pool 0|1]\n";
        return 2;
    }
    const std::string model_a = argv[1], model_b = argv[2], mode = argv[3];
    const int iterations = std::stoi(argv[4]);
    const int intra = std::stoi(argv[5]);
    const int global_pool = argc > 6 ? std::stoi(argv[6]) : 0;

    void *handle = dlopen("libspacemit_ep.so", RTLD_NOW);
    if (!handle) {
        std::cerr << "dlopen libspacemit_ep.so failed: " << dlerror() << "\n";
        return 1;
    }
    g_spacemit_init = reinterpret_cast<OrtSpacemitEnvInit>(
        dlsym(handle, "OrtSessionOptionsSpaceMITEnvInit"));
    if (!g_spacemit_init) {
        std::cerr << "OrtSessionOptionsSpaceMITEnvInit not found in libspacemit_ep.so\n";
        return 1;
    }
    g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "pipeline");

    std::vector<std::string> options_a = {"SPACEMIT_EP_INTRA_THREAD_NUM|" + std::to_string(intra),
                                          "SPACEMIT_EP_INTER_THREAD_NUM|1"};
    std::vector<std::string> options_b = options_a;
    if (global_pool) {
        options_a.push_back("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD|1");
        options_b.push_back("SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD|1");
    }
    Stage stage_a(model_a, options_a, "A", intra);
    Stage stage_b(model_b, options_b, "B", intra);

    std::cout << "mode=" << mode << " iterations=" << iterations << " intra=" << intra
              << " global_pool=" << global_pool << "\n";
    if (mode == "serial") {
        double a = time_stage(stage_a, iterations);
        double b = time_stage(stage_b, iterations);
        std::cout << "A_ms=" << a << " B_ms=" << b << " frame_ms=" << (a + b)
                  << " frames_per_s=" << 1000.0 / (a + b) << "\n";
    } else if (mode == "overlap") {
        double a = 0, b = 0;
        std::thread thread_a([&] { a = time_stage(stage_a, iterations); });
        std::thread thread_b([&] { b = time_stage(stage_b, iterations); });
        thread_a.join();
        thread_b.join();
        double frame = a > b ? a : b;
        std::cout << "A_ms=" << a << " B_ms=" << b << " frame_ms=" << frame
                  << " frames_per_s=" << 1000.0 / frame << "\n";
    } else {
        std::cerr << "unknown mode\n";
        return 2;
    }
    return 0;
}
