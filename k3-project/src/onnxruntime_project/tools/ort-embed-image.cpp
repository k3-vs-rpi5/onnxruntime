// S11 image embeddings on the AI cluster: one process, one provider-claimed session, vectors out.
//
//   ort-embed-image <model.onnx> <ppm>... --size 256|224 --norm clip|half [-o out.jsonl]
//
// Images are binary PPM (the fixture is converted on the host), so the harness needs no image
// library; the provider is loaded the documented way, because this board's Python ONNX Runtime
// reaches only the CPU EP. `--norm clip` is the CLIP/MobileCLIP mean-std pair, `half` is the
// siglip-style (x/255 - 0.5)/0.5.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using OrtSpacemitEnvInit = OrtStatus *(*)(OrtSessionOptions *, const char *const *,
                                          const char *const *, size_t);

static void check(OrtStatus *s, const char *what) {
    if (s) { std::cerr << what << ": " << Ort::GetApi().GetErrorMessage(s) << "\n"; std::exit(1); }
}

struct Image { int w = 0, h = 0; std::vector<unsigned char> rgb; };

static Image read_ppm(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    std::string magic; in >> magic;
    if (magic != "P6") { std::cerr << path << ": not a binary PPM\n"; std::exit(1); }
    auto next_int = [&]() {
        int v = 0; char c;
        while (in.get(c) && !std::isdigit(static_cast<unsigned char>(c))) {}
        while (std::isdigit(static_cast<unsigned char>(c))) { v = v * 10 + (c - '0'); if (!in.get(c)) break; }
        in.putback(c); return v;
    };
    int w = next_int(), h = next_int(), maxv = next_int();
    if (maxv > 255) { std::cerr << path << ": 16-bit PPM unsupported\n"; std::exit(1); }
    in.get();
    Image img; img.w = w; img.h = h; img.rgb.resize(static_cast<size_t>(w) * h * 3);
    in.read(reinterpret_cast<char *>(img.rgb.data()), img.rgb.size());
    return img;
}

// resize (stretch, the encoders were exported for square inputs) + normalise + NCHW
static std::vector<float> preprocess(const Image &img, int S, bool clip_norm) {
    static const float cm[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float cs[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    std::vector<float> out(static_cast<size_t>(3) * S * S, 0.f);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < S; ++y) {
            int sy = std::min(img.h - 1, static_cast<int>((y + 0.5f) * img.h / S));
            for (int x = 0; x < S; ++x) {
                int sx = std::min(img.w - 1, static_cast<int>((x + 0.5f) * img.w / S));
                float v = img.rgb[(static_cast<size_t>(sy) * img.w + sx) * 3 + c] / 255.f;
                v = clip_norm ? (v - cm[c]) / cs[c] : (v - 0.5f) / 0.5f;
                out[(static_cast<size_t>(c) * S + y) * S + x] = v;
            }
        }
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: ort-embed-image <model.onnx> <ppm>... --size N --norm clip|half "
                     "[-o out.jsonl]\n";
        return 2;
    }
    std::string model = argv[1], out_path = "embeddings.jsonl";
    int size = 256;
    bool clip_norm = true;
    std::vector<std::string> images;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--size" && i + 1 < argc) { size = std::stoi(argv[++i]); continue; }
        if (a == "--norm" && i + 1 < argc) { clip_norm = std::string(argv[++i]) == "clip"; continue; }
        if (a == "-o" && i + 1 < argc) { out_path = argv[++i]; continue; }
        images.push_back(a);
    }

    void *handle = dlopen("libspacemit_ep.so", RTLD_NOW);
    if (!handle) { std::cerr << "dlopen libspacemit_ep.so failed\n"; return 1; }
    auto init = reinterpret_cast<OrtSpacemitEnvInit>(
        dlsym(handle, "OrtSessionOptionsSpaceMITEnvInit"));
    if (!init) { std::cerr << "OrtSessionOptionsSpaceMITEnvInit missing\n"; return 1; }
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "embed");
    std::vector<std::string> opts = {"SPACEMIT_EP_INTRA_THREAD_NUM|8", "SPACEMIT_EP_INTER_THREAD_NUM|1"};
    std::vector<std::string> keys_s, vals_s;
    for (auto &o : opts) { auto p = o.find('|'); keys_s.push_back(o.substr(0, p)); vals_s.push_back(o.substr(p + 1)); }
    std::vector<const char *> keys, vals;
    for (auto &k : keys_s) keys.push_back(k.c_str());
    for (auto &v : vals_s) vals.push_back(v.c_str());
    Ort::SessionOptions so;
    check(init(so, keys.data(), vals.data(), keys.size()), "Spacemit EP init");
    so.SetIntraOpNumThreads(8);
    Ort::Session session(env, model.c_str(), so);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto in_name = session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    auto out_name = session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions());
    const char *in_names[] = {in_name.get()};
    const char *out_names[] = {out_name.get()};

    std::ofstream out(out_path);
    double total_ms = 0; int n = 0;
    for (const auto &path : images) {
        Image img = read_ppm(path);
        auto buf = preprocess(img, size, clip_norm);
        std::vector<int64_t> shape{1, 3, size, size};
        auto tensor = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(), shape.data(), shape.size());
        auto t0 = std::chrono::steady_clock::now();
        auto outs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
        auto t1 = std::chrono::steady_clock::now();
        const float *v = outs[0].GetTensorData<float>();
        auto dims = outs[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t dim = 1;
        for (auto d : dims) dim *= static_cast<size_t>(d);
        std::string base = path.substr(path.find_last_of('/') + 1);
        base = base.substr(0, base.find_last_of('.'));
        out << "{\"image\":\"" << base << "\",\"vector\":[";
        for (size_t i = 0; i < dim; ++i) out << (i ? "," : "") << v[i];
        out << "]}\n";
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++n;
    }
    std::cerr << "embedded " << n << " images, " << total_ms / std::max(n, 1)
              << " ms/image, dim=" << 0 << "\n";
    return 0;
}
