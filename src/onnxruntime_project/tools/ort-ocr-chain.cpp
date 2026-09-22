// The S12 own-chain harness: PP-OCRv6 detection + recognition on the AI cluster, in one process.
//
// Why a C++ harness and not Python: this board's Python ONNX Runtime exposes only
// `CPUExecutionProvider` (measured 2026-09-22), so a Python chain would report CPU numbers while
// looking like ours. This harness loads the provider the documented way -
// `dlopen("libspacemit_ep.so")` + `OrtSessionOptionsSpaceMITEnvInit` - exactly like
// `ort-two-session-pipeline.cpp`, so both sessions are claimed by the SpacemiT EP.
//
// Images are read as binary PPM (P6): the fixture is converted on the host, and the harness needs no
// image library at all (the board has no OpenCV headers in the cross sysroot).
//
//   ort-ocr-chain <det.onnx> <rec.onnx> <dict.txt> <page.ppm>... [-o out-dir]
//
// For every page it writes <out-dir>/<page-basename>.txt in the same shape `tools/cer-wer.py`
// reads, so the independent engine (tesseract) and this chain are compared by one instrument.
//
// Pipeline: letterbox to 640x640 -> det (1x1x640x640 prob map) -> threshold 0.3 -> connected
// components -> unclip 1.6 -> boxes mapped back to the page -> crop -> 48x320 rec -> argmax -> CTC
// greedy decode against the dictionary (class 0 = blank; classes 1..N map to dict[class-1], with the
// appended space at the end).

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <vector>

using OrtSpacemitEnvInit = OrtStatus *(*)(OrtSessionOptions *, const char *const *,
                                          const char *const *, size_t);
static Ort::Env *g_env = nullptr;

static void check(OrtStatus *status, const char *what) {
    if (status != nullptr) {
        std::cerr << what << ": " << Ort::GetApi().GetErrorMessage(status) << "\n";
        std::exit(1);
    }
}

struct Image {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;  // 3 bytes per pixel
};

static Image read_ppm(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    std::string magic;
    int w = 0, h = 0, maxv = 0;
    in >> magic;
    if (magic != "P6") { std::cerr << path << ": not a binary PPM\n"; std::exit(1); }
    auto next_int = [&]() {
        int v = 0; char c;
        while (in.get(c) && !std::isdigit(static_cast<unsigned char>(c))) {}
        while (std::isdigit(static_cast<unsigned char>(c))) { v = v * 10 + (c - '0'); if (!in.get(c)) break; }
        in.putback(c);
        return v;
    };
    w = next_int(); h = next_int(); maxv = next_int();
    if (maxv > 255) { std::cerr << path << ": 16-bit PPM not supported\n"; std::exit(1); }
    in.get();  // single whitespace after maxval
    Image img; img.w = w; img.h = h; img.rgb.resize(static_cast<size_t>(w) * h * 3);
    in.read(reinterpret_cast<char *>(img.rgb.data()), img.rgb.size());
    return img;
}

// --- det preprocessing: letterbox to 640x640, ImageNet normalisation, NCHW ---
struct Letterbox { float scale = 1.f; int pad_x = 0, pad_y = 0; };

static std::vector<float> det_input(const Image &img, Letterbox &lb, int S = 640) {
    lb.scale = std::min(S / static_cast<float>(img.w), S / static_cast<float>(img.h));
    int nw = std::max(1, static_cast<int>(std::round(img.w * lb.scale)));
    int nh = std::max(1, static_cast<int>(std::round(img.h * lb.scale)));
    lb.pad_x = (S - nw) / 2; lb.pad_y = (S - nh) / 2;
    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float std_[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> out(static_cast<size_t>(3) * S * S, 0.f);
    for (int y = 0; y < nh; ++y) {
        int sy = std::min(img.h - 1, static_cast<int>(y / lb.scale));
        for (int x = 0; x < nw; ++x) {
            int sx = std::min(img.w - 1, static_cast<int>(x / lb.scale));
            const unsigned char *p = &img.rgb[(static_cast<size_t>(sy) * img.w + sx) * 3];
            for (int c = 0; c < 3; ++c) {
                float v = p[c] / 255.f;
                out[(static_cast<size_t>(c) * S + (y + lb.pad_y)) * S + (x + lb.pad_x)] =
                    (v - mean[c]) / std_[c];
            }
        }
    }
    return out;
}

struct Box { int x0, y0, x1, y1; };

// --- DB post-processing: threshold, 8-connected components, unclip, map back to the page ---
static std::vector<Box> det_boxes(const float *prob, int S, const Letterbox &lb,
                                  float thr = 0.3f, float unclip = 1.6f) {
    std::vector<unsigned char> mask(static_cast<size_t>(S) * S, 0);
    for (size_t i = 0; i < static_cast<size_t>(S) * S; ++i) mask[i] = prob[i] > thr ? 1 : 0;
    std::vector<unsigned char> seen(static_cast<size_t>(S) * S, 0);
    std::vector<Box> boxes;
    for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
            size_t idx = static_cast<size_t>(y) * S + x;
            if (!mask[idx] || seen[idx]) continue;
            int x0 = x, x1 = x, y0 = y, y1 = y; long area = 0;
            std::queue<std::pair<int, int>> q; q.push({x, y}); seen[idx] = 1;
            while (!q.empty()) {
                auto [cx, cy] = q.front(); q.pop(); ++area;
                x0 = std::min(x0, cx); x1 = std::max(x1, cx);
                y0 = std::min(y0, cy); y1 = std::max(y1, cy);
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = cx + dx, ny = cy + dy;
                        if (nx < 0 || ny < 0 || nx >= S || ny >= S) continue;
                        size_t ni = static_cast<size_t>(ny) * S + nx;
                        if (mask[ni] && !seen[ni]) { seen[ni] = 1; q.push({nx, ny}); }
                    }
            }
            if (area < 12) continue;  // noise
            float w = x1 - x0 + 1.f, h = y1 - y0 + 1.f;
            float offset = area * unclip / (2.f * (w + h));
            float bx0 = x0 - offset, by0 = y0 - offset, bx1 = x1 + offset, by1 = y1 + offset;
            auto back = [&](float px, float py, int &ox, int &oy) {
                ox = static_cast<int>((px - lb.pad_x) / lb.scale);
                oy = static_cast<int>((py - lb.pad_y) / lb.scale);
            };
            int ox0, oy0, ox1, oy1; back(bx0, by0, ox0, oy0); back(bx1, by1, ox1, oy1);
            boxes.push_back({std::max(0, ox0), std::max(0, oy0), ox1, oy1});
        }
    }
    std::sort(boxes.begin(), boxes.end(), [](const Box &a, const Box &b) {
        int ca = (a.y0 + a.y1) / 2, cb = (b.y0 + b.y1) / 2;
        return std::abs(ca - cb) > 24 ? ca < cb : a.x0 < b.x0;
    });
    return boxes;
}

// --- rec preprocessing: crop, resize to 48 x W (<=320), pad to 320, normalise to [-1, 1] ---
static std::vector<float> rec_input(const Image &img, const Box &b, int H = 48, int W = 320) {
    int bw = std::max(1, b.x1 - b.x0), bh = std::max(1, b.y1 - b.y0);
    int nw = std::min(W, std::max(8, static_cast<int>(std::round(static_cast<float>(bw) * H / bh))));
    std::vector<float> out(static_cast<size_t>(3) * H * W, 0.f);
    for (int y = 0; y < H; ++y) {
        float sy = b.y0 + (y + 0.5f) * bh / H;
        for (int x = 0; x < nw; ++x) {
            float sx = b.x0 + (x + 0.5f) * bw / nw;
            int ix = std::min(img.w - 1, std::max(0, static_cast<int>(sx)));
            int iy = std::min(img.h - 1, std::max(0, static_cast<int>(sy)));
            const unsigned char *p = &img.rgb[(static_cast<size_t>(iy) * img.w + ix) * 3];
            for (int c = 0; c < 3; ++c)
                out[(static_cast<size_t>(c) * H + y) * W + x] = p[c] / 127.5f - 1.f;
        }
    }
    return out;
}

static std::vector<std::string> read_dict(const std::string &path) {
    std::ifstream in(path);
    std::vector<std::string> dict; std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) dict.push_back(line);
    }
    return dict;
}

static std::string ctc_decode(const float *logits, int T, int C,
                              const std::vector<std::string> &dict) {
    std::string out; int prev = -1;
    for (int t = 0; t < T; ++t) {
        int best = 0; float best_v = logits[static_cast<size_t>(t) * C];
        for (int c = 1; c < C; ++c) {
            float v = logits[static_cast<size_t>(t) * C + c];
            if (v > best_v) { best_v = v; best = c; }
        }
        if (best != 0 && best != prev) {
            if (best - 1 < static_cast<int>(dict.size())) out += dict[best - 1];
            else out += " ";  // the appended space class
        }
        prev = best;
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "usage: ort-ocr-chain <det.onnx> <rec.onnx> <dict.txt> <page.ppm>... "
                     "[-o out-dir]\n";
        return 2;
    }
    std::string det_path = argv[1], rec_path = argv[2], dict_path = argv[3];
    std::string out_dir = ".";
    int threads_det = 4, threads_rec = 4;  // the two sessions share eight TCM blocks
    std::vector<std::string> pages;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" && i + 1 < argc) { out_dir = argv[++i]; continue; }
        if (a == "--threads" && i + 2 < argc) {
            threads_det = std::stoi(argv[++i]); threads_rec = std::stoi(argv[++i]); continue;
        }
        pages.push_back(a);
    }

    void *handle = dlopen("libspacemit_ep.so", RTLD_NOW);
    if (!handle) { std::cerr << "dlopen libspacemit_ep.so failed\n"; return 1; }
    auto init = reinterpret_cast<OrtSpacemitEnvInit>(
        dlsym(handle, "OrtSessionOptionsSpaceMITEnvInit"));
    if (!init) { std::cerr << "OrtSessionOptionsSpaceMITEnvInit missing\n"; return 1; }
    g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ocr");
    // Each session asks for its own share: asking both for eight is refused outright with
    // "Not enough available AI cores for the thread pool: required 8, available 0" - the same
    // sharing rule the scenario registry records for pipelines.
    auto make_options = [&](int threads) {
        std::vector<std::string> opts = {"SPACEMIT_EP_INTRA_THREAD_NUM|" + std::to_string(threads),
                                         "SPACEMIT_EP_INTER_THREAD_NUM|1"};
        std::vector<std::string> keys_s, values_s;
        for (auto &o : opts) { auto p = o.find('|'); keys_s.push_back(o.substr(0, p));
                               values_s.push_back(o.substr(p + 1)); }
        std::vector<const char *> keys, values;
        for (auto &k : keys_s) keys.push_back(k.c_str());
        for (auto &v : values_s) values.push_back(v.c_str());
        Ort::SessionOptions so;
        check(init(so, keys.data(), values.data(), keys.size()), "SpacemiT EP init");
        so.SetIntraOpNumThreads(threads);
        return so;
    };
    auto so_det = make_options(threads_det), so_rec = make_options(threads_rec);
    std::cout << "threads det=" << threads_det << " rec=" << threads_rec << "\n";
    Ort::Session det(*g_env, det_path.c_str(), so_det), rec(*g_env, rec_path.c_str(), so_rec);
    auto dict = read_dict(dict_path);
    std::cout << "dict_entries=" << dict.size() << "\n";
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (const auto &page : pages) {
        Image img = read_ppm(page);
        Letterbox lb;
        auto in = det_input(img, lb);
        std::vector<int64_t> det_shape{1, 3, 640, 640};
        auto det_in = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(),
                                                      det_shape.data(), det_shape.size());
        const char *det_in_names[] = {"x"};
        const char *det_out_names[] = {"fetch_name_0"};
        auto t0 = std::chrono::steady_clock::now();
        auto det_out = det.Run(Ort::RunOptions{nullptr}, det_in_names, &det_in, 1, det_out_names, 1);
        auto t1 = std::chrono::steady_clock::now();
        auto boxes = det_boxes(det_out[0].GetTensorData<float>(), 640, lb);
        auto t2 = std::chrono::steady_clock::now();

        std::string text;
        for (const auto &b : boxes) {
            // Long lines must be recognised in chunks: a 50-character line squeezed into one
            // 320-wide crop has ~40 timesteps for 50 characters, and CTC alignment degrades into
            // word-shaped approximations (measured: "baseline" -> "beshie"). PP-OCR's own practice
            // is to cut the line into pieces whose aspect ratio stays near the model's 320/48.
            int bh = std::max(1, b.y1 - b.y0), bw = std::max(1, b.x1 - b.x0);
            const float max_aspect = 320.f / 48.f;
            int chunks = std::max(1, static_cast<int>(std::ceil((bw / static_cast<float>(bh))
                                                                / max_aspect)));
            int chunk_w = std::max(1, (bw + chunks - 1) / chunks);
            // Cut where the page is blank, not at a fixed pixel count: a word cut in half loses
            // characters in one chunk and gains fragments in the other (the WER 0.26 signature).
            // Channel order needs no hypothesis here - every fixture pixel is grey (R=G=B) - so the
            // only free variable left at the boundary is *where* the cut falls.
            auto ink_in_column = [&](int x) {
                int ink = 0;
                for (int y = b.y0; y < b.y1; ++y) {
                    const unsigned char *p = &img.rgb[(static_cast<size_t>(y) * img.w + x) * 3];
                    if (static_cast<int>(p[0]) + p[1] + p[2] < 3 * 128) ++ink;
                }
                return ink;
            };
            // Collect the blank columns once, then cut only at those: a chunk boundary inside a word
            // truncates it in one piece and finishes it in the next ("blo" + "ck"), which is worse
            // than a slightly over-wide chunk. If a stretch has no blank column at all, it stays
            // whole - the recogniser pads to its 320-wide input anyway.
            // Blank *runs*, not blank columns: letter gaps are 1-3 px wide and cutting there splits
            // a word ("b aseline", measured), while word gaps are much wider. Only runs of at least
            // 6 px count as a word boundary.
            std::vector<std::pair<int, int>> gaps;
            int run_start = -1;
            for (int x = b.x0 + 2; x <= b.x1 - 2; ++x) {
                if (ink_in_column(x) == 0) { if (run_start < 0) run_start = x; }
                else if (run_start >= 0) {
                    if (x - run_start >= 6) gaps.push_back({run_start, x - 1});
                    run_start = -1;
                }
            }
            if (run_start >= 0 && b.x1 - 2 - run_start >= 6)
                gaps.push_back({run_start, b.x1 - 3});
            std::vector<int> cuts{b.x0};
            int cursor = b.x0;
            while (cursor + chunk_w < b.x1 - 2) {
                int target = cursor + chunk_w;
                int pick = -1, best_dist = 1 << 30;
                for (auto &g : gaps) {
                    int centre = (g.first + g.second) / 2;
                    if (centre <= cursor + chunk_w / 3 || centre >= b.x1 - 2) continue;
                    int dist = std::abs(centre - target);
                    if (dist < best_dist) { best_dist = dist; pick = centre; }
                }
                if (pick <= 0) break;  // no clean gap left: keep the remainder as one chunk
                cuts.push_back(pick);
                cursor = pick;
            }
            cuts.push_back(b.x1);
            std::vector<std::string> parts;
            for (size_t k = 0; k + 1 < cuts.size(); ++k) {
                Box part{cuts[k], b.y0, cuts[k + 1], b.y1};
                auto rin = rec_input(img, part);
                std::vector<int64_t> rec_shape{1, 3, 48, 320};
                auto rec_in = Ort::Value::CreateTensor<float>(mem, rin.data(), rin.size(),
                                                              rec_shape.data(), rec_shape.size());
                const char *rec_in_names[] = {"x"};
                const char *rec_out_names[] = {"fetch_name_0"};
                auto rec_out = rec.Run(Ort::RunOptions{nullptr}, rec_in_names, &rec_in, 1,
                                       rec_out_names, 1);
                auto info = rec_out[0].GetTensorTypeAndShapeInfo();
                auto shape = info.GetShape();
                int T = static_cast<int>(shape[1]), C = static_cast<int>(shape[2]);
                parts.push_back(ctc_decode(rec_out[0].GetTensorData<float>(), T, C, dict));
            }
            // Chunk boundaries fall on blank columns, so the space that separated two words lives
            // outside both crops and must be put back - otherwise "pipeline evidence barrier"
            // arrives as "pipelineevidencebarrier" (measured).
            for (size_t k = 0; k < parts.size(); ++k) {
                std::string piece = parts[k];
                while (!piece.empty() && piece.front() == ' ') piece.erase(piece.begin());
                while (!piece.empty() && piece.back() == ' ') piece.pop_back();
                if (piece.empty()) continue;
                if (!text.empty() && text.back() != ' ') text += " ";
                text += piece;
            }
            text += "\n";
        }
        auto t3 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::string base = page.substr(page.find_last_of('/') + 1);
        base = base.substr(0, base.find_last_of('.'));
        std::ofstream(out_dir + "/" + base + ".txt") << text;
        std::cout << base << " boxes=" << boxes.size() << " det_ms=" << ms(t0, t1)
                  << " post_ms=" << ms(t1, t2) << " rec_ms=" << ms(t2, t3)
                  << " total_ms=" << ms(t0, t3) << "\n";
    }
    return 0;
}
