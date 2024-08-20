#include "../include/autox_nn.h"

// ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L164
// resize largest dimension to 1024
// normalize: x = (x - mean) / std
//     mean = [123.675, 116.28, 103.53]
//     std  = [58.395, 57.12, 57.375]
//     TODO: why are these hardcoded !?
// pad to 1024x1024
// TODO: for some reason, this is not numerically identical to pytorch's interpolation
bool sam_image_preprocess(const uint8_t* img, float* res, const int nx, const int ny,
    const int nx2, const int ny2) {

    const float scale = max(nx, ny) / 1024.0f;

    const int nx3 = (int)(nx / scale + 0.5f);
    const int ny3 = (int)(ny / scale + 0.5f);

    const float m3[3] = { 123.675f, 116.280f, 103.530f };
    const float s3[3] = { 58.395f,  57.120f,  57.375f };

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = (x + 0.5f) * scale - 0.5f;
                const float sy = (y + 0.5f) * scale - 0.5f;

                const int x0 = max(0, (int)floor(sx));
                const int y0 = max(0, (int)floor(sy));

                const int x1 = min(x0 + 1, nx - 1);
                const int y1 = min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = img[j00];
                const float v01 = img[j01];
                const float v10 = img[j10];
                const float v11 = img[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = min(max(round(v), 0.0f), 255.0f);

                const int i = 3 * (y * nx3 + x) + c;

                res[i] = ((float)(v2)-m3[c]) / s3[c];
            }
        }
    }

    return true;
}