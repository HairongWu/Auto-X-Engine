
#include "M5TimerCAM.h"
#include "esp_partition.h"
#include "esp_spi_flash.h"

#include "autox_models.h"

const void *model_ptr;
void setup() {
    TimerCAM.begin();

    if (!TimerCAM.Camera.begin()) {
        Serial.println("Camera Init Fail");
        return;
    }
    Serial.println("Camera Init Success");

    TimerCAM.Camera.sensor->set_pixformat(TimerCAM.Camera.sensor,
                                          PIXFORMAT_JPEG);
    TimerCAM.Camera.sensor->set_framesize(TimerCAM.Camera.sensor,
                                          FRAMESIZE_QVGA);

    TimerCAM.Camera.sensor->set_vflip(TimerCAM.Camera.sensor, 1);
    TimerCAM.Camera.sensor->set_hmirror(TimerCAM.Camera.sensor, 0);

    const esp_partition_t *partition = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    esp_partition_mmap_handle_t map_handle;
    // Map the partition to data memory
    ESP_ERROR_CHECK(esp_partition_mmap(partition, 0, partition->size, ESP_PARTITION_MMAP_DATA, &model_ptr, &map_handle));
    ESP_LOGI("TAG", "Mapped partition to data memory address %p", model_ptr);

}

void loop() {
    if (TimerCAM.Camera.get()) {
        Serial.printf("pic size: %d\n", TimerCAM.Camera.fb->len);
        uint8_t *image_ptr = (uint8_t *)calloc(TimerCAM.Camera.fb->height * TimerCAM.Camera.fb->width * 3, sizeof(uint8_t));
        if (image_ptr)
        {
            if (fmt2rgb888(TimerCAM.Camera.fb->buf, TimerCAM.Camera.fb->len, TimerCAM.Camera.fb->format, image_ptr))
            {
            }
            else
            {
                ESP_LOGE(TAG, "fmt2rgb888 failed");
                free(image_ptr);
            }
        }
        else
        {
            ESP_LOGE(TAG, "malloc memory for image image_ptr failed");
        }
        TimerCAM.Camera.free();
        
        // uint8_t *dst = (uint8_t *)calloc(224*224* 3, sizeof(uint8_t));
        // autox_resize_image(image_ptr, dst, TimerCAM.Camera.fb->height, TimerCAM.Camera.fb->width, 224, 224);
        // free(image_ptr);
        // float *out = (float *)calloc(224*224* 3, sizeof(float));
        // autox_normalize_image(dst, out, 224, 224, 3);
        // free(dst);
        // float *x = (float *)calloc(224*224* 3, sizeof(float));
        // autox_hwc2chw(out, x, 224, 224, 3);
        // free(out);

        // float cls = -1;
        // shufflenetv2_x_0_25(x, model_ptr, &cls);
        // ESP_LOGE(TAG, "ret:%f", cls);

        uint8_t *dst = (uint8_t *)calloc(320*320* 3, sizeof(uint8_t));
        autox_resize_image(image_ptr, dst, TimerCAM.Camera.fb->height, TimerCAM.Camera.fb->width, 320, 320);
        free(image_ptr);
        float *out = (float *)calloc(320*320* 3, sizeof(float));
        autox_normalize_image(dst, out, 320, 320, 3);
        free(dst);
        float *x = (float *)calloc(320*320* 3, sizeof(float));
        autox_hwc2chw(out, x, 320, 320, 3);
        free(out);

        float *dets = (float*)calloc(1*80 * 2125, sizeof(float));
	      float* boxes = (float*)calloc(1 * 2125 * 4, sizeof(float));
        picodet_xs_320(x, model_ptr, dets, boxes);
        free(dets);
	      free(boxes);
    }
}
