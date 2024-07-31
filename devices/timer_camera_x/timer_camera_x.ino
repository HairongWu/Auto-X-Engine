/**
 * @file capture.ino
 * @author SeanKwok (shaoxiang@m5stack.com)
 * @brief TimerCAM Take Photo Test
 * @version 0.1
 * @date 2023-12-28
 *
 *
 * @Hardwares: TimerCAM
 * @Platform Version: Arduino M5Stack Board Manager v2.0.9
 * @Dependent Library:
 * TimerCam-arduino: https://github.com/m5stack/TimerCam-arduino
 */
#include "M5TimerCAM.h"
#include "esp_partition.h"
#include "esp_spi_flash.h"

//#include "shufflenetv2.h"

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

        float ret = -1;
        //shufflenetv2(image_ptr, TimerCAM.Camera.fb->height, TimerCAM.Camera.fb->width, model_ptr, &ret);
        Serial.println("ret:");
    }
}
