#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>

#include "../../../engine/include/autox_models.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

char* read_file(const char* file_name) {
	FILE* file;
	fopen_s(&file, file_name, "rb");
	if (file == NULL)
		return NULL;

	if (fseek(file, 0, SEEK_END)) {
		return NULL; // fseek error
	}
	long tell = ftell(file);
	if (tell < 0 || tell + ((size_t)0) > SIZE_MAX - 1) {
		if (tell == -1L) return NULL;  // ftell error
		return NULL; // range error
	}
	size_t file_length = (size_t)tell;
	if (fseek(file, 0, SEEK_SET)) {
		return NULL; // fseek error
	}
	char* buf = (char*)calloc(file_length + 1, 1);
	if (buf == NULL) {
		return NULL; // alloc error
	}
	size_t bytes_read = fread(buf, 1, file_length, file);
	if (ferror(file)) {
		free(buf);
		return NULL; // fread error
	}
	buf[bytes_read] = '\0';

	fclose(file);
	return buf;
}

typedef struct
{
	int32_t category;              /*<! category index */
	float box[4];      /*<! [left_up_x, left_up_y, right_down_x, right_down_y] */
	float prob;
	float keypoints[34]; /*<! [x1, y1, x2, y2, ...] */
} result_t;

int main()
{
	int frame_h;
	int frame_w;
	int frame_c;

	///////////////////////////////////////////////////////////////////
	char* buffer = read_file("./shufflenetv2_x_0_25.bin");
	uint8_t* data = stbi_load("./Ball.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}

	 uint8_t *dst = (uint8_t *)calloc(224*224* 3, sizeof(uint8_t));
	 autox_resize_image(data, dst, frame_h, frame_w, 224, 224);
	 free(data);
	 float *out = (float *)calloc(224*224* 3, sizeof(float));
	 autox_normalize_image(dst, out, 224, 224, 3);
	 free(dst);
	 float *x = (float *)calloc(224*224* 3, sizeof(float));
	 autox_hwc2chw(out, x, 224, 224, 3);
	 free(out);

	float cls = -1;
	shufflenetv2_x_0_25(x, (void*)buffer, &cls);
	printf("%f\n", cls);

	//////////////////////////////////////////////////////////////////////
	buffer = read_file("./picodet_xs_320.bin");
	data = stbi_load("./000000000036.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}

	dst = (uint8_t*)calloc(320 * 320 * 3, sizeof(uint8_t));
	autox_resize_image(data, dst, frame_h, frame_w, 320, 320);
	free(data);
	out = (float*)calloc(320 * 320 * 3, sizeof(float));
	autox_normalize_image(dst, out, 320, 320, 3);
	free(dst);
	x = (float*)calloc(320 * 320 * 3, sizeof(float));
	autox_hwc2chw(out, x, 320, 320, 3);
	free(out);

	int class_num = 80;
	int box_num = 2125;
	float * scores = (float*)calloc(1*80 * 2125, sizeof(float));
	float* bboxes = (float*)calloc(1 * 2125 * 4, sizeof(float));

	picodet_xs_320(x, (void*)buffer, scores, bboxes);

	float im_scale_y = 320 / float(frame_h);
	float im_scale_x = 320 / float(frame_w);
	float nms_threshold = 0.5f;
	float score_threshold = 0.3f;
	result_t* detections = (result_t*)malloc(box_num * sizeof(result_t));
	//filter scores
	for (int num = 0; num < box_num; num++) {
		float score = scores[0];
		int index = 0;
		for (int c = 1; c < class_num; ++c) {
			if (score < scores[num + c * box_num])
			{
				score = scores[num + c * box_num];
				index = c;
			}

		}
		if (score > score_threshold)
		{
			float x1 = bboxes[num * 4] / im_scale_x;
			float y1 = bboxes[num * 4 + 1] / im_scale_y;
			float x2 = bboxes[num * 4 + 2] / im_scale_x;
			float y2 = bboxes[num * 4 + 3] / im_scale_y;

			detections[num].box[0] = x1;
			detections[num].box[1] = y1;
			detections[num].box[2] = x2;
			detections[num].box[3] = y2;
			detections[num].category = index;
			detections[num].prob = score;
			printf("%d, %f\n", index, score);
		}
		else
			detections[num].category = -1;
	}
	free(scores);
	free(bboxes);

	//////////////////////////////////////////////////////////////////////////
	frame_h = 96;
	frame_w = 128;
	frame_c = 3;

	buffer = read_file("./tinypose_128x96.bin");
	// char* frame = read_file("./image.bin");
	float *frame = (float*)calloc(frame_h * frame_w * frame_c, sizeof(float));
	for (int i = 0; i < frame_h * frame_w * frame_c; i++)
		frame[i] = 1;
	float* heatmap = (float*)calloc(17*32*24, sizeof(float));
	tinypose_128x96(frame, (float*)buffer, heatmap);
	free(heatmap);

	//////////////////////////////////////////////////////////////////////////
	frame_h = 48;
	frame_w = 1;
	frame_c = 3;

	buffer = read_file("./japan_PP_OCRv3_rec.bin");
	// char* frame = read_file("./image.bin");
	frame = (float*)calloc(frame_h * frame_w * frame_c, sizeof(float));
	for (int i = 0; i < frame_h * frame_w * frame_c; i++)
		frame[i] = 1;
	float* rec = (float*)calloc(1*1*4401, sizeof(float));
	japan_PP_OCRv3_rec(frame, (float*)buffer, rec);
	free(rec);

	/////////////////////////////////////////////////////////////////////////
	//frame_h = 512;
	//frame_w = 512;
	//frame_c = 3;

	//buffer = read_file("./mobileseg_tiny.bin");
	//// char* frame = read_file("./image.bin");
	//frame = (float*)calloc(frame_h * frame_w * frame_c, sizeof(float));
	//for (int i = 0; i < frame_h * frame_w * frame_c; i++)
	//	frame[i] = 1;
	//rec = (float*)calloc(1 * 512 * 512, sizeof(float));
	//mobileseg_tiny(frame, (float*)buffer, rec);
	//free(rec);

	//char checkpoint_path[128] = "stories15M.bin";
	//char tokenizer_path[128] = "tokenizer.bin";
	//char prompt[256] = "Once upon a time";
	//run_llama3(checkpoint_path, tokenizer_path, prompt);
}
