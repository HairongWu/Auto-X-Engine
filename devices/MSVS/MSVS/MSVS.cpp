﻿#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>

#include "../../../lib/include/autox_models.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../lib/include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../lib/include/stb_image_write.h"

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

	uint16_t in_h = 224;
	uint16_t in_w = 224;
	///////////////////////////////////////////////////////////////////
	uint8_t* data = stbi_load("./Ball.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}
	const float means[3] = { 0.485, 0.456, 0.406 };
	const float stds[3] = { 0.229, 0.224, 0.225 };
	uint8_t* dst = (uint8_t*)calloc(in_h * in_w * 3, sizeof(uint8_t));
	autox_resize_linear(data, dst, frame_w, frame_h, frame_c, in_w, in_h);
	free(data);
	float* out = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_normalize_image(dst, out, in_h, in_w, 3, means, stds);
	free(dst);
	float* x = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_hwc2chw(out, x, in_h, in_w, 3);
	free(out);

	float cls = -1;
	char* buffer = read_file("./shufflenetv2_x_0_25.bin");
	shufflenetv2_x_0_25(x, (void*)buffer, &cls);
	printf("%f\n", cls);

	//////////////////////////////////////////////////////////////////////

	data = stbi_load("./000000000036.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}
	in_h = 320;
	in_w = 320;
	dst = (uint8_t*)calloc(in_h * in_w * 3, sizeof(uint8_t));
	autox_resize_cubic(data, dst, frame_w, frame_h, frame_c, in_w, in_h);
	free(data);
	out = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_normalize_image(dst, out, in_h, in_w, 3, means, stds);
	free(dst);
	x = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_hwc2chw(out, x, in_h, in_w, 3);
	free(out);

	int class_num = 80;
	int box_num = 2125;
	float* scores = (float*)calloc(1 * 80 * 2125, sizeof(float));
	float* bboxes = (float*)calloc(1 * 2125 * 4, sizeof(float));
	buffer = read_file("./picodet_xs_320.bin");
	picodet_xs_320(x, (void*)buffer, scores, bboxes);

	float im_scale_y = 320 / float(frame_h);
	float im_scale_x = 320 / float(frame_w);
	float nms_threshold = 0.5f;
	float score_threshold = 0.3f;
	result_t* detections = (result_t*)calloc(box_num, sizeof(result_t));
	for (int i = 0; i < box_num; i++)
		detections[i].category = -1;
	//filter scores
	for (int num = 0; num < box_num; num++) {
		float score = scores[num];
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

		}
		else
			detections[num].category = -1;
	}
	free(scores);
	free(bboxes);
	//filter overlap
	for (int num = 0; num < box_num; ++num) {
		result_t box1 = detections[num];
		if (box1.category < 0)
			continue;
		for (int num2 = num + 1; num2 < box_num; ++num2) {
			result_t box2 = detections[num2];
			if (box2.category < 0 || box1.category != box2.category)
				continue;
			float overlap = cal_overlap(box1.box, box2.box);
			if (overlap > nms_threshold)
			{
				if (box1.prob > box2.prob)
					detections[num2].category = -1;
				else
				{
					detections[num].category = -1;
					break;
				}
			}
		}
	}
	for (int i = 0; i < box_num; i++)
		if (detections[i].category > -1)
			printf("%d, %f\n", detections[i].category, detections[i].prob);
	free(detections);
	//////////////////////////////////////////////////////////////////////////

	data = stbi_load("./hrnet_demo.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}
	in_h = 128;
	in_w = 96;
	dst = (uint8_t*)calloc(in_h * in_w * 3, sizeof(uint8_t));
	autox_resize_linear(data, dst, frame_w, frame_h, frame_c, in_w, in_h);
	free(data);
	out = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_normalize_image(dst, out, in_h, in_w, 3, means, stds);
	free(dst);
	x = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_hwc2chw(out, x, in_h, in_w, 3);
	free(out);

	float* heatmap = (float*)calloc(17 * 32 * 24, sizeof(float));
	buffer = read_file("./tinypose_128x96.bin");
	tinypose_128x96(x, (float*)buffer, heatmap);
	float* coords = (float*)calloc(17 * 2, sizeof(float));
	post_pose(heatmap, coords, 17, 32, 24);
	free(heatmap);
	for (int i = 0; i < 17 * 2; i++)
		printf("%f\n", coords[i]);
	free(coords);

	//////////////////////////////////////////////////////////////////////////
	in_h = 960;
	in_w = 960;

	data = stbi_load("./japan_2.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}

	dst = (uint8_t*)calloc(in_h * in_w * 3, sizeof(uint8_t));
	autox_resize_type0(data, dst, frame_w, frame_h, frame_c, 736, "min");
	free(data);
	out = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_normalize_image(dst, out, in_h, in_w, 3, means, stds);
	free(dst);
	x = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_hwc2chw(out, x, in_h, in_w, 3);
	float* pred = (float*)calloc(960 * 960, sizeof(float));

	buffer = read_file("./multilingual_det.bin");
	multilingual_det(x, (float*)buffer, pred);
	float* dt_boxes = (float*)calloc(960 * 960, sizeof(float));
	autox_db_postprocess(pred, dt_boxes, frame_h, frame_w, in_h, in_w, 0.3);
	free(pred);
	//////////////////////////////////////////////////////////////////////////
	in_h = 48;
	in_w = 320;
	data = stbi_load("./japan_2.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}

	dst = (uint8_t*)calloc(in_h * in_w * 3, sizeof(uint8_t));
	autox_resize_linear(data, dst, frame_w, frame_h, frame_c, in_w, in_h);
	free(data);
	out = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_normalize_image(dst, out, in_h, in_w, 3, means, stds);
	free(dst);
	x = (float*)calloc(in_h * in_w * 3, sizeof(float));
	autox_hwc2chw(out, x, in_h, in_w, 3);
	free(out);

	float* rec = (float*)calloc(1 * 1 * 4401, sizeof(float));
	buffer = read_file("./japan_PP_OCRv3_rec.bin");
	japan_PP_OCRv3_rec(x, (float*)buffer, rec);
	free(rec);
	/////////////////////////////////////////////////////////////////////////
	data = stbi_load("./example.jpg", &frame_w, &frame_h, &frame_c, 3);
	if (!data) {
		return 1;
	}

	float* img1 = (float*)calloc(1024 * 1024 * 3, sizeof(float));
	if (!sam_image_preprocess(data, img1, frame_w, frame_h, 1024, 1024)) {
		fprintf(stderr, "%s: failed to preprocess image\n", __func__);
		return AUTOX_FAIL;
	}
	free(data);

	buffer = read_file("E:\\app\\model_pool\\ggml-model-f32.bin");

	autox_sam(img1, buffer, frame_w, frame_h, 414.375f, 162.796875f);

	//std::string filename = fname + std::to_string(i) + ".png";
	//if (!stbi_write_png(filename.c_str(), res.nx, res.ny, 1, res.data.data(), res.nx)) {
	//	printf("%s: failed to write mask %s\n", __func__, filename.c_str());
	//	return false;
	//}

	/////////////////////////////////////////////////////////////////////////

	//char checkpoint_path[128] = "llama3_8b_base.bin";
	//char tokenizer_path[128] = "tokenizer_llama3.bin";
	//char prompt[256] = "Once upon a time";
	//run_llama3(checkpoint_path, tokenizer_path, prompt, 50);
}
