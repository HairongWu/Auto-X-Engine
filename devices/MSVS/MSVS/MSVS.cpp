#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>

#include "../../../engine/include/autox_models.h"

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

int main()
{
	uint16_t frame_h = 224;
	uint16_t frame_w = 224;
	uint16_t frame_c = 3;

	char* buffer = read_file("./shufflenetv2_x_0_25.bin");
	// char* frame = read_file("./image.bin");
	float* frame = (float*)calloc(frame_h * frame_w * frame_c, sizeof(float));
	for (int i = 0; i < frame_h * frame_w * frame_c; i++)
		frame[i] = 1;
	float cls = -1;
	shufflenetv2_x_0_25(frame, frame_h, frame_w, (void*)buffer, &cls);
	printf("%f\n", cls);

	frame_h = 320;
	frame_w = 320;
	frame_c = 3;

	buffer = read_file("./picodet_xs_320.bin");
	// char* frame = read_file("./image.bin");
	frame = (float*)calloc(frame_h * frame_w * frame_c, sizeof(float));
	for (int i = 0; i < frame_h * frame_w * frame_c; i++)
		frame[i] = 1;
	float *dets = (float*)calloc(1*80 * 2125, sizeof(float));
	float* boxes = (float*)calloc(1 * 2125 * 4, sizeof(float));
	picodet_xs_320(frame, frame_h, frame_w, (void*)buffer, dets, boxes);
	free(dets);
	free(boxes);

	frame_h = 96;
	frame_w = 128;
	frame_c = 3;

	buffer = read_file("./tinypose_128x96.bin");
	// char* frame = read_file("./image.bin");
	frame = (float*)calloc(frame_h * frame_w * frame_c, sizeof(float));
	for (int i = 0; i < frame_h * frame_w * frame_c; i++)
		frame[i] = 1;
	float* heatmap = (float*)calloc(17*32*24, sizeof(float));
	tinypose_128x96(frame, frame_h, frame_w, (float*)buffer, heatmap);
	free(heatmap);

	//frame_h = 48;
	//frame_w = 1;
	//frame_c = 3;

	//buffer = read_file("./japan_PP_OCRv3_rec.bin");
	//// char* frame = read_file("./image.bin");
	//frame = (float*)calloc(frame_h * frame_w * frame_c, sizeof(float));
	//for (int i = 0; i < frame_h * frame_w * frame_c; i++)
	//	frame[i] = 1;
	//float* rec = (float*)calloc(1*16*4401, sizeof(float));
	//japan_PP_OCRv3_rec(frame, frame_h, frame_w, (float*)buffer, rec);

	//char checkpoint_path[256] = "stories15M.bin";
	//char tokenizer_path[256] = "tokenizer.bin";
	//char prompt[256] = "Once upon a time";
	//run_llama3(checkpoint_path, tokenizer_path, prompt);
}
