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

	char* buffer = read_file("./model.bin");
	// char* frame = read_file("./image.bin");
	uint8_t* frame = (uint8_t*)calloc(224 * 224 * 3, 1);
	for (int i = 0; i < 224 * 224 * 3; i++)
		frame[i] = 1;
	float Out = -1;
	shufflenetv2_x_0_25(frame, frame_h, frame_w, (float*)buffer, &Out);
	printf("%f", Out);

	char checkpoint_path[256] = "";
	char tokenizer_path[256] = "";
	char prompt[256] = "";
	run_llama3(checkpoint_path, tokenizer_path, prompt);
}
