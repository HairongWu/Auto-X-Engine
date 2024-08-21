#include "../include/autox_models.h"

int32_t offset_t_ms = 0;
int32_t offset_n = 0;
int32_t duration_ms = 0;
int32_t progress_step = 5;
int32_t max_context = -1;
int32_t max_len = 0;
int32_t audio_ctx = 0;

float word_thold = 0.01f;
float entropy_thold = 2.40f;
float logprob_thold = -1.00f;
float grammar_penalty = 100.0f;
float temperature = 0.0f;
float temperature_inc = 0.2f;

bool debug_mode = false;
bool translate = false;
bool detect_language = false;
bool diarize = false;
bool tinydiarize = false;
bool split_on_word = false;
bool no_fallback = false;
bool output_txt = false;
bool output_vtt = false;
bool output_srt = false;
bool output_wts = false;
bool output_csv = false;
bool output_jsn = false;
bool output_jsn_full = false;
bool output_lrc = false;
bool no_prints = false;
bool print_special = false;
bool print_colors = false;
bool print_progress = false;
bool no_timestamps = false;
bool log_score = false;
bool use_gpu = true;
bool flash_attn = false;

char* language = "en";
char* prompt;
char* font_path = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf";
char* model = "models/ggml-base.en.bin";
char* grammar;
char* grammar_rule;

// [TDRZ] speaker turn string
char* tdrz_speaker_turn = " [SPEAKER_TURN]"; // TODO: set from command line

// A regular expression that matches tokens to suppress
char* suppress_regex;

char* openvino_encode_device = "CPU";

char* dtw = "";

autox_err_t autox_whisper(float* img1, void* weights, int nx, int ny, float x, float y)
{
}