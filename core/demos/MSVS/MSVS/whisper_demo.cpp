#include "whisper_demo.h"

#define GGML_FILE_MAGIC   0x67676d6c // "ggml"
#define GGML_QNT_VERSION_FACTOR 1000 // do not change this
#define GGML_MAX_CONTEXTS       64

// Available sampling strategies
enum whisper_sampling_strategy {
	WHISPER_SAMPLING_GREEDY,      // similar to OpenAI's GreedyDecoder
	WHISPER_SAMPLING_BEAM_SEARCH, // similar to OpenAI's BeamSearchDecoder
};

// available whisper models
enum e_model {
	MODEL_UNKNOWN,
	MODEL_TINY,
	MODEL_BASE,
	MODEL_SMALL,
	MODEL_MEDIUM,
	MODEL_LARGE,
};

// medium
// hparams: {
// 'n_mels': 80,
// 'n_vocab': 51864,
// 'n_audio_ctx': 1500,
// 'n_audio_state': 1024,
// 'n_audio_head': 16,
// 'n_audio_layer': 24,
// 'n_text_ctx': 448,
// 'n_text_state': 1024,
// 'n_text_head': 16,
// 'n_text_layer': 24
// }
//
// default hparams (Whisper tiny)
struct whisper_hparams {
	int32_t n_vocab = 51864;
	int32_t n_audio_ctx = 1500;
	int32_t n_audio_state = 384;
	int32_t n_audio_head = 6;
	int32_t n_audio_layer = 4;
	int32_t n_text_ctx = 448;
	int32_t n_text_state = 384;
	int32_t n_text_head = 6;
	int32_t n_text_layer = 4;
	int32_t n_mels = 80;
	int32_t ftype = 1;
	float   eps = 1e-5f;
};
// NOTE: always add types at the end of the enum to keep backward compatibility
enum ggml_type {
	GGML_TYPE_F32 = 0,
	GGML_TYPE_F16 = 1,
	GGML_TYPE_Q4_0 = 2,
	GGML_TYPE_Q4_1 = 3,
	// GGML_TYPE_Q4_2 = 4, support has been removed
	// GGML_TYPE_Q4_3 = 5, support has been removed
	GGML_TYPE_Q5_0 = 6,
	GGML_TYPE_Q5_1 = 7,
	GGML_TYPE_Q8_0 = 8,
	GGML_TYPE_Q8_1 = 9,
	GGML_TYPE_Q2_K = 10,
	GGML_TYPE_Q3_K = 11,
	GGML_TYPE_Q4_K = 12,
	GGML_TYPE_Q5_K = 13,
	GGML_TYPE_Q6_K = 14,
	GGML_TYPE_Q8_K = 15,
	GGML_TYPE_IQ2_XXS = 16,
	GGML_TYPE_IQ2_XS = 17,
	GGML_TYPE_IQ3_XXS = 18,
	GGML_TYPE_IQ1_S = 19,
	GGML_TYPE_IQ4_NL = 20,
	GGML_TYPE_IQ3_S = 21,
	GGML_TYPE_IQ2_S = 22,
	GGML_TYPE_IQ4_XS = 23,
	GGML_TYPE_I8 = 24,
	GGML_TYPE_I16 = 25,
	GGML_TYPE_I32 = 26,
	GGML_TYPE_I64 = 27,
	GGML_TYPE_F64 = 28,
	GGML_TYPE_IQ1_M = 29,
	GGML_TYPE_BF16 = 30,
	GGML_TYPE_COUNT,
};

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
	{ "en",  { 0,  "english",         } },
	{ "zh",  { 1,  "chinese",         } },
	{ "de",  { 2,  "german",          } },
	{ "es",  { 3,  "spanish",         } },
	{ "ru",  { 4,  "russian",         } },
	{ "ko",  { 5,  "korean",          } },
	{ "fr",  { 6,  "french",          } },
	{ "ja",  { 7,  "japanese",        } },
	{ "pt",  { 8,  "portuguese",      } },
	{ "tr",  { 9,  "turkish",         } },
	{ "pl",  { 10, "polish",          } },
	{ "ca",  { 11,  "catalan",        } },
	{ "nl",  { 12,  "dutch",          } },
	{ "ar",  { 13,  "arabic",         } },
	{ "sv",  { 14,  "swedish",        } },
	{ "it",  { 15,  "italian",        } },
	{ "id",  { 16,  "indonesian",     } },
	{ "hi",  { 17,  "hindi",          } },
	{ "fi",  { 18,  "finnish",        } },
	{ "vi",  { 19,  "vietnamese",     } },
	{ "he",  { 20,  "hebrew",         } },
	{ "uk",  { 21,  "ukrainian",      } },
	{ "el",  { 22,  "greek",          } },
	{ "ms",  { 23,  "malay",          } },
	{ "cs",  { 24,  "czech",          } },
	{ "ro",  { 25,  "romanian",       } },
	{ "da",  { 26,  "danish",         } },
	{ "hu",  { 27,  "hungarian",      } },
	{ "ta",  { 28,  "tamil",          } },
	{ "no",  { 29,  "norwegian",      } },
	{ "th",  { 30,  "thai",           } },
	{ "ur",  { 31,  "urdu",           } },
	{ "hr",  { 32,  "croatian",       } },
	{ "bg",  { 33,  "bulgarian",      } },
	{ "lt",  { 34,  "lithuanian",     } },
	{ "la",  { 35,  "latin",          } },
	{ "mi",  { 36,  "maori",          } },
	{ "ml",  { 37,  "malayalam",      } },
	{ "cy",  { 38,  "welsh",          } },
	{ "sk",  { 39,  "slovak",         } },
	{ "te",  { 40,  "telugu",         } },
	{ "fa",  { 41,  "persian",        } },
	{ "lv",  { 42,  "latvian",        } },
	{ "bn",  { 43,  "bengali",        } },
	{ "sr",  { 44,  "serbian",        } },
	{ "az",  { 45,  "azerbaijani",    } },
	{ "sl",  { 46,  "slovenian",      } },
	{ "kn",  { 47,  "kannada",        } },
	{ "et",  { 48,  "estonian",       } },
	{ "mk",  { 49,  "macedonian",     } },
	{ "br",  { 50,  "breton",         } },
	{ "eu",  { 51,  "basque",         } },
	{ "is",  { 52,  "icelandic",      } },
	{ "hy",  { 53,  "armenian",       } },
	{ "ne",  { 54,  "nepali",         } },
	{ "mn",  { 55,  "mongolian",      } },
	{ "bs",  { 56,  "bosnian",        } },
	{ "kk",  { 57,  "kazakh",         } },
	{ "sq",  { 58,  "albanian",       } },
	{ "sw",  { 59,  "swahili",        } },
	{ "gl",  { 60,  "galician",       } },
	{ "mr",  { 61,  "marathi",        } },
	{ "pa",  { 62,  "punjabi",        } },
	{ "si",  { 63,  "sinhala",        } },
	{ "km",  { 64,  "khmer",          } },
	{ "sn",  { 65,  "shona",          } },
	{ "yo",  { 66,  "yoruba",         } },
	{ "so",  { 67,  "somali",         } },
	{ "af",  { 68,  "afrikaans",      } },
	{ "oc",  { 69,  "occitan",        } },
	{ "ka",  { 70,  "georgian",       } },
	{ "be",  { 71,  "belarusian",     } },
	{ "tg",  { 72,  "tajik",          } },
	{ "sd",  { 73,  "sindhi",         } },
	{ "gu",  { 74,  "gujarati",       } },
	{ "am",  { 75,  "amharic",        } },
	{ "yi",  { 76,  "yiddish",        } },
	{ "lo",  { 77,  "lao",            } },
	{ "uz",  { 78,  "uzbek",          } },
	{ "fo",  { 79,  "faroese",        } },
	{ "ht",  { 80,  "haitian creole", } },
	{ "ps",  { 81,  "pashto",         } },
	{ "tk",  { 82,  "turkmen",        } },
	{ "nn",  { 83,  "nynorsk",        } },
	{ "mt",  { 84,  "maltese",        } },
	{ "sa",  { 85,  "sanskrit",       } },
	{ "lb",  { 86,  "luxembourgish",  } },
	{ "my",  { 87,  "myanmar",        } },
	{ "bo",  { 88,  "tibetan",        } },
	{ "tl",  { 89,  "tagalog",        } },
	{ "mg",  { 90,  "malagasy",       } },
	{ "as",  { 91,  "assamese",       } },
	{ "tt",  { 92,  "tatar",          } },
	{ "haw", { 93,  "hawaiian",       } },
	{ "ln",  { 94,  "lingala",        } },
	{ "ha",  { 95,  "hausa",          } },
	{ "ba",  { 96,  "bashkir",        } },
	{ "jw",  { 97,  "javanese",       } },
	{ "su",  { 98,  "sundanese",      } },
	{ "yue", { 99,  "cantonese",      } },
};

const char * whisper_lang_str(int id) {
	for (const auto & kv : g_lang) {
		if (kv.second.first == id) {
			return kv.first.c_str();
		}
	}
	return nullptr;
}

//struct ggml_context * ggml_init(struct ggml_init_params params) {
//	// make this function thread safe
//	ggml_critical_section_start();
//
//	static bool is_first_call = true;
//
//	if (is_first_call) {
//		// initialize GELU, Quick GELU, SILU and EXP F32 tables
//		{
//			for (int i = 0; i < (1 << 16); ++i) {
//				union {
//					uint16_t u16;
//					ggml_fp16_t fp16;
//				} u = { i };
//				float f = ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
//				ggml_table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
//				ggml_table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
//				ggml_table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
//				ggml_table_exp_f16[i] = GGML_FP32_TO_FP16(expf(f));
//			}
//
//			const uint64_t t_end = ggml_time_us(); UNUSED(t_end);
//
//			GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start) / 1000.0f);
//		}
//
//		// initialize g_state
//		{
//			const uint64_t t_start = ggml_time_us(); UNUSED(t_start);
//
//			g_state = (struct ggml_state) {
//				/*.contexts =*/ { { 0 } },
//					/*.numa =*/{
//						.n_nodes = 0,
//						.total_cpus = 0,
//					},
//			};
//
//			for (int i = 0; i < GGML_MAX_CONTEXTS; ++i) {
//				g_state.contexts[i].used = false;
//			}
//
//			const uint64_t t_end = ggml_time_us(); UNUSED(t_end);
//
//			GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start) / 1000.0f);
//		}
//
//		ggml_setup_op_has_task_pass();
//
//		is_first_call = false;
//	}
//
//	// find non-used context in g_state
//	struct ggml_context * ctx = NULL;
//
//	for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
//		if (!g_state.contexts[i].used) {
//			g_state.contexts[i].used = true;
//			ctx = &g_state.contexts[i].context;
//
//			GGML_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
//			break;
//		}
//	}
//
//	if (ctx == NULL) {
//		GGML_PRINT_DEBUG("%s: no unused context found\n", __func__);
//
//		ggml_critical_section_end();
//
//		return NULL;
//	}
//
//	// allow to call ggml_init with 0 size
//	if (params.mem_size == 0) {
//		params.mem_size = GGML_MEM_ALIGN;
//	}
//
//	const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);
//
//	*ctx = (struct ggml_context) {
//		/*.mem_size           =*/ mem_size,
//			/*.mem_buffer         =*/ params.mem_buffer ? params.mem_buffer : GGML_ALIGNED_MALLOC(mem_size),
//			/*.mem_buffer_owned   =*/ params.mem_buffer ? false : true,
//			/*.no_alloc           =*/ params.no_alloc,
//			/*.no_alloc_save      =*/ params.no_alloc,
//			/*.n_objects          =*/ 0,
//			/*.objects_begin      =*/ NULL,
//			/*.objects_end        =*/ NULL,
//			/*.scratch            =*/{ 0, 0, NULL, },
//			/*.scratch_save       =*/{ 0, 0, NULL, },
//	};
//
//	GGML_ASSERT(ctx->mem_buffer != NULL);
//
//	ggml_assert_aligned(ctx->mem_buffer);
//
//	GGML_PRINT_DEBUG("%s: context initialized\n", __func__);
//
//	ggml_critical_section_end();
//
//	return ctx;
//}
//
//void ggml_free(struct ggml_context * ctx) {
//	if (ctx == NULL) {
//		return;
//	}
//
//	// make this function thread safe
//	ggml_critical_section_start();
//
//	bool found = false;
//
//	for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
//		if (&g_state.contexts[i].context == ctx) {
//			g_state.contexts[i].used = false;
//
//			GGML_PRINT_DEBUG("%s: context %d has been freed. memory used = %zu\n",
//				__func__, i, ggml_used_mem(ctx));
//
//			if (ctx->mem_buffer_owned) {
//				GGML_ALIGNED_FREE(ctx->mem_buffer);
//			}
//
//			found = true;
//			break;
//		}
//	}
//
//	if (!found) {
//		GGML_PRINT_DEBUG("%s: context not found\n", __func__);
//	}
//
//	ggml_critical_section_end();
//}
//
//// load the model from a ggml file
////
//// file format:
////
////   - hparams
////   - pre-computed mel filters
////   - vocab
////   - weights
////
//// see the convert-pt-to-ggml.py script for details
////
//static bool whisper_model_load(auto model, auto vocab) {
//	// verify magic
//	{
//		uint32_t magic;
//		read_safe(loader, magic);
//		if (magic != GGML_FILE_MAGIC) {
//			return false;
//		}
//	}
//
//	//load hparams
//	{
//		whisper_hparams hparams = model.hparams;
//
//		read_safe(loader, hparams.n_vocab);
//		read_safe(loader, hparams.n_audio_ctx);
//		read_safe(loader, hparams.n_audio_state);
//		read_safe(loader, hparams.n_audio_head);
//		read_safe(loader, hparams.n_audio_layer);
//		read_safe(loader, hparams.n_text_ctx);
//		read_safe(loader, hparams.n_text_state);
//		read_safe(loader, hparams.n_text_head);
//		read_safe(loader, hparams.n_text_layer);
//		read_safe(loader, hparams.n_mels);
//		read_safe(loader, hparams.ftype);
//
//		assert(hparams.n_text_state == hparams.n_audio_state);
//
//		std::string mver = "";
//
//		if (hparams.n_audio_layer == 4) {
//			model.type = e_model::MODEL_TINY;
//		}
//
//		if (hparams.n_audio_layer == 6) {
//			model.type = e_model::MODEL_BASE;
//		}
//
//		if (hparams.n_audio_layer == 12) {
//			model.type = e_model::MODEL_SMALL;
//		}
//
//		if (hparams.n_audio_layer == 24) {
//			model.type = e_model::MODEL_MEDIUM;
//		}
//
//		if (hparams.n_audio_layer == 32) {
//			model.type = e_model::MODEL_LARGE;
//
//			if (hparams.n_vocab == 51866) {
//				mver = " v3";
//			}
//		}
//
//		const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;
//
//		hparams.ftype %= GGML_QNT_VERSION_FACTOR;
//
//		// for the big tensors, we have the option to store the data in 16-bit floats or quantized
//		// in order to save memory and also to speed up the computation
//		wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
//		if (wctx.wtype == GGML_TYPE_COUNT) {
//			WHISPER_LOG_ERROR("%s: invalid model (bad ftype value %d)\n", __func__, model.hparams.ftype);
//			return false;
//		}
//	}
//
//	// load mel filters
//	{
//		auto & filters = wctx.model.filters;
//
//		read_safe(loader, filters.n_mel);
//		read_safe(loader, filters.n_fft);
//
//		filters.data.resize(filters.n_mel * filters.n_fft);
//		loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
//		BYTESWAP_FILTERS(filters);
//	}
//
//	// load vocab
//	{
//		int32_t n_vocab = 0;
//		read_safe(loader, n_vocab);
//
//		//if (n_vocab != model.hparams.n_vocab) {
//		//    WHISPER_LOG_ERROR("%s: invalid model file '%s' (bad vocab size %d != %d)\n",
//		//            __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
//		//    return false;
//		//}
//
//		std::string word;
//		std::vector<char> tmp;
//
//		tmp.reserve(128);
//
//		for (int i = 0; i < n_vocab; i++) {
//			uint32_t len;
//			read_safe(loader, len);
//
//			if (len > 0) {
//				tmp.resize(len);
//				loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
//				word.assign(&tmp[0], tmp.size());
//			}
//			else {
//				// seems like we have an empty-string token in multi-language models (i = 50256)
//				//WHISPER_LOG_WARN("%s: warning: empty-string token in vocab, i = %d\n", __func__, i);
//				word = "";
//			}
//
//			vocab.token_to_id[word] = i;
//			vocab.id_to_token[i] = word;
//
//			//printf("%s: vocab[%d] = '%s'\n", __func__, i, word.c_str());
//		}
//
//		vocab.n_vocab = model.hparams.n_vocab;
//		if (vocab.is_multilingual()) {
//			vocab.token_eot++;
//			vocab.token_sot++;
//
//			// account for variable number of language tokens
//			const int dt = vocab.num_languages() - 98;
//
//			vocab.token_translate += dt;
//			vocab.token_transcribe += dt;
//			vocab.token_solm += dt;
//			vocab.token_prev += dt;
//			vocab.token_nosp += dt;
//			vocab.token_not += dt;
//			vocab.token_beg += dt;
//		}
//
//		if (n_vocab < model.hparams.n_vocab) {
//			for (int i = n_vocab; i < model.hparams.n_vocab; i++) {
//				if (i > vocab.token_beg) {
//					word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
//				}
//				else if (i == vocab.token_eot) {
//					word = "[_EOT_]";
//				}
//				else if (i == vocab.token_sot) {
//					word = "[_SOT_]";
//				}
//				else if (i == vocab.token_translate) {
//					word = "[_TRANSLATE_]";
//				}
//				else if (i == vocab.token_transcribe) {
//					word = "[_TRANSCRIBE_]";
//				}
//				else if (i == vocab.token_solm) {
//					word = "[_SOLM_]";
//				}
//				else if (i == vocab.token_prev) {
//					word = "[_PREV_]";
//				}
//				else if (i == vocab.token_nosp) {
//					word = "[_NOSP_]";
//				}
//				else if (i == vocab.token_not) {
//					word = "[_NOT_]";
//				}
//				else if (i == vocab.token_beg) {
//					word = "[_BEG_]";
//				}
//				else if (i > vocab.token_sot && i <= vocab.token_sot + vocab.num_languages()) {
//					word = "[_LANG_" + std::string(whisper_lang_str(i - vocab.token_sot - 1)) + "]";
//				}
//				else {
//					word = "[_extra_token_" + std::to_string(i) + "]";
//				}
//				vocab.token_to_id[word] = i;
//				vocab.id_to_token[i] = word;
//			}
//		}
//
//		WHISPER_LOG_INFO("%s: n_langs       = %d\n", __func__, vocab.num_languages());
//	}
//
//	const ggml_type wtype = wctx.wtype;
//	const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16; // conv type
//
//	// create the ggml context
//	{
//		const auto & hparams = model.hparams;
//
//		const int n_audio_layer = hparams.n_audio_layer;
//		const int n_text_layer = hparams.n_text_layer;
//
//		const size_t n_tensors = 10 /* input */ + 15 + 15 * n_audio_layer + 24 * n_text_layer;
//
//		model.ctx = ggml_init(params);
//		if (!model.ctx) {
//			WHISPER_LOG_ERROR("%s: ggml_init() failed\n", __func__);
//			return false;
//		}
//	}
//
//	// prepare tensors for the weights
//	{
//		auto & ctx = model.ctx;
//
//		const auto & hparams = model.hparams;
//
//		const int n_vocab = hparams.n_vocab;
//
//		const int n_audio_ctx = hparams.n_audio_ctx;
//		const int n_audio_state = hparams.n_audio_state;
//		const int n_audio_layer = hparams.n_audio_layer;
//
//		const int n_text_ctx = hparams.n_text_ctx;
//		const int n_text_state = hparams.n_text_state;
//		const int n_text_layer = hparams.n_text_layer;
//
//		const int n_mels = hparams.n_mels;
//
//		model.layers_encoder.resize(n_audio_layer);
//		model.layers_decoder.resize(n_text_layer);
//
//		// encoder
//		{
//			model.e_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_state, n_audio_ctx);
//
//			model.e_conv_1_w = ggml_new_tensor_3d(ctx, vtype, 3, n_mels, n_audio_state);
//			model.e_conv_1_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);
//
//			model.e_conv_2_w = ggml_new_tensor_3d(ctx, vtype, 3, n_audio_state, n_audio_state);
//			model.e_conv_2_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_audio_state);
//
//			model.e_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//			model.e_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//
//			// map by name
//			model.tensors["encoder.positional_embedding"] = model.e_pe;
//
//			model.tensors["encoder.conv1.weight"] = model.e_conv_1_w;
//			model.tensors["encoder.conv1.bias"] = model.e_conv_1_b;
//
//			model.tensors["encoder.conv2.weight"] = model.e_conv_2_w;
//			model.tensors["encoder.conv2.bias"] = model.e_conv_2_b;
//
//			model.tensors["encoder.ln_post.weight"] = model.e_ln_w;
//			model.tensors["encoder.ln_post.bias"] = model.e_ln_b;
//
//			for (int i = 0; i < n_audio_layer; ++i) {
//				auto & layer = model.layers_encoder[i];
//
//				layer.mlp_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//				layer.mlp_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//
//				layer.mlp_0_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state, 4 * n_audio_state);
//				layer.mlp_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_audio_state);
//
//				layer.mlp_1_w = ggml_new_tensor_2d(ctx, wtype, 4 * n_audio_state, n_audio_state);
//				layer.mlp_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//
//				layer.attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//				layer.attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//
//				layer.attn_q_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state);
//				layer.attn_q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//
//				layer.attn_k_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state);
//
//				layer.attn_v_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state);
//				layer.attn_v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//
//				layer.attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype, n_audio_state, n_audio_state);
//				layer.attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_state);
//
//				// map by name
//				model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.weight"] = layer.mlp_ln_w;
//				model.tensors["encoder.blocks." + std::to_string(i) + ".mlp_ln.bias"] = layer.mlp_ln_b;
//
//				model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.weight"] = layer.mlp_0_w;
//				model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.0.bias"] = layer.mlp_0_b;
//
//				model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.weight"] = layer.mlp_1_w;
//				model.tensors["encoder.blocks." + std::to_string(i) + ".mlp.2.bias"] = layer.mlp_1_b;
//
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.weight"] = layer.attn_ln_0_w;
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn_ln.bias"] = layer.attn_ln_0_b;
//
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.weight"] = layer.attn_q_w;
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn.query.bias"] = layer.attn_q_b;
//
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn.key.weight"] = layer.attn_k_w;
//
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.weight"] = layer.attn_v_w;
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn.value.bias"] = layer.attn_v_b;
//
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.weight"] = layer.attn_ln_1_w;
//				model.tensors["encoder.blocks." + std::to_string(i) + ".attn.out.bias"] = layer.attn_ln_1_b;
//			}
//		}
//
//		// decoder
//		{
//			model.d_pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_text_state, n_text_ctx);
//
//			model.d_te = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_vocab);
//
//			model.d_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//			model.d_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//			// map by name
//			model.tensors["decoder.positional_embedding"] = model.d_pe;
//
//			model.tensors["decoder.token_embedding.weight"] = model.d_te;
//
//			model.tensors["decoder.ln.weight"] = model.d_ln_w;
//			model.tensors["decoder.ln.bias"] = model.d_ln_b;
//
//			for (int i = 0; i < n_text_layer; ++i) {
//				auto & layer = model.layers_decoder[i];
//
//				layer.mlp_ln_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//				layer.mlp_ln_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.mlp_0_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, 4 * n_text_state);
//				layer.mlp_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_text_state);
//
//				layer.mlp_1_w = ggml_new_tensor_2d(ctx, wtype, 4 * n_text_state, n_text_state);
//				layer.mlp_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//				layer.attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.attn_q_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//				layer.attn_q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.attn_k_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//
//				layer.attn_v_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//				layer.attn_v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//				layer.attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.cross_attn_ln_0_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//				layer.cross_attn_ln_0_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.cross_attn_q_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//				layer.cross_attn_q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.cross_attn_k_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//
//				layer.cross_attn_v_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//				layer.cross_attn_v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				layer.cross_attn_ln_1_w = ggml_new_tensor_2d(ctx, wtype, n_text_state, n_text_state);
//				layer.cross_attn_ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_text_state);
//
//				// map by name
//				model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.weight"] = layer.mlp_ln_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".mlp_ln.bias"] = layer.mlp_ln_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.weight"] = layer.mlp_0_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.0.bias"] = layer.mlp_0_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.weight"] = layer.mlp_1_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".mlp.2.bias"] = layer.mlp_1_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.weight"] = layer.attn_ln_0_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn_ln.bias"] = layer.attn_ln_0_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.weight"] = layer.attn_q_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn.query.bias"] = layer.attn_q_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn.key.weight"] = layer.attn_k_w;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.weight"] = layer.attn_v_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn.value.bias"] = layer.attn_v_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.weight"] = layer.attn_ln_1_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".attn.out.bias"] = layer.attn_ln_1_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.weight"] = layer.cross_attn_ln_0_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn_ln.bias"] = layer.cross_attn_ln_0_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.weight"] = layer.cross_attn_q_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.query.bias"] = layer.cross_attn_q_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.key.weight"] = layer.cross_attn_k_w;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.weight"] = layer.cross_attn_v_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.value.bias"] = layer.cross_attn_v_b;
//
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.weight"] = layer.cross_attn_ln_1_w;
//				model.tensors["decoder.blocks." + std::to_string(i) + ".cross_attn.out.bias"] = layer.cross_attn_ln_1_b;
//			}
//		}
//	}
//
//	// load weights
//	{
//		size_t total_size = 0;
//
//		model.n_loaded = 0;
//
//		std::vector<char> read_buf;
//
//		while (true) {
//			int32_t n_dims;
//			int32_t length;
//			int32_t ttype;
//
//			read_safe(loader, n_dims);
//			read_safe(loader, length);
//			read_safe(loader, ttype);
//
//			if (loader->eof(loader->context)) {
//				break;
//			}
//
//			int32_t nelements = 1;
//			int32_t ne[4] = { 1, 1, 1, 1 };
//			for (int i = 0; i < n_dims; ++i) {
//				read_safe(loader, ne[i]);
//				nelements *= ne[i];
//			}
//
//			std::string name;
//			std::vector<char> tmp(length); // create a buffer
//			loader->read(loader->context, &tmp[0], tmp.size()); // read to buffer
//			name.assign(&tmp[0], tmp.size());
//
//			if (model.tensors.find(name) == model.tensors.end()) {
//				WHISPER_LOG_ERROR("%s: unknown tensor '%s' in model file\n", __func__, name.data());
//				return false;
//			}
//
//			auto tensor = model.tensors[name.data()];
//
//			if (ggml_nelements(tensor) != nelements) {
//				WHISPER_LOG_ERROR("%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
//				WHISPER_LOG_ERROR("%s: shape: [%d, %d, %d], expected: [%d, %d, %d]\n",
//					__func__, ne[0], ne[1], ne[2], (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2]);
//				return false;
//			}
//
//			if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1] || tensor->ne[2] != ne[2]) {
//				WHISPER_LOG_ERROR("%s: tensor '%s' has wrong shape in model file: got [%d, %d, %d], expected [%d, %d, %d]\n",
//					__func__, name.data(), (int)tensor->ne[0], (int)tensor->ne[1], (int)tensor->ne[2], ne[0], ne[1], ne[2]);
//				return false;
//			}
//
//			const size_t bpe = ggml_type_size(ggml_type(ttype));
//
//			if ((nelements*bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
//				WHISPER_LOG_ERROR("%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
//					__func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
//				return false;
//			}
//
//			//ggml_backend_t backend = wctx.backend;
//
//			//printf("%s: [%5.5s] %s\n", __func__, ggml_backend_name(backend), name.c_str());
//
//			if (ggml_backend_buffer_is_host(model.buffer)) {
//				// for the CPU and Metal backend, we can read directly into the tensor
//				loader->read(loader->context, tensor->data, ggml_nbytes(tensor));
//				BYTESWAP_TENSOR(tensor);
//			}
//			else {
//				// read into a temporary buffer first, then copy to device memory
//				read_buf.resize(ggml_nbytes(tensor));
//
//				loader->read(loader->context, read_buf.data(), read_buf.size());
//
//				ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
//			}
//
//			//printf("%48s - [%5d, %5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype), ggml_nbytes(tensor)/1e6);
//			total_size += ggml_nbytes(tensor);
//			model.n_loaded++;
//		}
//
//		WHISPER_LOG_INFO("%s: model size    = %7.2f MB\n", __func__, total_size / 1e6);
//
//		if (model.n_loaded == 0) {
//			WHISPER_LOG_WARN("%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
//		}
//		else if (model.n_loaded != (int)model.tensors.size()) {
//			WHISPER_LOG_ERROR("%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
//			return false;
//		}
//	}
//
//	return true;
//}
//// Parameters for the whisper_full() function
//// If you change the order or add new parameters, make sure to update the default values in whisper.cpp:
//// whisper_full_default_params()
//struct whisper_full_params {
//	enum whisper_sampling_strategy strategy;
//
//	int n_threads;
//	int n_max_text_ctx;     // max tokens to use from past text as prompt for the decoder
//	int offset_ms;          // start offset in ms
//	int duration_ms;        // audio duration to process in ms
//
//	bool translate;
//	bool no_context;        // do not use past transcription (if any) as initial prompt for the decoder
//	bool no_timestamps;     // do not generate timestamps
//	bool single_segment;    // force single segment output (useful for streaming)
//	bool print_special;     // print special tokens (e.g. <SOT>, <EOT>, <BEG>, etc.)
//	bool print_progress;    // print progress information
//	bool print_realtime;    // print results from within whisper.cpp (avoid it, use callback instead)
//	bool print_timestamps;  // print timestamps for each text segment when printing realtime
//
//	// [EXPERIMENTAL] token-level timestamps
//	bool  token_timestamps; // enable token-level timestamps
//	float thold_pt;         // timestamp token probability threshold (~0.01)
//	float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
//	int   max_len;          // max segment length in characters
//	bool  split_on_word;    // split on word rather than on token (when used with max_len)
//	int   max_tokens;       // max tokens per segment (0 = no limit)
//
//	// [EXPERIMENTAL] speed-up techniques
//	// note: these can significantly reduce the quality of the output
//	bool speed_up;          // speed-up the audio by 2x using Phase Vocoder
//	bool debug_mode;        // enable debug_mode provides extra info (eg. Dump log_mel)
//	int  audio_ctx;         // overwrite the audio context size (0 = use default)
//
//	// [EXPERIMENTAL] [TDRZ] tinydiarize
//	bool tdrz_enable;       // enable tinydiarize speaker turn detection
//
//	// A regular expression that matches tokens to suppress
//	const char * suppress_regex;
//
//	// tokens to provide to the whisper decoder as initial prompt
//	// these are prepended to any existing text context from a previous call
//	// use whisper_tokenize() to convert text to tokens
//	// maximum of whisper_n_text_ctx()/2 tokens are used (typically 224)
//	const char * initial_prompt;
//	const whisper_token * prompt_tokens;
//	int prompt_n_tokens;
//
//	// for auto-detection, set to nullptr, "" or "auto"
//	const char * language;
//	bool detect_language;
//
//	// common decoding parameters:
//	bool suppress_blank;    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L89
//	bool suppress_non_speech_tokens; // ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253
//
//	float temperature;      // initial decoding temperature, ref: https://ai.stackexchange.com/a/32478
//	float max_initial_ts;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/decoding.py#L97
//	float length_penalty;   // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L267
//
//	// fallback parameters
//	// ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L274-L278
//	float temperature_inc;
//	float entropy_thold;    // similar to OpenAI's "compression_ratio_threshold"
//	float logprob_thold;
//	float no_speech_thold;  // TODO: not implemented
//
//	struct {
//		int best_of;    // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L264
//	} greedy;
//
//	struct {
//		int beam_size;  // ref: https://github.com/openai/whisper/blob/f82bc59f5ea234d4b97fb2860842ed38519f7e65/whisper/transcribe.py#L265
//
//		float patience; // TODO: not implemented, ref: https://arxiv.org/pdf/2204.05424.pdf
//	} beam_search;
//
//	// called for every newly generated text segment
//	whisper_new_segment_callback new_segment_callback;
//	void * new_segment_callback_user_data;
//
//	// called on each progress update
//	whisper_progress_callback progress_callback;
//	void * progress_callback_user_data;
//
//	// called each time before the encoder starts
//	whisper_encoder_begin_callback encoder_begin_callback;
//	void * encoder_begin_callback_user_data;
//
//	// called each time before ggml computation starts
//	ggml_abort_callback abort_callback;
//	void * abort_callback_user_data;
//
//	// called by each decoder to filter obtained logits
//	whisper_logits_filter_callback logits_filter_callback;
//	void * logits_filter_callback_user_data;
//
//	const whisper_grammar_element ** grammar_rules;
//	size_t                           n_grammar_rules;
//	size_t                           i_start_rule;
//	float                            grammar_penalty;
//};
//autox_err_t run_whisper(char* checkpoint_path, char* tokenizer_path, char* wav)
//{
//	autox_err_t err = AUTOX_OK;
//	params = whisper_init_from_file_with_params(checkpoint_path);
//
//
//	if (!params.grammar.empty()) {
//		auto & grammar = params.grammar_parsed;
//		if (is_file_exist(params.grammar.c_str())) {
//			// read grammar from file
//			std::ifstream ifs(params.grammar.c_str());
//			const std::string txt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
//			grammar = grammar_parser::parse(txt.c_str());
//		}
//		else {
//			// read grammar from string
//			grammar = grammar_parser::parse(params.grammar.c_str());
//		}
//
//		// will be empty (default) if there are parse errors
//		if (grammar.rules.empty()) {
//			fprintf(stderr, "error: failed to parse grammar \"%s\"\n", params.grammar.c_str());
//			return 4;
//		}
//		else {
//			fprintf(stderr, "%s: grammar:\n", __func__);
//			grammar_parser::print_grammar(stderr, grammar);
//			fprintf(stderr, "\n");
//		}
//	}
//	if (!whisper_is_multilingual(ctx)) {
//		if (params.language != "en" || params.translate) {
//			params.language = "en";
//			params.translate = false;
//			fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
//		}
//	}
//	if (params.detect_language) {
//		params.language = "auto";
//	}
//
//	// run the inference
//	{
//		whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
//
//		const bool use_grammar = (!params.grammar_parsed.rules.empty() && !params.grammar_rule.empty());
//		wparams.strategy = (params.beam_size > 1 || use_grammar) ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
//
//		wparams.print_realtime = false;
//		wparams.print_progress = params.print_progress;
//		wparams.print_timestamps = !params.no_timestamps;
//		wparams.print_special = params.print_special;
//		wparams.translate = params.translate;
//		wparams.language = params.language.c_str();
//		wparams.detect_language = params.detect_language;
//		wparams.n_threads = params.n_threads;
//		wparams.n_max_text_ctx = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
//		wparams.offset_ms = params.offset_t_ms;
//		wparams.duration_ms = params.duration_ms;
//
//		wparams.token_timestamps = params.output_wts || params.output_jsn_full || params.max_len > 0;
//		wparams.thold_pt = params.word_thold;
//		wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
//		wparams.split_on_word = params.split_on_word;
//		wparams.audio_ctx = params.audio_ctx;
//
//		wparams.debug_mode = params.debug_mode;
//
//		wparams.tdrz_enable = params.tinydiarize; // [TDRZ]
//
//		wparams.suppress_regex = params.suppress_regex.empty() ? nullptr : params.suppress_regex.c_str();
//
//		wparams.initial_prompt = params.prompt.c_str();
//
//		wparams.greedy.best_of = params.best_of;
//		wparams.beam_search.beam_size = params.beam_size;
//
//		wparams.temperature_inc = params.no_fallback ? 0.0f : params.temperature_inc;
//		wparams.temperature = params.temperature;
//
//		wparams.entropy_thold = params.entropy_thold;
//		wparams.logprob_thold = params.logprob_thold;
//
//		wparams.no_timestamps = params.no_timestamps;
//
//		whisper_print_user_data user_data = { &params, &pcmf32s, 0 };
//
//		const auto & grammar_parsed = params.grammar_parsed;
//		auto grammar_rules = grammar_parsed.c_rules();
//
//		if (use_grammar) {
//			if (grammar_parsed.symbol_ids.find(params.grammar_rule) == grammar_parsed.symbol_ids.end()) {
//				fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, params.grammar_rule.c_str());
//			}
//			else {
//				wparams.grammar_rules = grammar_rules.data();
//				wparams.n_grammar_rules = grammar_rules.size();
//				wparams.i_start_rule = grammar_parsed.symbol_ids.at(params.grammar_rule);
//				wparams.grammar_penalty = params.grammar_penalty;
//			}
//		}
//
//		// this callback is called on each new segment
//		if (!wparams.print_realtime) {
//			wparams.new_segment_callback = whisper_print_segment_callback;
//			wparams.new_segment_callback_user_data = &user_data;
//		}
//
//		if (wparams.print_progress) {
//			wparams.progress_callback = whisper_print_progress_callback;
//			wparams.progress_callback_user_data = &user_data;
//		}
//
//		// examples for abort mechanism
//		// in examples below, we do not abort the processing, but we could if the flag is set to true
//
//		// the callback is called before every encoder run - if it returns false, the processing is aborted
//		{
//			static bool is_aborted = false; // NOTE: this should be atomic to avoid data race
//
//			wparams.encoder_begin_callback = [](struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, void * user_data) {
//				bool is_aborted = *(bool*)user_data;
//				return !is_aborted;
//			};
//			wparams.encoder_begin_callback_user_data = &is_aborted;
//		}
//
//		// the callback is called before every computation - if it returns true, the computation is aborted
//		{
//			static bool is_aborted = false; // NOTE: this should be atomic to avoid data race
//
//			wparams.abort_callback = [](void * user_data) {
//				bool is_aborted = *(bool*)user_data;
//				return is_aborted;
//			};
//			wparams.abort_callback_user_data = &is_aborted;
//		}
//
//		if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
//			fprintf(stderr, "%s: failed to process audio\n", argv[0]);
//			return 10;
//		}
//	}
//
//	return err;
//}
//
//
