# Convert a SAM model checkpoint to a ggml compatible file
#

import sys
import torch
import struct
import numpy as np

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

class GGMLModel:

    file_format: GGMLFormat
    format_version: int

    def __init__(self):
        self.hyperparameters = None
        self.vocab = None
        self.tensor_map = {}
        self.tensors = []

    def validate_header(self, data, offset):
        magic = bytes(data[offset:offset + 4])
        if magic == b'GGUF':
            raise ValueError('File is already in GGUF format.')
        if magic == b'lmgg':
            self.file_format = GGMLFormat.GGML
            self.format_version = 1
            return 4
        version = struct.unpack('<I', data[offset + 4:offset + 8])[0]
        if magic == b'fmgg':
            if version != 1:
                raise ValueError(f'Cannot handle unexpected GGMF file version {version}')
            self.file_format = GGMLFormat.GGMF
            self.format_version = version
            return 8
        if magic == b'tjgg':
            if version < 1 or version > 3:
                raise ValueError(f'Cannot handle unexpected GGJT file version {version}')
            self.file_format = GGMLFormat.GGJT
            self.format_version = version
            return 8
        raise ValueError(f"Unexpected file magic {magic!r}! This doesn't look like a GGML format file.")

    def validate_conversion(self, ftype):
        err = ''
        if (self.file_format < GGMLFormat.GGJT or self.format_version < 2):
            if ftype not in (GGMLFType.ALL_F32, GGMLFType.MOSTLY_F16):
                err = 'Quantizations changed in GGJTv2. Can only convert unquantized GGML files older than GGJTv2.'
        elif (self.file_format == GGMLFormat.GGJT and self.format_version == 2):
            if ftype in (GGMLFType.MOSTLY_Q4_0, GGMLFType.MOSTLY_Q4_1,
                         GGMLFType.MOSTLY_Q4_1_SOME_F16, GGMLFType.MOSTLY_Q8_0):
                err = 'Q4 and Q8 quantizations changed in GGJTv3.'
        if len(err) > 0:
            raise ValueError(f'{err} Sorry, your {self.file_format.name}v{self.format_version} file of type {ftype.name} is not eligible for conversion.')

    def load(self, data, offset):
        offset += self.validate_header(data, offset)
        hp = Hyperparameters()
        offset += hp.load(data, offset)
        logger.info(f'* File format: {self.file_format.name}v{self.format_version} with ftype {hp.ftype.name}')
        self.validate_conversion(hp.ftype)
        vocab = Vocab(load_scores = self.file_format > GGMLFormat.GGML)
        offset += vocab.load(data, offset, hp.n_vocab)
        tensors: list[Tensor] = []
        tensor_map = {}
        while offset < len(data):
            tensor = Tensor(use_padding = self.file_format > GGMLFormat.GGMF)
            offset += tensor.load(data, offset)
            tensor_map[tensor.name] = len(tensors)
            tensors.append(tensor)
        self.hyperparameters = hp
        self.vocab = vocab
        self.tensors = tensors
        self.tensor_map = tensor_map
        hp.set_n_ff(self)
        return offset


class GGMLToGGUF:
    def __init__(self, ggml_model, data):
        hp = ggml_model.hyperparameters
        self.model = ggml_model
        self.data = data

        if params_override is not None:
            n_kv_head = params_override.n_head_kv
        else:
            if cfg.gqa == 1:
                n_kv_head = hp.n_head
            else:
                gqa = float(cfg.gqa)
                n_kv_head = None
                for x in range(1, 256):
                    if float(hp.n_head) / float(x) == gqa:
                        n_kv_head = x
                assert n_kv_head is not None, "Couldn't determine n_kv_head from GQA param"
                logger.info(f'- Guessed n_kv_head = {n_kv_head} based on GQA {cfg.gqa}')
        self.n_kv_head = n_kv_head
        self.name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, ggml_model.hyperparameters.n_layer)

    def save(self):
        logger.info('* Preparing to save GGUF file')
        gguf_writer = gguf.GGUFWriter(
            self.cfg.output,
            gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.LLAMA],
            use_temp_file = False)
        self.add_params(gguf_writer)
        self.add_vocab(gguf_writer)
        if self.special_vocab is not None:
            self.special_vocab.add_to_gguf(gguf_writer)
        self.add_tensors(gguf_writer)
        logger.info("    gguf: write header")
        gguf_writer.write_header_to_file()
        logger.info("    gguf: write metadata")
        gguf_writer.write_kv_data_to_file()
        logger.info("    gguf: write tensors")
        gguf_writer.write_tensors_to_file()
        gguf_writer.close()

    def add_params(self, gguf_writer):
        hp = self.model.hyperparameters
        cfg = self.cfg
        if cfg.desc is not None:
            desc = cfg.desc
        else:
            desc = f'converted from legacy {self.model.file_format.name}v{self.model.format_version} {hp.ftype.name} format'
        try:
            # Filenames aren't necessarily valid UTF8.
            name = cfg.name if cfg.name is not None else cfg.input.name
        except UnicodeDecodeError:
            name = None
        logger.info('* Adding model parameters and KV items')
        if name is not None:
            gguf_writer.add_name(name)
        gguf_writer.add_description(desc)
        gguf_writer.add_file_type(int(hp.ftype))
        if self.params_override is not None:
            po = self.params_override
            assert po.n_embd == hp.n_embd, 'Model hyperparams mismatch'
            assert po.n_layer == hp.n_layer, 'Model hyperparams mismatch'
            assert po.n_head == hp.n_head, 'Model hyperparams mismatch'
            gguf_writer.add_context_length      (po.n_ctx)
            gguf_writer.add_embedding_length    (po.n_embd)
            gguf_writer.add_block_count         (po.n_layer)
            gguf_writer.add_feed_forward_length (po.n_ff)
            gguf_writer.add_rope_dimension_count(po.n_embd // po.n_head)
            gguf_writer.add_head_count          (po.n_head)
            gguf_writer.add_head_count_kv       (po.n_head_kv)
            gguf_writer.add_layer_norm_rms_eps  (po.f_norm_eps)
            return
        gguf_writer.add_context_length(cfg.context_length)
        gguf_writer.add_embedding_length(hp.n_embd)
        gguf_writer.add_block_count(hp.n_layer)
        gguf_writer.add_feed_forward_length(hp.n_ff)
        gguf_writer.add_rope_dimension_count(hp.n_embd // hp.n_head)
        gguf_writer.add_head_count(hp.n_head)
        gguf_writer.add_head_count_kv(self.n_kv_head)
        gguf_writer.add_layer_norm_rms_eps(float(cfg.eps))

    def add_vocab(self, gguf_writer):
        hp = self.model.hyperparameters
        gguf_writer.add_tokenizer_model('llama')
        gguf_writer.add_tokenizer_pre('default')
        tokens = []
        scores = []
        toktypes = []
        if self.vocab_override is not None:
            vo = self.vocab_override
            logger.info('* Adding vocab item(s)')
            for (_, (vbytes, score, ttype)) in enumerate(vo.all_tokens()):
                tokens.append(vbytes)
                scores.append(score)
                toktypes.append(ttype)
            assert len(tokens) == hp.n_vocab, \
                f'Override vocab has a different number of items than hyperparameters - override = {len(tokens)} but n_vocab={hp.n_vocab}'
            gguf_writer.add_token_list(tokens)
            gguf_writer.add_token_scores(scores)
            if len(toktypes) > 0:
                gguf_writer.add_token_types(toktypes)
            return
        logger.info(f'* Adding {hp.n_vocab} vocab item(s)')
        assert len(self.model.vocab.items) >= 3, 'Cannot handle unexpectedly short model vocab'
        for (tokid, (vbytes, vscore)) in enumerate(self.model.vocab.items):
            tt = 1 # Normal
            # Special handling for UNK, BOS, EOS tokens.
            if tokid <= 2:
                if tokid == 0:
                    vbytes = b'<unk>'
                    tt = 2
                elif tokid == 1:
                    vbytes = b'<s>'
                    tt = 3
                else:
                    vbytes = b'</s>'
                    tt = 3
            elif len(vbytes) == 0:
                tt = 3 # Control
            elif tokid >= 3 and tokid <= 258 and len(vbytes) == 1:
                vbytes = bytes(f'<0x{vbytes[0]:02X}>', encoding = 'UTF-8')
                tt = 6 # Byte
            else:
                vbytes = vbytes.replace(b' ', b'\xe2\x96\x81')
            toktypes.append(tt)
            tokens.append(vbytes)
            scores.append(vscore)
        gguf_writer.add_token_list(tokens)
        gguf_writer.add_token_scores(scores)
        gguf_writer.add_token_types(toktypes)
        gguf_writer.add_unk_token_id(0)
        gguf_writer.add_bos_token_id(1)
        gguf_writer.add_eos_token_id(2)

    def add_tensors(self, gguf_writer):
        tensor_map = self.name_map
        data = self.data
        logger.info(f'* Adding {len(self.model.tensors)} tensor(s)')
        for tensor in self.model.tensors:
            name = str(tensor.name, 'UTF-8')
            mapped_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
            assert mapped_name is not None, f'Bad name {name}'
            tempdims = list(tensor.dims[:])
            if len(tempdims) > 1:
                temp = tempdims[1]
                tempdims[1] = tempdims[0]
                tempdims[0] = temp
            gguf_writer.add_tensor(
                mapped_name,
                data[tensor.start_offset:tensor.start_offset + tensor.len_bytes],
                raw_shape = tempdims,
                raw_dtype = tensor.dtype)

if len(sys.argv) < 3:
    print("Usage: convert-pth-to-ggml.py file-model dir-output [ftype]\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
fname_model = sys.argv[1]
dir_out     = sys.argv[2]
fname_out   = dir_out + "/ggml-model.bin"

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 3:
    ftype = int(sys.argv[3])

if ftype < 0 or ftype > 1:
    print("Invalid ftype: " + str(ftype))
    sys.exit(1)

fname_out = fname_out.replace(".bin", "-" + ftype_str[ftype] + ".bin")

# Default params are set to sam_vit_b checkpoint
n_enc_state = 768
n_enc_layers = 12
n_enc_heads = 12
n_enc_out_chans = 256
n_pt_embd = 4

model = torch.load(fname_model, map_location="cpu")
for k, v in model.items():
    print(k, v.shape)
    if k == "image_encoder.blocks.0.norm1.weight":
        n_enc_state = v.shape[0]

if n_enc_state == 1024: # sam_vit_l
    n_enc_layers = 24
    n_enc_heads  = 16
elif n_enc_state == 1280: # sam_vit_h
    n_enc_layers = 32
    n_enc_heads  = 16

hparams = {
    "n_enc_state":      n_enc_state,
    "n_enc_layers":     n_enc_layers,
    "n_enc_heads":      n_enc_heads,
    "n_enc_out_chans":  n_enc_out_chans,
    "n_pt_embd":        n_pt_embd,
}

print(hparams)

for k, v in model.items():
    print(k, v.shape)

#exit()
#code.interact(local=locals())

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["n_enc_state"]))
fout.write(struct.pack("i", hparams["n_enc_layers"]))
fout.write(struct.pack("i", hparams["n_enc_heads"]))
fout.write(struct.pack("i", hparams["n_enc_out_chans"]))
fout.write(struct.pack("i", hparams["n_pt_embd"]))
fout.write(struct.pack("i", ftype))

for k, v in model.items():
    name = k
    shape = v.shape

    if name[:19] == "prompt_encoder.mask":
        continue

    print("Processing variable: " + name + " with shape: ", shape, " and type: ", v.dtype)

    #data = tf.train.load_variable(dir_model, name).squeeze()
    #data = v.numpy().squeeze()
    data = v.numpy()
    n_dims = len(data.shape)

    # for efficiency - transpose some matrices
    # "model/h.*/attn/c_attn/w"
    # "model/h.*/attn/c_proj/w"
    # "model/h.*/mlp/c_fc/w"
    # "model/h.*/mlp/c_proj/w"
    #if name[-14:] == "/attn/c_attn/w" or \
    #   name[-14:] == "/attn/c_proj/w" or \
    #   name[-11:] == "/mlp/c_fc/w" or \
    #   name[-13:] == "/mlp/c_proj/w":
    #    print("  Transposing")
    #    data = data.transpose()

    dshape = data.shape

    # default type is fp16
    ftype_cur = 1
    if ftype == 0 or n_dims == 1 or \
            name == "image_encoder.pos_embed" or \
            name.startswith("prompt_encoder") or \
            name.startswith("mask_decoder.iou_token") or \
            name.startswith("mask_decoder.mask_tokens"):
        print("  Converting to float32")
        data = data.astype(np.float32)
        ftype_cur = 0
    else:
        print("  Converting to float16")
        data = data.astype(np.float16)

    # reshape the 1D bias into a 4D tensor so we can use ggml_repeat
    # keep it in F32 since the data is small
    if name == "image_encoder.patch_embed.proj.bias":
        data = data.reshape(1, data.shape[0], 1, 1)
        n_dims = len(data.shape)
        dshape = data.shape

    print("  New shape: ", dshape)

    # header
    str = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", dshape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")

data = np.memmap(fname_out, mode = 'r')
model = GGMLModel()
logger.info('* Scanning GGML input file')
offset = model.load(data, 0)  # noqa
logger.info(f'* GGML model hyperparameters: {model.hyperparameters}')

converter = GGMLToGGUF(
    model, data
)
converter.save()
logger.info(f'* Successful completion. Output saved to: {fname_out}')