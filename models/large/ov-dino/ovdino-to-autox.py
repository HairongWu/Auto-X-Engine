# Convert a SAM model checkpoint to a ggml compatible file
#

import sys
import torch
import os
import numpy as np

fname_model = sys.argv[1]
dir_out     = sys.argv[2]
fname_out   = os.path.join(dir_out, "./autox-model.bin")
ftype = 0
ftype_str = ["f32", "f16"]
fname_out = fname_out.replace(".bin", "-" + ftype_str[ftype] + ".bin")

fout = open(fname_out, "wb")
model = torch.load(fname_model, map_location="cpu")
for k, v in model.items():
    for name, vvv in v.items():
        shape = vvv.shape

        print("Processing variable: " + name + " with shape: ", shape)

        #data = tf.train.load_variable(dir_model, name).squeeze()
        #data = v.numpy().squeeze()
        data = vvv.numpy()

        # data
        data.tofile(fout)

fout.close()