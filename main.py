# Regular python libraries
from utils_cv.similarity.model import compute_features, compute_feature, compute_features_learner
from utils_cv.similarity.metrics import compute_distances, vector_distance
from utils_cv.similarity.data import Urls
from utils_cv.common.gpu import which_processor, db_num_workers
from utils_cv.common.data import unzip_url
import math
import os
import random
import sys
import torch
import numpy as np
from pathlib import Path
from read_config import read_py_config
import torch.nn as nn
from IPython.core.debugger import set_trace
from PIL import ImageOps, Image
import cv2
from typing import List

from fastapi import FastAPI, File, UploadFile
from starlette.responses import HTMLResponse
from pdf2image import convert_from_bytes

# Fast.ai
import fastai
from fastai.layers import FlattenedLoss
from fastai.vision import (
    cnn_learner,
    DatasetType,
    ImageList,
    imagenet_stats,
    models,
    open_image,
    load_learner
)

# Computer Vision repository
sys.path.extend([".", "../.."])  # to access the utils_cv library


class EmbeddedFeatureWrapper(nn.Module):
    """
    DNN head: pools, down-projects, and normalizes DNN features to be of unit length.
    """

    def __init__(self, input_dim, output_dim, dropout=0):
        super(EmbeddedFeatureWrapper, self).__init__()
        self.output_dim = output_dim
        self.dropout = dropout
        if output_dim != 4096:
            self.pool = nn.AdaptiveAvgPool2d(1)
        self.standardize = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.remap = None
        if input_dim != output_dim:
            self.remap = nn.Linear(input_dim, output_dim, bias=False)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.output_dim != 4096:
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.standardize(x)
        if self.remap:
            x = self.remap(x)
        if self.dropout > 0:
            x = self.dropout(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class L2NormalizedLinearLayer(nn.Module):
    """
    Apply a linear layer to the input, where the weights are normalized to be of unit length.
    """

    def __init__(self, input_dim, output_dim):
        super(L2NormalizedLinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)

    def forward(self, x):
        norm_weight = nn.functional.normalize(self.weight, dim=1)
        prediction_logits = nn.functional.linear(x, norm_weight)
        return prediction_logits


class NormSoftmaxLoss(nn.Module):
    """
    Apply temperature scaling on logits before computing the cross-entropy loss.
    """

    def __init__(self, temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, prediction_logits, instance_targets):
        loss = self.loss_fn(prediction_logits /
                            self.temperature, instance_targets)
        return loss


app = FastAPI()
config = read_py_config('config.py')

similarity_learn = load_learner(
    config.sig_config.model_path, config.sig_config.similarity_model_name)
classify_learn = load_learner(
    config.sig_config.model_path, config.sig_config.classify_model_name)
embedding_layer = similarity_learn.model[1][-2]


def extract_signature(file, device):
    header_offset = device and 3000 or 2200
    footer_offset = device and 3151 or 2305
    pages = convert_from_bytes(file)
    for page in pages:
        h_o = device and (page.width/2)+171 or (page.width/2)+550
        f_o = device and (page.width/2)+171+707 or (page.width/2)+550+370
        page = page.crop((h_o, header_offset, f_o, footer_offset))
        return page


def center_image(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

    # desired_size = img.shape[:2][::-1]
    desired_size = [706, 151]

    # new_size should be in (width, height) format

    retval, thresh_gray = cv2.threshold(
        gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    # find where the signature is and make a cropped region
    points = np.argwhere(thresh_gray == 0)  # find where the black pixels are
    # store them in x,y coordinates instead of row,col indices
    points = np.fliplr(points)
    # create a rectangle around those points
    x, y, w, h = cv2.boundingRect(points)
    crop = gray[y:y+h, x:x+w]  # create a cropped region of the gray image
    try:
        retval, thresh_crop = cv2.threshold(
            crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

        # old_size is in (height, width) format
        old_size = thresh_crop.shape[:2]

        old_image = Image.fromarray(thresh_crop)
        deltaw = desired_size[0]-old_size[1]
        deltah = desired_size[1]-old_size[0]
        ltrb_border = (deltaw//2, deltah//2, deltaw -
                       (deltaw//2), deltah-(deltah//2))
        new_im = ImageOps.expand(old_image, border=ltrb_border, fill='white')
        return new_img
    except:
        new_im = Image.fromarray(thresh_gray)
        return new_img


@app.post("/compare_signature/")
async def signature_compute(app: UploadFile = File(...), device: UploadFile = File(...)):

    app_img = center_image(extract_signature(app, False))
    device_img = center_image(extract_signature(device, True))

    signature_class_1 = classify_learn.predict(app_img)[1]
    signature_class_2 = classify_learn.predict(device_img)[1]

    if(config.sig_config.classify_response[signature_class_1] != config.sig_config.success_case
       and config.sig_config.classify_response[signature_class_2] != config.sig_config.success_case):
        return {"status": config.sig_config.classify_response[signature_class_1] if config.sig_config.classify_response[signature_class_1] != config.sig_config.success_case else config.sig_config.classify_response[signature_class_2]}

    app_emb = compute_feature(app_img, similarity_learn, embedding_layer)
    device_emb = compute_feature(device_img, similarity_learn, embedding_layer)

    similarity_score = vector_distance(app_emb, device_emb)

    return {"status": config.sig_config.success_case if similarity_score < config.sig_config.threshold else config.sig_config.fail_case, "similarity": similarity_score}


@app.post("/check_signature/")
async def signature_compute(file: UploadFile = File(...)):
    device_img = open_image(file.file)

    signature_class = classify_learn.predict(device_img)[1]

    return {"status": config.sig_config.classify_response[signature_class]}


@app.get("/")
async def main():
    content = """
<body>
<form action="/signature/" enctype="multipart/form-data" method="post">
<input name="app" type="file" multiple>
<input name="device" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
