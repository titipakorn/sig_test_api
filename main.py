# Regular python libraries
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
from torch.nn import Module
from torch import Tensor

from fastapi import FastAPI, File, UploadFile, Body
from starlette.responses import HTMLResponse, StreamingResponse
from pdf2image import convert_from_bytes
import time

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
    load_learner,
    image,
    pil2tensor
)

import io


class SaveFeatures:
    """Hook to save the features in the intermediate layers

    Source: https://forums.fast.ai/t/how-to-find-similar-images-based-on-final-embedding-layer/16903/13

    Args:
        model_layer (nn.Module): Model layer

    """

    features = None

    def __init__(self, model_layer: Module):
        self.hook = model_layer.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module: Module, input: Tensor, output: Tensor):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

    def remove(self):
        self.hook.remove()


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
print('MY CONFIG', config)
similarity_learn = load_learner(
    config['sig_config']['model_path'], config['sig_config']['similarity_model_name'])
classify_learn = load_learner(
    config['sig_config']['model_path'], config['sig_config']['classify_model_name'])
embedding_layer = similarity_learn.model[1][-2]
featurizer = SaveFeatures(embedding_layer)


def extract_signature(file, device):
    start_time = time.time()
    offset = device and config['sig_config']['device'] or config['sig_config']['app']
    pages = convert_from_bytes(file.read())
    for page in pages:
        page = page.crop((offset['start_width'], offset['start_height'],
                          offset['end_width'], offset['end_height']))
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
    crop = gray[y:y + h, x:x + w]  # create a cropped region of the gray image
    try:
        retval, thresh_crop = cv2.threshold(
            crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

        # old_size is in (height, width) format
        old_size = thresh_crop.shape[:2]

        old_image = Image.fromarray(thresh_crop)
        deltaw = desired_size[0] - old_size[1]
        deltah = desired_size[1] - old_size[0]
        ltrb_border = (deltaw // 2, deltah // 2, deltaw -
                       (deltaw // 2), deltah - (deltah // 2))
        new_img = ImageOps.expand(old_image, border=ltrb_border, fill='white')
        return new_img
    except:
        new_img = Image.fromarray(thresh_gray)
        return new_img


def compare_sig(app_img, device_img):
    start_time = time.time()
    signature_class_1 = None
    signature_class_2 = None
    if app_img:
        signature_class_1 = classify_learn.predict(app_img)[1]
        signature_class_1 = str(signature_class_1.item())
    if device_img:
        signature_class_2 = classify_learn.predict(device_img)[1]
        signature_class_2 = str(signature_class_2.item())
    result = ""
    similarity_score = -1
    if(app_img and device_img):
        if(config['sig_config']['similarity_response'][signature_class_1] != config['sig_config']['success_case']
           and config['sig_config']['similarity_response'][signature_class_2] != config['sig_config']['success_case']):
            result = config['sig_config']['similarity_response'][signature_class_1] if config['sig_config']['similarity_response'][
                signature_class_1] != config['sig_config']['success_case'] else config['sig_config']['similarity_response'][signature_class_2]
        if(result == ""):
            featurizer.features = None
            similarity_learn.predict(app_img)
            app_emb = featurizer.features[0][:]
            assert len(app_emb) > 1
            featurizer.features = None
            similarity_learn.predict(device_img)
            device_emb = featurizer.features[0][:]
            assert len(device_emb) > 1
            featurizer.features = None
            similarity_score = vector_distance(app_emb, device_emb)
            if similarity_score <= config['sig_config']['accept_threshold']:
                result = config['sig_config']['success_case']
            elif similarity_score >= config['sig_config']['deny_threshold']:
                result = config['sig_config']['fail_case']
            else:
                result = config['sig_config']['unknown_case']
    elif app_img or device_img:
        result = app_img and config['sig_config']['similarity_response'][
            signature_class_1] or config['sig_config']['similarity_response'][signature_class_2]
    else:
        return {"status": "error", 'description': "no files"}
    print("compare signature  --- %s seconds ---" % (time.time() - start_time))
    return {"status": result, "similarity": str(similarity_score), "app_class": signature_class_1 and config['sig_config']['classify_response'][signature_class_1] or "", "device_class": signature_class_2 and config['sig_config']['classify_response'][signature_class_2] or ""}


@app.post("/compare_signature/")
async def signature_compute(app: UploadFile = File(...), device: UploadFile = File(...)):
    try:
        app_img = open_image(app.file)
    except:
        app_img = None
    try:
        device_img = open_image(device.file)
    except:
        device_img = None
    result = compare_sig(app_img, device_img)
    app.file.close()
    device.file.close()
    return result


@app.post("/compare_signature_pdf/")
async def signature_compute(app: UploadFile = File(...), device: UploadFile = File(...)):
    try:
        app_img = center_image(extract_signature(app.file, False))
        cv2img = np.array(app_img)
        res, im_png = cv2.imencode(".jpg", cv2img)
        app_img = open_image(io.BytesIO(im_png.tobytes()))
    except:
        app_img = None
    try:
        device_img = center_image(extract_signature(device.file, True))
        cv2img = np.array(device_img)
        res, im_png = cv2.imencode(".jpg", cv2img)
        device_img = open_image(io.BytesIO(im_png.tobytes()))
    except:
        device_img = None
    result = compare_sig(app_img, device_img)
    app.file.close()
    device.file.close()
    return result


@app.post("/check_signature/")
async def signature_compute(file: UploadFile = File(...)):
    start_time = time.time()
    device_img = open_image(file.file)
    classify_result = classify_learn.predict(device_img)
    signature_class = classify_result[1]
    print("classify signature  --- %s seconds ---" %
          (time.time() - start_time))
    file.file.close()

    return {"status": config['sig_config']['classify_response'][str(signature_class.item())]}


@app.post("/extract_signature")
def image_endpoint(device: int = Body(...), file: UploadFile = File(...)):
    # Returns a cv2 image array from the document vector
    start_time = time.time()
    if(device > 0):
        device = True
    else:
        device = False
    ex_sig = extract_signature(file.file, device)
    file.file.close()
    vector = center_image(ex_sig)
    cv2img = np.array(vector)
    res, im_png = cv2.imencode(".jpg", cv2img)
    print("extract signature  --- %s seconds ---" % (time.time() - start_time))
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/jpg")


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
