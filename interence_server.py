import signal
import cv2
from flask import Flask, Response, request
import numpy as np

from hf import download_model, download_dataset
from interence import MyModelONNX
from crack import Crack
from data_collector import HINTS_DIR, GUESS_CATEGORY_THRESHOLD
from image_processor import crop_v3_nine_image

import os
import time
import json

MODEL_NAME = "model.onnx"
ENSURE_DURATION = 4.0
TOTAL_CATEGORIES = 90

app = Flask("geetest-v3-nine-pic-crack")
model = None
category_hint_images = []

def guess_category_id(hint_img: cv2.typing.MatLike) -> tuple[int, float]:
    if len(category_hint_images) == 0:
        return (-1, 1000)
    else:
        diffs = [cv2.absdiff(img, hint_img) for img in category_hint_images]
        stds = np.std(diffs, axis=(1,2,3))
        t = np.argmin(stds)
        # np.int64 cannot be serialized
        return (int(t), stds[t])


def solve_challenge(crack: Crack, retry: int):
    raw_bytes, raw_data = crack.get_pic(retry)
    app.logger.debug(f'raw_data={raw_data}')
    assert raw_data['spec'] == '3*3'
    assert raw_data['pic_type'] == 'nine'

    t1 = time.perf_counter()
    r = crop_v3_nine_image(raw_bytes)

    cat_idx, cat_guess_std = guess_category_id(r['hint'])
    app.logger.info(f'cat_idx={cat_idx}, cat_guess_std={cat_guess_std}')
    assert cat_guess_std <= GUESS_CATEGORY_THRESHOLD

    prediction = model.predict(r['options'])
    app.logger.info(f'target={cat_idx}, predictions={prediction}')

    submission = []
    for idx, v in enumerate(prediction):
        if v == cat_idx:
            col = idx % 3
            row = idx // 3
            submission.append(f"{col+1}_{row+1}")
    
    app.logger.info(f"submission={submission}")

    t2 = time.perf_counter()
    if t2 - t1 < ENSURE_DURATION:
        app.logger.info(f"solved in {t2-t1 :.2f}s, too fast")
        time.sleep(ENSURE_DURATION - (t2 - t1))
    return json.loads(crack.verify(submission))


@app.route('/health')
def health():
    assert model is not None
    return "OK"


@app.route('/stop')
def stop():
    os.kill(os.getpid(), signal.SIGINT)


@app.route('/crack_it')
def crack_it():
    gt = request.args.get('gt', '')
    challenge = request.args.get('challenge', '')
    assert len(gt) > 0 and len(challenge) > 0

    app.logger.info(f'gt={gt}, challenge={challenge}')

    crack = Crack(gt, challenge)
    crack.get_type()
    crack.get_c_s()
    crack.ajax()

    for retry in range(6):
        ret = solve_challenge(crack, retry)
        app.logger.info(f"retry={retry}, geetest says: {ret}")
        if ret["data"]["result"] == "success":
            return ret

    return Response(
        f"Could not solve this challenge, ret={ret}",
        status=500,
    )


if __name__ == "__main__":
    model_path = download_model(MODEL_NAME)
    model = MyModelONNX(model_path)

    download_dataset(only_hint_images=True)
    hint_filenames = os.listdir(HINTS_DIR)
    for idx in range(len(hint_filenames)): # Important! keep idx in order
        category_hint_images.append(cv2.imread(f'{HINTS_DIR}/hint_{idx}.jpg'))

    assert len(category_hint_images) == TOTAL_CATEGORIES

    app.run(debug=True, host="127.0.0.1", port=3333)
