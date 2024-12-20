import httpx
import cv2
from PIL import Image 
import numpy as np
import io
import os
import json
from typing import Tuple

from crack import Crack
import image_processor

HINTS_DIR = './dataset/hints'
ANNOTED_DIR = './dataset/annoted'
RAW_DIR = './dataset/raw'
UNKNOWN_DIR = './dataset/unknown'
RAW_IMAGE_METADATA_PATH = './dataset/raw_images.json'

GUESS_CATEGORY_THRESHOLD = 10
FREEZE_CATEGORY_NUM = 90
SHOW_GEETEST_RESULT = True

class DataSetManager:
    def __init__(self,
                 raw_dir: str = RAW_DIR, annoted_dir: str = ANNOTED_DIR,
                 hints_dir: str = HINTS_DIR, unknown_dir: str = UNKNOWN_DIR):
        self.raw_dir = raw_dir
        self.annoted_dir = annoted_dir
        self.hints_dir = hints_dir
        self.unknown_dir = unknown_dir
        self.check_all_dirs()

        self.raw_images_url_seen = set()
        self.category_hint_imgs = []
        self.raw_image_metadatas = []
    
    def check_all_dirs(self):
        assert os.path.exists(self.raw_dir)
        assert os.path.exists(self.annoted_dir)
        assert os.path.exists(self.hints_dir)
        assert os.path.exists(self.unknown_dir)

    def guess_category_id(self, hint_img: cv2.typing.MatLike) -> Tuple[int, float]:
        if len(self.category_hint_imgs) == 0:
            return (-1, 1000)
        else:
            diffs = [cv2.absdiff(img, hint_img) for img in self.category_hint_imgs]
            stds = np.std(diffs, axis=(1,2,3))
            t = np.argmin(stds)
            # np.int64 cannot be serialized
            return (int(t), stds[t])

    def add_raw_image(self, raw_bytes: bytes, raw_data: dict) -> int:
        assert raw_data['spec'] == '3*3'
        assert raw_data['pic_type'] == 'nine'

        if raw_data['pic'] in self.raw_images_url_seen:
            return -1
        else:
            self.raw_images_url_seen.add(raw_data['pic'])

        idx = len(self.raw_image_metadatas)
        raw_decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f'{RAW_DIR}/{idx}.jpg', raw_decoded)

        self.raw_image_metadatas.append({
            'path': f'{RAW_DIR}/{idx}.jpg',
            'category_id': None,
            'annoted_dir': None,
            # 'img': raw_decoded,
            'geetest_metadata': raw_data
        })
        # r = image_processor.crop_v3_nine_image(raw_bytes)
        # return idx, r['hint'], r['options']
        return idx

    def assign_category(self, raw_idx: int, category_id: int, hint_img: cv2.typing.MatLike):
        assert raw_idx < len(self.raw_image_metadatas)
        assert category_id < len(self.category_hint_imgs)

        if category_id == -1:
            category_id = len(self.category_hint_imgs)
            os.makedirs(f'{ANNOTED_DIR}/{category_id}', exist_ok=True)
            cv2.imwrite(f'{HINTS_DIR}/hint_{category_id}.jpg', hint_img)
            self.category_hint_imgs.append(hint_img)

        self.raw_image_metadatas[raw_idx]['category_id'] = category_id
        self.raw_image_metadatas[raw_idx]['annoted_dir'] = f'{ANNOTED_DIR}/{category_id}'

    def add_annoted_image(self,
                          raw_idx: int,
                          cat_idx: int,
                          hint_img: cv2.typing.MatLike, option_imgs: list[cv2.typing.MatLike],
                          select: list[bool]):
        assert raw_idx < len(self.raw_image_metadatas)
        assert cat_idx < len(self.category_hint_imgs)

        cv2.imwrite(f'{ANNOTED_DIR}/{cat_idx}/hint_{cat_idx}_{raw_idx}.jpg', hint_img)

        for idx, v in enumerate(option_imgs):
            if select[idx]:
                cv2.imwrite(f'{ANNOTED_DIR}/{cat_idx}/option_{cat_idx}_{raw_idx}_{idx}.jpg', v)
            else:
                cv2.imwrite(f'{UNKNOWN_DIR}/option_unk_{raw_idx}_{idx}.jpg', v)
    
    def add_fail_image(self,
                       raw_idx: int,
                       cat_idx: int,
                       hint_img: cv2.typing.MatLike, option_imgs: list[cv2.typing.MatLike]):
        cv2.imwrite(f'{UNKNOWN_DIR}/hint_fail_{cat_idx}_{raw_idx}.jpg', hint_img)
        for idx, v in enumerate(option_imgs):
            cv2.imwrite(f'{UNKNOWN_DIR}/option_fail_{cat_idx}_unk_{raw_idx}_{idx}.jpg', v)

    def load_progress(self, raw_img_json_path: str = RAW_IMAGE_METADATA_PATH):
        if not os.path.exists(raw_img_json_path):
            return

        with open(raw_img_json_path, 'r') as f:
            self.raw_image_metadatas = json.load(f)

        if FREEZE_CATEGORY_NUM:
            assert len(os.listdir(f'{ANNOTED_DIR}/')) == FREEZE_CATEGORY_NUM
            assert len(os.listdir(f'{HINTS_DIR}/')) == FREEZE_CATEGORY_NUM

        for r in self.raw_image_metadatas:
            self.raw_images_url_seen.add(r['geetest_metadata']['pic'])

        hint_filenames = os.listdir(HINTS_DIR)
        for idx in range(len(hint_filenames)): # Important! keep idx in order
            self.category_hint_imgs.append(cv2.imread(f'{HINTS_DIR}/hint_{idx}.jpg'))

        # print("Progressed loaded.", "Categories:", len(self.category_stats), "Raw images:", len(self.raw_image_metadatas))

    def flush(self):
        with open(RAW_IMAGE_METADATA_PATH, 'w') as fp:
            json.dump(self.raw_image_metadatas, fp)


def new_crack() -> Crack:
    reg = httpx.get(
        "https://bbs-api.miyoushe.com/misc/api/createVerification?is_high=true"
    ).json()

    reg = reg["data"]
    crack = Crack(reg["gt"], reg["challenge"])
    return crack


def fetch_challenge_data(crack: Crack):
    crack_type = crack.get_type()

    crack.get_c_s()
    crack.ajax()

    with open("current_challenge.jpg", 'wb') as f:
        pic_content, data = crack.get_pic()
        f.write(pic_content)
        assert data['spec'] == '3*3'
        assert data['pic_type'] == 'nine'
    
    return pic_content, data


def collect_once(ds: DataSetManager) -> bool:
    crack = new_crack()
    raw_bytes, raw_data = fetch_challenge_data(crack)
    r = image_processor.crop_v3_nine_image(raw_bytes)

    cat_idx = -1
    guess_result = ds.guess_category_id(r['hint'])
    if guess_result[1] <= GUESS_CATEGORY_THRESHOLD:
        cat_idx = guess_result[0]
    else:
        cat_idx = -1

    if cat_idx == -1:
        print("Possibly new category detected.")
        assert FREEZE_CATEGORY_NUM == 0
    
    raw_idx = ds.add_raw_image(raw_bytes, raw_data)
    if raw_idx == -1:
        print(raw_data['pic'], "seen previously.")
        return False

    sel, cat_idx, stop = image_processor.annote_v3_nine_raw_image(raw_bytes, guessed_category_id=cat_idx)

    if SHOW_GEETEST_RESULT:
        submission = []
        for v, idx in enumerate(sel):
            if v:
                col = idx % 3
                row = idx // 3
                submission.append(f"{col+1}_{row+1}")

        geetest_result = json.loads(crack.verify(submission))
        print("geetest says:", geetest_result)

    ds.assign_category(raw_idx, cat_idx, r['hint'])
    ds.add_annoted_image(raw_idx, cat_idx, r['hint'], r['options'], sel)
    ds.flush()

    return stop


if __name__ == '__main__':
    os.makedirs(f'{ANNOTED_DIR}', exist_ok=True)
    os.makedirs(f'{RAW_DIR}', exist_ok=True)
    os.makedirs(f'{HINTS_DIR}', exist_ok=True)
    os.makedirs(f'{UNKNOWN_DIR}', exist_ok=True)

    ds = DataSetManager()
    ds.load_progress()
    stop = False
    while not stop:
        stop = collect_once(ds)
