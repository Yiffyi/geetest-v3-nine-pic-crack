import json
import time

from data_collector import new_crack, fetch_challenge_data, DataSetManager
from data_collector import GUESS_CATEGORY_THRESHOLD, FREEZE_CATEGORY_NUM, SHOW_GEETEST_RESULT
from inference import MyModelONNX
import image_processor

MODEL_PATH = './model/model_e10.onnx'
# 关闭人工介入
# 设为 False 则在提交答案前可以人工修改模型推理的结果
AUTO_COLLECT = True
AUTO_COLLECT_STOP_AT_VALID_COUNT = 100

ENSURE_DURATION = 4.0

# 设为 True，则会将通过验证的数据保存到数据集中，以便再次训练
UPDATE_DATASET = False


class ValidationStat:
    def __init__(self):
        self.total_count = 0
        self.total_dup = 0
        self.correct_count = 0
        self.correct_count_by_category = [0] * FREEZE_CATEGORY_NUM

    @property
    def total_valid(self) -> int:
        return self.total_count - self.total_dup

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.total_valid

    def get_accuracy(self, category_id: int = None) -> float:
        if category_id is not None:
            return self.correct_count_by_category[category_id] / self.total_valid
        else:
            return self.correct_count / self.total_valid


def collect_once(ds: DataSetManager, model: MyModelONNX, stat: ValidationStat) -> bool:
    crack = new_crack()
    raw_bytes, raw_data = fetch_challenge_data(crack)

    stat.total_count += 1
    t1 = time.perf_counter()

    r = image_processor.crop_v3_nine_image(raw_bytes)

    cat_idx = -1
    guess_result = ds.guess_category_id(r['hint'])
    if guess_result[1] <= GUESS_CATEGORY_THRESHOLD:
        cat_idx = guess_result[0]
    else:
        cat_idx = -1

    # I Wanna New:
    if cat_idx == -1:
        print("Warning: possibly new category detected")
        assert FREEZE_CATEGORY_NUM == 0
        # return False
    
    if UPDATE_DATASET:
        raw_idx = ds.add_raw_image(raw_bytes, raw_data)
        if raw_idx == -1:
            print(raw_data['pic'], "seen previously.")
            stat.total_dup += 1
            return False

    prediction = model.predict(r['options'])
    print(f'target={cat_idx}, predictions={prediction}')

    if AUTO_COLLECT:
        sel = list(prediction == cat_idx)
        stop = False
    else:
        sel, cat_idx, stop = image_processor.annote_v3_nine_raw_image(raw_bytes, guessed_category_id=cat_idx, default_selection=list(prediction == cat_idx))

    if SHOW_GEETEST_RESULT:
        submission = []
        for idx, v in enumerate(sel):
            if v:
                col = idx % 3
                row = idx // 3
                submission.append(f"{col+1}_{row+1}")
        print(f"selected={submission}")
        t2 = time.perf_counter()
        if t2 - t1 < ENSURE_DURATION:
            print(f"solved in {t2-t1 :.2f}s, too fast")
            time.sleep(ENSURE_DURATION - (t2 - t1))
            # correct = False
        geetest_result = json.loads(crack.verify(submission))
        if geetest_result['data']['result'] == 'success':
            stat.correct_count += 1
            stat.correct_count_by_category[cat_idx] += 1
        print("geetest says:", geetest_result)
    
    if UPDATE_DATASET:
        ds.assign_category(raw_idx, cat_idx, r['hint'])
        if correct:
            ds.add_annoted_image(raw_idx, cat_idx, r['hint'], r['options'], sel)
        else:
            ds.add_fail_image(raw_idx, cat_idx, r['hint'], r['options'])
        ds.flush()
    
    return stop


if __name__ == '__main__':
    ds = DataSetManager()
    ds.check_all_dirs()
    ds.load_progress()
    model = MyModelONNX(MODEL_PATH)
    stat = ValidationStat()
    stop = False

    total_challenges = 0
    total_correct = 0
    correct = False
    try:
        while not stop:
            total_challenges += 1
            stop = collect_once(ds, model, stat)
            total_correct += correct

            if AUTO_COLLECT and stat.total_valid >= AUTO_COLLECT_STOP_AT_VALID_COUNT:
                stop = True
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except Exception as e:
        print(e)
    
    print(f"total={stat.total_count}, valid={stat.total_valid}, accuracy={stat.accuracy*100 : .2f}%")