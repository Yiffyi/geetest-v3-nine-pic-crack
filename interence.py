import onnxruntime
import numpy as np
import cv2


class MyModelONNX:
    def __init__(self, model_path: str):
        self.model = onnxruntime.InferenceSession(model_path)
        assert len(self.model.get_inputs()) == 1
        self.model_inputs = self.model.get_inputs()
        assert len(self.model_inputs) == 1

    @staticmethod
    def data_transforms(image: cv2.typing.MatLike) -> np.ndarray:
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_array = np.array(image)
        image_array = image_array.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        image_array = np.transpose(image_array, (2, 0, 1))
        # image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def predict(self, options_imgs: list[cv2.typing.MatLike]):
        options_imgs = [self.data_transforms(x) for x in options_imgs]
        outputs = self.model.run(None, {self.model_inputs[0].name: options_imgs})[0]
        return np.argmax(outputs, axis=1)
