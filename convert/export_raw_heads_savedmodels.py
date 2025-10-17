# convert/export_raw_heads_savedmodel.py
import os, sys, pathlib, tensorflow as tf

# --- TODO EDIT these three paths ---
MODELS_ROOT = r"C:\Users\yxl21\Documents\School\DroneRS\models"  # TF Models repo root
PIPELINE_CFG = r"C:\Users\yxl21\Documents\School\DroneRS\tflite-aideck\person_ssd.config"
CKPT_DIR     = r"C:\Users\yxl21\Documents\School\DroneRS\tflite-aideck\training\person_ssd"
OUT_DIR      = r"C:\Users\yxl21\Documents\School\DroneRS\tflite-aideck\export\export_raw_heads_savedmodel"

# Put OD API on sys.path
sys.path.insert(0, str(pathlib.Path(MODELS_ROOT, "research")))
sys.path.insert(0, str(pathlib.Path(MODELS_ROOT, "research", "slim")))

from object_detection.utils import config_util
from object_detection.builders import model_builder

def build_model_from_config(pipeline_cfg_path):
    cfg = config_util.get_configs_from_pipeline_file(pipeline_cfg_path)
    model = model_builder.build(model_config=cfg['model'], is_training=False)
    return model

class RawHeadsModule(tf.Module):
    def __init__(self, detection_model):
        super().__init__()
        self.model = detection_model

    @tf.function(input_signature=[tf.TensorSpec([1,320,320,3], tf.uint8, name="input_tensor")])
    def __call__(self, x_uint8):
        # Cast once; OD API preprocess handles normalization
        x = tf.cast(x_uint8, tf.float32)
        images, shapes = self.model.preprocess(x)
        preds = self.model.predict(images, shapes)

        # Try standard SSD keys
        boxes  = preds.get('box_encodings', None)                         # [1, N, 4]
        logits = preds.get('class_predictions_with_background', None)     # [1, N, num_classes+1]

        # Fallback alt names (print keys if still missing)
        if boxes is None or logits is None:
            tf.print("Available prediction keys:", list(preds.keys()))
            boxes  = boxes  if boxes  is not None else preds.get('raw_box_predictions', None)
            logits = logits if logits is not None else preds.get('raw_cls_predictions', None)
        tf.debugging.assert_type(x_uint8, tf.uint8)
        if boxes is None or logits is None:
            raise ValueError("Could not find raw heads in preds. See printed keys above.")

        # Return float32 raw heads; TFLite will quantize them with your rep dataset
        return {
            "raw_box_encodings": tf.identity(boxes,  name="raw_box_encodings"),
            "raw_class_logits":  tf.identity(logits, name="raw_class_logits")
        }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    detection_model = build_model_from_config(PIPELINE_CFG)
    ckpt = tf.train.Checkpoint(model=detection_model)
    latest = tf.train.latest_checkpoint(CKPT_DIR)
    if latest is None:
        raise SystemExit(f"No checkpoint found in {CKPT_DIR}")
    ckpt.restore(latest).expect_partial()
    print("Restored:", latest)

    module = RawHeadsModule(detection_model)
    tf.saved_model.save(
        module,
        OUT_DIR,
        signatures={"serving_default": module.__call__}
    )
    print("Saved:", OUT_DIR)

if __name__ == "__main__":
    main()
