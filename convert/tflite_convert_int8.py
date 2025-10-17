import tensorflow as tf
from rep_dataset import rep_data_uint8_rgb

SAVED_MODEL_DIR = "export/export_raw_heads_savedmodel"
OUT_PATH        = "export/person_ssd_int8.tflite"

loaded = tf.saved_model.load(SAVED_MODEL_DIR)
infer  = loaded.signatures['serving_default']

# Introspect the real input name/dtype
in_key, in_spec = list(infer.structured_input_signature[1].items())[0]
INPUT_SHAPE = [1, 320, 320, 3]
INPUT_DTYPE = tf.uint8
target_dtype = in_spec.dtype  # likely tf.uint8

# Get the output keys so we return exactly the two raw heads
out_keys = list(infer.structured_outputs.keys())
# Expect something like: ["raw_box_encodings", "raw_class_logits"]
# If different, replace the two names below accordingly.
BOX_KEY = "raw_box_encodings"
CLS_KEY = "raw_class_logits"

@tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, INPUT_DTYPE, name=in_key)])
def wrapped(x):
    outs = infer(**{in_key: tf.cast(x, target_dtype)})
    # Return ONLY the two raw heads (no extras)
    return {BOX_KEY: outs[BOX_KEY], CLS_KEY: outs[CLS_KEY]}

conc = wrapped.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([conc], loaded)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data_uint8_rgb
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.int8
converter._experimental_lower_tensor_list_ops = True
converter.experimental_enable_resource_variables = True
converter._experimental_disable_resource_variables = False
converter.experimental_new_quantizer = True  # ← keep this

open(OUT_PATH, "wb").write(converter.convert())
print("Wrote", OUT_PATH)