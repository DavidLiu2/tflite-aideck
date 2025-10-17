import tensorflow as tf
i = tf.lite.Interpreter(model_path="export/person_ssd_int8.tflite")
i.allocate_tensors()
print("INPUT:", i.get_input_details())
for k, t in enumerate(i.get_output_details()):
    print(f"OUT{k}:", {x: t[x] for x in ("name","shape","dtype","quantization")})