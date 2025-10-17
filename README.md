1. install dependencies
download python 3.10 and set up venv
$ pip install -r requirements.txt

2. acquire image dataset
download wget and unzip (highly recommend using git bash as your terminal https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058)
$ cd data
$ source coco.sh

3. filter data
$ python coco_person_filter.py data/coco/annotations/instances_train2017.json data/coco/annotations/train_person.json
$ python coco_person_filter.py data/coco/annotations/instances_val2017.json   data/coco/annotations/val_person.json

4. create tf records
$  python create_coco_tf_record.py \
    --images_dir data/coco/images/train2017 \
    --annotations_json data/coco/annotations/train_person.json \
    --label_map_out annotations/label_map.pbtxt \
    --output_path annotations/train.record \
    --include_masks False \
    --filter_categories person \
    --num_shards 1 \
    --drop_crowd True \
    --min_area 0
$  python create_coco_tf_record.py \
    --images_dir data/coco/images/val2017 \
    --annotations_json data/coco/annotations/val_person.json \
    --label_map_out annotations/label_map.pbtxt \
    --output_path annotations/val.record \
    --include_masks False \
    --filter_categories person \
    --num_shards 1 \
    --drop_crowd True \
    --min_area 0

6. change the following in person_ssd.config to point to the correct file paths (absolute path)
fine_tune_checkpoint: TODO.../tflite-aideck/models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0
train_input_reader
    label_map_path: TODO.../tflite-aideck/annotations/label_map.pbtxt
    input_path: TODO.../tflite-aideck/annotations/train.record
eval_input_reader
    label_map_path: TODO.../tflite-aideck/annotations/label_map.pbtxt
    input_path: TODO.../tflite-aideck/annotations/val.record
**note: only use / or \\ for file paths

5. set up tf models repo
(in a different project folder)
$ git clone https://github.com/tensorflow/models.git
$ cd models/research
do [$ sudo apt install -y protobuf-compiler] if using a linux system, otherwise:
    1. Go to: https://github.com/protocolbuffers/protobuf/releases/tag/v3.20.2
    2. Download the protoc-3.20.2-XXXXX.zip (for your operating system).
    3. Unzip the .exe file to your C:\Users\name\AppData\Local\Programs\Git\mingw64 directory (refer to step 2 tutorial)
$ protoc object_detection/protos/*.proto --python_out=.
$ cd object_detection/packages/tf2
$ pip install --no-deps -e .
(ignore the error message for tf-models-official)

6. train the model
$ pip install --no-deps "tf-models-official==2.13.2"
$ pip install --no-deps "gin-config==0.5.0" "clu==0.0.12" "ml-collections==0.1.1" "etils==1.6.0"
$ cd models/research
$ python -m object_detection.model_main_tf2   --pipeline_config_path=YOUR_PATH/tflite-aideck/person_ssd.config   --model_dir=YOUR_PATH/tflite-aideck/training/person_ssd

7. use the custom script to export the model (modify the filepaths)
in your tflite-aideck repo
$ python convert/export_raw_heads_savedmodels.py

8. create representative dataset to quantize model
run from tflite-aideck dir
$ python convert/coco_to_rep_persons.py \
  --ann_json data/coco/annotations/train_person.json \
  --images_dir data/coco/images/train2017 \
  --out_dir data/rep_images \
  --num_full 400 \
  --crops_per_image 2 \
  --crop_pad 0.15 \
  --seed 7
$ python convert/rep_dataset.py

9. quantize model to export/ folder
$ python convert/tflite_convert_int8.py
$ python convert/verify_tflite_outputs.py
make sure that the only tensors that are dtype float are outputs (no float weights or activations)
ex. OUT0: {'name': 'StatefulPartitionedCall:1', 'shape': array([    1, 12804,     2]), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}

10. use the NNTool