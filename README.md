1. install dependencies
download python 3.10 and set up venv
$ pip install -r requirements.txt

2. acquire image dataset
download wget and unzip (https://gist.github.com/evanwill/0207876c3243bbb6863e65ec5dc3f058)
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

6. change the following in person_ssd.config to point to the correct file paths
fine_tune_checkpoint: TODO
train_input_reader
    label_map_path: TODO
    input_path: TODO
eval_input_reader
    label_map_path: TODO
    input_path: TODO

5. set up tf models repo
(in a different project folder)
$ git clone https://github.com/tensorflow/models.git
$ cd models/research
$ protoc object_detection/protos/*.proto --python_out=.
$ cd object_detection/packages/tf2
$ pip install -e .

6. train the model
$ cd models/research
$ python object_detection/model_main_tf2.py   --pipeline_config_path=YOUR_PATH/tflite-aideck/person_ssd.config   --model_dir=YOUR_PATH/tflite-aideck/training/person_ssd