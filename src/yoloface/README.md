# YOLOv7 Face Detection

## Data preparation

- Download and extract [winderface](http://shuoyang1213.me/WIDERFACE/index.html) dataset.
- Download and extract annotation file [retinaface_gt_v1.1.zip](https://github.com/deepinsight/insightface/tree/master/detection/retinaface).
- Download [ground_truth](https://github.com/deepcam-cn/yolov5-face/tree/master/widerface_evaluate/ground_truth).
- Folder after download and extract all:
  ```
  - ./winderface
      - WIDER_test/
          - images/
              - 0--Parade/
              - ...
      - WIDER_train/
          - images/
              - 0--Parade/
              - ...
      - WIDER_val/
          - images/
              - 0--Parade/
              - ...
      - train/
          - labels.txt
      - val/
          - labels.txt
      - test/
          - labels.txt
      - ground_truth/
          - wider_easy_val.mat
          - wider_medium_val.mat
          - wider_hard_val.mat
          - wider_face_val.mat
  ```
- Convert label to yolo format:
  - `python3 utils/convert_to_yolo.py --root ./widerface --image-folder WIDER_train/images --label-file train/label.txt --txt-file train.txt`
  - `python3 utils/convert_to_yolo.py --root ./widerface --image-folder WIDER_val/images --label-file val/label.txt --txt-file val.txt`
- Modify path of `winderface` folder in [data/winderface.yaml](data/winderface.yaml)

## Detect

- `python3 detect.py --weights ./weights/yolov7-tiny.pt --source inference/images --img-size 640 --conf-thres 0.2 --iou-thres 0.5 --device 1 --no-trace`

## Testing

- `python3 eval.py --weights ./weights/yolov7-tiny.pt --data-root ./winderface --img-size 640 --conf-thres 0.02 --iou-thres 0.5 --device 0 --no-trace`

## Training

- Single GPU training: `python3 ./train.py --device 0 --batch-size 16 --data data/widerface.yaml --img 640 640 --cfg cfg/yolov7-tiny-landmark.yaml --weights ./weights/yolov7-tiny-origin.pt --name yolov7-tiny --hyp data/hyp.scratch.tiny.yaml --noautoanchor --linear-lr --epochs 80`

- Multiple GPU training: `torchrun --standalone --nnodes=1 --nproc_per_node 2 ./train.py --device 0,1 --batch-size 16 --data data/widerface.yaml --img 640 640 --cfg cfg/yolov7-tiny-landmark.yaml --weights ./weights/yolov7-tiny-origin.pt --name yolov7-tiny --hyp data/hyp.scratch.tiny.yaml --noautoanchor --sync-bn --linear-lr --epochs 80`

## Export

### ONNX

```bash
python3 export.py --weights ./weights/yolov7-tiny.pt --img-size 640 --batch-size 1 --dynamic-batch --grid --end2end --max-wh 640 --topk-all 100 --iou-thres 0.5 --conf-thres 0.2 --device 0 --simplify --cleanup
```

### TensorRT
 
```bash
/usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-face-tiny.onnx --saveEngine=./weights/yolov7-face-tiny.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:4x3x640x640 --shapes=images:1x3x640x640
```

## Acknowledgments

- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- [Hiennguyen9874](https://github.com/hiennguyen9874/yolov7-face-detection/tree/using-landmark)
