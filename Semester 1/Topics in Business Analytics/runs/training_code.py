##### First Run
%cd yolov5
# This time: --device 0 (using GPU instead of CPU)
!python train.py --img 640 --batch 8 --epochs 50 --data "C:/Users/10/OneDrive - University of Exeter/Exeter University/Units/Topics in Business Analytics/dataset/data.yaml" --weights yolov5s.pt --device 0 --name first_run
# Result: Success

##### Second Run
%cd yolov5
# 10x more epochs to increase performance, --noplots to reduce the volume of logs sent to the notebook
!python train.py --img 640 --batch 8 --epochs 500 --data "C:/Users/10/OneDrive - University of Exeter/Exeter University/Units/Topics in Business Analytics/dataset/data.yaml" --weights "C:/Users/10/OneDrive - University of Exeter/yolov5/runs/train/first_run/weights/best.pt" --device 0 --name second_run --noplots
# Result: Fail, stopped at 100 epochs due to yolov5's patience (by default, the script automatically stops if performance stops improving for several epochs)

##### Third Run
%cd yolov5
# This time: --weights yolov5l.pt (large model) on 50 epochs and reduced batch size (6)
!python train.py --img 640 --batch 6 --epochs 50 --data "C:/Users/10/OneDrive - University of Exeter/Exeter University/Units/Topics in Business Analytics/dataset/data.yaml" --weights yolov5l.pt --device 0 --name third_run 
# Result: Success, time needed = approx. 6h, results not as good as expected, but achieved the highest mAP_0.5:0.95 value of 0.41816 (see descriptives)


