# PY3-Faster-RCNN-TF

This repo is a Python3 implementation of [Faster-RCNN_TF by smallcorgi](https://github.com/smallcorgi/Faster-RCNN_TF) which implements 

[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

The full repository has been made compaitable for python3, and if you want to train a network you can follow the following instructions. In case, if you want to do the same using python2, then you can fork or download the following repo.
https://github.com/dxyang/Faster-RCNN-COCO_TF





### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))

2. Python packages you might not have: `pillow` `scipy` `cython`, `opencv3`, `easydict`

This repo has been tested using python3.5, tensorflow1.4, cudNN-6.

### Installation




1. Clone the PY3-Faster-RCNN-TF repository
	```bash
	# Make sure to clone with --recursive
	git clone --recursive 
	```
	
2. Build the Cython modules
	```bash
	cd $PY3-Faster-RCNN-TF/lib
	make
	```	

3. To run the demo 
   Download the COCO trained pretrained model from https://drive.google.com/file/d/0Bw0qMqgwZcafZlRqRDYxSnBkNFE/view
   rename VGGnet_fast_rcnn_iter_490000.ckpt.data-00000-of-00001 as VGGnet_fast_rcnn_iter_490000.ckpt
   
   Run the following command.
 ```bash
  python ./tools/demo.py --model "path to the VGGnet_fast_rcnn_iter_490000.ckpt"  --net VGGnet_test  --img-path path_to_img_folder
  ```	
  The results will be saved in a directory called ```detections_test```
   
 
Follow the instructions to train your own model.
1. Build pycocotools modules
	```bash
	cd $PY3-Faster-RCNN-TF/lib
	git clone https://github.com/cocodataset/cocoapi.git
	cd cocoapi/PythonAPI
	make
	cd ../..
	mv cocoapi/PythonAPI/pycocotools pycocotools
	rm -rf cocoapi
	```

2. Build the Cython modules
	```bash
	cd $PY3-Faster-RCNN-TF/lib
	make
	```


### Training Model
3. Install gsutil if you haven't already
	```bash
	curl https://sdk.cloud.google.com | bash
	```

4. Download the training, validation, test data for MS COCO
	```bash
	cd $PY3-Faster-RCNN-TF/data
	mkdir coco; cd coco
	mkdir images; cd images
	mkdir train2014
	mkdir test2014
	mkdir val2014
	gsutil -m rsync gs://images.cocodataset.org/train2014 train2014
	gsutil -m rsync gs://images.cocodataset.org/test2014 test2014
	gsutil -m rsync gs://images.cocodataset.org/val2014 val2014
	```

5. Download the annotations for MS COCO and unzip
	```bash
	cd $PY3-Faster-RCNN-TF/data
	gsutil -m rsync gs://images.cocodataset.org/annotations coco
	cd coco
	unzip annotations_trainval2014.zip
	unzip image_info_test2014.zip
	rm *.zip
	```

6. Download the annotations for the 5000 image minival subset of COCO val2014 as mentioned [here](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data)
	```bash
	cd $PY3-Faster-RCNN-TF/data/coco/annotations
	wget https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip
	wget https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip
	unzip instances_minival2014.json.zip; rm instances_minival2014.json.zip
	unzip instances_valminusminival2014.json.zip; rm instances_valminusminival2014.json.zip
	```

7. Download the pre-trained ImageNet model [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)
	```bash
	cd $PY3-Faster-RCNN-TF
	wget https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy
	mkdir data/pretrain_model
	mv VGG_imagenet.npy data/pretrain_model/VGG_imagenet.npy
	```

8. Create an output directory for log files
	```bash
	cd $PY3-Faster-RCNN-TF
	mkdir experiments/logs
	```

9. Run script to train and test model
	```bash
	cd $PY3-Faster-RCNN-TF
	./experiments/scripts/faster_rcnn_end2end.sh $DEVICE $DEVICE_ID VGG16 coco
	```
  - DEVICE is either cpu/gpu

### Testing Model
Run the following command.

```bash
python ./tools/test_net.py \
		--device gpu \
		--device_id 0 \
		--weights output/faster_rcnn_end2end/coco_2014_train/VGGnet_fast_rcnn_iter_490000.ckpt \
		--cfg experiments/cfgs/faster_rcnn_end2end.yml \
		--imdb coco_2014_minival \
		--network VGGnet_test	\
		--vis False
```

- Changing ```vis``` to ```True``` will save images with all detections above 0.8 for every image in the testing set.
- The checkpoint files folder contains the following:
	```bash
	cd output/faster_rcnn_end2end/coco_2014_train
	ls
	# VGGnet_fast_rcnn_iter_490000.ckpt.data-00000-of-00001
	# VGGnet_fast_rcnn_iter_490000.ckpt.index
	# VGGnet_fast_rcnn_iter_490000.ckpt.meta
	```





 
