if [ ! -f "../datasets/7Scenes/$1/mean_image_depth.npy" ]; then
    python util/compute_image_depth_mean.py --dataroot ../datasets/7Scenes/$1 --height 256 --width 341 --save_resized_imgs
fi
python train.py --model posenet --dataset_mode rgbd_posenet --input_nc 4 --dataroot ../datasets/7Scenes/$1 --name posenet/$1/rgbdbeta500 --beta 500 --gpu $2
python test.py --model posenet --dataset_mode rgbd_posenet --input_nc 4 --dataroot ../datasets/7Scenes/$1 --name posenet/$1/rgbdbeta500 --gpu $2 