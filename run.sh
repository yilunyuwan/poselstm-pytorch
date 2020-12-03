if [ ! -f "../datasets/7Scenes/$1/mean_image.png" ]; then
    python util/compute_image_mean.py --dataroot ../datasets/7Scenes/$1 --height 256 --width 455 --save_resized_imgs
fi
python train.py --model posenet --dataroot ../datasets/7Scenes/$1 --name posenet/$1/beta500 --beta 500 --gpu $2
python test.py --model posenet  --dataroot ../datasets/7Scenes/$1 --name posenet/$1/beta500 --gpu $2