
# Eliceiri's data

## Train
nohup python train.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_train --batch_size 32 --resize_size 256 --crop_size 256 --name eliceiri_drit_train --concat 0 --n_ep 1200 --gpu 1 > out_eliceiri.file 2>&1 &

## Test
### rotation
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/rotation/ --resize_size 834 --crop_size 834 --a2b 1 --name eliceiri_drit_rotation_a2b --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
python test_transfer.py --dataroot ../../pytorch-CycleGAN-and-pix2pix/datasets/eliceiri_cyc_test/rotation/ --resize_size 834 --crop_size 834 --a2b 0 --name eliceiri_drit_rotation_b2a --concat 0 --resume ../results/eliceiri_drit_train/01199.pth
