# pix2pix

## For Zurich data
### Train
#!./scripts/train_pix2pix.sh
nohup python train.py --display_id -1 --dataroot ./datasets/zurichP2P --name zurich_p2p_a2b --model pix2pix --direction AtoB --output_nc 1 --n_epochs 10 --n_epochs_decay 10 --serial_batches --load_size 256 --crop_size 256 --batch_size 16 --gpu_ids 2 > out_a2b.file 2>&1 &
nohup python train.py --display_id -1 --dataroot ./datasets/zurichP2P --name zurich_p2p_b2a --model pix2pix --direction BtoA --output_nc 3 --n_epochs 10 --n_epochs_decay 10 --serial_batches --load_size 256 --crop_size 256 --batch_size 16 --gpu_ids 1 > out_b2a.file 2>&1 &

### test patches
#!./scripts/test_pix2pix.sh
mkdir checkpoints/zurich_p2p_rotated_a2b/
cp checkpoints/zurich_p2p_a2b/latest_net_* checkpoints/zurich_p2p_rotated_a2b/
python test.py --dataroot ./datasets/zurich_rotated --name zurich_p2p_rotated_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1

mkdir checkpoints/zurich_p2p_standard_a2b/
cp checkpoints/zurich_p2p_a2b/latest_net_* checkpoints/zurich_p2p_standard_a2b/
python test.py --dataroot ./datasets/zurich_standard --name zurich_p2p_standard_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2

mkdir checkpoints/zurich_p2p_rotated_b2a/
cp checkpoints/zurich_p2p_b2a/latest_net_* checkpoints/zurich_p2p_rotated_b2a/
python test.py --dataroot ./datasets/zurich_rotated --name zurich_p2p_rotated_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1

mkdir checkpoints/zurich_p2p_standard_b2a/
cp checkpoints/zurich_p2p_b2a/latest_net_* checkpoints/zurich_p2p_standard_b2a/
python test.py --dataroot ./datasets/zurich_standard --name zurich_p2p_standard_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2


### test whole image
mkdir checkpoints/zurich_p2p_whole_a2b/
cp checkpoints/zurich_p2p_a2b/latest_net_* checkpoints/zurich_p2p_whole_a2b/
python test.py --dataroot ./datasets/zurich_whole --name zurich_p2p_whole_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2

mkdir checkpoints/zurich_p2p_whole_b2a/
cp checkpoints/zurich_p2p_b2a/latest_net_* checkpoints/zurich_p2p_whole_b2a/
python test.py --dataroot ./datasets/zurich_whole --name zurich_p2p_whole_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1
#python test.py --dataroot ./datasets/zurichP2P --name zurich_pix2pix --model pix2pix --direction AtoB --output_nc 1 --batch_size 1 --gpu_ids 2 --preprocess none



## For Eliceiri's data 
### prepare training data
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_temp/A --fold_B ../Datasets/Eliceiri_temp/B --fold_AB ./datasets/eliceiri_train

### Train
nohup python train.py --display_id -1 --dataroot ./datasets/eliceiri_train --name eliceiri_p2p_train_a2b --model pix2pix --direction AtoB --output_nc 3 --n_epochs 100 --n_epochs_decay 100 --serial_batches --load_size 256 --crop_size 256 --batch_size 16 --gpu_ids 1 > out_a2b.file 2>&1 &
nohup python train.py --display_id -1 --dataroot ./datasets/eliceiri_train --name eliceiri_p2p_train_b2a --model pix2pix --direction BtoA --output_nc 1 --n_epochs 100 --n_epochs_decay 100 --serial_batches --load_size 256 --crop_size 256 --batch_size 16 --gpu_ids 1 > out_b2a.file 2>&1 &

### prepare test data
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_test/Eliceiri_Both/A --fold_B ../Datasets/Eliceiri_test/Eliceiri_Both/B --fold_AB ./datasets/eliceiri_test/both
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_test/Eliceiri_RotationOnly/A --fold_B ../Datasets/Eliceiri_test/Eliceiri_RotationOnly/B --fold_AB ./datasets/eliceiri_test/rotation
python datasets/combine_A_and_B.py --fold_A ../Datasets/Eliceiri_test/Eliceiri_TranslationOnly/A --fold_B ../Datasets/Eliceiri_test/Eliceiri_TranslationOnly/B --fold_AB ./datasets/eliceiri_test/translation

### Test
#### rotation
mkdir checkpoints/eliceiri_p2p_rotation_a2b
cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_rotation_a2b
python test.py --dataroot ./datasets/eliceiri_test/rotation --name eliceiri_p2p_rotation_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2
mkdir checkpoints/eliceiri_p2p_rotation_b2a
cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_rotation_b2a
python test.py --dataroot ./datasets/eliceiri_test/rotation --name eliceiri_p2p_rotation_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1

#### translation
mkdir checkpoints/eliceiri_p2p_translation_a2b
cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_translation_a2b
python test.py --dataroot ./datasets/eliceiri_test/translation --name eliceiri_p2p_translation_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2
mkdir checkpoints/eliceiri_p2p_translation_b2a
cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_translation_b2a
python test.py --dataroot ./datasets/eliceiri_test/translation --name eliceiri_p2p_translation_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1

#### both
mkdir checkpoints/eliceiri_p2p_both_a2b
cp checkpoints/eliceiri_p2p_train_a2b/latest_net_* checkpoints/eliceiri_p2p_both_a2b
python test.py --dataroot ./datasets/eliceiri_test/both --name eliceiri_p2p_both_a2b --model pix2pix --num_test 99999 --direction AtoB --output_nc 3 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 2
mkdir checkpoints/eliceiri_p2p_both_b2a
cp checkpoints/eliceiri_p2p_train_b2a/latest_net_* checkpoints/eliceiri_p2p_both_b2a
python test.py --dataroot ./datasets/eliceiri_test/both --name eliceiri_p2p_both_b2a --model pix2pix --num_test 99999 --direction BtoA --output_nc 1 --batch_size 16 --preprocess pad --divisor 256 --gpu_ids 1




# cycleGAN

## Eliceiri's data

### prepare training data
mkdir datasets/eliceiri_cyc_train
cp -r ../Datasets/Eliceiri_temp/A/train/ ./datasets/eliceiri_cyc_train/trainA
cp -r ../Datasets/Eliceiri_temp/B/train/ ./datasets/eliceiri_cyc_train/trainB

### Train
nohup python train.py --display_id -1 --dataroot ./datasets/eliceiri_cyc_train --name eliceiri_cyc_train --model cycle_gan --serial_batches --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 2 > out.file 2>&1 &

### prepare test data
mkdir datasets/eliceiri_cyc_test
mkdir datasets/eliceiri_cyc_test/both
mkdir datasets/eliceiri_cyc_test/rotation
mkdir datasets/eliceiri_cyc_test/translation
cp -r ../Datasets/Eliceiri_test/Eliceiri_Both/A/test/ ./datasets/eliceiri_cyc_test/both/testA
cp -r ../Datasets/Eliceiri_test/Eliceiri_Both/B/test/ ./datasets/eliceiri_cyc_test/both/testB

cp -r ../Datasets/Eliceiri_test/Eliceiri_RotationOnly/A/test/ ./datasets/eliceiri_cyc_test/rotation/testA

cp -r ../Datasets/Eliceiri_test/Eliceiri_RotationOnly/B/test/ ./datasets/eliceiri_cyc_test/rotation/testB

cp -r ../Datasets/Eliceiri_test/Eliceiri_TranslationOnly/A/test/ ./datasets/eliceiri_cyc_test/translation/testA

cp -r ../Datasets/Eliceiri_test/Eliceiri_TranslationOnly/B/test/ ./datasets/eliceiri_cyc_test/translation/testB

### Test
#### rotation
mkdir checkpoints/eliceiri_cyc_rotation
cp checkpoints/eliceiri_cyc_train/latest_net_* checkpoints/eliceiri_cyc_rotation
python test.py --dataroot ./datasets/eliceiri_cyc_test/rotation/ --name eliceiri_cyc_rotation --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --gpu_ids 2


##### re-train and re-test with weighted loss function
nohup python train.py --display_id -1 --dataroot ./datasets/eliceiri_cyc_train --name eliceiri_cyc_train_weighted --model cycle_gan --serial_batches --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 2 > out_weighted.file 2>&1 &

mkdir checkpoints/eliceiri_cyc_weighted_rotation
cp checkpoints/eliceiri_cyc_train_weighted/latest_net_* checkpoints/eliceiri_cyc_weighted_rotation
python test.py --dataroot ./datasets/eliceiri_cyc_test/rotation/ --name eliceiri_cyc_weighted_rotation --model cycle_gan --num_test 99999 --batch_size 1 --preprocess pad --divisor 256 --gpu_ids 2
