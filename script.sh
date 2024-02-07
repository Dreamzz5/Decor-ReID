# The code is builded with DistributedDataParallel. 
# Reprodecing the results in the paper should train the model on 2 GPUs.
# You can also train this model on single GPU and double config.DATA.TRAIN_BATCH in configs.
# For LTCC dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 main.py --dataset ltcc --cfg configs/res50_cels_cal.yaml --gpu 0,1 #
# For PRCC dataset
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12391 main.py  --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 7 --lambda1 0.05 --lambda2 0.67 --ra_gray 0.1 --re_color 0.9 #
