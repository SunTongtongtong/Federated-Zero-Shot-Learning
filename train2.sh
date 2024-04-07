MODEL_NAME=$1
# CUDA_VISIBLE_DEVICES=1 python train.py --name ${MODEL_NAME} --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --syn_num 300 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA2 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa --dataroot /import/sgg-homes/ss014/datasets/ZSL/xlsa17/data/ 

# CUDA_VISIBLE_DEVICES=0 python train.py --gzsl --name ${MODEL_NAME} --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataset AWA2 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa --dataroot /import/sgg-homes/ss014/datasets/ZSL/xlsa17/data/ 

# echo 'cub'
# python train.py --name ${MODEL_NAME} --manualSeed 3483 --val_every 1 --cls_weight 0.01 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 --dataset CUB --batch_size 64 --nz 312 --attSize 312 --resSize 2048 --syn_num 300 --outname cub --dataroot /data/DERI-Gong/acw557/datasets/xlsa17/xlsa17/data/ 

 
echo 'sun'
# python train.py --name ${MODEL_NAME} --manualSeed 4115 --cls_weight 0.01 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --classifier_lr 0.0005 --syn_num 400 --outname sun --dataroot /data/DERI-Gong/acw557/datasets/xlsa17/xlsa17/data/ 
python train.py --global_ep 100  --name ${MODEL_NAME} --manualSeed 4115 --cls_weight 0.01 --val_every 1 --preprocessing --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC  --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset SUN --batch_size 64 --nz 102 --attSize 102 --resSize 2048 --lr 0.0002 --syn_num 400 --classifier_lr 0.001 --nclass_all 717 --outname sun  --dataroot /data/DERI-Gong/acw557/datasets/xlsa17/xlsa17/data/ 

 
# echo 'apy'
# python train.py --name ${MODEL_NAME} --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset APY --batch_size 64 --nz 64 --attSize 64 --resSize 2048 --outname apy --dataroot /data/DERI-Gong/acw557/datasets/xlsa17/xlsa17/data/


# echo 'awa2'
# python train.py --global_ep 100 --name ${MODEL_NAME} --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA2 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa --dataroot  /data/DERI-Gong/acw557/datasets/xlsa17/xlsa17/data/

# echo 'awa1'
# python train.py --name ${MODEL_NAME} --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa --dataroot  /data/DERI-Gong/acw557/datasets/xlsa17/xlsa17/data/ 
   
# echo 'apy'
# python train.py --name ${MODEL_NAME} --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --dataset APY --batch_size 64 --nz 64 --attSize 64 --resSize 2048 --outname apy --dataroot /data/DERI-Gong/acw557/datasets/xlsa17/xlsa17/data/
