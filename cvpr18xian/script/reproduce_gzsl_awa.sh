CUDA_VISIBLE_DEVICES=0 python clswgan.py --gzsl --manualSeed 9182 --cls_weight 0.01 --preprocessing --val_every 1 --lr 0.00001 --cuda --image_embedding res101 --class_embedding att --netG_name MLP_G --netD_name MLP_CRITIC --nepoch 30 --syn_num 1800 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 --nclass_all 50 --dataset AWA1 --batch_size 64 --nz 85 --attSize 85 --resSize 2048 --outname awa 
