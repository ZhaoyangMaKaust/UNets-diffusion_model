#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o Augresultlosssecondmoment.%J.out
#SBATCH -e Augresultlosssecondmoment.%J.err
#SBATCH --time=48:02:01
#SBATCH --mem=180G
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu

#run the application:
module load machine_learning
#pip install blobfile
#ifconfig eth0
#nvidia-smi
echo "http://"`hostname -i`":9000"
python -m visdom.server -p 9000&
python sample.py --out_dir /ibex/user/maz0a/DiffusionAugmentation/diffAug/Augresultlosssecondmoment/ --num_samples 10000 --model_path "/ibex/user/maz0a/DiffusionAugmentation/diffAug/ckptslosssecondmoment/emasavedmodel_0.9999_060000.pt" --dev 0,1,2,3
#python train_loss_second_moment.py --data_dir /ibex/user/maz0a/DiffusionAugmentation/diffAug/Data/MoNuSeg/Trainset/ --lr_anneal_steps 60000 --batch_size 8 --resume_checkpoint "/ibex/user/maz0a/DiffusionAugmentation/diffAug/ckptslosssecondmoment/savedmodel045000.pt" --multi_gpu 0,1,2,3
# python train.py --data_dir /ibex/user/maz0a/DiffusionAugmentation/diffAug/Data/MoNuSeg/Trainset/ --lr_anneal_steps 60000 --batch_size 2

