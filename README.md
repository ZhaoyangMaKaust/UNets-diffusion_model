# UNets-diffusion_model


In digital rock physics, analysing microstructures from CT and SEM scans is crucial for estimating properties like porosity and pore connectivity. Traditional segmentation methods like thresholding and CNNs often fall short in accurately detailing rock microstructures and are prone to noise. U-Net improved segmentation accuracy but required many expert-annotated samples, a laborious and error-prone process due to complex pore shapes. Our study employed an advanced generative AI model, the diffusion model, to overcome these limitations. This model generated a vast dataset of CT/SEM and binary segmentation pairs from a small initial dataset. We assessed the efficacy of three neural networks: U-Net, Attention-U-net, and TransUNet, for segmenting these enhanced images. The diffusion model was proved to be an effective data augmentation technique, improving the generalization and robustness of deep learning models. TransU-Net, incorporating Transformer structures, demonstrated superior segmentation accuracy and IoU metrics, outperforming both U-Net and Attention-U-net. Our research advances rock image segmentation by combining the diffusion model with cutting-edge neural networks, reducing dependency on extensive expert data and boosting segmentation accuracy and robustness. TransU-Net sets a new standard in digital rock physics, paving the way for future geoscience and engineering breakthroughs.

![image](https://github.com/ZhaoyangMaKaust/UNets-diffusion_model/assets/112864738/52ef1807-b83f-4062-8c52-7b72ad495ea2)
![image](https://github.com/ZhaoyangMaKaust/UNets-diffusion_model/assets/112864738/986c2a6b-76a8-4061-9470-e4f4d1ba4f96)

For data augumentation using diffusion model:

Required package:
pytorch
torchvision
numpy
visdom

Please organe the data in this format:)
/Path_To_TrainSet/
    images/
        img1.png
        img2.png
        ...
    masks/
        img1.png
        img2.png
        ...

Training Steps:
1. Activate the visualization tool: python -m visdom.server -p 9000
2. Run the training file. python train.py --data_dir /Path_To_TrainSet/ --lr_anneal_steps 60000 --batch_size 10

Sampling:
python sample.py --out_dir /Path_to_Save/ --num_samples 1000 --model_path "./ckpts/emasavedmodel_0.9999_060000.pt" --dev "gpuindex"

Note:
If you want to change the image size, please change it in the "./guided_diffusion/script_util-model_and_diffusion_defaults"
