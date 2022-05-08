# SAGAN: Adversarial Spatial-asymmetric Attention 
This is the official implementation of paper title "SAGAN: Adversarial Spatial-asymmetric
Attention for Noisy Nona-Bayer
Reconstruction". The paper has been accepted and to be published in the proceedings of BMVC21. 

**Download links : [[Full Paper](https://www.bmvc2021-virtualconference.com/assets/papers/1020.pdf)] | [[arxiv](https://arxiv.org/abs/2110.08619)] | [[Supplemental](https://www.bmvc2021-virtualconference.com/assets/supp/1014_supp.zip)] | [[presentation](https://docs.google.com/presentation/d/1IIzhOVnqTVRhc2v6H2sptlu-UrSm58hxsGKPq3GsAzc/edit?usp=sharing)]**


**Please consider to cite this paper as follows:**
```
@inproceedings{a2021beyond,
  title={SAGAN: Adversarial Spatial-asymmetric Attention for Noisy Nona-Bayer Reconstruction},
  author={Sharif, SMA and Naqvi, Rizwan Ali and Biswas, Mithun},
  booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
  pages={},
  year={2021}
}
```

# Overview
Despite the substantial advantages, such non-Bayer CFA patterns are susceptible to produce visual artefacts while reconstructing RGB images from noisy sensor data. SAGAN addresses the challenges of learning RGB image reconstruction from noisy Nona-Bayer CFA comprehensively.



<p align="center">
<img width=400 align="center" src = "https://user-images.githubusercontent.com/15001857/140978518-dc871bdd-9d71-4d51-afbb-93b24d64b0b1.png" alt="CFA"> </br>
</p>



<p align="center">
<img width=1000 align="center" src = "https://user-images.githubusercontent.com/15001857/140979382-c725083b-3488-481f-8ccd-318f276ef749.png" alt="Overview"> </br>
</p>



# Nona-Bayer Reconstruction with Real-world Denoising </br>

<p align="center">
<img width=1000 align="center"  src="https://user-images.githubusercontent.com/15001857/140975651-1a7936e5-1537-43a0-8d24-366289de6f17.png" alt="Real Reconstruction"> </br>
</p>



# Comparison with state-of-the-art deep JDD methods </br>

<p align="center">
<img width=1000 align="center" src = "https://user-images.githubusercontent.com/15001857/140975214-9c555403-44dc-4498-9831-fb49feeb43aa.png" alt="comp"> </br>
</p>





# Prerequisites
```
Python 3.8
CUDA 10.1 + CuDNN
pip
Virtual environment (optional)
```

# Installation
**Please consider using a virtual environment to continue the installation process.**
```
git clone https://github.com/sharif-apu/SAGAN_BMVC21.git
cd SAGAN_BMVC21
pip install -r requirement.txt
```

# Testing with Synthesised Images


**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=sigma(s)``` </br>
Here,**-ns** specifies the standard deviation of a Gaussian distribution (i.e., -ns=10, 20, 30),**-s** specifies the root directory of the source images
 (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).


# Training
To start training we need to sampling the images according to the CFA pattern and have to pair with coresponding ground-truth images.
To sample images for pair training please execute the following command:

```python main.py -ds -s /path/to/GTimages/ -d /path/to/saveSamples/ -g 3 -n 10000 ```
</br> Here **-s** flag defines your root directory of GT images, **-d** flag defines the directory where sampled images should be saved, and **-g** flag defines the binnig factr (i.e., 1 for bayer CFA, 2 for Quad-Bayer, 3 for Nona-Bayer), **-n** defines the number of images have to sample (optional)</br>


</br> After extracting samples, please execute the following commands to start training:

```python main.py -ts -e X -b Y```
To specify your trining images path, go to mainModule/config.json and update "gtPath" and "targetPath" entity. </br>You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., -b 16).</br>


**For transfer learning execute:**</br>
```python main.py -tr -e -b ```


# Traaning with Real-world Noisy Images
To train our model with real-world noisy images, please download "Smartphone Image Denoising Dataset" and comment out line-29 of dataTools/customDataloader.py. The rest of the training procedure should remain the same as learning from synthesized images.

follow the training/data extraction procedure similar to the synthesized images. 

** To inference with real-world Noisy images execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=0``` </br>
Here,**-s** specifies the root directory of the source images
 (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).

A few real-world noisy images can be downloaded from the following link **[[Click Here](https:)]**

# Others
**Check model configuration:**</br>
```python main.py -ms``` </br>
**Create new configuration file:**</br>
```python main.py -c```</br>
**Update configuration file:**</br>
```python main.py -u```</br>
**Overfitting testing** </br>
```python main.py -to ```</br>

# Contact
For any further query, feel free to contact us through the following emails: sma.sharif.cse@ulab.edu.bd, rizwanali@sejong.ac.kr, or mithun.bishwash.cse@ulab.edu.bd
