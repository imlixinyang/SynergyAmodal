<p align="center">
<!--   <h1 align="center"><img height="100" src="https://github.com/imlixinyang/director3d-page/raw/master/assets/icon.ico"></h1> -->
  <h2 align="center"> <b> SynergyAmodal 😷⇒☺️ </b>: Deocclude Anything with Text Control</h2>
  <p align="center">
<!--         <a href="https://arxiv.org/pdf/2406.17601"><img src='https://img.shields.io/badge/arXiv-Director3D-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://imlixinyang.github.io/director3d-page'><img src='https://img.shields.io/badge/Project_Page-Director3D-green' alt='Project Page'></a>
        <a href='https://colab.research.google.com/drive/1LtnxgBU7k4gyymOWuonpOxjatdJ7AI8z?usp=sharing'><img src='https://img.shields.io/badge/Colab_Demo-Director3D-yellow?logo=googlecolab' alt='Project Page'></a> -->
</p>

![teaser_](https://github.com/user-attachments/assets/6f94e6b0-f26e-426b-aa84-7429f0f854b5)

**⭐ Key components of SynergyAmodal**:

- A full completion diffusion model achieving zero-shot generalization and textual controllability.

**🔥 News**:

- 🥰 Check out our new gradio demo by simply running ```python app.py```.

- 😊 Our paper is accepted by ACMMM 2025.

## 🔧 Installation
- create a new conda enviroment
```
conda create -n synergyamodal python=3.10
conda activate synergyamodal
```

- install pytorch (or use your own if it is compatible with ```xformers```)
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- install ```xformers``` for momory-efficient attention
```
conda install xformers -c xformers
```

- clone this repo:
```
git clone https://github.com/imlixinyang/SynergyAmodal.git
cd SynergyAmodal
```

## Model
Download the pre-trained model by:
```
wget https://huggingface.co/cloudyfall/DeoccAnything/resolve/main/vae_ckpt_dir/epoch%3D5-step%3D100000.ckpt?download=true -O vae.ckpt
wget https://huggingface.co/cloudyfall/DeoccAnything/resolve/main/ldm_ckpt_dir/epoch%3D8-step%3D58000.ckpt?download=true -O ldm.ckpt
```

## Dataset
Download and extract our dataset with:
```
wget https://huggingface.co/datasets/cloudyfall/SynergyAmodal16K/resolve/main/dataset.tar.gz
tar -xzvf dataset.tar.gz
```

## Citation

```
@article{li2025synergyamodal,
  title={SynergyAmodal: Deocclude Anything with Text Control},
  author={Li, Xinyang and Yi, Chengjie and Lai, Jiawei and Lin, Mingbao and Qu, Yansong and Zhang, Shengchuan and Cao, Liujuan},
  journal={arXiv preprint arXiv:2504.19506},
  year={2025}
}
```


## License

Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)

The code is released for academic research use only. 

If you have any questions, please contact me via [imlixinyang@gmail.com](mailto:imlixinyang@gmail.com). 
