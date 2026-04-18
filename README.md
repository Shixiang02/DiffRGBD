# DiffRGBD

**[TCSVT 2026] DiffRGBD: Diffusion-driven RGB-D Salient Object Detection with Temporal Modulation**  
🔗 [IEEE Paper](https://ieeexplore.ieee.org/document/11483127/)

---

## 🧠 Network Architecture
<div align="center">
  <img src="https://github.com/Shixiang02/DiffRGBD/blob/main/image/overview.png" width="80%">
</div>

---

## ⚙️ Requirements
- Python 3.10  
- PyTorch 2.4.0  

---

## 📊 Saliency Maps

We provide the predicted saliency maps of:

- **DiffRGBD (ours)**: [Download Link](https://pan.baidu.com/s/1WWeKr6ArIvP0p0_ABPYpRQ) (code: `XHCL`)  
- **Other RGB-D methods**: [Download Link](https://pan.baidu.com/s/1xtzq61GrHID1w5wPtkGOLg) (code: `XHCL`)  

Evaluated on the following datasets:
- DUT  
- LFSD  
- NJU2K  
- NLPR  
- SIP  
- SSD  
- STERE1000  

<div align="center">
  <img src="https://github.com/Shixiang02/DiffRGBD/blob/main/image/table.png" width="80%">
</div>

---

## 🚀 Training

1. Download the pretrained backbone:
   - [sam2_hiera_large.pt](https://pan.baidu.com/s/1v8_rKwupi9LMifkJv0B7-g) (code: `XHCL`)

2. Modify the checkpoint loading path in:./model/net.py
3. Set your dataset paths in the configuration file.
4. Run training:
accelerate launch train.py \
  --config config/camoDiffusion_352x352.yaml \
  --num_epoch=YOUR_EPOCHS \
  --batch_size=YOUR_BATCH_SIZE \
  --gradient_accumulate_every=1



## 🧪 Pre-trained Model & Testing
  Download pretrained model: [our_checkpoint](https://pan.baidu.com/s/1eDM5FUqjBLkRtiejegTziw)(code: XHCL)
  
  Run inference：
  accelerate launch sample.py \
  --config config/camoDiffusion_352x352.yaml \
  --results_folder YOUR_OUTPUT_PATH \
  --checkpoint YOUR_CHECKPOINT_PATH \
  --num_sample_steps 10 \
  --target_dataset DATASET_NAME 

   
# 📏 Evaluation
 We recommend using the following toolkit for evaluation: [Evaluation Tool](https://github.com/lartpang/PySODEvalToolkit).

# 🙏 Acknowledgements
   
This work is built upon: [Camodiffusion](https://github.com/Rapisurazurite/CamoDiffusion) and [SAM2UNet](https://github.com/WZH0120/SAM2-UNet).
We sincerely thank the authors for their great contributions.

# 📬 Contact
If you have any questions, encounter issues, or find bugs, please feel free to contact: shixiang_joy@163.com.


