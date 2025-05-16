pip install -r ./requirements.txt &&
pip install albumentations==1.4.4 timm einops &&
pip install -U openmim &&
mim install mmengine &&
mim install mmcv==2.2.0 &&
#git clone https://github.com/open-mmlab/mmcv.git
#cd mmcv && pip install -r requirements/optional.txt && pip install -e . -v
pip install xformers==0.0.24 # torch 2.2
#pip install PyWavelets
pip install -v -e .
pip install numpy==1.26.0