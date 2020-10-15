1.install pytorch
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
I don't have gpu, if you have one and installed cuda, then
(I assume coda version is 10.2, the last one supported by pytorch)
pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

2.install other dependencies
pip install -r requirements.txt

3.install package for bert summrizationn
pip install git+https://github.com/dmmiller612/bert-extractive-summarizer.git@small-updates

4.get model
https://drive.google.com/drive/folders/189VY65I_n4RTpQnmLGj7IzVnOF6dmePC
in the same folder with python script

5.dataset
unzip dataset in the same folder with python script and model, like
main.py
ManyModalImages/
ManyModalQAData/

5.how to run
python caption.py --model=BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar --word_map=WORDMAP_coco_5_cap_per_img_5_min_word_freq.json --beam_size=5 --img=ManyModalImages
