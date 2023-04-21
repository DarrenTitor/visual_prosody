# visual_prosody

## Ego4D_final_v2

pip install pyworld的时候安装最新版，就不会failed to build pyworld。
```sh
pip install g2p-en inflect librosa matplotlib numba numpy pyworld PyYAML scikit-learn scipy soundfile tensorboard tgt tqdm unidecode pandas
```


```sh
python prepare_align.py config/Ego4D_final_v4/0421preprocess.yaml
```

得到的结果在`./raw_data/Ego4D_final_v3`.

```sh
python preprocess.py config/Ego4D_final_v4/0421preprocess.yaml
```

```sh
unzip -q /content/visual_prosody/hifigan/generator_universal.pth.tar.zip -d /content/visual_prosody/hifigan/
unzip -q ./hifigan/generator_LJSpeech.pth.tar.zip -d ./hifigan/  
```

```sh
python3 train.py -p config/Ego4D_final_v4/0421preprocess.yaml -m config/Ego4D_final_v4/0421model.yaml -t config/Ego4D_final_v4/0421train.yaml
```




```sh
tensorboard --logdir .\output\0421a\log\Ego4D_final_v4\
```