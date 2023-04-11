import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, ego4d_final_v2, ego4d_final_v3


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "Ego4D_final_v2" in config["dataset"]:
        ego4d_final_v2.prepare_align(config)
    if "Ego4D_final_v3" in config["dataset"]:
        ego4d_final_v3.prepare_align(config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()
    print(args)
    
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
