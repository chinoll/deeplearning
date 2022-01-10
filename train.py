import json
import importlib
from utils import weights_init,load_dataset
import visdom
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name',required=True)
args = parser.parse_args()

with open("config/" + args.model_name + '.json') as f:
    config = json.loads(f.read())

weight = weights_init(config["weight_init_function"])
config["weights_init"] = weight

m = importlib.import_module("model.cv.gan." + args.model_name,"*")

model = m.model_init(**config)
model["dataloader"] = load_dataset("MNIST",**config)
model["vis"] = visdom.Visdom(env=args.model_name)

for i in tqdm.tqdm(range(config["epochs"])):
    m.train(epoch=i,**model,**config)