from trainer import evaluate
import json
from dataloader import Flickr8KDataset
import torch
import torchvision.models as models
from decoder import CaptionDecoder

config_path = "config.json"

with open(config_path, "r", encoding="utf8") as f:
    config = json.load(f)
use_gpu = config["use_gpu"] and torch.cuda.is_available()
valid_set = Flickr8KDataset(config, config["split_save"]["validation"], training=False)

device = torch.device("cuda" if use_gpu else "cpu")


encoder = models.resnet50(pretrained=True)
# Extract only the convolutional backbone of the model
encoder = torch.nn.Sequential(*(list(encoder.children())[:-2]))
encoder = encoder.to(device)
# Freeze encoder layers
for param in encoder.parameters():
    param.requires_grad = False
encoder.eval()

######################
# Set up the decoder
######################
# Instantiate the decoder
decoder = CaptionDecoder(config)
decoder = decoder.to(device)


checkpoint_path = "/home/gs285/image_captioning/pytorch-image-captioning/checkpoints/Dec-10_09-26-13/model_19.pth"
decoder.load_state_dict(torch.load(checkpoint_path))
    

valid_bleu = evaluate(valid_set, encoder, decoder, config, device)