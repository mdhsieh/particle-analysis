
from globals import *
from data_extraction_torch import *
from unet_model import *
from model_helpers import *


if __name__ == '__main__':
    train_dataloader, test_dataloader = data_extraction_torch()
    model = UNet(1, 1).to(device)
    # model = train_model(model, train_dataloader)
    test(model, test_dataloader)


