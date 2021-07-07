import random

from unet_parts import *
import main
from globals import *
from unet_model import *


def train_model(model, dataload):
    best_model = model
    min_loss = 1000
    optimizer = optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch, EPOCHS - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in tqdm(dataload):
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))

        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))

        if (epoch_loss/step) < min_loss:
            min_loss = (epoch_loss/step)
            best_model = model
            torch.save(model.state_dict(), './model/temp%d_model.pt' % epoch)
    torch.save(best_model.state_dict(), MODEL_PATH)

    print()
    print("... Model trained")
    print()

    return best_model


""" 
Process the results into more easily recognizable results as required. 
Since this experiment is image segmentation, set num_class to 2, which is divided into two colors.
"""


def labelVisualize(num_class, color_dict, pred):
    pred = pred[:,:,0] if len(pred.shape) == 3 else pred
    pred_out = np.zeros(pred.shape + (3,))
    for i in range(num_class):
        pred_out[pred == i, :] = color_dict[i]
    for i in range(num_class):
        pred_out[pred == i, :] = color_dict[i]

    A = np.random.rand(5, 5)

    plt.show()

    return pred_out / 255


def test(model, dataloaders):
    # model = UNet(1, 1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    plt.ion()
    idx = 0
    predictions = []

    with torch.no_grad():
        for index, x in enumerate(dataloaders):
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            img_y = img_y[:, :, np.newaxis]
            pred = labelVisualize(2, COLOR_DICT, img_y) if False else img_y[:, :, 0]
            print(pred.dtype)
            io.imsave("./test_preds/" + str(index) + "_predict.png", pred)
            plt.pause(0.01)
            predictions.append(pred)
            if idx == 10:
                break
            idx += 1
        plt.show()

    # visualize the predictions
    id = random.randint(0, 9)
    image = Image.open(test_data+'images/t%.3d.tif' % id)
    mask = Image.open(test_data+'masks/SEG/man_seg%.3d.tif' % id)
    pred = predictions[id]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    for ax, img, name in zip(axs, [image, mask, pred], ['Image', 'Mask', 'Prediction']):
        ax.imshow(img, cmap='gray')
        ax.set_title(name)
        ax.grid(False)
    plt.savefig("./test_preds/comparison")

    print()
    print("... Particles predicted")
    print()