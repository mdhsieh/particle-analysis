from globals import *

"""
Reference:
https://www.programmersought.com/article/54904167186/
"""

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


y_transforms = transforms.ToTensor()

def train_dataset(img_root, label_root):
    imgs = []
    n = len(os.listdir(img_root))-1
    print(n)
    for i in range(0, n):
        img = os.path.join(img_root, "t%.3d.tif" % i)
        label = os.path.join(label_root, "man_seg%.3d.tif" % i)
        imgs.append((img, label))
        # print(img, label)
    return imgs


def test_dataset(img_root):
    imgs = []
    n = len(os.listdir(img_root))-1
    for i in range(n):
        img = os.path.join(img_root, "t%.3d.tif" % i)
        imgs.append((img))
    return imgs


class TrainDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None, target_transform=None):
        imgs = train_dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_x = np.array(img_x).astype(np.float32)
        img_y = Image.open(y_path)
        img_y = np.array(img_y).astype(np.float32)

        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, img_root, transform=None, target_transform=None):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        # https://discuss.pytorch.org/t/pil-image-to-floattensor-uint16-to-float32/54577
        img_x = Image.open(x_path)
        img_x = np.array(img_x).astype(np.float32)
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x

    def __len__(self):
        return len(self.imgs)


def data_extraction_torch():
    IMG_PATH = train_data + 'images/t000.tif'
    MASK_PATH = train_data + 'masks/SEG/man_seg000.tif'

    sample_image = Image.open(IMG_PATH)
    sample_mask = Image.open(MASK_PATH)

    print("Information about data: ")

    TRAINSET_SIZE = len(glob(train_data + 'images/*.tif'))
    print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

    TESTSET_SIZE = len(glob(test_data + 'images/*.tif'))
    print(f"The Test Dataset contains {TESTSET_SIZE} images.")

    input_size = sample_image.size
    print("Image size: ", input_size)
    print("Image format: ", sample_image.format)
    print(f"Image mode: {sample_image.mode} (16-bit unsigned integer pixels)")
    print()

    mask_size = sample_mask.size
    print("Mask size: ", mask_size)
    print("Mask format: ", sample_mask.format)
    print(f"Mask mode: {sample_mask.mode} (16-bit unsigned integer pixels)")
    print()

    # sample_image = train_data + 'images/t000.tif'
    # image = imread(sample_image)    # read the image using skimage
    # plt.imshow(image)
    # plt.waitforbuttonpress()
    #
    # sample_mask = train_data + 'masks/SEG/man_seg000.tif'
    # mask = imread(sample_mask)
    # plt.imshow(mask)
    # plt.waitforbuttonpress()

    print("... Data extracted")


    train_dataset = TrainDataset(train_data+'images/', train_data+'masks/SEG/', transform=x_transforms, target_transform=y_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)

    test_dataset = TestDataset(test_data+'images/', transform=x_transforms,target_transform=y_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    print()
    print("... Torch Dataloaders created")

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0].squeeze()

    # plt.imshow(img)
    # plt.waitforbuttonpress()
    #
    # plt.imshow(label)
    # plt.waitforbuttonpress()

    plt.show()
    print(f"Label: {label}")
    print()
    print("... Torch Dataloaders viewed")

    return train_dataloader, test_dataloader


