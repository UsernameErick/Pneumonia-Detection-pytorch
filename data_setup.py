from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
# Подготовка данных к работе. приведение их в один вид, 

# стандарт для моделей обученных на ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# трансформация для ТРЕНИРОВОЧНЫХ данных - с аугментацией (случайные искажения картинки, чтобы модель находила общие признаки лучше)
train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# трансформация для ТЕСТА, без аугментации(эти данные используются для честной проверки)
eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
# ^ Normalize делается потому что многие модели в том числе и ResNet обучались на диапазонах чисел со средним знач. 0, а после ToTensor у нас значения от 0 до 1 

# формируем train датасет (картинка+класс). full потому что train еще не поделен на train и val(val - промежуточные проверки)
full_train_dataset = datasets.ImageFolder("data/chest_xray/train", transform=train_transform)

# т.к. валидационных данных всего 16 шт., мы вручную разделим 85% на 15% и отдадим 15% в val_dataset
val_size = int(0.15 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
generator = torch.Generator().manual_seed(42) # random state 42

# формируем уже финальные train, val, test датасеты (картинка+класс)
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator) # что режем - на какие доли - по какому сиду
test_dataset = datasets.ImageFolder("data/chest_xray/test", transform=eval_transform) # загружаем отдельно тест

# dataloader - то что реально будет скармливать данные модели пачками(батч - по 32 фотки за раз, для современной GPU в самый раз)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    # посмотрим на 1 батч
    images, labels = next(iter(train_loader))
    print("Размер одного батча картинок:", images.shape)
    print("Размер одного батча меток:", labels.shape)
