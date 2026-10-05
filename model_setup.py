import torch
import torch.nn as nn
from torchvision import models

# Подготовка модели, создание, скачивание весов, заморозка слоев, пееренос на GPU. возвращает модель готовую к работе

# использовать GPU, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model():
    # загружаем ResNet18 предобученную на ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # замораживаем все веса, запрещаем модели их менять
    for param in model.parameters():
        param.requires_grad = False

    # первые слои модель видит просты линии, границы; средние слои - фигуры, углы узоры; поздние - на что похоже; финальный - учитывая все найденное, это вероятно PNEUMONIA
    # изначально fc(fully connected, последний слой) выдает 1000 чисел(классов), надо изменить чтобы было всего 2(NORMAL / PNEUMONIA)
    # если разморозить более ранние блоки слоев, то он будет цепляться за базовые признаки типа потемнений на снимке, обучается пристальнее, но риск переобучения если данных мало как у нас
    
    # новый блок кода, разморозка 4 слоя(раньше был только финальный слой fc) для лучшего результата в evaluate
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2) # создали fc слой с нуля со случайными весами

    # перенос модели на GPU
    model = model.to(device)

    return model

if __name__ == "__main__":
    model = create_model()
    print(model.fc)
    print("Модель на устройстве:", device)