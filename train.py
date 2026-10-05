import torch
import torch.nn as nn
import torch.optim as optim
from data_setup import train_loader, val_loader
from model_setup import create_model, device

print("Размер val_dataset прямо сейчас:", len(val_loader.dataset))

# файл обучения сети, корректировок

model = create_model()

# функция ошибки, насколько сильно ошиблась
criterion = nn.CrossEntropyLoss()

# оптимизатор - как именно поправлять веса
# lr - размер шага перед каждой поправкой ошибки
# weight_decay - регуляризация, наказание за очень большие веса
# optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4) # передаем только model.fc.parameters. воплощаем заморозку, оптимизатор не трогает заморож. слои
optimizer = optim.Adam([{"params": model.fc.parameters(), "lr": 0.001}, {"params": model.layer4.parameters(), "lr": 0.0001}], weight_decay=1e-4)
# ^ разные learning-rate для разных слоев. layer4 уже что-то умеет и его lr низкий, осторожно учится. fc учится заново, его lr обычный


NUM_EPOCHS = 8 # 1 батч проходится 5 раз, исправляя свои ошибки 5 раз. прошлый = 5

for epoch in range(NUM_EPOCHS):
    # этап обучения, создание счетчиков чтобы в конце посчитать среднее 
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader: # проходим по всем батчам которые мы заготовили в data_setup.py
        images, labels = images.to(device), labels.to(device) # модель и наши данные(на батчах) должны быть в 1 месте(у нас - в GPU)

        optimizer.zero_grad() # перед каждым новым батчем обнулять счетчик поправок с прошлого раза
        outputs = model(images) # скармливаем по картинке и получаем мнение модели в числовом виде 
        loss = criterion(outputs, labels) # сравниваем ответ модели с правильным ответом. число насколько сильно ошиблась модель
        loss.backward() # PyTorch сам подправляет вес в нужную сторону, чтобы ошибка была меньше в след. раз (это и есть обучение)
        optimizer.step() # применение этих поправок по весам 

        running_loss += loss.item() # сумма ошибок этого батча за эпоху
        _, predicted  = torch.max(outputs, 1) # outputs - 2 числа(увер-сть NORMAL, увер-сть PNEUMONIA). torch.max() возвращает его значение и позицию. _, - значение не возвращаем
        # ^ нам нужна только позиция, потому что мы узнаем это 0 нормал или 1 пневмония и сравним это с 0 и 1 в ImageFolder. позиция - и есть класс.
        correct += (predicted == labels).sum().item() # сравниваем все с labels(правильными ответами), подсчет
        total += labels.size(0) # считаем в общем сколько всех

    train_acc = correct/total # просто общая точность эпохи
    print(f"Эпоха {epoch+1}/{NUM_EPOCHS} | Train loss: {running_loss/len(train_loader)}") # средняя ошибка за эпоху. сумма ошибок / число батчей. train_loader - внутри батчи

    # этап проверки на val
    model.eval() # model.train() и model.eval() - переключение между режимами модели. разные режимы по-разному ведут себя в режиме обучения и проверки
    val_correct = 0
    val_total = 0

    with torch.no_grad(): # на этапе проверки запрещаем читать поправки, потому что мы не обучаемся, а проверяем + экономит память и время
        for images, labels in val_loader: # все то же самое что и выше только без backward() и step() потому что здесь мы не обучаемся, а проверяем
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    print(f"             Val acc: {val_acc:.4f}")

torch.save(model.state_dict(), "model.pth") # сохраняем обученную модель в файл
print("Модель сохранена в model.pth")