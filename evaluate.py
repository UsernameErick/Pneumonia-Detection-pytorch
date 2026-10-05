import torch
from sklearn.metrics import confusion_matrix, classification_report
from data_setup import test_loader
from model_setup import create_model, device

# файл прогона test данных, построение confusion_matrix

model = create_model()
model.load_state_dict(torch.load("model.pth"))
model.eval()

all_predicted = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_predicted.extend(predicted.cpu().numpy()) # возвращаем список предсказаний в cpu т.к. numpy не умеет работать с gpu
        all_labels.extend(labels.cpu().numpy())

accuracy = sum(p == l for p, l in zip(all_predicted, all_labels)) / len(all_labels)
print(f"Точность на test: {accuracy:.4f}")

print("\nConfusion matrix:\n", confusion_matrix(all_labels, all_predicted))
print("\nПодробный отчёт:\n", classification_report(all_labels, all_predicted, target_names=["NORMAL", "PNEUMONIA"]))

# разморозка layer4 позволила уменьшить FN(модель посчитала здоровым больного) с 6 до 2. при этом общая accuracy снизилась за счет повышенной ложной тревоги,
# что в медицине более предпочтительно чем отпускать настоящих больных