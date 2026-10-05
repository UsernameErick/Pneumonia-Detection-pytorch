from torchvision import datasets, transforms
import matplotlib.pyplot as plt

train_path = "data/chest_xray/train"

# изменяем размер картинки под стандарт
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# загрузка датасета. каждая подпапка (Normal, Pneumonia) - отдельный класс. ImageFolder - умная функция, сама понимает что подпапки это классы для классификации
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)

print("Классы:", train_dataset.classes)
print("Всего снимков:", len(train_dataset))

# посмотрим на первую картинку
image, label = train_dataset[0] # распаковка пары (image, label) на две отдельные составляющие, image - цифровой формат картинки, label - номер класса
plt.imshow(image.permute(1, 2, 0)) # pytorch хранит картинку как(канал, высота, ширина), а plt нужно (высота, ширина, канал), поэтому 0(канал) отпр. в конец списка
plt.title(f"Класс: {train_dataset.classes[label]}") # взяли числовой label и перевели его обратно в слово (Normal/Pneumonia), чтобы на график было написано NORMAL
plt.show()