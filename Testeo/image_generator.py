import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Definir el generador
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.gen(x)

# Definir el discriminador
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.dis(x)

# Definir los hiperparámetros
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0002
batch_size = 64
image_size = 784
hidden_size = 256
latent_size = 64
epochs = 100

# Cargar los datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Inicializar el generador y el discriminador
generator = Generator(input_size=latent_size, hidden_size=hidden_size, output_size=image_size).to(device)
discriminator = Discriminator(input_size=image_size, hidden_size=hidden_size, output_size=1).to(device)

# Definir la función de pérdida y los optimizadores
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
dis_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Cargar el modelo si este existe
try:
    cargado = True
    generator.load_state_dict(torch.load('generator.ckpt'))
    print('Modelo cargado exitosamente')
except:
    cargado = False
    print('No se ha encontrado un modelo guardado')

# Entrenar la GAN
if not cargado:
    for epoch in range(epochs) :
        for batch_idx, (real, _) in enumerate(train_loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # Entrenar el discriminador
            dis_optimizer.zero_grad()
            noise = torch.randn(batch_size, latent_size).to(device)
            fake = generator(noise)
            dis_real = discriminator(real).view(-1)
            dis_fake = discriminator(fake).view(-1)
            dis_loss = criterion(dis_real, torch.ones_like(dis_real)) + criterion(dis_fake, torch.zeros_like(dis_fake))
            dis_loss.backward()
            dis_optimizer.step()

            # Entrenar el generador
            gen_optimizer.zero_grad()
            noise = torch.randn(batch_size, latent_size).to(device)
            fake = generator(noise)
            gen_loss = criterion(discriminator(fake).view(-1), torch.ones(batch_size).to(device))
            gen_loss.backward()
            gen_optimizer.step()

            # Imprimir el progreso del entrenamiento
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Batch {batch_idx}/{len(train_loader)}, Loss D: {dis_loss:.4f}, Loss G: {gen_loss:.4f}')

# Guardar el modelo
if not cargado:
    torch.save(generator.state_dict(), 'generator.ckpt')

with torch.no_grad():
    noise = torch.randn(1, latent_size).to(device)
    fake = generator(noise).view(28, 28)
    fake = 0.5 * (fake + 1)
    fake = transforms.ToPILImage()(fake.cpu())
    fake.save('fake.png')
