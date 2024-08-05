import matplotlib.pyplot as plt
import numpy as np

# Generar datos de ejemplo
np.random.seed(0)
months = np.arange(1, 13)
line1 = np.random.rand(12) * 10
line2 = np.random.rand(12) * 10
line3 = np.random.rand(12) * 10
line4 = np.random.rand(12) * 10

# Crear la figura y los ejes
plt.figure(figsize=(10, 6))

# Graficar las cuatro líneas
plt.plot(months, line1, label='Line 1', linewidth=4, color='blue')
plt.plot(months, line2, label='Line 2', linewidth=4, color='orange')
plt.plot(months, line3, label='Line 3', linewidth=4, color='green')
plt.plot(months, line4, label='Line 4', linewidth=4, color='red')

# Añadir anotaciones a cada punto
for i in range(len(months)):
    plt.text(months[i], line1[i] + 0.3, f'{line1[i]:.2f}', color='blue', ha='center',
             va='bottom', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
    plt.text(months[i], line2[i] + 0.3, f'{line2[i]:.2f}', color='orange', ha='center',
             va='bottom', bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.3'))
    plt.text(months[i], line3[i] + 0.3, f'{line3[i]:.2f}', color='green', ha='center',
             va='bottom', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.3'))
    plt.text(months[i], line4[i] + 0.3, f'{line4[i]:.2f}', color='red', ha='center',
             va='bottom', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

# Personalizar los ejes
plt.xticks(months)
plt.yticks(np.arange(0, 11, 1))
plt.xlabel('Month')
plt.ylabel('Value')

# Añadir la leyenda y configurar su posición
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=4, frameon=False, prop={'size': 12})

# Añadir título
plt.title('Generic Multi-Line Plot')

# Mostrar el gráfico
plt.tight_layout()
plt.show()
