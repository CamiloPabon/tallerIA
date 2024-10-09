# Usar una imagen base de Python oficial (slim es una versión ligera con dependencias mínimas)
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app/

# Copiar el archivo de requisitos de dependencias y los scripts necesarios al contenedor
COPY requirements.txt ./
COPY main.py ./
COPY model.py ./
COPY /models/AdaBoostClassifier.pkl /app/models/AdaBoostClassifier.pkl
COPY /models/scaler.pkl /app/models/scaler.pkl

# Instalar las dependencias especificadas en el archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Añadir un usuario no-root para ejecutar la aplicación
RUN useradd -ms /bin/bash appuser
USER appuser

# Exponer el puerto 8000 donde la aplicación FastAPI se ejecutará
EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI usando Uvicorn como servidor ASGI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

