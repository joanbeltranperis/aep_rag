## Cómo ejecutar el proyecto

Requisitos:
- Python 3.11+
- Clave de Google AI (Gemini)

1) Instalar uv (herramienta para manejar versiones de python)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Configurar variables de entorno
Crea un archivo `.env` en la raíz del proyecto:
```bash
echo "GEMINI_API_KEY=tu_api_key_aqui" > .env
```

3) Ejecutar
```bash
uv run main.py
```

4) Para cambiar la pregunta
Edita el archivo `main.py` en la línea 24 donde está definida la variable `question`.

Notas:
- En el primer arranque se descargarán modelos y se crearán los índices vectoriales automáticamente. Puede tardar varios minutos y requiere conexión a internet.

