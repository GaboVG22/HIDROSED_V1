# HIDROSED_V1

Primera versión base modular de HidroSed:

1. **Cuenca y morfometría**: delimita dos cuencas desde PC cuenca soporte y PC hidrológico usando DEM, calcula propiedades morfométricas y prepara datos para hidrología.
2. **Eje del cauce y secciones**: usa eje obligatorio, recorta tramo útil entre los dos PC, genera perfil longitudinal, secciones naturales/prismáticas/por puntos y modelo geométrico compatible con módulos hidráulicos.
3. **Base técnica exportable**: JSON y Excel con cuenca, eje, perfil, secciones y QA para interacción posterior con hidrología, hidráulica, sedimentos y socavación.

## Streamlit Cloud

- Main file path: `app.py`
- Python version: `3.11`

## Recomendación

Para cuencas grandes, usar DEM COP30 con resolución interna 60–120 m y generar curvas/secciones solo en el tramo útil del eje.
