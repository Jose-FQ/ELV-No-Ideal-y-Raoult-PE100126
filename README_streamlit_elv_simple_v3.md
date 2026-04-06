# App Streamlit: ELV binario y ternario

## Archivos necesarios en la misma carpeta
- `app_streamlit_elv_simple_v3.py`
- `nrtl_virial_vle.py`
- `experimental_elv_from_sheet.json`
- `precomputed_binary_elv_30pts.json`
- `requirements_streamlit_nrtl_virial.txt`
- `Escudo-UNAM-escalable.svg.png`
- `Logo-Buho.png`

## Qué hace esta versión
1. **Comparación binaria precalculada**
   - Sistemas: agua+etanol, agua+metanol, metanol+etanol
   - Modelos:
     - ideal líquido + ideal vapor
     - ideal líquido + vapor no ideal
     - líquido no ideal + vapor ideal
     - líquido no ideal + vapor no ideal (γ–φ)
   - Gráficas:
     - `y-x` a 1.013 bar
     - `T-x-y` a 1.013 bar
   - Los cálculos ya vienen guardados en `precomputed_binary_elv_30pts.json`
   - Se usan **30 puntos fijos** por curva

2. **Mezcla ternaria**
   - Componentes: metanol, etanol y agua
   - El usuario selecciona:
     - modelo termodinámico
     - presión constante
     - composición líquida para punto de burbuja
     - composición de vapor para punto de rocío
   - La app calcula:
     - temperatura de burbuja
     - composición de vapor en equilibrio
     - temperatura de rocío
     - composición líquida en equilibrio

## Ejecución en Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_streamlit_nrtl_virial.txt
python3 -m streamlit run app_streamlit_elv_simple_v3.py
```

## Observaciones
- La comparación binaria es inmediata porque usa una base precalculada.
- En la pestaña ternaria, las composiciones se normalizan automáticamente si la suma no es exactamente 1.
- El encabezado incluye los escudos, el título, el laboratorio, el año 2026 y la leyenda de agradecimientos debajo del año.
