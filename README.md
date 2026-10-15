# Albumer

Seleccionador/clasificador local de fotos. Usa CLIP (modelo local, sin APIs ni tokens) para mover imágenes a subcarpetas según criterios de texto que indiques.

## Requisitos
- Node.js 18+ (probado en v24)
- Conexión a internet solo la primera vez (descarga el modelo ~150MB a caché local). Después funciona 100% offline.

## Instalación
```bash
cd albumer
npm install
```

## Ejecutar
```bash
npm start
```
Abrir http://localhost:3000

## Uso
1. Indicá la carpeta con fotos.
2. Escribí un criterio por línea (ej: `perro`, `gato`, `paisaje`). Sugerencia: usar términos en inglés para mejor precisión (`dog`, `cat`, `landscape`).
3. Ajustá el umbral (0-1) y el modo.
4. Pulsá "Clasificar" para ver el plan. Luego "Aplicar" para mover los archivos.

Las fotos se mueven (no se copian). Se crean subcarpetas con el nombre del criterio. Las que no superan el umbral quedan en la carpeta original (modo mejor coincidencia).

## Sin consumir tokens
Todo el procesamiento ocurre en tu máquina con `@xenova/transformers` (CLIP). No se envía ninguna imagen ni texto a servicios externos.
