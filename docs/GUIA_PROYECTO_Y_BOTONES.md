# Guia del proyecto OW RAG Multimodal

## 1. Que hace el proyecto
Este proyecto recomienda heroes de Overwatch usando tres señales:

1. `Query` en lenguaje natural (tu estilo de juego).
2. Heroes que ya juegas (perfil del jugador).
3. Contexto semantico recuperado con RAG a partir de esos heroes.


## 2. Flujo general (extremo a extremo)

1. Se cargan heroes desde `data/heroes.json`.
2. Se construye (o reutiliza) un indice de embeddings en `data/cache/`.
3. Si el usuario manda heroes jugados, se construye un perfil RAG.
4. Se fusionan vectores para consulta final:
   - Query: peso 0.6
   - Heroes jugados: peso 0.3
   - Contexto recuperado: peso 0.1
5. Se calcula similitud contra todos los heroes y se devuelve el Top-K.

## 3. Modulos principales

- `src/ow_rag_multimodal/data.py`: carga y normaliza el dataset.
- `src/ow_rag_multimodal/embeddings.py`: crea embeddings y cache.
- `src/ow_rag_multimodal/rag.py`: recupera contexto y arma perfil de jugador.
- `src/ow_rag_multimodal/recommender.py`: orquesta el ranking final.
- `src/ow_rag_multimodal/history.py`: guarda historial en `data/player_history.json`.
- `src/ow_rag_multimodal/cli.py`: entrada por consola.
- `src/ow_rag_multimodal/ui.py`: interfaz Gradio.

## 4. Como funciona la UI (flujo actual)

1. Seleccionas heroes en `Heroes que ya juegas`.
2. Escribes la query de estilo.
3. Pulsas `Recomendar` (o Enter en la query).
4. El sistema muestra:
   - Recomendaciones
   - Perfil RAG
   - Perfil usado para recomendar
   - Historial RAG

## 5. Explicacion de botones y controles que preguntaste

### `Refrescar cache embeddings`
Sirve para forzar una reconstruccion completa del indice vectorial.

Cuando esta activo:
- Ignora `hero_vectors.npy` aunque exista.
- Recalcula embeddings de texto.
- Reescribe metadatos de cache (`index_meta.json`).

Usalo cuando:
- Cambiaste `data/heroes.json`.
- Cambiaste modelo de embedding.
- Sospechas cache desactualizada/corrupta.

### `Incluir historial en RAG`
Sirve para enriquecer el perfil con tus heroes mas usados historicamente.

Cuando esta activo:
- Al perfil no solo entran heroes seleccionados en esta sesion.
- Tambien entran los `Top historial` heroes de `player_history.json`.

Impacto:
- Mejor memoria de estilo a largo plazo.
- Recomendaciones mas estables aunque selecciones pocos heroes en la sesion.

### `Mostrar contexto RAG`
Es un control de visualizacion, no de logica de ranking.

Cuando esta activo:
- Muestra en pantalla los heroes recuperados como contexto RAG (con score).

Cuando esta inactivo:
- El RAG se sigue usando para recomendar.
- Solo se oculta ese detalle en la UI.

## 6. Datos y archivos de cache

### Datos de entrada
- `data/heroes.json`: heroes, rol y texto.

### Historial del usuario
- `data/player_history.json`:
  - `played_counts`
  - `played_sequence`
  - `updated_at`

### Cache vectorial
- `data/cache/hero_vectors.npy`: matriz de embeddings de heroes.
- `data/cache/index_meta.json`: firma del dataset, modelo y cantidad de heroes.

## 7. Comandos utiles

### Ejecutar UI
```bash
ow-rag-ui --host 127.0.0.1 --port 7860
```

### Ejecutar por CLI
```bash
ow-rag --query "presion constante, picks rapidos" --played tracer ana --top-k 5
```

### Forzar regeneracion de cache por CLI
```bash
ow-rag --query "dive agresivo" --refresh-cache
```

## 8. Recomendaciones de uso

1. Deja `Incluir historial en RAG` activo en uso normal.
2. Usa `Refrescar cache embeddings` solo cuando hagas cambios de datos/modelo.
3. Activa `Mostrar contexto RAG` cuando quieras depurar por que salio una recomendacion.
