# OW RAG Multimodal
## Arquitectura, flujo y controles de UI

---

# Objetivo del proyecto

- Recomendar heroes de Overwatch con contexto de estilo de jugador.
- Combinar texto + historial + RAG.

---

# Problema que resuelve

- Evitar recomendaciones genericas solo por descripcion corta.
- Incorporar memoria de lo que el jugador ya usa.
- Explicar parcialmente el resultado con contexto RAG.

---

# Componentes principales

- `data.py`: carga y normalizacion de heroes.
- `embeddings.py`: embeddings y cache.
- `rag.py`: recuperacion y perfil del jugador.
- `recommender.py`: fusion de señales y ranking final.
- `history.py`: memoria de uso del jugador.
- `ui.py` y `cli.py`: capas de entrada.

---

# Flujo de recomendacion

1. Cargar heroes.
2. Construir/reusar indice vectorial.
3. Construir perfil RAG (si hay heroes jugados).
4. Fusionar vectores (query + played + contexto).
5. Rankear por similitud y devolver Top-K.

---

# Formula de fusion

- Query: `0.6`
- Heroes jugados: `0.3`
- Contexto RAG recuperado: `0.1`

`final_query = normalize(0.6*query + 0.3*played + 0.1*context)`

---

# UI simplificada (actual)

- Seleccion multiple de heroes.
- Query libre.
- Boton principal: `Recomendar`.
- Panel `Ajustes avanzados` para opciones tecnicas.

---

# Boton: Refrescar cache embeddings

Para que sirve:
- Forzar reconstruccion completa del indice.

Cuando usarlo:
- Cambio de dataset.
- Cambio de modelo.
- Cache sospechosa/inconsistente.

---

# Control: Incluir historial en RAG

Para que sirve:
- Agregar heroes historicos mas usados al perfil efectivo.

Beneficio:
- Recomendaciones mas estables y personalizadas en el tiempo.

---

# Control: Mostrar contexto RAG

Para que sirve:
- Mostrar detalle del contexto recuperado (héroe, rol, score).

Importante:
- No cambia el ranking.
- Solo cambia el nivel de explicabilidad en UI.

---

# Datos y cache

- `data/heroes.json`
- `data/player_history.json`
- `data/cache/hero_vectors.npy`
- `data/cache/index_meta.json`

---

# Riesgos y buenas practicas

- No refrescar cache en cada corrida (costoso).
- Mantener textos de heroes consistentes y ricos semánticamente.
- Activar contexto RAG para depurar resultados inesperados.

---

# Comandos clave

```bash
ow-rag-ui --host 127.0.0.1 --port 7860
ow-rag --query "dive agresivo" --played tracer genji --top-k 5
ow-rag --query "frontline" --refresh-cache
```

---

# Cierre

- El proyecto combina recomendacion vectorial + memoria de jugador + contexto RAG.
- Los controles de UI permiten balancear velocidad, personalizacion y explicabilidad.
