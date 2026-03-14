# Diagrama de flujo del proyecto

```mermaid
flowchart TD
    A[Usuario: UI o CLI] --> B{Entradas}
    B --> B1[Query]
    B --> B2[Heroes jugados]
    B --> B3[Opciones: rol, top-k, historial]

    B --> C[Cargar heroes.json]
    C --> D[Construir/Reusar indice de embeddings]
    D --> F[hero_vectors normalizados]

    B2 --> G{Hay heroes jugados?}
    G -- Si --> H[Construir perfil RAG]
    H --> H1[Query interna = textos heroes + extra context]
    H1 --> H2[Retrieve top_k contexto]
    H2 --> H3[Resumen de perfil + rasgos]

    B1 --> I[Embedding de query]
    H --> J[Vector promedio de heroes jugados]
    H2 --> K[Vector promedio de contexto]

    I --> L[Combinacion ponderada final]
    J --> L
    K --> L

    L --> M[Score = hero_vectors dot final_query]
    M --> N[Filtrar por rol / excluir jugados]
    N --> O[Top-K recomendaciones]

    O --> P[Salida: recomendaciones]
    H3 --> Q[Salida: perfil RAG]
```

## Lectura rapida

1. El motor siempre trabaja en espacio vectorial normalizado.
2. RAG se usa para enriquecer el perfil, no para reemplazar la query.
3. La salida final es ranking por similitud semantica.
