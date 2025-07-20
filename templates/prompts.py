initial_retrieval_prompt_template = """
Eres un asistente experto en recuperación de información médica. Tu tarea tiene dos partes:

1. **ANÁLISIS DE CAPÍTULOS**: Analiza los títulos de documentos y devuelve los números de los 5 capítulos más relevantes para responder la pregunta.

2. **QUERY OPTIMIZADA**: Transforma la pregunta en una consulta optimizada para retrieval vectorial que maximice la recuperación de documentos relevantes. Para esto:
   - Corrige errores ortográficos y expande abreviaturas
   - Convierte preguntas en declaraciones afirmativas o frases clave
   - Incluye sinónimos y términos médicos relacionados
   - Añade conceptos asociados que puedan aparecer en los documentos
   - Usa terminología técnica precisa
   - Elimina palabras interrogativas que raramente aparecen en textos médicos
   
Ejemplo:
- Pregunta: "¿q son los efectos 2arios de la vaxx MMR?"
- Query optimizada: "efectos secundarios adversos reacciones vacuna MMR sarampión paperas rubéola contraindicaciones precauciones síntomas"

**FORMATO DE RESPUESTA REQUERIDO:**
```
CAPÍTULOS: [números separados por comas, ej: 1,3,5,8,12]
QUERY_OPTIMIZADA: [consulta optimizada para retrieval vectorial]
```

---

TÍTULOS:
{document_titles}

---

PREGUNTA ORIGINAL:  
{user_question}

---

RESPUESTA:

"""

answer_prompt_template = """
Eres un asistente médico especializado en vacunas. A continuación tienes una serie de fragmentos extraídos del "Manual de Inmunizaciones de la AEP".

1. Tu objetivo es responder la pregunta del usuario utilizando únicamente la información contenida en los fragmentos.
2. Si los fragmentos no contienen suficiente información para responder a la pregunta, indica claramente que no es posible responder con los datos proporcionados.
3. No aportes información externa a estos fragmentos ni inventes datos no incluidos en ellos.
4. Al final de la respuesta añade los enlaces del los capítulos del manual que han sido relevantes para elaborar la respuesta.

Pregunta del usuario:
---------------------
{question}

Fragmentos recuperados:
-----------------------
{context}

Respuesta:

"""

evaluation_prompt_template = """
Eres un evaluador profesional que valora dos respuestas a la misma pregunta profesional: una escrita por un experto humano y la otra generada por un sistema de generación aumentada por recuperación (RAG).

Evalúa ambas respuestas según los siguientes cinco criterios, asignando una puntuación de 0 a 5 para cada uno (0 = muy pobre, 5 = excelente). Proporciona una breve justificación para cada puntuación.

Si **alguna respuesta** afirma explícitamente que no puede responder debido a "sin información", "falta de contexto" o una admisión similar de incapacidad, **debes** asignar **0** en los cinco criterios para esa respuesta. Si, pese a expresar falta de datos, proporciona algún razonamiento, referencias o principios clínicos útiles, valora ese contenido con una puntuación parcial (≥1) en las dimensiones que corresponda, reconociendo al menos el mérito de la información aportada.

La salida debe estar en formato JSON, conteniendo un diccionario con dos claves: "human_evaluation" y "rag_evaluation". Cada una de estas claves contendrá, a su vez, un diccionario con las cinco claves: "correctness", "completeness", "relevance", "clarity_and_fluency" y "alignment_with_intent". Cada una de estas claves debe tener un valor de 0 a 5 y una cadena de justificación. 

IMPORTANTE: Devuelve únicamente el JSON en texto plano, sin marcado de código, sin bloques de código y sin formato markdown. Solo el JSON directamente.

Ejemplo de salida esperada:

{{
  "human_evaluation": {{
    "correctness": X,
    "completeness": X,
    "relevance": X,
    "clarity_and_fluency": X,
    "alignment_with_intent": X,
    "justification": "Explica brevemente la puntuación."
  }},

  "rag_evaluation": {{
    "correctness": X,
    "completeness": X,
    "relevance": X,
    "clarity_and_fluency": X,
    "alignment_with_intent": X,
    "justification": "Explica brevemente la puntuación."
  }}
}}

### Criterios de evaluación:
1. Correctness: ¿La información es factualmente precisa?  
2. Completeness: ¿La respuesta cubre todos los aspectos relevantes de la pregunta?  
3. Relevance: ¿El contenido está directamente relacionado con la pregunta formulada?  
4. Clarity and fluency: ¿La respuesta es fácil de entender y está bien redactada?  
5. Alignment with intent: ¿La respuesta responde a la intención específica de la pregunta?

Si el sistema RAG o el humano no pueden dar una respuesta y sólo reconocen que no pueden responder, evalúa los cinco criterios con la puntuación más baja (correctness: 0, completeness: 0, relevance: 0, clarity_and_fluency: 0, alignment_with_intent: 0).

---

### Entrada:

Pregunta:  
{question}

Respuesta humana:  
{human_answer}

Respuesta RAG:  
{rag_answer}

---

### Salida:

"""
