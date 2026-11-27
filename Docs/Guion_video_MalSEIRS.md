# Guion de grabación de MalSEIRS
CE-1111 · Análisis Numérico para Ingeniería · Instituto Tecnológico de Costa Rica

Duración prevista: 12 min 40 s, incluida la demostración y los cambios de pantalla.
El enunciado exige entre 10 y 15 minutos. Archivo final: pi_video_grupo_#.mp4.
Reparto por responsabilidades: persona 1, problema y modelo; persona 2, Euler, Heun y metodología; persona 3, RK4, demostración, análisis y cierre. Asignar los nombres al reparto acordado por el equipo.

## PERSONA 1 · 0:00–3:35 · Problema y formulación

### 0:00–0:35 | Portada
«En este proyecto se estudia la propagación de malware en una red de computadoras con el modelo MalSEIRS. Se busca estimar cuántos nodos pueden estar infectados al mismo tiempo y en qué momento se alcanza el pico. También se compara qué tan bien Euler, Heun y RK4 aproximan esa dinámica.»

### 0:35–1:25 | Problema de ingeniería y fuente
«El momento del pico ayuda a interpretar cuánto margen habría para reaccionar en el escenario simulado. La fuente del modelo es el artículo de Martínez Martínez y colaboradores, publicado en Complexity en 2021. El artículo propone adaptar un modelo compartimental a redes de computadoras, incorporando tasas que cambian durante la evolución de la infección. El proyecto toma el caso base de ese trabajo para concentrar la comparación en los métodos numéricos.»

### 1:25–2:30 | Diagrama de compartimentos
*Señalar cada flecha mientras se explica.*
«S reúne los nodos susceptibles. Cuando ocurre la exposición, el nodo pasa a E, donde permanece durante una fase de latencia. El paso de E a I representa la activación de la infección. La recuperación lleva el nodo de I a R. Si pierde la protección, puede volver a S. También hay inmunización preventiva que lleva nodos de S directamente a R. El modelo incorpora entradas y salidas de nodos, por lo que la población inicial de 33 puede cambiar durante el intervalo. Lo que se busca es la trayectoria de cada compartimento, en especial la curva de I.»

### 2:30–3:35 | Problema de valor inicial
«La red comienza con treinta susceptibles, dos expuestos, un infectado y cero recuperados. Se estudian treinta unidades de tiempo. Para describir la evolución se escribe un sistema de cuatro ecuaciones de primer orden, con un vector formado por S, E, I y R. Los flujos se acoplan. Por ejemplo, el término de exposición depende simultáneamente de susceptibles e infectados. La tasa de recuperación depende del número de infectados, y la pérdida de inmunidad tiene una oscilación amortiguada. Para el caso estudiado se aproxima la solución por pasos. A continuación se explica cómo se construyen esas aproximaciones.»

## PERSONA 2 · 3:35–7:20 · Métodos y comparación

### 3:35–4:25 | Euler explícito
«En Euler se evalúa la derivada vectorial en el punto actual. Luego se multiplica esa pendiente por el tamaño de paso y se suma al estado actual. Las cuatro componentes avanzan juntas. El método requiere una evaluación de la función por paso. Su orden global es uno bajo las condiciones habituales, así que al dividir el paso entre dos se espera una reducción del error cercana a la mitad en el régimen de convergencia.»

### 4:25–5:20 | Heun
«Heun comienza con la misma pendiente que Euler. Con ella forma un estado provisional al final del paso. Después evalúa la derivada en ese tiempo y con ese estado provisional, y corrige el avance usando el promedio de las dos pendientes. Esto requiere dos evaluaciones por paso. Bajo las condiciones de regularidad correspondientes, su orden es dos y por eso se espera que el error baje aproximadamente por un factor de cuatro al dividir el paso entre dos.»

### 5:20–6:15 | Diseño experimental
«Para comparar los métodos se conservaron el mismo modelo, los parámetros, las condiciones iniciales y el intervalo de cero a treinta. Cada método se ejecutó con pasos de 0.1, 0.05, 0.025 y 0.0125. Esos pasos corresponden a 300, 600, 1200 y 2400 avances. Se registraron el número de pasos, el tiempo, el error, el pico y su instante. El error es la mayor diferencia absoluta entre la aproximación y la referencia sobre los cuatro estados y los tiempos comunes.»

### 6:15–7:20 | Pseudocódigo del experimento
*Mostrar este bloque en pantalla. Leer el flujo, sin recitar la sintaxis.*

```text
referencia ← integrar con RK4 usando h = 0.00125
para cada método en Euler, Heun y RK4:
    para cada h en 0.1, 0.05, 0.025 y 0.0125:
        trayectoria, tiempo ← integrar el modelo en [0, 30]
        referencia_común ← extraer la referencia en los tiempos de la malla
        error ← máximo de |trayectoria − referencia_común|
        pico, instante ← máximo de I y tiempo de ese punto
        registrar métricas
guardar CSV y gráficas
```

«La referencia se construye una vez y se toma únicamente en los puntos que comparte con la malla de cada corrida. Para el pico se recorre la columna de infectados y se recupera el tiempo asociado a su máximo. El CSV reúne las doce configuraciones. Con esta base se puede explicar qué aporta RK4 y verificarlo ejecutando el programa.»

## PERSONA 3 · 7:20–12:40 · RK4, demostración y resultados

### 7:20–8:25 | RK4 y referencia
«RK4 evalúa cuatro pendientes. La primera usa el estado actual. Las dos siguientes se evalúan en el punto medio, cada una con un estado provisional distinto. La cuarta se evalúa al final del paso. El avance combina las pendientes con pesos uno, dos, dos y uno, divididos entre seis. Para la referencia se usó un paso de 0.00125. Al reducirlo a 0.000625, la diferencia máxima entre las dos referencias fue de aproximadamente 6.8 por diez a la menos doce, mucho menor que el menor error experimental.»

### 8:25–9:40 | Demostración computacional
*Cambiar a VS Code. Abrir la terminal en la raíz del proyecto y ejecutar python experimentos.py. Mostrar el diagnóstico de la referencia mientras se genera la salida.*
«El programa recorre los tres métodos y los cuatro pasos. En la terminal se puede ver la verificación de la referencia y la tabla de doce filas. En esta fila de RK4 con el paso de 0.0125, el error es aproximadamente 7.48 por diez a la menos ocho. La misma corrida registra el pico en 5.01426 y su instante en 2.0125.»
*Abrir resultados_experimentos.csv y la gráfica comparacion_infectados.png. Mostrar la fila y señalar el pico. Reservar tiempo para el cambio de ventana.*
«El CSV conserva los resultados y las figuras se generan a partir de la misma ejecución. En la gráfica de infectados, las curvas casi se superponen para el paso más fino.»

### 9:40–10:35 | Trayectoria y pico
*Mostrar la trayectoria RK4 o la comparación de infectados.*
«Los susceptibles caen al comienzo y los expuestos alcanzan su máximo antes que los infectados. La curva de infectados llega cerca de cinco nodos alrededor de la segunda unidad de tiempo. Para h igual a 0.0125, la diferencia entre el mayor y el menor pico de los tres métodos es inferior a 0.012 nodos. El valor de t máximo corresponde a un punto de la malla, por lo que el paso condiciona la resolución con la que se puede reportar.»

### 10:35–11:45 | Convergencia y costo
*Mostrar comparacion_errores.png y luego comparacion_tiempos.png.*
«En los últimos refinamientos, el error baja cerca de dos veces para Euler, cuatro para Heun y dieciséis para RK4 cada vez que se divide el paso entre dos. El salto de RK4 entre 0.1 y 0.05 es más fuerte. Con el paso más grueso, algunos intervalos cruzan los puntos angulosos de la tasa de pérdida de inmunidad. Con los pasos más finos, esos puntos coinciden con los nodos temporales. Para un mismo paso, Euler tardó menos y RK4 dio el menor error. La comparación de tiempos corresponde a una ejecución por configuración.»

### 11:45–12:40 | Conclusiones y fuentes
«Los tres métodos describieron el mismo pico general en el caso base, pero la precisión de toda la trayectoria sí cambió entre métodos. RK4 con h de 0.05 obtuvo un error de alrededor de 2.05 por diez a la menos cinco respecto de la referencia. Ese paso es una opción cuando la tolerancia del ejercicio admite esa diferencia. Los resultados describen el escenario simulado. Para evaluar la capacidad de predicción en una red concreta se necesitarían observaciones de esa red. Las fuentes son el artículo de MalSEIRS y el material de métodos numéricos del curso.»

Preparación de pantalla: tener abiertas la presentación, la terminal y las gráficas trayectorias_rk4.png, comparacion_infectados.png, comparacion_errores.png y comparacion_tiempos.png. Ensayar los cambios de ventana y cronometrar el video completo. Las marcas son tiempos objetivo para la explicación, el señalamiento y las pausas de la demostración.

Secuencia de pantallas en Canva: la persona 1 usa las diapositivas 1–4. La persona 2 usa las diapositivas 5–6 y, al terminar la 6, comparte el bloque de pseudocódigo de este guion. La persona 3 retoma en la diapositiva 7, pasa de la 8 a VS Code para la demostración y vuelve a la 9. En la 10 alterna con las gráficas reales de error y tiempo. Cierra con las diapositivas 11–12.
