
# Prime ChainSentinel — Benchmarks

Suite de pruebas de rendimiento (*benchmarks*) desarrollada para el prototipo de Prime ChainSentinel, diseñada para realizar mediciones con un consumo controlado de recursos computacionales y créditos de GPU.

## ¿Qué mide esta suite?

- **Rendimiento criptográfico en CPU:** tasa de procesamiento real de los algoritmos SHA-256, SHA3-256 y BLAKE3.
- **Construcción de árboles de Merkle:** tiempo de construcción en CPU y errores de predicción de los modelos ingenuo y corregido.
- **Prueba de trabajo (Proof of Work):** medición de tiempos mediante una simulación local, sin conexiones de red ni minería real.
- **Carga sintética en GPU (opcional):** medición de una carga de trabajo paralela sobre bytes mediante PyTorch en una AMD MI300X, utilizada como referencia para estudiar la aceleración de simulaciones.

## Alcance y limitaciones científicas

El modo GPU mide actualmente el rendimiento de una carga de trabajo sintética con procesamiento paralelo de bytes implementada en PyTorch.

**Esta prueba no mide directamente el rendimiento de los algoritmos SHA-256, SHA3-256 o BLAKE3 ejecutados mediante kernels criptográficos nativos en GPU.**

Para realizar afirmaciones sobre el rendimiento criptográfico nativo de estos algoritmos en GPU, sería necesario implementar y evaluar kernels específicos utilizando tecnologías como HIP, CUDA u OpenCL.

Por tanto, los resultados de la carga sintética deben interpretarse como mediciones experimentales de su propia carga de trabajo y no como mediciones directas del rendimiento criptográfico nativo.

## Comandos de ejecución

### Prueba rápida en entorno local

```bash
python main.py --quick
```

### Benchmark completo en CPU

```bash
python main.py --mode full --out exports_cpu_full
```

### Prueba rápida en AMD MI300X

```bash
python main.py --gpu --quick --out exports_mi300x_smoke
```

### Benchmark completo en AMD MI300X

```bash
python main.py --gpu --mode full --out exports_mi300x_full
```

## Archivos generados

La ejecución de las pruebas genera los siguientes archivos:

| Archivo | Descripción |
|---------|-------------|
| `hardware_profile.json` | Información del hardware y del entorno de ejecución. |
| `benchmark_full.json` | Resultados completos de las pruebas de rendimiento. |
| `summary.json` | Resumen general de los resultados obtenidos. |
| `hash_benchmark.csv` | Resultados de rendimiento de los algoritmos criptográficos en CPU. |
| `raw_hash_benchmark.csv` | Mediciones individuales de las pruebas criptográficas. |
| `gpu_synthetic_benchmark.csv` | Resultados de la carga sintética en GPU. Se genera cuando se utiliza `--gpu`. |
| `merkle_benchmark.csv` | Tiempos de construcción y resultados de los modelos de predicción de árboles de Merkle. |
| `pow_benchmark.csv` | Resultados de las pruebas locales de simulación de Proof of Work. |

## Hallazgo experimental: modelado de árboles de Merkle

Las predicciones ingenuas del tiempo de construcción de árboles de Merkle, basadas únicamente en la cantidad de nodos internos `(N - 1)`, tienden a subestimar el tiempo total de ejecución.

Esto ocurre porque la implementación evaluada realiza operaciones de hash tanto sobre las hojas, que representan las transacciones, como sobre los nodos internos del árbol.

Para un árbol binario completo con `N` hojas, el modelo corregido contempla aproximadamente:

**2N - 1 operaciones de hash**

Este ajuste permite representar de manera más completa el trabajo criptográfico requerido durante la construcción del árbol.

Sin embargo, la corrección no elimina completamente la diferencia entre el tiempo predicho y el tiempo medido.

El error residual puede atribuirse a factores adicionales, entre ellos:

- Sobrecarga propia de la implementación.
- Movimiento y copia de datos en memoria.
- Asignación y administración de listas.
- Comportamiento de la memoria caché.
- Costos de despacho y ejecución de instrucciones en Python.

En consecuencia, el modelo corregido constituye una aproximación más representativa del costo de construcción, aunque no reemplaza la medición experimental del tiempo real de ejecución.

## Consideraciones de reproducibilidad

Los archivos exportados permiten conservar las mediciones y la información del entorno de ejecución para su posterior análisis.

Los resultados deben interpretarse considerando el hardware empleado, la configuración del benchmark y las diferencias entre cargas criptográficas reales en CPU y cargas sintéticas ejecutadas en GPU.

Esta distinción es fundamental para mantener la trazabilidad de los resultados experimentales y evitar extrapolaciones de rendimiento que no hayan sido verificadas directamente.
