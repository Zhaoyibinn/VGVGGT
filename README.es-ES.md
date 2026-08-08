
</think>

<div align="center">

# $\text{VG}^2\text{GT}$: Transformer con Geometría Visual Basado en Splatting de Gaussiano Voxel

<!-- <a href="https://jytime.github.io/data/VGGT_CVPR25.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a> -->
<a href="https://arxiv.org/abs/2606.01573"><img src="https://img.shields.io/badge/arXiv-2606.01573-red?logo=arxiv" alt="arXiv"></a>
<!-- <a href="https://vgg-t.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/facebook/vggt'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> -->


**[Grupo de Visión 3D, Universidad de Ciencia y Tecnología de China Oriental](https://www.ecust.edu.cn/)**; 


[Yibin Zhao](https://github.com/Zhaoyibinn)
</div>

```bibtex
@misc{zhao2026textvg2gtvoxelgaussiansplattingvisual,
      title={$\text{VG}^2$GT: Voxel-Gaussian Splatting Visual Geometry Grounded Transformer}, 
      author={Yibin Zhao and Yihan Pan and Jun Nan and Wenli Yang and Liwei Chen and Jianjun Yi},
      year={2026},
      eprint={2606.01573},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.01573}, }
```



## Descripción General

$\text{VG}^2\text{GT}$ (Voxel-Gaussian Splatting Visual Geometry Grounded Transformer) es un método de reconstrucción mediante Splatting de Gaussiano feed-forward, **no alineado por píxeles**, que mapea imágenes a escenas de Splatting de Gaussiano en cuestión de segundos, demostrando ventajas significativas en la síntesis de vistas nuevas (NVS), la estimación de mapas de profundidad y la reconstrucción de superficies, manteniendo a la vez costos eficientes de entrenamiento e inferencia.

## Configuración del Entorno

Este código ha sido probado con Torch 2.5.0 + CUDA 11.8 y Torch 2.7.1 + CUDA 12.8.

Primero, clona este repositorio en tu máquina local e instala las dependencias. 

```bash
git clone https://github.com/Zhaoyibinn/VGVGGT.git
cd VGVGGT
conda env create -f environment.yaml
```

Instala el renderizador diferenciable basado en renderizado volumétrico estocástico
```bash
git clone --recursive https://github.com/Zhaoyibinn/Geometry-Grounded-Gaussian-Splatting.git
cd Geometry-Grounded-Gaussian-Splatting
pip install . --no-build-isolation
```

(Opcional) Instalar el renderizador de rasterización 3DGS original (con renderizado de mapas de profundidad)
```bash
git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
git checkout 9c5c202
pip install . --no-build-isolation
```

(Opcional) Instalar el renderizador de rasterización 2DGS
```bash
git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git
cd 2d-gaussian-splatting
pip install . --no-build-isolation
```

## Descarga de Datos
### Descarga de Datos de Entrenamiento
Tomando el conjunto de datos IGGT infinigen como ejemplo, todos los datos de entrenamiento deben colocarse bajo la carpeta `train_data`. La estructura de datos predeterminada es `train_data/iggt/processed_infinigen/extracted/******`, donde cada escena se almacena en su propia subcarpeta.

<a href="https://huggingface.co/datasets/lifuguan/InsScene-15K"><img src="https://img.shields.io/badge/Data-IGGT-blue?logo=huggingface" alt="Data IGGT"></a>

### Descarga de Pesos Preentrenados
Utilizamos DA3_Giant_1.1 como pesos preentrenados predeterminados. Todos los pesos preentrenados deben colocarse bajo la carpeta `weight`. La estructura predeterminada es `weight/da3/weights_gaint_1.1/model.safetensors`.

<a href="https://huggingface.co/depth-anything/DA3-GIANT-1.1"><img src="https://img.shields.io/badge/PreTrained-IGGT-blue?logo=huggingface" alt="PreTrained IGGT"></a>

## Entrenamiento del Modelo

El marco de entrenamiento de nuestro modelo se basa en VGGT, con modificaciones realizadas sobre él.

El modelo se encuentra principalmente bajo `vggt/models`. De forma predeterminada, entrenamos y realizamos inferencias utilizando `vggt/models/depthanything3.py`.


El archivo de configuración de entrenamiento predeterminado se encuentra en `training/config/demo.yaml`.

Puedes iniciar el entrenamiento directamente:
```bash
source training/launch.sh
python training/launch.py --config demo
```

Si tienes múltiples GPU, también puedes usar el entrenamiento con múltiples GPU para mejorar la eficiencia:
```bash
bash training/launch2.sh
```

Los resultados del entrenamiento se guardarán bajo `logs/demo`, donde también se guardará una copia de la configuración YAML de forma predeterminada para la inferencia.

### Descripción del Archivo de Configuración
En el archivo de configuración, puedes ajustar los siguientes parámetros clave:
1. `gs_mode`: Controla el tipo de renderizador de Splatting de Gaussiano. Opciones disponibles: `2DGS`, `3DGS`, `GGGS`
2. `img_size`, `aspects`: Controlan la resolución del lado más largo y la relación entre los lados cortos y largos de las imágenes de entrenamiento, respectivamente. Si sigues encontrando errores de falta de memoria (OOM), considera reducir estos valores
3. `img_nums`: El rango del número de imágenes muestreadas durante el entrenamiento. Del mismo modo, reduce este rango si encuentras problemas de memoria OOM
4. `low_resolution_voxelsize`: El tamaño de voxel utilizado para la decodificación de la escena de Gaussiano
5. `frozen_module_names`: Módulos a congelar durante el entrenamiento. Solo `model.backend` y `*backend_gs_decoder*` deben ser entrenados

## Inferencia del Modelo
También proporcionamos inferencia y guardado del modelo basado en Colmap, junto con la evaluación del rendimiento de NVS y la evaluación de la síntesis de mapas de profundidad (nota: para conjuntos de datos que requieren máscaras, como DTU y T&T, aún podrías necesitar volver a calcular las métricas).

La carpeta de datos de entrada solo necesita contener una carpeta `images`.
```bash
python demo_colmap.py --scene_dir <data_folder> --shared_camera --conf_thres_value 0.0 --config_file <config_yaml_path> --checkpoint_path <checkpoint_path> --sparse_subdir <colmap_output_folder>
```

## Agradecimientos

Gracias a estos grandes repositorios: [VGGT](https://github.com/facebookresearch/vggt), [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3), [amb3r](https://github.com/HengyiWang/amb3r), [IGGT](https://github.com/lifuguan/IGGT_official), [GGGS](https://baowenz.github.io/geometry_grounded_gaussian_splatting/), [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [2DGS](https://github.com/hbb1/2d-gaussian-splatting/) y muchas otras obras inspiradoras de la comunidad.
