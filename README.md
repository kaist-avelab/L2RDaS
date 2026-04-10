# L2RDaS: LiDAR-to-Radar Synthesis for Building Large-Scale Tensor Datasets

L2RDaS is a novel framework designed to synthesize high-fidelity 4D radar tensors (C-RAE) from LiDAR point clouds. It addresses the critical challenges of data scarcity, providing a robust data augmentation solution for downstream 3D object detection tasks.

## 🌟 Key Features

* **3D Tensor Representation:** Unlike conventional point cloud extraction methods (e.g., CFAR) that suffer from inevitable information loss, L2RDaS directly synthesizes 3D spatial tensors, perfectly preserving the structural integrity of the scene.
* **OBIS Module (Object Bounding-box-based Injection and Sampling):** Overcomes the naturally blurry characteristics of radar data. By injecting class-specific, Gaussian-distributed auxiliary points, it ensures spatial continuity and realistic object generation.
* **Hardware-Agnostic Generation:** While trained solely on the K-Radar dataset, the L2RDaS generator is not constrained by specific sensor hardware or training data distributions. It can seamlessly synthesize meaningful radar tensors across entirely different collection environments.
* **Boosts Downstream Perception:** Demonstrated to significantly improve object detection performance (e.g., RTNH, Radarpillar-Net, RPFA-Net, and DaDan) through high-fidelity data augmentation.

## 📖 Overview

4D radar is emerging as a crucial sensor for autonomous driving due to its robustness in adverse weather conditions. However, compared to cameras and LiDARs, massive 4D radar datasets are extremely scarce. 

L2RDaS bridges this gap. By leveraging LiDAR point clouds, our L2RDaS learns the cross-modal correspondence and translates LiDAR voxels into 4D radar C-RAE tensors. 

---
## ⚙️ Getting Started (To be updated after publication)
* Installation
* Dataset Preparation
* Training & Evaluation
