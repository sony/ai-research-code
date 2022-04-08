# Low Memory Footprint Quantized Depth Completion
<img src="images\spotnet_edited.png" width="1200"/>

This repository hosts supplementary resources for the paper:
> [**A Low Memory Footprint Quantized Neural Network for Depth Completion of Very Sparse Time-of-Flight Depth Maps**](https://sites.google.com/view/ecv2022/home),            
> X. Jiang, V. Cambareri, G. Agresti, C. I. Ugwu, A. Simonetto, F. Cardinaux, and P. Zanuttigh    
> In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops 2022.    
> Presented at the 5th Efficient Deep Learning for Computer Vision Workshop.   

## Abstract 

Sparse active illumination enables precise time-of-flight depth sensing as it maximizes signal-to-noise ratio for low power budgets. However, depth completion is required to produce dense depth maps for 3D perception. 
We address this task with realistic illumination and sensor resolution constraints by simulating ToF datasets for indoor 3D perception with challenging sparsity levels. 
We propose a quantized convolutional encoder-decoder network for this task. Our model achieves optimal depth map quality by means of input pre-processing and carefully tuned training with a geometry-preserving loss function. 
We also achieve low memory footprint for weights and activations by means of mixed precision quantization-at-training techniques. 
The resulting quantized models are comparable to the state of the art in terms of quality, but they require very low GPU times and achieve up to 14-fold memory size reduction for the weights w.r.t. their floating point counterpart with minimal impact on quality metrics.

## Citation

The paper may be cited with the BibTeX key:
```
@inproceedings{jiang_low_2022,
	title = {A {Low} {Memory} {Footprint} {Quantized} {Neural} {Network} for {Depth} {Completion} of {Very} {Sparse} {Time}-of-{Flight} {Depth} {Maps}},
	booktitle = {Proceedings of the {IEEE}/{CVF} {Conference} on {Computer} {Vision} and {Pattern} {Recognition} ({CVPR}) {Workshops}},
	author = {Jiang, Xiaowen and Cambareri, Valerio and Agresti, Gianluca and Ugwu, Cynthia I and Simonetto, Adriano and Zanuttigh, Pietro and Cardinaux, Fabien},
	month = jun,
	year = {2022},
}
```

## Visual Examples

### Depth Maps
We report the ground truth depth map $D_{\rm GT}$ and predicted depth maps $\hat{D}$ for different depth completion models: float32, mixed-precision models $\rm W_i A_j (i = 4, 8; j = 4, 8)$ and $\rm W_{\star}$ (14 MBytes for weights), [NLSPN][1].

| Color     | GT Depth | float32 | W<sub>8</sub> A<sub>8</sub> |
| --------- | -------- | ------- | ------- |
| <img src="images\color_wref_molly_485.png" width="400"/> | <img src="images\gt_depth_wref_molly_485.png" width="400"/> | <img src="images\pred_depth_float32_wref_molly_485.png" width="400"/> | <img src="images\pred_depth_w8a8_wref_molly_485.png" width="400"/> |

| W<sub>4</sub> A<sub>8</sub> | W<sub>4</sub> A<sub>4</sub> | W<sub>\*</sub> | NLSPN | 
| --------- | -------- | ------- | ------- |
 <img src="images\pred_depth_w4a8_wref_molly_485.png" width="400"/> | <img src="images\pred_depth_w4a4_wref_molly_485.png" width="400"/> | <img src="images\pred_depth_wq_wref_molly_485.png" width="400"/> | <img src="images\pred_depth_nlspn_wref_molly_485.png" width="400"/> |


### Error Maps
The error maps (_i.e._, $\hat{D}-D_{\rm GT}$) are reported with the red-white-blue colormap in the symmetric range $[-500, 500] mm$.

| Color     | GT Depth | float32 | W<sub>8</sub> A<sub>8</sub> |
| --------- | -------- | ------- | ------- |
| <img src="images\color_wref_molly_485.png" width="400"/> | <img src="images\gt_depth_wref_molly_485.png" width="400"/> | <img src="images\err_float32_wref_molly_485.png" width="400"/> | <img src="images\err_w8a8_wref_molly_485.png" width="400"/> |

| W<sub>4</sub> A<sub>8</sub> | W<sub>4</sub> A<sub>4</sub> | W<sub>\*</sub>  | NLSPN | 
| --------- | -------- | ------- | ------- |
 <img src="images\err_w4a8_wref_molly_485.png" width="400"/> | <img src="images\err_w4a4_wref_molly_485.png" width="400"/> | <img src="images\err_wq_wref_molly_485.png" width="400"/> | <img src="images\err_nlspn_wref_molly_485.png" width="400"/> |

## Datasets
We shall host a download link for a subset of the SDS-ST1k dataset samples which is used for validation of our results. 
 <!-- The samples will be released on [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). -->

## Copyrights
All content released on this page is &copy; 2022 Sony Europe B.V., &copy; 2022 Sony Depthsensing Solutions NV. 

[1]: https://github.com/zzangjinsun/NLSPN_ECCV20
