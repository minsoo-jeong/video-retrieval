#### Video-level (Average+L2Norm)

|Model|DSVR|CSVR|ISVR|
|---|---|---|---|
|Resnet50-IRMAC + PCA/w <sup>(1)</sup> |0.5974|0.6105|0.5812|
|Resnet50 + GAP |0.4016|0.4191|0.3943|
|Resnet50 + RMAC | 0.4329|0.4501|0.4219|
|Resnet50<sup>(2)</sup> + GAP |0.4041|0.4273|0.4123|
|Resnet50<sup>(3)</sup> + GAP |0.4189|0.4430|0.4241|
|Resnet50<sup>(4)</sup> + GAP |0.4649|0.4858|0.4551|
|Resnet50<sup>(4)</sup> + RMAC |0.4863|0.5058|0.4760|
|Resnet50<sup>(4)</sup> + GeM(p=3) |0.4760|0.4919|0.4688|
|Resnet50<sup>(5)</sup> + GAP |0.4535|0.4718|0.4462|
|Resnet50<sup>(6)</sup> + GAP |0.4503|0.4656|0.4332|
|Resnet50<sup>(7)</sup> + GAP |0.3619|0.3805|0.3632|
|X3d_s_13_6_182_160 |0.4572|0.4735|0.4624|
|X3d_m_16_5_256_224 |0.4645|0.4742|0.4529|

#### Clip-level

|Model|DSVR|CSVR|ISVR|
|---|---|---|---|
|Resnet50-IRMAC + PCA/w <sup>(1)</sup> |0.8569|0.8441|0.7711|
|Resnet50 + GAP |0.7408|0.7380|0.6819|
|Resnet50 + RMAC | 0.7798|0.7758|0.7157|
|Resnet50<sup>(2)</sup> + GAP |0.7376|0.7337|0.6800|
|Resnet50<sup>(3)</sup> + GAP |0.7392|0.7411|0.6843|
|Resnet50<sup>(4)</sup> + GAP |0.8107|0.8049|0.7427|
|Resnet50<sup>(4)</sup> + RMAC |0.8400|0.8371|0.7748|
|Resnet50<sup>(4)</sup> + GeM(p=3) |0.8400|0.8360|0.7684|
|Resnet50<sup>(5)</sup> + GAP |0.7960|0.7915|0.7327|
|Resnet50<sup>(6)</sup> + GAP |0.8001|0.7907|0.7312|
|Resnet50<sup>(7)</sup> + GAP |0.7320|0.7207|0.6516|
|X3d_s_13_6_182_160 |0.7434|0.7437|0.6815|
|X3d_m_16_5_256_224 |0.7488|0.7414|0.6849|

PCA/w<sup>(1)</sup> parameter: `/mldisk/nfs_shared_/dh/graduation_thesis/vcdb/pca_params_vcdb89325_resnet50_rmac_3840.npz`

Resnet50<sup>(2)</sup>: `moco_v2/moco_v2_200ep_pretrain`

Resnet50<sup>(3)</sup>: `moco_v2/moco_v2_800ep_pretrain`

Resnet50<sup>(4)</sup>: `moco_v3/r-50-1000ep`

Resnet50<sup>(5)</sup>: `dino/dino_resnet50_pretrain`

Resnet50<sup>(6)</sup>: `byol/pretrain_res50x1` [github](https://github.com/ajtejankar/byol-convert.git)

Resnet50<sup>(7)</sup>: `moco_v1/moco_v1_200ep_pretrain`



resnet + norm + rmac(origin)

Video-level mAP (DSVR|CSVR|ISVR): 0.3892|0.4067|0.3852
Clip-level mAP (DSVR|CSVR|ISVR): 0.7474|0.7453|0.6862

resnet + rmac(origin)
0.4329	0.4501	0.4219
0.7798	0.7758	0.7157

resnet + norm + rmac(simple)
Video-level mAP (DSVR|CSVR|ISVR): 0.3994|0.4189|0.3941
Clip-level mAP (DSVR|CSVR|ISVR): 0.7563|0.7539|0.6938

resnet +  rmac(simple)
Video-level mAP (DSVR|CSVR|ISVR): 0.4331|0.4526|0.4219
Clip-level mAP (DSVR|CSVR|ISVR): 0.7811|0.7769|0.7163


