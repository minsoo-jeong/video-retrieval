## Clip Feature Representation

### Image Based

#### Resnet50 Backbone

- R50_pretrained
- R50_MoCov1
- R50_MoCov2
- R50_MoCov3
- R50_Byol
- R50_Dino

#### Global Feature Representation

- Gap(Global Average Pooling, SPoC)
- RMAC
    - Basic
    - TCA (input norm o)
    - TCA (input norm x)
- GeM(Generalized Mean Pooling)

&rarr; from Last Conv / Intermediate Layer

#### PCA/w

- apply PCA/Whitening

#### &rarr; 72 methods: 6(backbone) * 3(gfr) * 2(layer) * 2(pca/w)

#### Experiments

#### 1. TCA framework 비교

- TCA framework mAP 높음

   ```
    # TCA evalutation
    # resnet50-irmac-tca-norm-pca
    # cosine 
    ===== FIVR-5K Dataset =====
    Queries: 50 videos
    Database: 5049 videos
    ----------------
    DSVR mAP: 0.6042
    CSVR mAP: 0.6173
    ISVR mAP: 0.5857
    # chamfer
    ===== FIVR-5K Dataset =====
    Queries: 50 videos
    Database: 5049 videos
    ----------------
    DSVR mAP: 0.8604
    CSVR mAP: 0.8479
    ISVR mAP: 0.7716
    
    # evalutation Results
    |Resnet50-IRMAC + PCA/w <sup>(1)</sup> |0.5974|0.6105|0.5812|
    |Resnet50-IRMAC + PCA/w <sup>(1)</sup> |0.8569|0.8441|0.7711|
    ```

#### 2. Backbone/Spatial Representation 비교

- Clip: R50 + last layer Global Average Pooling
- Video: Average

|  backbone  | DSVR(v) | CSVR(v) | ISVR(v) | DSVR(c) | CSVR(c) | ISVR(c) |
|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| supervised | 0.4016  | 0.4191  | 0.3943  | 0.7408  | 0.7380  | 0.6819  |
|  MoCo v1   | 0.3619  | 0.3805  | 0.3632  | 0.7320  | 0.7207  | 0.6516  |
|  MoCo v2   | 0.4189  | 0.4430  | 0.4241  | 0.7392  | 0.7411  | 0.6843  |
|  MoCo v3   | 0.4649  | 0.4858  | 0.4551  | 0.8107  | 0.8049  | 0.7427  |
|    DINO    | 0.4535  | 0.4718  | 0.4462  | 0.7960  | 0.7916  | 0.7327  |
|    BYOL    | 0.4503  | 0.4656  | 0.4333  | 0.8001  | 0.7907  | 0.7312  |

- Clip: R50 + last layer Gem(p=3)
- Video: Average

|  backbone  | DSVR(v) | CSVR(v) | ISVR(v) | DSVR(c) | CSVR(c) | ISVR(c) |
|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| supervised | 0.4063  | 0.4198  | 0.3981  | 0.7448  | 0.7427  | 0.6886  |
|  MoCo v1   | 0.3973  | 0.4158  | 0.4016  | 0.7721  | 0.7663  | 0.6977  |
|  MoCo v2   | 0.4216  | 0.4434  | 0.4271  | 0.7632  | 0.7662  | 0.7048  |
|  MoCo v3   | 0.4760  | 0.4919  | 0.4688  | 0.8400  | 0.8360  | 0.7684  |
|    DINO    | 0.4717  | 0.4832  | 0.4624  | 0.8140  | 0.8090  | 0.7503  |
|    BYOL    | 0.4676  | 0.4775  | 0.4403  | 0.8288  | 0.8166  | 0.7531  |

- Clip: R50 + last layer RMAC
- Video: Average

2. Representation
    - choose the best backbone at test # 1 (아마도 mocov3)

==> 22.08.18 까지 할것

3. PCA/w

4. transformer 학습

4. blah blah..