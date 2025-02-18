                                                                        Contents   lists   available   at   ScienceDirect
                                                                                        Methods
                                                              journal homepage: www.elsevier.com/locate/ymeth
Local  augmented  graph  neural  network  for  multi-omics  cancer  prognosis
prediction  and  analysis
Yongqing    Zhang   a, Shuwen    Xiong   a, Zixuan    Wang   a, Yuhang    Liu   a, Hong    Luo   a, Beichen    Li   a,
Quan    Zou   b,∗
a       School of Computer Science, Chengdu University of Information Technology, Chengdu, 610225, China
b       Institute of Fundamental and Frontier Sciences, University of Electronic Science and Technology of China, Chengdu, 610054, China
A  R  T  I  C  L  E               I  N  F  O                  A  B  S  T  R  A  C  T
Dataset link: https://                                        Cancer prognosis prediction and analysis can help patients understand expected life and help clinicians provide
tcga -data .nci .nih .gov /tcga/                              correct  therapeutic  guidance.  Thanks  to  the  development  of  sequencing  technology,  multi-omics  data,  and
Dataset link: https://                                        biological  networks  have  been  used  for  cancer  prognosis  prediction.  Besides,  graph  neural  networks  can
github .com /ZhangLab312 /LAGProg                             simultaneously  consider  multi-omics  features  and  molecular  interactions  in  biological  networks,  becoming
                                                              mainstream in cancer prognosis prediction and analysis. However, the limited number of neighboring genes
Keywords:                                                     in biological networks restricts the accuracy of graph neural networks. To solve this problem, a local augmented
Cancer prognosis                                              graph   convolutional   network   named   LAGProg   is   proposed   in   this   paper   for   cancer   prognosis   prediction
Survival risk prediction                                      and analysis. The process follows: ﬁrst, given a patient’s multi-omics data features and biological network,
Biomarker discovery                                           the  corresponding  augmented  conditional  variational  autoencoder  generates  features.  Then,  the  generated
Local augmentation                                            augmented features and the original features are fed into a cancer prognosis prediction model to complete the
Graph neural network                                          cancer prognosis prediction task. The conditional variational autoencoder consists of two parts: encoder-decoder.
Multi-omics data                                              In the encoding phase, an encoder learns the conditional distribution of the multi-omics data. As a generative
                                                              model, a decoder takes the conditional distribution and the original feature as inputs to generate the enhanced
                                                              features. The cancer prognosis prediction model consists of a two-layer graph convolutional neural network and
                                                              a Cox proportional risk network. The Cox proportional risk network consists of fully connected layers. Extensive
                                                              experiments on 15 real-world datasets from TCGA demonstrated the eﬀectiveness and eﬃciency       of the proposed
                                                              method in predicting cancer prognosis. LAGProg improved the C-index values by an average of 8.5% over the
                                                              state-of-the-art graph neural network method. Moreover, we conﬁrmed that the local augmentation technique
                                                              could enhance the model’s ability to represent multi-omics features, improve the model’s robustness to missing
                                                              multi-omics features, and prevent the model’s over-smoothing during training. Finally, based on genes identiﬁed
                                                              through diﬀerential expression analysis, we discovered 13 prognostic markers highly associated with breast
                                                              cancer, among which ten genes have been proved by literature review.
1.     Introduction                                                                                  available,  including  mRNA  [2],  miRNA  [3]and  copy  number  varia-
                                                                                                     tion  (CNV)  [4].  Compared  to  single-omics  data,  multi-omics  data  is
    Cancer  is  a  heterogeneous  disease  involving  complex  interactions                          beneﬁcial for capturing potential molecular correlations in the cancer
between genes and environments, leading to signiﬁcant diﬀerences in                                  prognosis.
cancer  prognosis  among  patients  with  the  same  type  of  cancer  [1].                               Many  statistical  methods  have  been  developed  in  recent  years  to
Therefore, it is necessary to develop accurate and robust computational                              predict cancer survival risk based on multi-omics data. For example, Si-
methods based on molecular features for cancer prognosis prediction                                  mon et al. proposed a Cox proportional hazards model with elastic net
and analysis. This can help patients understand expected life and help                               penalties, named Cox-EN, for eﬃcient        and stable survival risk estima-
clinicians   give   correct   therapeutic   guidance.   With   the   development                     tion [5]. Tong et al. presented a Cox proportional hazards model with
of high-throughput sequencing technology, multi-omics information is                                 Harrell’s  concordance  index-based  clustering  integrating  high  dimen-
 *     Corresponding author.
    E-mail address:   zouquan@nclab.net  (Q.   Zou).
https://doi.org/10.1016/j.ymeth.2023.02.011
Received 29 September 2022; Received in revised form 30 December 2022; Accepted 25 February 2023

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
sional multi-omics data for improving colon cancer prognosis prediction
performance [6]. Lu et al. designed a genetic algorithm-based online
gradient boosting model for incremental breast cancer prognosis [7].
These statistical approaches have been carefully developed to combine
multi-omics data for cancer prognosis prediction and analysis. However,
these approaches have limitations in accurately learning the represen-
tative features from high-dimensional and heterogeneous multi-omics
data.
    Many deep learning approaches have been successfully applied in
multi-omics cancer prognosis prediction and analysis in recent years.
For  example,  Cheerla  et  al.  combined  a  Cox  proportional  risk  model
with a deep learning model and proposed Cox-DL for pan-cancer (20
diﬀerent cancer types) prognosis prediction [8]. Lee et al. hybridized
the Cox proportional hazards model with a deep autoencoder and pro-
posed Cox-AE for the prognosis of lung adenocarcinoma [9]. Tong et al.
developed a concatenation autoencoder to preserve modality-unique in-
formation from each single-omics data. Then they used a cross-modality
autoencoder to achieve the modality-invariant representation for breast
cancer  survival  analysis  [10].  However,  traditional  autoencoders  are
susceptible  to  data  noise.  To  address  this  problem,  Chai  et  al.  em-
ployed the denoising autoencoder. They proposed a framework named                             Fig.        1.   An example of a gene interaction network in breast cancer. As can be
DCAP  for  accurate  cancer  prognosis  prediction.  It  ﬁrst  captures  the                  seen from the ﬁgure, the EGFR node has nine neighbors, while the HGF node
robust  representation  of  the  multi-omics  data  through  the  denoising                   has only two neighbors.
autoencoder,  then  predicts  survival  risks  based  on  learned  represen-
tative features [11]. However, multi-omics features are the outcomes                              To verify the eﬀeteness of LAGProg on the prediction of cancer sur-
of molecular interactions, which belong to the non-Euclidean manifold                         vival risks and the discovery of prognostic biomarkers, we conducted
represented by biological networks. Such biological networks include                          extensive experiments on 15 cancers from The Cancer Genome Atlas
the protein-protein interaction (PPI) network [12], the Kyoto Encyclo-                        (TCGA) database. By comparison, LAGProg averagely improved C-index
pedia of Genes and Genomes (KEGG) network [13], etc. Deep learning                            values by 8.5% over the state-of-the-art method GraphSurv in predict-
methods  based  on  Euclidean  manifolds  are  limited  to  learning  non-                    ing cancer survival risks. Besides, based on genes identiﬁed diﬀerential
Euclidean features in cancer prognosis prediction and analysis.                               expression analysis, we discovered 13 prognostic markers (F7, AGR3,
    Fortunately, Graph Neural Network (GNN) is an eﬀective method                             NAT1,  TFF3,  TFF1,  GFRA1,  GYP2B7P1,  AGR2,  GP2,  KCNJ3,  VGLL1,
to deal with non-Euclidean manifolds [14]and has been widely used                             STAC2, FABP7) highly associated with breast cancer, among which ten
in bioinformatics. However, few studies have applied GNN to cancer                            genes have been proved by literature review.
prognosis  prediction  tasks.  Wang  et  al.  developed  a  Graph  Survival                       In Summary, our main contributions to this article are threefold: (1)
Network,  GraphSurv,  by  integrating  the  Graph  Convolution  Network                       We presented a local augmented graph convolutional network frame-
(GCN) with the Cox proportional hazards model for pan-cancer prog-                            work for cancer prognosis prediction and analysis by considering the
nosis analysis [15]. The key of GraphSurv is to introduce association                         KEGG  pathway  and  multi-omics  molecular  features.  (2)  We  applied
relationships between genes in biological networks as a priori informa-                       a Conditional Variational AutoEncoder for Local Augmentation to en-
tion into the model to construct a non-Euclidean manifold. The GCN                            hance  the  prediction  power  of  the  graph  convolutional  network  by
and Cox proportional risk model is then used to predict cancer prog-                          better learning the interaction of the genes. (3) We conducted exten-
nosis. However, due to some genes’ limited number of neighbors, local                         sive experiments on 15 cancer datasets to validate the eﬀectiveness of
neighbors’ multi-omics features still need to be improved to represent                        our approach in survival risk prediction and discovered 13 prognostic
the molecular interactions between diﬀerent genes (Fig.   1). Therefore,                      markers of breast cancer by diﬀerential expression analysis.
we argue that the limited number of neighbor genes restricts the accu-
rate prediction of cancer survival risk and the discovery of prognostic                       2.     Materials and methods
biomarkers.
    This  paper  proposes  LAGProg,  a  local  augmented  graph  convolu-
tional  network-based  framework  for  cancer  prognosis  prediction  and                     2.1.     Data preparation and preprocessing
analysis.  Speciﬁcally,  given  any  patient’s  multi-omics  data  and  bio-
logical network, the corresponding augmented conditional variational                              To construct our framework, we collected the mRNA, CNV, and DNA
autoencoder  generates  features.  Then,  the  generated  augmented  fea-                     methylation data from the TCGA database and the KEGG pathway map
tures and the original features are fed into a cancer prognosis prediction                    from the KEGG database. The statistics of the TCGA datasets and the
model to complete the cancer prognosis prediction task. The conditional                       number of features per data type are summarized in Table   1. Details of
variational autoencoder consists of two parts: encoder-decoder. In the                        the datasets are as follows:
encoding phase, an encoder learns the conditional distribution of the                             TCGA.       We    downloaded    15    level-3    cancer    datasets    from    TCGA
multi-omics  data.  As  a  generative  model,  a  decoder  takes  the  condi-                 (https://portal .gdc .cancer .gov/) through the R package “TCGA-Assem-
tional  distribution  and  the  original  feature  as  inputs  to  generate  the              bler 2” version 2.0.6 [16]. Each dataset contains three types of multi-
enhanced  features.  To  ensure  that  the  generated  augmented  features                    omics data, mRNA, CNV, and DNA methylation. mRNA data indicates
are consistent with the original features, two target functions used in                       RNA sequencing data generated by the UNC Illumina HiSeq_RNASeq.
this paper. The BCE loss and Kullback Leibler (KL) divergence penalty                         CNV data denotes copy number variation generated by the BROAD-MIT
are used to constrain the diﬀerence between the generated enhanced                            Genome-wide SNP_6. DNA methylation data was generated by the USC
features and the original ones. The cancer prognosis prediction model                         HumanMethylation450.  Then  we  pre-processed  the  multi-omics  data
consists of a two-layer graph convolutional neural network and a Cox                          following the previous study [11].
proportional risk network. The Cox proportional risk network consists                             mRNA. To quantify the expression level of each gene in each sample,
of fully connected layers.                                                                    we  used  the  dataset  from  TCGA.  For  each  gene,  the  expression  was

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
              Table      1                                                                                                                  •   Cox-GCN.      Cox-GCN   employs   a   graph   convolutional   network   to
              The statistics of the TCGA dataset.                                                                                              compress and recode multi-omics and the generated features (de-
                 Cancer                 Samples                 UncensoredUncensored            Feature                                        rived from LACVAE) and then inputs them to the Cox proportional
                                                                         Rate                   Number                                         hazard layer for predicting cancer prognosis.
                 BLCA                       418                                   110                                                  26.3%                                        15487
                 BRCA                      658                                   72                                                       10.9%                                        15951In the following, we detail the two components sequentially.
                 CESC                       296                                   58                                                       19.6%                                        15584
                 COAD                    255                                   36                                                       14.1%                                        156562.2.1.     Conditional variational AutoEncoder for local augmentation
                 ESCA                       189                                   58                                                       30.7%                                        16027Supposing  the  number  of  genes  is   𝑁     .  The  input  of  the  LACVAE
                 HNSC                     493                                   165                                                  33.5%                                        15755is the interaction network of gene products from the KEGG pathway,
                 LGG                            520                                   97                                                       18.7%                                        16155
                 LIHC                         397                                   109                                                  27.5%                                        14937which is deﬁned as a graph with 𝑁         genes as nodes and mRNA, CNV,
                 LUAD                     452                                   105                                                  23.2%                                        15876and DNA methylation as node features. Thus, the feature matrix of the
                 LUSC                       335                                   93                                                       27.8%                                        16088input graph can be deﬁned as 𝑿         ∈        ℝ     𝑁     ×3. Given a central node 𝑔, LAC-
                 MESO                    87                                        58                                                       66.7%                                        15801VAE exploits the conditional variational autoencoder [17]to learn the
                 PAAD                     181                                   59                                                       32.6%                                        16334conditional distribution of the gene multi-omics features of connected
                 SARC                      254                                   72                                                       28.3%                                        15323
                 SKCM                     449                                   151                                                  33.6%                                        15267neighbors  𝑢      (𝑢 ∈       𝑁     𝑔 ). The symbol  𝑁     𝑔      denotes the set of neighboring
                 STAD                      366                                   72                                                       19.7%                                        15959nodes   of   the   given   central   node   𝑔.   Let   𝑿     𝒈      ∈          ℝ     1×3      and   𝑿     𝒖      ∈          ℝ     𝐷   𝑔 ×3
                 Total                        5350                              1315                                            24.6%                                        -represent the feature matrices of node g and its connected neighbors,
                                                                                                                                       respectively. Where 𝐷    𝑔    denotes the node degree of node 𝑔. Speciﬁcally,
transformed through the log function. If the expression of a gene was                                                                  LACVAE ﬁrst generates the latent variable 𝒛   from the prior Gaussian dis-
not measured, then the missing value was set to zero.                                                                                  tribution 𝑝𝜃 (𝒛|𝑿     𝒖 ,𝑿     𝒈 ). The latent variable 𝒛    represents the distribution
      CNV.   Gene-associated   copy   number   variations   were   downloaded                                                          of the node representations of g and its neighbors. LACVAE then gener-
from TCGA. The copy number values were already transferred by base2                                                                    ates the data 𝑿     𝒖   by the generative distribution 𝑝𝜃 (𝑿     𝒖  |𝑿     𝒈 , 𝒛)  conditioned
log(copyNumber/2),   centered   on   0.   Gene   coordinates   identiﬁed   the                                                         on 𝒛   and 𝑿     𝒈 . Let 𝜙     and 𝜃   denote the variational and the generative pa-
original copy number values. Then, we used the gene coordinates to                                                                     rameters, respectively. The evidence lower bound (ELBO) is deﬁned as
calculate an average copy number of each gene in each sample and out-                                                                  follows:
                                                                                                                                                 (  𝑿     𝑢,  𝑿     𝑔 ;  𝜃, 𝜙   )    =−  𝐾𝐿               (  𝑞𝜙(  𝒛   ∣𝑿     𝑢,  𝑿     𝑔)       ‖   𝑝𝜃(  𝒛   ∣𝑿     𝑔))
put the gene-level CNA data.
      DNA methylation. We collected DNA methylation data from TCGA.                                                                                                               ∑𝐿              (  𝑿     𝑢   ∣𝑿     𝑔 ,  𝒛(𝑙) )                                                                                                                 (1)
The original methylation values were represented by a beta (𝛽) value                                                                                                      +       1𝐿    log  𝑝𝜃
and identiﬁed by a CpG site. For each gene, we deﬁned a window as the                                                                                                             𝑙=1
±1,500 base pair region ahead of a Transcription Start Site. We then                                                                   where 𝒛(𝑙)    =   𝑔𝜙  (𝑿     𝒈 , 𝑿     𝒖 , 𝜖(𝑙)), 𝜖(𝑙)    ∼                  (0, 𝐈), and 𝐿     is the number of neigh-
averaged the beta (𝛽) values of all CpG sites within the deﬁned window                                                                 boring  nodes of node 𝑔. In the training stage, the objective of LACVAE is
to compute the average promoter methylation per gene.                                                                                  to use the neighboring pairs (𝑿     𝑔 , 𝑿     𝑢)   to maximize the ELBO. In the gen-
      In each cancer data, we removed features that were missing in more                                                               eration stage, LACVAE uses the feature 𝑿     𝑔    as the condition and samples
than  20%  of  the  patients  and  then  removed  patient  samples  if  they                                                           a hidden variable 𝒛    as input for the decoder to generate feature matrix
missed  more  than  20%  of  the  remaining  multi-omics  features.  Then,                                                             𝑿     𝑔    ∈        ℝ     1×3    associated with node 𝑔.
missing values of genes from all omics types were set to zero. Subse-
quently, Z-score normalization was performed on all features. Finally,                                                                 2.2.2.     Graph convolutional network with Cox proportional hazard layer
we used the common features of mRNA, CNV, and DNA methylation.                                                                               LAGProg takes the original multi-omics feature matrix  𝑿         and the
      KEGG.   The KEGG pathway map is a molecular interaction network                                                                  generated  feature  matrix 𝑿          as  input.  Cox-GCN  only  makes  a  small
diagram represented in terms of the KEGG Orthology groups [13]. We                                                                     change on the ﬁrst graph convolutional layer of the GCN. It directly
downloaded  the  KEGG  pathway  from  https://www .genome .jp /kegg/                                                                   sums 𝑿        and 𝑿        as the input. Then, Cox-GCN obtains the new embed-
and generated the interaction network of gene products from the path-                                                                  ding of gene nodes through two graph convolutional layers [18].
way through the R package “parseKGML2Graph”. After removing re-
dundant interactions, the network contains 6336 nodes and 71455 in-                                                                    𝑯       (𝟏)   =RELU( ̃𝑷    (𝑿         +𝑿     )𝑾          (𝟎))                                                                                                                                                                                                            (2)
teractions.                                                                                                                            𝑯       (𝟐)   =SELU( ̃𝑷𝑯         (𝟏)𝑾          (𝟏))                                                                                                                                                                                                                                (3)
      Concatenation of the omics and network features.    We built an
undirected graph where each node corresponds to a gene in the KEGG                                                                     where    ̃𝑷        =      ̃𝑫      −     𝟏𝟐   𝑨      ̃𝑫      −     𝟏𝟐  .     ̃𝑫         and  𝑨        indicate the degree matrix and the
pathway of choice and edges between genes to pathway interactions.                                                                     adjacency matrix of the graph. 𝑾          (𝟎)   and 𝑾          (𝟏)   denote the parameter ma-
Each gene was assigned the three omics types, mRNA, CNV, and DNA                                                                       trix of the ﬁrst and the second graph convolutional layers, respectively.
methylation computed for each cancer type.                                                                                             𝑯       (𝟏)    and 𝑯       (𝟐)    refer to the output of the ﬁrst and the second graph con-
                                                                                                                                       volutional layers, respectively.  RELU(.)     and  SELU(.)     deﬁne the RELU
2.2.     The architecture of LAGProg                                                                                                   and SELU functions, respectively. In practice, Cox-GCN uses  ̃𝑨       =   𝑨       +   𝑰
                                                                                                                                       instead of 𝑨      to preserve the node features.
      In this section, we introduce the proposed LAGProg framework in                                                                        Finally, Cox-GCN inputs the embeddings compressed by the graph
detail. The architecture of LAGProg is shown in Fig.   2. The framework                                                                convolutional layers into the Cox proportional hazard layer to predict
comprises Conditional Variational AutoEncoder for Local Augmentation                                                                   the survival risk of suﬀerers. This layer is a multi-layer perceptron with
(LACVAE)  and  Graph  Convolutional  Network  with  Cox  proportional                                                                  a non-linear proportional hazard objective function based on the Cox
hazard layer (Cox-GCN) to predict patients’ risk scores.                                                                               proportional hazard model [19]. Given the embeddings and the survival
                                                                                                                                       time of the suﬀerer 𝑖 and 𝑗, the objective function is deﬁned as follows:
    •   LACVAE.     Given  a  gene  and  corresponding  multi-omics  features                                                                                 ∑
        (mRNA, CNV, and DNA methylation), LACVAE learns the condi-                                                                     𝐿    =−                    1𝑁     𝐸   =1(𝛽𝑯       (𝟐)𝒊        −log                        ∑𝑒𝛽𝑯     (𝟐)𝒋        )                                                                                                                                         (4)
        tional distribution of the multi-omics features with the neighbor                                                                                   𝑖∶𝐸  𝑖=1                       𝑗∶𝑡𝑗>𝑡𝑖)
        genes. According to the learnt distribution, LACVAE generates fea-                                                             where  𝑁     𝐸   =1     indicates the number of uncensored suﬀerers.  𝛽     denotes
        tures associated with the given gene as the additional input.                                                                  the parameter vector of the multi-layer perceptron. 𝑯       (𝟐)𝒊          and 𝑯       (𝟐)𝒋          refer

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
Fig.       2.   The architecture of the proposed LAGProg framework. The top part is the Conditional Variational AutoEncoder for Local Augmentation (LACVAE), which is
useful for obtaining the generated features 𝑥′. The bottom part is Graph Convolutional Network with Cox proportional hazard layer (Cox-GCN) to predict patients’
risk scores.
to the embeddings of the suﬀerer 𝑖 and 𝑗, respectively. 𝑡𝑖  and 𝑡𝑗   are the                              2.4.     Training and inference
survival time of the suﬀerer 𝑖 and 𝑗. The Adam algorithm [20]is used
to optimize the parameters of the LAGProg.                                                                    The training and inference processes are summarized in Algorithm  1.
                                                                                                          During the training process, LAGProg ﬁrst trains the LACVAE and then
2.3.     Prognostic biomarker discovery                                                                   samples one feature matrix generated by LACVAE as additional input to
                                                                                                          train the Cox-GCN. Supervised loss functions are computed on the orig-
                                                                                                          inal multi-omics feature matrix 𝑿        and the generated feature matrix 𝑿    .
    To  discover  the  prognostic  biomarkers  of  cancers,  we  ﬁrst  distin-                            The Adam optimizer is used for backpropagation on 15 TCGA datasets,
guish high-risk suﬀerers from low-risk suﬀerers according to the median                                   where  the  L2  norm  regulation  coeﬃcient         and  momentum  are  set  to
survival risk forecasted by the LAGProg. Then, we apply the diﬀerential                                   1.2 ×10−4     and 0.7, respectively. During the validation process, LAGProg
expression analysis to identify genes with the most signiﬁcant expres-                                    resamples another feature matrix that is diﬀerent from the one to com-
sion diﬀerences according to the distinguished sub-groups. These genes                                    pute the training loss to compute the validation loss. At the inference
are  considered  possible  biomarkers  that  may  aﬀect  cancer  suﬀerers’                                process, LAGProg is unnecessary to generate 𝑿       again. Because LAGProg
prognostic. In practice, the diﬀerential expression genes are detected                                    can select 𝑿       with the minor average validation loss on TCGA datasets.
by the “limma” package in R [21]. Speciﬁcally, genes with |log2(fold                                          Considering  the  small  sample  size  of  some  datasets,  we  combine
change)| >2 and corrected p-value <   0.05 are regarded as the potential                                  bootstrapping with K-fold cross-validation to reduce the eﬀects of the
biomarkers. The log2(fold change) is deﬁned as follows:                                                   diﬀerences in training size. For each cancer type 𝐷   , we use the boot-
                                                                                                          strapping algorithm to randomly sample 10% of the samples from 𝐷       at
log2 (f old   change)   =   log2   ExphExpl                                                                                                                                                                                                                    (5)a time and then repeat this process ten times to obtain a new dataset
                                                                                                          𝐷    ′. The samples in  𝐷    ′    are trained and validated using 10-fold cross-
where 𝐸𝑥𝑝    ℎ    and 𝐸𝑥𝑝    𝑙  indicate the average gene expression of high-risk                         validation.  The  unsampled  samples  in  D  are  used  as  the  test  set  for
and low-risk suﬀerers, respectively.                                                                      testing.

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
    In the experiment, we implement LAGProg based on PyTorch [22].
All the experiments are conducted on a Windows computer with 32GB
memory and NVIDIA GeForce GTX 1660 SUPER.
Algorithm 1 Local Augmented Graph Convolutional Network for Can-
cer Prognosis Prediction and Analysis.
Require:    Adjacency matrix 𝑨     and multi-omics feature matrix 𝑿
Ensure:    Predicted Survival risk 𝒛
  Train LACVAE by Eq. (1), given 𝑨     and 𝑿      as input
  while  not convergence      do
     for   𝑠  in range (𝑆  )      do
        Generate the augmented feature matrix 𝑿    𝒔
        Predict survival risk 𝑧𝑠   using Cox-GCN by Eq. (2)and Eq. (3)
     end      for
     Compute the training loss by proportional hazard objective function by Eq. (4)
     Update parameter Θ    by Adam
     Resample the augmented feature matrix 𝑿
     Compute the validation loss by proportional hazard objective function by Eq. (4)
  end      while                                                                                     Fig.        3.   The contribution of each omics data for cancer outcome evaluation by
  Predict survival risk 𝒛   using Cox-GCN by Eq. (2)and Eq. (3), Cox-GCN selects 𝑿      with         using the individual type of omics data or excluding one type from the ﬁnal
  the smallest validation loss                                                                       model. The curves represent the C-index values, and the histogram represents
                                                                                                     the change in C-index values.
2.5.     Baseline methods                                                                            proportion of all pairs of suﬀerers whose predicted survival times are
    We  compared  the  cancer  prognosis  prediction  performance  with                              correctly ordered based on Harrell’s C statistic. The C-index is deﬁned
statistical approaches (Cox-PCA, Cox-EN, and Cox-HC), deep learning                                  as follows.
approaches  (Cox-AE,  ConcatAE,  and  DCAP),  and  a  graph  neural  net-                            C   −   index   =        1∑     ∑    𝐼 [𝑟𝑖 >𝑟𝑗]                                                                                                                                                                  (6)
work approach (GraphSurv). We carefully tuned these approaches to                                                   𝑛  𝑖𝜀{1…     𝑛|𝛿𝑖=1}𝑡𝑖>𝑡𝑗
ensure a fair comparison and reported the best performance.                                          where 𝑟𝑖   and 𝑟𝑗     indicates the predicted risks for suﬀerers 𝑖  and 𝑗, re-
    Cox-PCA.    It reconstructs multi-omics data by PCA and then inputs                              spectively. 𝑡𝑖   and 𝑡𝑗    denotes the true survival times for suﬀerers 𝑖  and
it into the Cox proportional hazards model for predicting cancer prog-                               𝑗, respectively. 𝛿    refers to whether the suﬀerer is uncensored. 𝑛    is the
nosis.                                                                                               number of the matching suﬀerer pairs. 𝐼 [.]    deﬁnes the indicator func-
    Cox-EN.  It is a variant of the Cox proportional hazards model, which                            tion. The C-index ranges from 0 to 1, and a higher C-index means a
employs  𝑙1     and  𝑙2     penalties for regularization and the coordinate de-                      better prediction performance.
scent for model optimization.
    Cox-HC.   It uses Harrell’s concordance index-based clustering to re-                            3.     Results
duce the dimension of multi-omics data and constructs the prognostic
model by integrating low-dimensional data.                                                           3.1.     LAGProg can estimate survival risks by multi-omics data
    Cox-AE.   It combines multi-omics data by an autoencoder and then
inputs it into the univariate Cox proportional hazards model for analysis                                As shown in Table   2, LAGProg achieves the essentially identical C-
of cancer prognostication. There are ﬁve layers in the model; one is the                             index for the 10-fold cross-validation and independent tests with the
input layer, three hidden layers (1000, 500, 1000) of the middle are the                             average C-index of 0.718 and 0.715 for 15 cancer datasets, respectively.
bottleneck layer, and the last is the output layer.                                                  The results indicate the generalization capability of LAGProg.
    ConcatAE.   It separately maps each single-omics data into a hidden                                  We further performed the feature importance analysis to detail each
space by an independent autoencoder and then concatenates the fea-                                   omics  type’s  contribution  in  LAGProg.  As  shown  in  Fig.    3,  when  us-
tures from each hidden space for cancer survival analysis. It uses the                               ing single-omics data, the mRNA serves best with an average C-index
concatenation autoencoder to integrate the complementary information                                 of  0.684,  followed  by  DNA  methylation  with  an  average  C-index  of
from each data modality. Then, it uses the cross-modality autoencoder                                0.677, and CNV has the lowest performance with an average C-index
to integrate the consensus information from each data modality.                                      of 0.622. Consistently, when excluding single-omics data from the LAG-
    DCAP.   It captures representative features of multi-omics data by the                           Prog, mRNA leads to the most signiﬁcant decrease in C-index by 0.014,
unsupervised denoising autoencoder (DAE) and then uses these features                                while CNV caused the smallest decrease by 0.004. These results indicate
to estimate cancer risks by the Cox model. The DAE is a deep neural                                  that mRNA plays the most crucial role in cancer prognosis prediction,
network with three hidden layers [500, 200, 500]. The DAE was trained                                while CNV makes a minor contribution.
by backpropagation via the Adam optimizer and RMSE were used as the
loss function.                                                                                       3.2.     LAGProg can signiﬁcantly         outperform all baselines in prognosis
    GraphSurv.   It introduces the KEGG pathway as a priori knowledge                                prediction
into the network, uses GCN to recode multi-omics data, and then inputs
it into the Cox layer for predicting cancer prognosis. The model consists                                Table    3    summarizes the comparison results over 15 TCGA datasets
of a GCN extraction module and a deep proportional hazard prediction                                 regarding C-index. The main observations from Table   3   are as follows.
module. The GCN extraction model comprises two graph convolutional
layers, and the deep proportional hazard prediction module comprises                                    ∙    The deep learning approaches (Cox-AE, ConcatAE, and DCAP) can
four fully connected layers.                                                                               consistently outperform the statistical ones (Cox-PCA, Cox-EN, and
                                                                                                           Cox-HC). This indicates that deep learning approaches can more ac-
2.6.     Evaluation metrics                                                                                curately capture the representative features from high-dimensional
                                                                                                           and heterogeneous multi-omics data than statistical approaches.
    In cancer prognosis prediction, the performance is often measured                                   ∙    GraphSurv     (a  graph  neural  network  approach)  can  signiﬁcantly
by  the  concordance  index  (C-index)  [23].  The  C-index  indicates  the                                outperform the deep learning approaches (Cox-AE, ConcatAE, and

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
                                                                     Table      2
                                                                     The C-index of the cross-validations and tests on 15 cancers by LAGProg.
                                                                          Cancer                 Validation                 Test                    Test                                           Cancer                      Validation                 Test                  Test
                                                                                                                                                    (95% conﬁdence)                                                                                                             (95% conﬁdence)
                                                                          BLCA                       0.703                                    0.674                 0.621-0.728                                          LUAD                          0.701                                    0.667                 0.583-0.751
                                                                          BRCA                      0.684                                    0.704                 0.660-0.748                                          LUSC                            0.714                                    0.625                 0.567-0.683
                                                                          CESC                       0.727                                    0.740                 0.675-0.805                                          MESO                         0.767                                    0.809                 0.752-0.867
                                                                          COAD                    0.757                                    0.774                 0.732-0.861                                          PAAD                          0.766                                    0.736                 0.650-0.821
                                                                          ESCA                       0.724                                    0.729                 0.666-0.792                                          SARC                            0.693                                    0.720                 0.655-0.785
                                                                          HNSC                     0.655                                    0.654                 0.585-0.723                                          SKCM                          0.623                                    0.619                 0.590-0.649
                                                                          LGG                            0.785                                    0.815                 0.771-0.858                                          STAD                           0.706                                    0.710                 0.604-0.818
                                                                          LIHC                         0.762                                    0.756                 0.724-0.788                                          Average                 0.718                                    0.715                 0.647-0.787
                                                                          Table      3
                                                                          Performance comparisons by C-index achieved in 15 TCGA cancer datasets.
                                                                              Cancer                      Cox-PCA                 Cox-EN                 Cox-HC                 Cox-AE                 ConcatAE                 DCAP                    GraphSurv                 LAGProg
                                                                              BLCA                            0.582                              0.605                        0.611                         0.626                        0.634                                  0.646                    0.624                                                                                    0.674
                                                                              BRCA                           0.603                              0.611                        0.616                         0.653                        0.658                                  0.662                    0.696                                                                                    0.704
                                                                              CESC                             0.595                              0.633                        0.647                         0.661                        0.672                                  0.685                    -                                                                                                                                  0.740
                                                                              COAD                         0.568                              0.580                        0.591                         0.628                        0.622                                  0.622                    -                                                                                                                                  0.774
                                                                              ESCA                            0.557                              0.564                        0.572                         0.571                        0.584                                  0.594                    -                                                                                                                                  0.729
                                                                              HNSC                          0.553                              0.573                        0.580                         0.602                        0.608                                  0.628                    0.637                                                                                    0.654
                                                                              LGG                                 0.691                              0.719                        0.731                         0.805                        0.797                                  0.823                    0.837                                                                                    0.815
                                                                              LIHC                              0.593                              0.615                        0.629                         0.703                        0.701                                  0.710                    0.684                                                                                    0.756
                                                                              LUAD                          0.559                              0.573                        0.583                         0.612                        0.621                                  0.629                    0.646                                                                                    0.667
                                                                              LUSC                            0.541                              0.554                        0.559                         0.582                        0.580                                  0.597                    0.529                                                                                    0.625
                                                                              MESO                         0.660                              0.675                        0.708                         0.752                        0.747                                  0.765                    -                                                                                                                                  0.809
                                                                              PAAD                          0.562                              0.591                        0.606                         0.645                        0.636                                  0.665                    -                                                                                                                                  0.736
                                                                              SARC                            0.585                              0.597                        0.631                         0.706                        0.694                                  0.719                    0.695                                                                                    0.720
                                                                              SKCM                          0.554                              0.568                        0.595                         0.631                        0.638                                  0.644                    0.646                                                                                    0.619
                                                                              STAD                           0.559                              0.568                        0.571                         0.577                        0.589                                  0.591                    0.592                                                                                    0.710
                                                                              Average                 0.584                              0.602                        0.615                         0.650                        0.652                                  0.665                    0.659                                                                                    0.715
                                                                              P-value                    1.7E-9                          1.3E-8                     8.2E-8                      7.4E-5                     6.8E-5                              9.3E-4                 4.6E-3                                                                             -
           DCAP)  over  most  TCGA  datasets.  This  indicates  that  the  graph
           neural network approach can learn the molecular interactions rep-
           resented   by   the   KEGG   network.   However,   GraphSurv      performs
           worse than the deep learning approaches over BLCA, LIHC, LUSC,
           and    SARC    datasets.    This    is    because    genes    (in    the    KEGG    net-
           work)   in   such   datasets   have   limited   neighborhoods,   which   re-
           stricts the representation of the multi-omics features of the central
           genes.
      ∙    Across  most  of  the  datasets  (apart  from  LGG  and  SKCM),  LAG-
           Prog can achieve the highest C-index between 0.619 (SKCM) and
           0.815   (LGG),   with   an   average   of   0.715.   Compared   to   Graph-
           Surv,  LAGProg  can  improve  C-index  by  8.5%  on  average.   This
           indicates    that    combining    generated    features    (from    local    aug-
           mentation)   with   original   multi-omics   features   can   enhance   the
           power of the graph neural network to learn the representation of                                                                                                                           Fig.         4.    Performance comparison between LAGProg with variants. The curves
           genes.                                                                                                                                                                                     represent the C-index values of each variant, and the bars indicate the number
                                                                                                                                                                                                      of parameters for each variant.
3.3.     Local augmentation can improve the performance of prognosis
prediction
                                                                                                                                                                                                               We compared our LAGProg with three variants (LOWFC, LAWFA,
        To demonstrate our local augmentation, we decomposed our LAG-                                                                                                                                 and  COAFA).  The  performance  of  these  three  variants  and  our  ﬁnal
Prog with various variants, including the following.                                                                                                                                                  model is shown in Fig.   4. The main observations are threefold.
      ∙         Leverage            Original            features            Without            Feature            Combination                                                                                ∙    LOWFC  can  perform  well  compared  to  LAWFA,  which  indicates
           (LOWFC).    It prunes LACVAE and only uses original multi-omics                                                                                                                                        that the augmented features contain less useful information than
           features as the input to train Cox-GCN.                                                                                                                                                                the original multi-omics features.
      ∙         Leverage Augmented features Without Feature Augmentation                                                                                                                                     ∙    By consideration of original multi-omics features and augmented
           (LAWFA).    It prunes the multi-omics features from Cox-GCN and                                                                                                                                        features, i.e., COAFA and LAGProg, we observed them have better
           only uses augmented features as the input to train Cox-GCN.                                                                                                                                            performance than LOWFC and LAWFA, which reveals the inﬂuence
      ∙         Combine   Original   features   with   Augmented   features   as   Fea-                                                                                                                           of  the  local  augmentation  on  enhancing  the  power  of  the  graph
           ture  Augmentation  (COAFA).    It is a variant of LAGProg, which                                                                                                                                      neural network to learn the representation of genes.
           concatenates original features and augmented features (from LAC-                                                                                                                                  ∙    LAGProg    can    signiﬁcantly    outperform    COAFA    with    fewer    pa-
           VAE) on the second dimension as the input to train Cox-GCN.                                                                                                                                            rameters, which indicates that the summation operation is more

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
                                                                                                                  Fig.       6.   The C-index of GCN and LAGProg on diﬀerent graph convolutional lay-
Fig.     5.  Summary of results of GCN and LAGProg on recovering study in terms of                                ers.
C-index. The curves represent the C-index values, and the histogram represents
the change in C-index values.                                                                                                       Table      4
                                                                                                                                    The identiﬁed prognostic markers in breast cancer.
      appropriate than the concatenation operation for the feature com-                                                                Gene                                        logFC                    AveExp                 P-value                        Ref
      bination.                                                                                                                        VGLL1                                 2.251                    -0.685                      3.90E-14                 [24]
                                                                                                                                       FABP7                                 2.111                    -0.795                      4.05E-10                 [25]
3.4.     Local augmentation can improve the robustness of missing multi-omics                                                          STAC2                                 2.057                    2.454                         3.35E-10                 [26]
data                                                                                                                                   AGR3                                     -2.998                 4.544                         7.60E-19                 [27]
                                                                                                                                       TFF1                                        -2.989                 4.222                         8.16E-15                 [28]
     To verify whether local augmentation is robust against missing in-                                                                CYP2B7P1                 -2.737                 3.410                         1.52E-14                 -
formation in the multi-omics features, we masked a certain percentage                                                                  TFF3                                        -2.364                 5.686                         2.07E-16                 [29]
                                                                                                                                       GP2                                            -2.244                 1.699                         5.18E-12                 [30]
(10%, 20%, 30%, and 40%) of the attributes of each multi-omics fea-                                                                    AGR2                                     -2.119                 7.142                         4.83E-14                 [31]
ture   vector.   We   used   the   same   strategy   for   local   augmentation   for                                                  KCNJ3                                 -2.092                 -0.281                      3.95E-09                 [32]
the masked feature matrix. As shown in Fig.    5, we can observe from                                                                  F7                                                   -2.091                 1.183                         2.21E-21                 -
the two curves that as the mask ratio increases, the gap of the aver-                                                                  GFRA1                                -2.081                 6.489                         1.12E-14                 [33]
                                                                                                                                       NAT1                                     -2.033                 5.207                         2.16E-20                 [34]
age C-index between the GCN and the LAGProg consistently enlarges,
which conﬁrms that the local augmentation can complement the miss-
ing multi-omics information of the local neighborhoods in the biological                                          and  195  diﬀerentially  expressed  genes  are  down-regulated.  Thirteen
network. In addition, it can be observed from the histogram that the C-                                           genes (|log2    (fold change)| >2 and P-value < 0.05) are selected as the
index of GCN changes increasingly with increasing mask ratio, while                                               possible cancer prognostic biomarkers. The heat map drawn based on
the C-index of LAGProg remains relatively small. This phenomenon in-                                              the expression of these genes is shown in Fig.   7B. As shown in Table   4,
dicates that local augmentation can improve the model’s robustness to                                             the  literature  review  survey  demonstrated  that  10  (76%)  genes  had
missing multi-omics features.                                                                                     been proven to be related to breast cancer prognosis (VGLL1, FABP7,
                                                                                                                  STAC2, AGR3, TFF1, TFF3, AGR2, KCNJ3, GFRA1, and NAT1). Among
3.5.     Local augmentation can alleviate the over-smoothing problem                                              these 13 genes, GP2 is associated with speciﬁc breast cancer suﬀerers’
                                                                                                                  subgroups with improved disease-free clinical outcomes. Therefore, re-
     To discuss how local augmentation can prevent the over-smoothing                                             searchers can further investigate the potential use of GP2 in vaccination
problem of Cox-GCN in cancer prognosis prediction, we analyzed the re-                                            strategies. As shown in Fig.   7C, the discovered diﬀerentially expressed
sults of LAGProg and Cox-GCN with distinct graph convolutional layers.                                            genes are signiﬁcantly enriched in the KEGG pathways related to breast
Fig.    6    shows the C-index variation with increasing the graph convolu-                                       cancer.
tional layers (2, 3, 4, 5, and 6). We can observe from the two curves that
with the increase of the graph convolutional layers, the gap in C-index                                           3.7.     Feature number sensitivity
between Cox-GCN and LAGProg is more signiﬁcant. This shows that our
model is less aﬀected by the number of the graph convolutional layers                                                  We investigated the sensitivity of the diﬀerent number of features
with the help of local augmentation. The histogram shows that the C-                                              for cancer prognosis prediction. To achieve this goal, the threshold for
index diﬀerence between GCN and LAGProg increases from 0.056 to                                                   discarding features was set to 10%-50% when dealing with missing val-
0.182 when the number of graph convolution layers increases from 2 to                                             ues. If the missing rate of a feature exceeded the threshold, the feature
6. The results indicate that though local augmentation is not specially                                           was discarded. The experiment results are shown in Fig.  8. It can be seen
designed for solving over-smoothing, it can enrich local neighborhood                                             that the model’s performance increases when the threshold increases
information and thus can enhance the locality of the gene representa-                                             from 10% to 20% because more features are capable of encoding more
tions. Therefore, local augmentation can alleviate the over-smoothing                                             information. However, as the threshold keeps rising, the performance
problem.                                                                                                          tends to drop because the model might suﬀer from overﬁtting and thus
                                                                                                                  performs poorly. So the threshold was ﬁnally set to 20% when dealing
3.6.     LAGProg can discover novel prognostic biomarkers                                                         with missing values.
     We performed diﬀerential expression analysis on breast cancer to                                             4.     Discussion
verify  the  eﬀectiveness  of  LAGProg  on  cancer  prognostic  biomarkers
discovery. Based on the sub-groups (low-risk and high-risk) grouped by                                                 The  graph  neural  network  has  been  used  for  multi-omics  cancer
LAGProg, we discovered 331 diﬀerentially expressed genes that met the                                             prognosis prediction and analysis, which can capture informative rep-
conditions of |log2    (fold change)| >1 and corrected p-value <    0.05. As                                      resentation through aggregating multi-omics features from local neigh-
shown in Fig.    7A, 136 diﬀerentially expressed genes are up-regulated,                                          borhoods (genes) in the KEGG network. However, the local neighbor-

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
Fig.        7.   Biomarker discovery. (A) Diﬀerentially expressed gene selection in breast cancer. The red and blue nodes repressively represent the selected up-regulated
and down-regulated diﬀerential expression genes. (B) The heat map of discovered biomarkers in both up-regulated and down-regulated groups sorted based on |log2
(fold change)|. (C) The KEGG-enriched pathways in breast cancer are displayed. (For interpretation of the colors in the ﬁgure(s), the reader is referred to the web
version of this article.)
                                                                                                   perform GraphSurv (the second-best approach) by 8.5% C-index value.
                                                                                                   Moreover, we conﬁrmed that the local augmentation technique could
                                                                                                   enhance the model’s ability to represent multi-omics features, improve
                                                                                                   the  model’s  robustness  to  missing  multi-omics  features,  and  prevent
                                                                                                   the  model’s  over-smoothing  during  training  in  cancer  prognosis  pre-
                                                                                                   diction. Finally, LAGProg can signiﬁcantly separate high-risk suﬀerers
                                                                                                   from low-risk ones. Based on the risk subgroups divided by LAGProg,
                                                                                                   we discovered ten prognostic biomarkers associated with breast cancer
                                                                                                   through diﬀerential expression analysis.
                                                                                                       Though LAGProg has been indicated accurate and robust for multi-
                                                                                                   omics cancer prognosis prediction and analysis, there is still room for
                                                                                                   improvement. Firstly, integrating clinic data of suﬀerers into the LAG-
                                                                                                   Prog may lead to better performance [35]. Second, we plan to extend
                                                                                                   the architecture of LAGProg to model the relationship between tran-
                                                                                                   scriptional regulation and cancer survival risks [36].
                    Fig.      8.   Impact   of   diﬀerent   feature   numbers.                     CRediT authorship contribution statement
hood features need to be improved for learning representations of genes                                Yongqing  Zhang:    Conceptualization, Methodology, Writing origi-
with few neighbors. Considering this problem, we designed a local aug-                             nal draft preparation.         Shuwen Xiong:    Methodology, Experiment.         Zix-
mented  graph  convolutional  network  framework  named  LAGProg  to                               uan Wang:  Data curation, Experiment.      Yuhang Liu:  Experiment.      Hong
predict cancer prognosis. LAGProg ﬁrst exploits a local augmentation                               Luo:   Experiment.       Beichen Li:   Validation.       Quan Zou:   Supervision, Val-
strategy  to  generate  augmented  features.  It  then  combines  the  origi-                      idation, Writing review & editing.
nal multi-omics features with generated features as the inputs of the
GCN  to  enhance  the  prediction  performance.  Without  additional  pa-                          Declaration of competing interest
rameters, LAGProg can signiﬁcantly improve the current state-of-the-art
approaches.
    To  validate  the  eﬀectiveness  of  LAGProg,  we  conducted  a  series                            The authors declare that they have no known competing ﬁnancial
of  experiments  on  15  cancer  datasets  from  TCGA.  By  comparing  the                         interests or personal relationships that could have appeared to inﬂuence
prognosis prediction performance, the results achieved by LAGProg out-                             the work reported in this paper.

Y.      Zhang,      S.      Xiong,      Z.      Wang      et      al.
Data availability                                                                                                 [19]    Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting
                                                                                                                        Jiang, Yuval Kluger, Deepsurv: personalized treatment recommender system using
     All   the   data   analyzed   during   the   current   study   are   available   in                                a Cox proportional hazards deep neural network, BMC Med. Res. Methodol. 18   (1)
the TCGA dataset (https://tcga -data .nci .nih .gov /tcga/). The codes are                                              (2018) 1–12.
                                                                                                                  [20]    Imran Khan Mohd Jais, Amelia Ritahani Ismail, Syed Qamrun Nisa, Adam optimiza-
available at https://github .com /ZhangLab312 /LAGProg.                                                                 tion algorithm for wide and deep neural network, Knowl. Eng. Data Sci. 2(1) (2019)
                                                                                                                        41–46.
Acknowledgement                                                                                                   [21]    Matthew E. Ritchie, Belinda Phipson, D.I. Wu, Yifang Hu, Charity W. Law, Wei Shi,
                                                                                                                        Gordon K. Smyth, Limma powers diﬀerential expression analyses for rna-sequencing
     This work is supported by the National Natural Science Foundation                                                  and microarray studies, Nucleic Acids Res. 43  (7) (2015) e47.
                                                                                                                  [22]    Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
of China under Grant No. 62272067, 62131004, 62250028; the Sichuan                                                      Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al., Py-
Science and Technology Program under Grant No. 2023NSFSC0499; the                                                       torch: an imperative style, high-performance deep learning library, Adv. Neural Inf.
Scientiﬁc  Research  Foundation  of  Sichuan  Province  under  Grant  No.                                               Process. Syst. 32 (2019).
2022001; the Scientiﬁc Research Foundation of Chengdu University of                                               [23]    Adam R. Brentnall, Jack Cuzick, Use of the concordance index for predictors of
                                                                                                                        censored survival data, Stat. Methods Med. Res. 27  (8) (2018) 2359–2373.
Information Technology under Grant No. KYQN202208.                                                                [24]    María Ángeles Castilla, María Ángeles López-García, María Reina Atienza, Juan
                                                                                                                        Manuel Rosa-Rosa, Juan Díaz-Martín, María Luisa Pecero, Begona Vieites, Laura
References                                                                                                              Romero-Pérez, Javier Benítez, Annarica Calcabrini, et al., Vgll1 expression is asso-
                                                                                                                        ciated with a triple-negative basal-like phenotype in breast cancer, Endocr.-Relat.
 [1]    Ibiayi Dagogo-Jack, Alice T. Shaw, Tumour heterogeneity and resistance to cancer                                Cancer 21  (4) (2014) 587–599.
      therapies, Nat. Rev. Clin. Oncol. 15  (2) (2018) 81–94.                                                     [25]    Alex Cordero, Deepak Kanojia, Jason Miska, Wojciech K. Panek, Annie Xiao, Yu
 [2]    Samuel Marguerat, Jürg Bähler, Rna-seq: from technology to biology, Cell. Mol. Life                             Han, Nicolas Bonamici, Weidong Zhou, Ting Xiao, Meijing Wu, et al., Fabp7 is a
      Sci. 67  (4) (2010) 569–579.                                                                                      key metabolic regulator in her2+ breast cancer brain metastasis, Oncogene 38  (37)
 [3]    Susanne Motameny, Stefanie Wolters, Peter Nürnberg, Björn Schumacher, Next gen-                                 (2019) 6445–6460.
      eration sequencing of mirnas–strategies, resources and methods, Genes 1(1) (2010)                           [26]    Yulong  Bao,  Li  Wang,  Lin  Shi,  Fen  Yun,  Xia  Liu,  Yongxia  Chen,  Chen  Chen,
      70–84.                                                                                                            Yanni Ren, Yongfeng Jia, Transcriptome proﬁling revealed multiple genes and ecm-
 [4]    Chao Xie, Martti T. Tammi, Cnv-seq, a new method to detect copy number variation                                receptor interaction pathways that may be associated with breast cancer, Cell. Mol.
      using high-throughput sequencing, BMC Bioinform. 10  (1) (2009) 1–9.                                              Biol. Lett. 24  (1) (2019) 1–20.
 [5]    Noah Simon, Jerome Friedman, Trevor Hastie, Rob Tibshirani, Regularization paths                          [27]    Paul J. Adam, Robert Boyd, Kerry L. Tyson, Graham C. Fletcher, Alasdair Stamps,
      for Cox’s proportional hazards model via coordinate descent, J. Stat. Softw. 39   (5)                             Lindsey Hudson, Helen R. Poyser, Nick Redpath, Matthew Griﬃths,       Graham Steers,
      (2011) 1.                                                                                                         et al., Comprehensive proteomic analysis of breast cancer cell membranes reveals
 [6]    Danyang Tong, Yu Tian, Tianshu Zhou, Qiancheng Ye, Jun Li, Kefeng Ding, Jing-                                   unique proteins with potential roles in clinical cancer, J. Biol. Chem. 278  (8) (2003)
      song Li, Improving prediction performance of colon cancer prognosis based on the                                  6482–6489.
      integration of clinical and multi-omics data, BMC Med. Inform. Decis. Mak. 20   (1)                         [28]    E. Buache, N. Etique, F. Alpy, I. Stoll, M. Muckensturm, B. Reina-San-Martin, M.P.
      (2020) 1–15.                                                                                                      Chenard, C. Tomasetto, M.C. Rio, Deﬁciency in trefoil factor 1 (tﬀ1) increases tu-
 [7]    Hongya Lu, Haifeng Wang, Sang Won Yoon, A dynamic gradient boosting machine                                     morigenicity of human breast cancer cells and mammary tumor development in
      using genetic optimizer for practical breast cancer prognosis, Expert Syst. Appl. 116                             tﬀ1-knockout mice, Oncogene 30  (29) (2011) 3261–3273.
      (2019) 340–350.                                                                                             [29]    Felicity E.B. May, Bruce R. Westley, Tﬀ3 is a valuable predictive biomarker of en-
 [8]    Anika Cheerla, Olivier Gevaert, Deep learning with multimodal representation for                                docrine response in metastatic breast cancer, Endocr.-Relat. Cancer 22   (3) (2015)
      pancancer prognosis prediction, Bioinformatics 35  (14) (2019) i446–i454.                                         465.
 [9]    Tzong-Yi Lee, Kai-Yao Huang, Cheng-Hsiang Chuang, Cheng-Yang Lee, Tzu-Hao                                 [30]    Tommy A. Brown, Elizabeth A. Mittendorf, Diane F. Hale, John W. Myers, Kaitlin M.
      Chang, Incorporating deep learning and multi-omics autoencoding for analysis of                                   Peace, Doreen O. Jackson, Julia M. Greene, Timothy J. Vreeland, G.  Travis Clifton,
      lung adenocarcinoma prognostication, Comput. Biol. Chem. 87 (2020) 107277.                                        Alexandros Ardavanis, et al., Prospective, randomized, single-blinded, multi-center
[10]    Li Tong, Jonathan Mitchel, Kevin Chatlin, May D. Wang, Deep learning based                                      phase ii trial of two her2 peptide vaccines, gp2 and ae37, in breast cancer patients
      feature-level integration of multi-omics data for breast cancer patients survival anal-                           to prevent recurrence, Breast Cancer Res. Treat. 181  (2) (2020) 391–401.
      ysis, BMC Med. Inform. Decis. Mak. 20  (1) (2020) 1–12.                                                     [31]    Florian Rudolf Fritzsche, Edgar Dahl, Stefan Pahl, Mick Burkhardt, Jun Luo, Empar
[11]    Hua Chai, Xiang Zhou, Zhongyue Zhang, Jiahua Rao, Huiying Zhao, Yuedong Yang,                                   Mayordomo, Tserenchunt Gansukh, Anja Dankof, Ruth Knuechel, Carsten Denkert,
      Integrating multi-omics data through deep learning for accurate cancer prognosis                                  et al., Prognostic relevance of agr2 expression in breast cancer, Clin. Cancer Res.
      prediction, Comput. Biol. Med. 134 (2021) 104481.                                                                 12  (6) (2006) 1728–1734.
[12]    Damian Szklarczyk, Annika L. Gable, Katerina C. Nastou, David Lyon, Rebecca                               [32]    Sarah Kammerer, Armin Sokolowski, Hubert Hackl, Dieter Platzer, Stephan  Wenzel
      Kirsch, Sampo Pyysalo, Nadezhda T. Doncheva, Marc Legeay, Tao Fang, Peer Bork,                                    Jahn, Amin El-Heliebi, Daniela Schwarzenbacher, Verena Stiegelbauer, Martin Pich-
      et al., The string database in 2021: customizable protein–protein networks, and                                   ler, Simin Rezania, et al., Kcnj3 is a new independent prognostic marker for estrogen
      functional characterization of user-uploaded gene/measurement sets, Nucleic Acids                                 receptor positive breast cancer patients, Oncotarget 7  (51) (2016) 84705.
      Res. 49  (D1) (2021) D605–D612.                                                                             [33]    Tan-Chi Fan, Hui Ling Yeo, Huan-Ming Hsu, Jyh-Cherng Yu, Ming-Yi Ho, Wen-
[13]    Minoru Kanehisa, Susumu Goto, Kegg: Kyoto encyclopedia of genes and genomes,                                    Der Lin, Nai-Chuan Chang, John Yu, L. Yu Alice, Reciprocal feedback regulation of
      Nucleic Acids Res. 28  (1) (2000) 27–30.                                                                          st3gal1 and gfra1 signaling in breast cancer cells, Cancer Lett. 434 (2018) 184–195.
[14]    Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, S. Yu                                  [34]    Ida Johansson, Cecilia Nilsson, Pontus Berglund, Martin Lauss, Markus Ringnér,
      Philip, A comprehensive survey on graph neural networks, IEEE Trans. Neural Netw.                                 Håkan Olsson, Lena Luts, Edith Sim, Sten Thorstensson, Marie-Louise Fjällskog, et
      Learn. Syst. 32  (1) (2020) 4–24.                                                                                 al., Gene expression proﬁling of primary male breast cancers reveals two unique sub-
[15]    Yi Wang, Zhongyue Zhang, Hua Chai, Yuedong Yang, Multi-omics cancer prognosis                                   groups and identiﬁes n-acetyltransferase-1 (nat1) as a novel prognostic biomarker,
      analysis based on graph convolution network, in: 2021 IEEE International Confer-                                  Breast Cancer Res. 14  (1) (2012) 1–15.
      ence on Bioinformatics and Biomedicine (BIBM), IEEE, 2021, pp.  1564–1568.                                  [35]    Ricardo Ramirez, Yu-Chiao Chiu, SongYao Zhang, Joshua Ramirez, Yidong Chen,
[16]    Lin Wei, Zhilin Jin, Shengjie Yang, Yanxun Xu, Yitan Zhu, Yuan Ji, Tcga-assembler                               Yufei Huang, Yu-Fang Jin, Prediction and interpretation of cancer survival using
      2: software pipeline for retrieval and processing of tcga/cptac data, Bioinformatics                              graph convolution neural networks, Methods 192 (2021) 120–130.
      34  (9) (2018) 1615–1617.                                                                                   [36]    Zixuan Wang, Meiqin Gong, Yuhang Liu, Shuwen Xiong, Maocheng Wang, Jiliu
[17]    Songtao Liu, Rex Ying, Hanze Dong, Lanqing Li, Tingyang Xu, Yu Rong, Peilin Zhao,                               Zhou, Yongqing Zhang, Towards a better understanding of tf-dna binding predic-
      Junzhou Huang, Dinghao Wu, Local augmentation for graph neural networks, in:                                      tion from genomic features, Comput. Biol. Med. (2022) 105993.
      International Conference on Machine Learning, PMLR, 2022, pp.  14054–14072.
[18]    Si Zhang, Hanghang Tong, Jiejun Xu, Ross Maciejewski, Graph convolutional net-
      works: a comprehensive review, Comput. Soc. Netw. 6(1) (2019) 1–23.

