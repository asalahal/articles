                                                                                                   Contents lists available at            ScienceDirect
                                                                                ComputersinBiologyandMedicine
                                                                              journal homepage:               www.elsevier.com/locate/compbiomed
Integrationofmulti-omicsdatausingadaptivegraphlearningandattention
mechanismforpatientclassificationandbiomarkeridentification
DongOuyang               a,b,YongLiang           a,∗,LeLi    b,NingAi       b,ShanghuiLu            b,MingkunYu             b,XiaoyingLiu           c,
ShengliXie          d
a Peng Cheng Laboratory, Shenzhen, 518055, China
b School of Computer Science and Engineering, Faculty of Innovation Engineering, Macau University of Science and Technology, 999078,
 Macao Special Administrative Region of China
c Computer Engineering Technical College, Guangdong Polytechnic of Science and Technology, Zhuhai, 519090, China
d Guangdong-HongKong-Macao Joint Laboratory for Smart Discrete Manufacturing, Guangzhou, 510000, China
A R T I C L E   I N F O                                                               A B S T R A C T
Dataset link:      https://github.com /O uyang-D ong                                  With the rapid development and accumulation of high-throughput sequencing technology and omics data,
/M O G LA M                                                                           many studies have conducted a more comprehensive understanding of human diseases from a multi-omics
Keywords:                                                                             perspective. Meanwhile, graph-based methods have been widely used to process multi-omics data due to its
Multi-omicsdata                                                                       powerful expressive ability. However, most existing graph-based methods utilize fixed graphs to learn sample
Featureselection                                                                      embedding representations, which often leads to sub-optimal results. Furthermore, treating embedding repre-
Dynamicgraphconvolutionalnetwork                                                      sentationsofdifferentomicsequallyusuallycannotobtainmorereasonableintegratedinformation.Inaddition,
Multi-omicsattentionmechanism                                                         the complex correlation between omics is not fully taken into account. To this end, we propose an end-
Omic-integratedrepresentationlearning                                                 to-end interpretable multi-omics integration method, named MOGLAM, for disease classification prediction.
                                                                                      Dynamic graph convolutional network with feature selection is first utilized to obtain higher quality omic-
                                                                                      specific embedding information by adaptively learning the graph structure and discover important biomarkers.
                                                                                      Then, multi-omics attention mechanism is applied to adaptively weight the embedding representations of
                                                                                      differentomics,therebyobtainingmorereasonableintegratedinformation.Finally,weproposeomic-integrated
                                                                                      representation learning to capture complex common and complementary information between omics while
                                                                                      performing multi-omics integration. Experimental results on three datasets show that MOGLAM achieves
                                                                                      superior performance than other state-of-the-art multi-omics integration methods. Moreover, MOGLAM can
                                                                                      identify important biomarkers from different omics data types in an end-to-end manner.
1. Introduction                                                                                                                              However,eachsampleofomicsdatausuallycontainstensofthousands
                                                                                                                                             of molecular features, and the sample size is too small. Therefore, for
      The continuous advancement of technologies for high-throughput                                                                         multi-omics data with the ‘‘curse of dimensionality’’, there is an urgent
biological data generation has made it possible to generate multiple                                                                         need for a novel method to comprehensively understand complex bio-
omics data for the same set of samples [                    1–3], such as genomics, epige-                                                   logicalprocessesanddiscoverpotentialbiomarkersthroughintegrating
nomics, transcriptomics, proteomics and metabolomics, etc. Moreover,                                                                         these omics data.
differenttypesofomicsdatarevealdifferentaspectsofthesameclinical                                                                                   Existing methods have made great efforts in multi-omics data in-
sampleatthemolecularlevel.Uncoveringthemolecularmechanismsof
complex diseases by integrating different omics data types for compre-                                                                       tegration,  which  are  mainly  divided  into  two  categories:  unsuper-
hensiveanalysiscanleadtobetterdiseasetreatmentandmoreaccurate                                                                                vised learning and supervised learning methods [                          12 ]. Unsupervised
clinical decision-making [             4–6]. Meanwhile, many studies have shown                                                              learning-based methods perform clustering and classification tasks by
that mRNA expression and DNA methylation play important roles in                                                                             integrating multi-omics data into a single low-dimensional embedding
human cancer and provide meaningful biomarkers for cancer diag-                                                                              space [   13 ,14 ]. However, due to the lack of additional information on
nosis and prognosis [           7–9]. In addition, microRNAs (miRNAs) silence                                                                sample labels, the model cannot achieve end-to-end training, which
or degrade mRNAs by binding to target mRNAs. Its overexpression                                                                              often leads to sub-optimal results. Supervised learning-based methods
or dysregulation may also lead to various human diseases [                               10 ,11 ].
   ∗  Corresponding author.
       E-mail address:          yongliangresearch@gmail.com                    (Y. Liang).
https://doi.org/10.1016/j.compbiomed.2023.107303
Received 16 March 2023; Received in revised form 8 July 2023; Accepted 28 July 2023

D. Ouyang et al.
mainlyutilizethesimplestintegrationmethodoffeatureconcatenation,                                                                       considering the correlation between features by using inner product
oranalyzeeachdatatypeindependentlyandcombinethepredictionre-                                                                           regularization. In addition, the embedding information of different
sultsbasedontheaverageormajorityvotingstrategy[5,15].However,                                                                          omics usually has different contributions to the downstream classifica-
these two types of methods cannot effectively consider the correlation                                                                 tion performance. Thus, multi-omics attention mechanism is employed
betweendifferentomicsdatatypes,resultingintheintegratedinforma-                                                                        to weight different embedding information for more reasonable omics
tionisnotrichandreasonable.Moreover,thepredictionresultsmaybe                                                                          integration and improving classification performance. Finally, the pro-
biasedtowardscertainomics,therebyleadingtounreliableconclusions.                                                                       posed omic-integrated representation learning can not only integrate
      Inthepastfewyears,researchershavebeguntoproposemoremulti-                                                                        multi-omics data, but also further capture the common and comple-
omics data integration methods based on supervised learning, which                                                                     mentaryinformationbetweendifferenttypesofomicsdata.Meanwhile,
focus on the correlation between different omics. For example, van de                                                                  the effectiveness of MOGLAM is demonstrated by conducting extensive
Wiel 𝑒𝑡𝑎𝑙.[4]developedamethodcalledGRridge,whichintroducesan                                                                           disease classification tasks on three datasets, including kidney cancer
adaptive group-regularized (logistic) ridge regression method to clas-                                                                 type classification, pan-cancer classification related to squamous cells
sify cervical cancer using methylation microarray data. Singh              𝑒𝑡𝑎𝑙. [5]                                                   carcinomas,andbreastinvasivecarcinomasubtypeclassification.Com-
extended the sparse generalized canonical correlation analysis into a                                                                  paredwithothermulti-omicsintegrationmethods,experimentalresults
supervised classification framework and proposed the Data Integration                                                                  show that our proposed MOGLAM achieves superior classification per-
Analysis for Biomarker discovery using Latent cOmponents (DIABLO)                                                                      formance. Moreover, the results of ablation experiments further verify
method. DIABLO seeks common information between different omics                                                                        the necessity of the three modules of MOGLAM. In addition, MOGLAM
data types and distinguishes different phenotype groups by selecting                                                                   can select important biomarkers during end-to-end training without
a subset of features. Inspired by self-paced learning, Yang             𝑒𝑡𝑎𝑙. [16]                                                     using independent feature selection methods.
proposed a robust multimodal data integration method named SMSPL,
which interactively recommends high-confidence samples from differ-                                                                    2. Materials
ent data types in a soft-weighted manner to predict cancer subtypes.
However, the simple linearity assumption among omic features may                                                                       2.1. Dataset collection
not be applicable in complex biomedical studies.
      Recently, deep learning has been applied to multi-omics data in-                                                                       In this paper, we utilized the following three different biomedical
tegration due to its powerful nonlinear capturing capability [17,18].                                                                  classification datasets to demonstrate the effectiveness of the proposed
Furthermore, existing methods have begun to exploit similarity re-                                                                     MOGLAM method, including KIPAN dataset for kidney cancer type
lationships between samples based on graph convolutional network                                                                       classification, SCC dataset for pan-cancer classification related to squa-
(GCN), which can capture more effective and reasonable sample em-                                                                      mouscellscarcinomas,andBRCAdatasetforbreastinvasivecarcinoma
bedding information compared to fully connected neural network [19,                                                                    PAM50subtypeclassification.KIPANandSCCdatasetscanbeobtained
20].  For  example,  Li      𝑒𝑡𝑎𝑙.  [19]  developed  a  graph  convolutional                                                           from Pan-Cancer data provided on UCSC Xena (https://xenabrowser.
network-based multi-omics integration model MoGCN. It adopts au-                                                                       net/hub/). The BRCA dataset was obtained through TCGAbiolinks [20,
toencoder (AE) and similarity network fusion (SNF) methods to obtain                                                                   22]. Among them, the KIPAN dataset has chromophobe renal cell
multi-omics integrated embedding information and similarity network,                                                                   carcinoma (KICH), clear renal cell carcinoma (KIRC) and papillary
respectively. Wang     𝑒𝑡𝑎𝑙. [20] designed a method called MOGONET,                                                                    renal cell carcinoma (KIRP). The SCC dataset mainly includes cervical
which utilizes three separate graph convolution networks to analyze                                                                    squamous cell carcinoma and endocervical adenocarcinoma (CESC),
different omics data and integrates the learned label space in a ten-                                                                  lung squamous cell carcinoma (LUSC) and head and neck squamous
sor manner for performing the final classification. Nevertheless, most                                                                 cell carcinoma (HNSC). The BRCA dataset is for subtype classification
of these methods based on graph convolutional network use fixed                                                                        withnormal-like,basal-like,humanepidermalgrowthfactorreceptor2
sample similarity networks to learn sample embedding information.                                                                      (HER2)-enriched, Luminal A, and Luminal B subtypes. For these three
Obviously, if the constructed graph is incorrect, the learning results                                                                 datasets, mRNA expression, DNA methylation and miRNA expression
will be affected. Moreover, simply treating the embedding information                                                                  data were collected for multi-omics analysis. Note that only samples
ofdifferentomicsequallyintheprocessofomicsintegrationoftenfails                                                                        with matched mRNA expression, DNA methylation and miRNA expres-
toobtainmorereasonableandrichintegratedinformation.Inaddition,                                                                         sion data were included in our study. The details of the three datasets
when performing multi-omics integration, the common and comple-                                                                        used in this paper are shown in Supplementary Table S2.
mentary  information  between  omics  is  effectively  captured,  which
contribute to improving the prediction performance of downstream                                                                       2.2. Pre-processing
classification tasks. Meanwhile, although graph convolutional network
can capture complex nonlinear information and reduce the dimension-                                                                          To remove noise and redundant features in the data, we prepro-
ality of multi-omics data, standard graph convolutional network [21]                                                                   cessed the omics data appropriately to obtain better interpretation
is not easy to select biomarkers during the model training. Therefore,                                                                 results. For three omics data, we first removed features with missing
improving  biological  interpretability  while  mitigating  the  ‘‘disaster                                                            values,wheremissingvaluesrefertonotanumber(NaN)values.Then,
of dimensionality’’ is crucial for understanding the pathogenesis of                                                                   since a probe may correspond to multiple genes for DNA methylation
diseases at the molecular level across different omics.                                                                                data, we only kept probes corresponding to one gene for better in-
      To alleviate the above problems, we propose                          𝐌  ulti- 𝐎 mics    𝐆 raph                                   terpreting the results and averaged all duplicate genes. In addition,
𝐋 earning  and         𝐀 ttention     𝐌  echanism  (MOGLAM),  a  multi-omics  in-                                                      we believed that features with zero or low variance do not contribute
tegration method for patient classification and important biomarker                                                                    substantially for the prediction results. Therefore, additional variance
identification. MOGLAM learns an optimal sample similarity network                                                                     filtering thresholds were applied to different omics data. To be more
in an end-to-end manner to obtain higher quality embedding informa-                                                                    specific,thethresholdformRNAexpressiondatais0.1,andthethresh-
tion. Moreover, multi-omics attention mechanism (MOAM) and omic-                                                                       old for DNA methylation data is 0.001. In particular, due to the small
integrated representation learning (OIRL) are used to explore more                                                                     number of miRNA features, only features with variance equal to zero
reasonable omics integration strategies and capture common and com-                                                                    were filtered out for miRNA expression data.
plementary information between omics, respectively. First, MOGLAM                                                                            Although we have preprocessed the omics data through the above
utilizes dynamic graph convolutional network with feature selection                                                                    steps, the retained data is still high-dimensional and may contain
(FSDGCN) to learn an omic-specific optimal sample similarity net-                                                                      redundantfeaturesthatnegativelyaffecttheclassificationperformance.
work in adaptive graph learning manner, and select biomarkers while                                                                    Therefore, we further preselected the omics features using the analysis

D. Ouyang et al.
ofvariance(ANOVA)methodinstatisticaltests.WecalculatedANOVA                                                                        Although the initial graph      𝐴 may be noisy, it usually contains rich
F-values for each omic sequentially to evaluate whether a feature has                                                         andusefulinformationregardingtruegraphtopology.Thus,wefurther
significant differences between different categories. In addition, the                                                        performed the graph learning through integrating the initial graph                𝐴
featuresselectedbytheANOVAmethodmayonlyincludehighlycorre-                                                                    and the learned graph      𝑆 as below:
lated features, which may miss the less correlated but more important                                                          ̂𝐴 = 𝜂𝐴+(1−  𝜂)𝑆                                                                            (4)
biomarkers and limit the model to utilize the interactive information
fromdifferentfeatures.Hence,wecalculatedthenumberofpreselected                                                                where hyper-parameter      𝜂is a trade-off to balance     𝑆 and 𝐴.
featuresbasedonpreviouswork[20],thatis,thefeaturesofpreselected
datashouldhaveafirstprincipalcomponentthatmakesuparound50%
of the variance. Finally, we normalized each type of omics data to the                                                        3.3. Graph regularization
range of 0–1.
3. Methods                                                                                                                         Although combining the learned graph          𝑆 with the initial graph     𝐴 is
                                                                                                                              an effective way to approach the optimal graph, the constraint on the
     In this section, we developed an end-to-end interpretable multi-                                                         smoothnessandsparsityofthelearnedgraph           𝑆 isimportanttoimprove
omics integration method, called MOGLAM, for the disease classifi-                                                            the quality of the final graph [25,27]. The smoothness constraint                                  𝑠𝑚
cation task. As shown inFig.3, MOGLAM mainly consists of three                                                                aims to establish links between similar nodes as much as possible. For
modules: dynamic graph convolutional network with feature selec-                                                              the input data    𝑋   =  [ 𝒙 1,𝒙 2,… ,𝒙𝑛], the Dirichlet energy is used to
tion (FSDGCN), multi-omics attention mechanism (MOAM), and omic-                                                              measure the smoothness:
integrated representation learning (OIRL).                                                                                                                  𝑛∑
                                                                                                                              𝑠𝑚(𝑆,𝑋) =     12𝑛2               𝑆𝑖𝑗‖𝒙𝑖−  𝒙𝑗‖22                                                    (5)
3.1. FSDGCN                                                                                                                                               𝑖,𝑗=1
                                                                                                                                   As can be seen, larger distance           ‖𝒙𝑖−  𝒙𝑗‖22 between feature vector              𝒙𝑖
     Most of the existing multi-omics integration methods do not con-                                                         and   𝒙𝑗forces a smaller value     𝑆𝑖𝑗. However, only minimizing              𝑠𝑚(𝑆,𝑋)
sider the correlation between samples, while using the sample sim-                                                            may result in the trivial solution (i.e.,       𝑆 = 0  ). To avoid such problem,
ilarity network as the input of GCN can improve the performance                                                               we integrated information from the initial graph           𝐴 by considering the
of the model [20]. Nevertheless, there are the following challenges                                                           regularization term. Furthermore, the sparsity regularization term is
when directly applying GCN in relevant biological problems. On the                                                            alsoimposedon    𝑆.Then,thetworegularizationtermscanbecombined
one hand, although GCN can achieve better predictions by exploiting                                                           as follows:
the correlation between samples, it is not trivial to build a reliable
graph for a specific task. Moreover, the quality of the constructed                                                           𝑟𝑒(𝑆,𝐴) = 𝜆                         𝐹 + 𝛾             𝐹                                                  (6)
graph has a significant impact on the prediction results of downstream                                                                            𝑛2‖𝑆 −𝐴‖2               𝑛2‖𝑆‖2
tasks. On the other hand, the standard GCN [21] can only learn the                                                            where   ‖  ⋅‖𝐹 denotes the Frobenius norm of a matrix. Finally, the total
embeddinginformation,buthardtoperformfeatureselection,resulting                                                               graph regularization          𝑔𝑙is defined as:
in poor interpretability. Therefore, we propose dynamic graph neural
networkwithfeatureselection(FSDGCN)tolearnoptimalomic-specific                                                                𝑔𝑙(𝑆,𝑋,𝐴) =  𝑠𝑚(𝑆,𝑋)+  𝑟𝑒(𝑆,𝐴)                                                 (7)
embedding information while identifying important biomarkers.
3.2. Graph structure learning                                                                                                 3.4. FSDGCN architecture
     Some previous methods have adopted such as K-Nearest-Neighbor                                                                 Given the input data     𝑋 ∈  R𝑛×𝑑, where  𝑛and 𝑑denote the number
(KNN) [23], radial basis function (RBF) kernel [24] and cosine sim-                                                           of patients and input features, respectively. The adjacency matrix               ̂𝐴 ∈
ilarity [20] to construct corresponding graphs. However, the graphs                                                           R𝑛×𝑛obtained by graph structure learning. Then, the propagation rules
constructedbythesemethodsarestillfixed.Obviously,ifthegraphwe                                                                 for first layer in FSDGCN can be defined as:
constructed is incorrect, the embedding information learned based on                                                          𝐻 (1)  = 𝜎(̃𝐴𝐻 (0)𝑊  (0) )                                                                        (8)
GCNmodelswillhaveadverseeffectsondownstreamtasks.Toaddress
this issue, we designed a dynamic GCN method that can adaptively                                                              where  ̃𝐴 = ̂𝐷−  12 (̂𝐴 + 𝐼)̂𝐷−  12 , ̂𝐷 is a diagonal matrix of     ̂𝐴.𝜎(⋅) denotes
adjust the graph structure based on the classification results. Inspired                                                      a nonlinear activation function. In this paper, we had            𝐻 (0)   = 𝑋𝑊𝑓
bypreviousworks[25,26],weusedweightedcosinesimilarityasnode                                                                   and 𝑊𝑓 is the feature indicator matrix that aims to perform feature
similarity metric function:                                                                                                   selection.
         ⎧⎪⎨⎪⎩cos(𝑊𝑠𝒙𝑖,𝑊𝑠𝒙𝑗),  if𝑖≠𝑗𝑎𝑛𝑑                                                                                            ThepropagationrulesofthesecondlayerarethesameasGCN.The
𝑆𝑖𝑗=                                      cos(𝑊𝑠𝒙𝑖,𝑊𝑠𝒙𝑗) ≥𝜖                                                     (1)           encoding process can be formulated as:
           0,                       otherwise  .                                                                              𝑍 = 𝐻 (2)  = 𝜎(̃𝐴𝐻 (1)𝑊  (1) )                                                                 (9)
where  𝑊𝑠is a learnable weight matrix,                𝒙𝑖and    𝒙𝑗indicate the feature                                              Finally, the dynamic graph convolutional network with feature
vectors of node    𝑖and node   𝑗, respectively.       cos(    ⋅) is the cosine similarity                                     selection  𝐹𝑆𝐷𝐺𝐶𝑁 (⋅) is trained with   𝑋 and  ̃𝐴 as shown inFig.1, the
and its calculation formula as follows:                                                                                       final prediction result can be formulated as:
cos(𝑊𝑠𝒙𝑖,𝑊𝑠𝒙𝑗) =   𝑊𝑠𝒙𝑖⋅𝑊𝑠𝒙𝑗‖𝑊𝑠𝒙𝑖‖2‖𝑊𝑠𝒙𝑗‖2                                                  (2)                                ̂𝑌𝐹𝑆𝐷𝐺𝐶𝑁  = 𝐹𝑆𝐷𝐺𝐶𝑁 (𝑋,̃𝐴)                                                         (10)
and the threshold    𝜖can be determined given a parameter          𝑘:                                                         where  ̂𝑌𝐹𝑆𝐷𝐺𝐶𝑁   ∈  R𝑛×𝑐 represents the predicted label probability,         𝑐
𝑘= ∑        𝐼(cos( 𝑊𝑠𝒙𝑖,𝑊𝑠𝒙𝑗) ≥𝜖)∕𝑛                                                      (3)                                  represents the number of classification categories.
       𝑖,𝑗                                                                                                                         Previous studies [28,29] have applied the             𝐿2,1  norm to GCN for
where  𝐼(⋅) refers to the indicator function,       𝑛denotes the number of                                                    feature selection, but the     𝐿2,1  norm did not consider the correlation
patients, and the hyper-parameter         𝑘 represents the sparsity of the                                                    between features. Inspired by previous work [30], we applied inner
graph.                                                                                                                        productregularizationonthefeatureindicatormatrixtoselectfeatures

D. Ouyang et al.
Fig. 1.     Illustration of FSDGCN. FSDGCN includes feature selection, adaptive graph learning and graph regularization. We take mRNA expression data as an example to show the
FSDGCN module of MOGLAM. The cosine similarity and weighted cosine similarity functions are adopted to calculate the initial graph                                 𝐴 and the learned graph      𝑆, respectively.
Fig. 2.     The detailed process of feature selection. For each type of omics data, we select important biomarkers by ranking the score matrix. To consider the correlation between
features during feature selection, the inner product regularization is constrained on the feature indicator matrix                         𝑊𝑓.
with high similarity (see             Fig.   2). The formulation of inner product                                               data, where   𝑀   is the number of omics. To fully capture the impor-
regularization is as follows:                                                                                                   tance ofembedding information, weemployed attention mechanismto
                 𝑑∑      𝑑∑                                                                                                     compute the attention weights of embedding information:
𝛺(𝑊𝑓) =                        |⟨𝑤𝑓𝑖,𝑤𝑓𝑗⟩|                                                                                      𝐺𝑎𝑡𝑡= 𝜙𝑎𝑡𝑡𝑒𝑛(𝐺,𝑊𝑎𝑡𝑡) = 𝛿2(𝑊  2𝑎𝑡𝑡𝛿1(𝑊  1𝑎𝑡𝑡𝐺))                                         (14)
                𝑖=1  𝑗=1,𝑗≠𝑖
                 𝑑∑    𝑑∑                          𝑑∑                                                           (11)            where  𝛿2(⋅)  is  Sigmoid  activation,     𝛿1(⋅)  is  Relu  activation,    𝑊𝑎𝑡𝑡 =
            =             |⟨𝑤𝑓𝑖,𝑤𝑓𝑗⟩|−                 ‖𝑤𝑓𝑖‖22                                                                  {𝑊  1𝑎𝑡𝑡,𝑊2𝑎𝑡𝑡}  is the model training parameter. Finally, multi-omics at-
                𝑖=1  𝑗=1                          𝑖=1                                                                           tention can be obtained as      𝐺𝑎𝑡𝑡= [𝑔1𝑎𝑡𝑡,𝑔2𝑎𝑡𝑡,… ,𝑔𝑀𝑎𝑡𝑡].
            = ‖𝑊𝑓𝑊𝑇𝑓‖1 − ‖𝑊𝑓‖2                                                                                                        To avoid downstream classification results over-reliance on high-
                                               2                                                                                quality embedding information while ignoring the role of other omics,
where    |⋅| means to take the absolute value and               ⟨⋅⟩ denotes inner                                               we combined the     𝑚-th omic embedding information with attention to
product.                                                                                                                        normalize it, which is defined as follows:
     After the above analysis, the final loss function of FSDGCN is as                                                           ̃𝑍𝑚 = 𝜙𝑠𝑐𝑎𝑙𝑒(𝑍𝑚,𝑔𝑚𝑎𝑡𝑡) = 𝑔𝑚𝑎𝑡𝑡⋅𝑍𝑚                                                      (15)
follows:
                                                                                                                                      Through the above steps, we can get normalized multi-omics em-
𝐹𝑆𝐷𝐺𝐶𝑁  =  𝑐𝑒(𝑌,̂𝑌𝐹𝑆𝐷𝐺𝐶𝑁 )+ 𝛼𝑔𝑙(𝑆,𝑋,𝐴)+ 𝛽𝑓𝑒(𝑊𝑓)               (12)                                                          bedding information      ̃𝑍 = [ ̃𝑍1,̃𝑍2,… ,̃𝑍𝑀 ].
where  𝑌  = [𝑦1,𝑦2,… ,𝑦𝑛] means the corresponding true labels,                  𝑐𝑒(⋅)                                          3.6. Omic-integrated representation learning
represents the cross-entropy loss function,                   𝑓𝑒(𝑊𝑓)  =  ( ‖𝑊𝑓𝑊𝑇𝑓‖1  −
‖𝑊𝑓‖22),𝛼and 𝛽aretwohyper-parameterstobalancetheregularization                                                                        Existing studies have shown that different types of omics data
terms.                                                                                                                          usually contain commonalities between omics and complementarity of
                                                                                                                                different omics [        26 ,32 ]. In general, commonality among multi-omics
3.5. Multi-omics attention mechanism                                                                                            means consistent information about the patient’s phenotype, while
                                                                                                                                complementarity represents each omic contains information that other
                                                                                                                                omics does not. Furthermore, when we performed multi-omics integra-
     Although most of the existing methods can integrate different types                                                        tion,itiscrucialtoeffectivelycapturecommonandcomplementaryin-
of omics data, these methods simply treat the label or embedding in-                                                            formationbetweenomicsforimprovingclassificationperformance[                                     14 ,
formationofdifferentomicsequallyduringmulti-omicintegration[                                   19 ,                             33 ].
20 ]. However, different types of omics data will generate the embed-                                                                 Inspired by Transformer encoder [                   34 ] in natural language process-
ding representations of different quality and have different contribu-                                                          ing, we captured common and complementary information between
tionsfordownstreamclassificationtasks.InspiredbyHu             𝑒𝑡𝑎𝑙.[31 ],the                                                   multipleomicsbyleveragingfeedforwardnetworkandmulti-headself-
attention mechanism is used to obtain the importance of embedding                                                               attention. Based on previous work [                   26 ], for a patient   𝑝, its multi-omics
information in different omics. For the embedding information               𝑍𝑚 ∈                                                feature embedding matrix can be expressed as            ̂𝑍𝑝 = [̃𝒛 1𝑝,̃𝒛 2𝑝,… ,̃𝒛𝑀𝑝 ] ∈
R𝑛×𝑑𝑓 of the 𝑚-th omic learned by FSDGCN, where          𝑑𝑓 refers to the size                                                  R𝑑𝑓×𝑀 , where  ̃𝒛𝑚𝑝 represents the feature embedding vector of the           𝑚-th
of the embedding dimension. We first need to use squeeze’s strategy to                                                          omic. Then, we can get the query matrix          𝑄𝑝= 𝑊𝑞̂𝑍𝑝= [ 𝒒 1𝑝,𝒒 2𝑝,… ,𝒒𝑀𝑝 ],
obtain the global embedding information          𝑔𝑚 of the 𝑚-th omic.                                                           the key matrix    𝐾𝑝 = 𝑊𝑘̂𝑍𝑝 =  [ 𝒌 1𝑝,𝒌 2𝑝,… ,𝒌𝑀𝑝 ] and the value matrix
                                        𝑛∑   𝑑𝑓∑                                                                                𝑉𝑝  = 𝑊𝑣̂𝑍𝑝  =  [ 𝒗 1𝑝,𝒗 2𝑝,… ,𝒗𝑀𝑝 ]. Meanwhile, we applied the scaled
𝑔𝑚 = 𝜙𝑠𝑞(𝑍𝑚) =       1                            𝑍𝑚(𝑖,𝑗)                                             (13)                      dot product function as the attention function. Finally, the inter-omic
                           𝑛×𝑑𝑓        𝑖=1  𝑗=1                                                                                 attention matrix    𝐻𝑝can be calculated as follows:
     Throughtheabovecomputationalanalysis,wecanobtaintheglobal                                                                  𝐻𝑝(𝑖,𝑗) =       exp[(  𝒒𝑖𝑝)𝑇  ⋅ 𝒌𝑗𝑝∕√            𝑑𝑓]
embeddinginformation      𝐺 = [𝑔1,𝑔2,… ,𝑔𝑀 ] ∈  R 1×1× 𝑀  ofmultipleomics                                                                        ∑𝑀                                   𝑑𝑓]                                                (16)
                                                                                                                                                     𝑗=1  exp[(  𝒒𝑖𝑝)𝑇  ⋅ 𝒌𝑗𝑝∕√

D. Ouyang et al.
where  𝐻𝑝(𝑖,𝑗) represents how much concern the         𝑖-th omic has for the                                                    hand, we fixed omic-specific FSDGCN and updated multi-omics atten-
𝑗-th omic of patient    𝑝. Note that we can obtain      𝑛inter-omic attention                                                   tion mechanism and omic-integrated representation learning to min-
matrices for   𝑛patients.                                                                                                       imize the loss function          𝑀𝐿𝑃 (𝑌,̂𝑌𝐻 ). Finally, omic-specific FSDGCN,
     According to inter-omic attention matrix          𝐻𝑝, the inter-omic ag-                                                   multi-omics attention mechanism and omic-integrated representation
gregation for the value vector of each omic could be calculated. In                                                             learning are updated alternately until convergence.
addition, in order to obtain a more accurate attention matrix and
robust learning process, we extended the inter-omic aggregation to a                                                            4. Results
multi-head version similar to previous work [34].
                                                 𝑁∑                                                                             4.1. Evaluation metrics and experimental details
 ̂𝑉𝑝= 𝐻𝑝 ⋅𝑉𝑇𝑝 ;   ̂𝑉𝑎𝑣𝑒𝑝    =    1𝑁                  (̂𝑉𝑇𝑝 )𝑖                                            (17)
                                                𝑖=1                                                                                   To reasonably compare the classification performance of different
where  𝑁  refers to the number of head. Moreover, different heads can                                                           methods, 70% of the samples were randomly selected as the training
capture different perspective information of inter-omic aggregation.                                                            set and the remaining 30% of the samples as the test set. It is worth
     For ease of understanding, we represented the parameters in the                                                            noting that the proportions of classes in the training and test sets
feedforwardnetworkas      𝑊ℎ ∈ R𝑑ℎ×𝑑𝑓𝑀 ,wherethefeedforwardnetwork                                                              were the same. To evaluate the performance of all methods, accuracy
is a two-layer neural network separated by a GeLU activation. Finally,                                                          (ACC),averageF1scoreweightedbysupport(F1_weighted)andmacro-
the omics common representation vector           ℎ𝑐𝑚𝑝    ∈   R𝑑ℎ×1   for the  𝑝-th                                              averaged F1 score (F1_macro) were utilized to perform biomedical
patient can be obtained as follows:                                                                                             classification tasks. In addition, we randomly generated five different
                                                                                                                                training and test sets, and evaluated all methods using the average of
ℎ𝑐𝑚𝑝  = 𝑊ℎ ⋅ Vec( ̂𝑉𝑎𝑣𝑒𝑝   )                                                                      (18)                          the calculated evaluation metrics. Note that we evaluated the classifi-
where      Vec(      ⋅) represents the vectorization of row-wise concatenation.                                                 cation performance of the model on the test set for conducting a fairer
Then, the omics common representation matrix can be expressed as                                                                comparative analysis.
𝐻𝑐𝑚= [ℎ𝑐𝑚1 ,ℎ𝑐𝑚2 ,… ,ℎ𝑐𝑚𝑝,… ,ℎ𝑐𝑚𝑛 ] for 𝑛patients.                                                                              4.2. Baseline methods
     Based on Eq.(16), we observed that the elements in               𝐻𝑝 represent
the degree of concern between different omics for patient             𝑝. Therefore,                                                   To evaluate the performance of MOGLAM, we chose to compare
for each patient, the attention matrix can be regarded as the concern                                                           with the following multi-omic integration methods.
between omics. Further, we can regard the attention matrix               𝐻𝑝 as                                                        𝐍𝐍    [36]: Fully connected neural networks (NN). We directly con-
morecomplexcomplementaryinformationbetweenomicsforpatient                  𝑝.                                                   catenated the preprocessed multi-omics data for training.
Finally, we can get complementary information            ℎ𝑐𝑝𝑝  ∈  R𝑀 2 ×1   between                                                   𝐗𝐆𝐁𝐨𝐨𝐬𝐭        [37]: eXtreme Gradient Boosting (XGBoost). We imple-
omics for the   𝑝-th patient as below:                                                                                          mented it using the xgboost sklearn package.
ℎ𝑐𝑝𝑝  = Vec( 𝐻𝑝)                                                                                (19)                                  𝐁𝐏𝐋𝐒𝐃𝐀         [5]:  Block  partial  least  squares  discriminant  analysis
                                                                                                                                (BPLSDA)methodthatprojectsmulti-omicsdatameasuredonthesame
     Similarly, we can obtain the omics complementary representation                                                            set of samples into the latent space through discriminant analysis.
matrix  𝐻𝑐𝑝 =  [ℎ𝑐𝑝1,ℎ𝑐𝑝2,… ,ℎ𝑐𝑝𝑝,… ,ℎ𝑐𝑝𝑛 ] between different omics for       𝑛                                                       𝐁𝐒𝐏𝐋𝐒𝐃𝐀         [5]: Block sparse partial least squares discriminant anal-
patients.Finally,theomic-integratedrepresentation           𝐻 canbeobtained                                                     ysis (BSPLSDA) selects the most relevant features by adding sparse
by concatenating    𝐻𝑐𝑚and 𝐻𝑐𝑝:                                                                                                 regularization to BPLSDA. BPLSDA and BSPLSDA are supervised anal-
𝐻  = 𝑐𝑜𝑛𝑐𝑎𝑡(𝐻𝑐𝑚,𝐻𝑐𝑝)                                                                    (20)                                    ysis methods included in DIABLO. The mixOmics R package was used
                                                                                                                                to implement BPLSDA and BSPLSDA.
3.7. Model optimization                                                                                                               𝐒𝐌𝐒𝐏𝐋        [16]: Robust multimodal data integration (SMSPL) was
                                                                                                                                trained by  adopting a  new soft  weighting scheme  to interactively
     After the learning of FSDGCN, multi-omics attention mechanism                                                              recommend high-confidence samples.
and omic-integrated representation, we can get the final embedding                                                                    𝐌𝐎𝐆𝐎𝐍𝐄𝐓              [20]:   Multi-omics   graph   convolutional   networks
representation    𝐻 . To obtain the final prediction result, we used a                                                          (MOGONET) constructs a graph for each omic and utilizes correla-
classifier 𝑓(⋅) to output the predicted result       ̂𝑌𝐻  = 𝑓(𝐻 ). For simplicity                                               tion discovery network for disease classification based on the label
andend-to-endtraining,themulti-layerperceptron(MLP)isutilizedas                                                                 space of tensor fusion. Note that the hyper-parameter            𝑘also exists in
classifier 𝑓(⋅). Finally, we constructed a joint loss function to optimize                                                      MOGONET,weusedcross-validationtoselecttheoptimal              𝑘forafairer
all modules in MOGLAM:                                                                                                          comparative analysis in Supplementary Figure S3.
                                                                                                                                      𝐌𝐨𝐆𝐂𝐍        [19]: Multi-omics integration model based on graph con-
        𝑀∑                                                                                                                      volutional network (MoGCN) was used to reduce dimensionality and
  =          (𝑚)𝐹𝑆𝐷𝐺𝐶𝑁  +𝜇𝑀𝐿𝑃 (𝑌,̂𝑌𝐻 )                                                (21)                                   construct a patient similarity network by utilizing autoencoder and
       𝑚=1
                                                                                                                                similarity network fusion methods for cancer subtype classification.
where     (𝑚)𝐹𝑆𝐷𝐺𝐶𝑁  =   (𝑚)𝑐𝑒(𝑌,̂𝑌𝐹𝑆𝐷𝐺𝐶𝑁 )+ 𝛼 (𝑚)𝑔𝑙(𝑆,𝑋,𝐴)+ 𝛽 (𝑚)𝑓𝑒(𝑊𝑓),𝜇is                                                      𝐌𝐌𝐆𝐋         [26]: Multi-modal Graph Learning (MMGL) method ap-
a hyper-parameter to balance the omic-specific FSDGCN loss and the                                                              plies modality-aware representation learning to explore the shared and
final MLP-based classification loss.                                                                                            complementaryinformationbetweendifferentmodalities,andadaptive
     MOGLAM is implemented on the PyTorch platform and Adam [35]                                                                graph learning to learn latent graph structure.
as the optimizer is applied. For the selection of hyper-parameters, both
𝛼and 𝜇of the balanced graph structure learning loss and classification                                                          4.3. Parameters analysis
loss are set to 1, and the     𝛽 of the balanced feature selection loss
is set to 1e-4 by experience. For the training of MOGLAM, we first                                                                    In our method, there are two important hyper-parameters that
pre-trained each omic-specific FSDGCN to obtain a good FSDGCN ini-                                                              can significantly affect the classification performance. One important
tialization. Then, we adopted an alternate update strategy to optimize                                                          hyper-parameter is     𝑘, which refers to the average number of edges re-
each module in MOGLAM. On the one hand, during the each epoch                                                                   tainedbyeachsample.Obviously,thesizeofthe           𝑘valuedeterminesthe
of training process, we fixed multi-omics attention mechanism and                                                               averagenumberofaggregatedneighbornodesfortheFSDGCNmodule,
omic-integrated representation learning, and updated             𝐹𝑆𝐷𝐺𝐶𝑁  (𝑚)(⋅)                                                 which influences the final prediction performance. Another important
by minimizing the omic-specific loss function                      (𝑚)𝐹𝑆𝐷𝐺𝐶𝑁 . On the other                                    hyper-parameteris     𝑓𝑒𝑎_𝑛𝑢𝑚,whichindicatestheembeddingdimension

D. Ouyang et al.
Fig. 3.     Workflow of MOGLAM. MOGLAM is mainly composed of three modules: FSDGCN for omic-specific learning, multi-omics attention mechanism for different omics importance
learning and omic-integrated representation learning for multi-omics integration.
Fig. 4.     The influence of different hyper-parameters on MOGLAM based on BRCA dataset. (a) The impact of hyper-parameter                              𝑘 on MOGLAM. (b) The impact of hyper-parameter
𝑓𝑒𝑎_𝑛𝑢𝑚on MOGLAM.
of the feature indicator matrix       𝑊𝑓. Note that different embedding                                                                4.4. Comparison experiments
dimensionscontaindifferentamountsofinformation,therebyaffecting
the model training. Therefore, choosing appropriate            𝑘and 𝑓𝑒𝑎_𝑛𝑢𝑚are                                                               As shown in        Tables    1, 2 and Supplementary Table S1, we observed
important for the classification performance of model. In our experi-                                                                  that  MOGLAM  outperformed  eight  comparison  methods  on  BRCA,
ments, hyper-parameters      𝑘and 𝑓𝑒𝑎_𝑛𝑢𝑚were determined by perform-                                                                   KIPAN and SCC datasets. To be more specific, MOGLAM consistently
ing cross-validation on the training data. We took the BRCA dataset                                                                    outperformed BPLSDA and BSPLSDA methods on different datasets,
as an example to show the impact of these two hyper-parameters on                                                                      which indicates that integrating omic-specific FSDGCN, multi-omics
the classification performance. First, we searched for the optimal value                                                               attention mechanism and omic-integrated representation learning can
of 𝑘 from     {2,3,4,5,6,7,8,9,10}  . From     Fig.   4(a), we found that when                                                         achieve better classification performance during multi-omics integra-
𝑘 = 8  , the model obtained the best prediction results. Next, varying                                                                 tion. In addition, although MMGL and MOGLAM yielded the same
𝑓𝑒𝑎_𝑛𝑢𝑚within     {200 ,300 ,400 ,500 ,600 ,700 ,800}    and set  𝑓𝑒𝑎_𝑛𝑢𝑚= 400                                                         mean F1_weighted value on the KIPAN dataset, the corresponding
in  Fig.   4(b).  In  addition,  the  detailed  hyper-parameter  adjustment                                                            standard deviation of MOGLAM was smaller. Moreover, the ACC and
results of other experimental datasets are available in Supplementary                                                                  F1_macro values of MOGLAM were better than MMGL, which shows
Figures S1 and S2.                                                                                                                     that using multi-omics attention mechanism to adaptively weight the

D. Ouyang et al.
Table 1                                                                                                                                                Table 3
Classification performance of all methods on BRCA dataset.                                                                                             The results of ablation study on BRCA dataset.
  Method                 ACC                              F1_weighted                  F1_macro                                                                                                      ACC                        F1_weighted            F1_macro
  XGBoost                0.7566             ±  0.029             0.7488          ±  0.030             0.6876          ±  0.038                           FSNN_MOAM_OIRL            0.8198                       ±  0.004      0.8280          ±  0.004      0.7978          ±  0.005
  NN                        0.7604         ±  0.018             0.7574          ±  0.019             0.7030          ±  0.027                            FSGCN_MOAM_OIRL          0.8282                        ±  0.028      0.8372          ±  0.027      0.8116          ±  0.033
  BPLSDA                0.6234             ±  0.006             0.4906          ±  0.006             0.3074          ±  0.010                            FSDGCN_MOAM_concat     0.8122                          ±  0.017      0.8194          ±  0.015      0.7914          ±  0.020
  BSPLSDA              0.6266              ±  0.005             0.4938          ±  0.005             0.3146          ±  0.007                            FSDGCN_OIRL                   0.8342                  ±  0.020      0.8420          ±  0.019      0.8060          ±  0.018
  SMSPL                  0.7310            ±  0.031             0.7468          ±  0.028             0.7104          ±  0.031                            MOGLAM                                   0.8380       ±  0.023     0.8456           ±  0.022     0.8124           ±  0.028
  MOGONET            0.7886                 ±  0.021             0.7740          ±  0.029             0.7254          ±  0.037
  MoGCN                 0.8190              ±  0.025             0.8196          ±  0.027             0.7930          ±  0.026
  MMGL                  0.8030             ±  0.050             0.7912          ±  0.069             0.7398          ±  0.082                          Table 4
  MOGLAM                       0.8380       ±  0.023            0.8456           ±  0.022            0.8124           ±  0.028                         The results of ablation study on KIPAN dataset.
                                                                                                                                                                                                     ACC                        F1_weighted            F1_macro
Table 2                                                                                                                                                  FSNN_MOAM_OIRL            0.9608                       ±  0.008      0.9610          ±  0.008      0.9500          ±  0.012
Classification performance of all methods on KIPAN dataset.                                                                                              FSGCN_MOAM_OIRL          0.9616                        ±  0.010      0.9620          ±  0.010      0.9534          ±  0.014
  Method                 ACC                              F1_weighted                  F1_macro                                                          FSDGCN_MOAM_concat     0.9606                          ±  0.012      0.9610          ±  0.012      0.9518          ±  0.014
                                                                                                                                                         FSDGCN_OIRL                   0.9618                  ±  0.010      0.9620          ±  0.009      0.9528          ±  0.013
  XGBoost                0.9526             ±  0.003             0.9524          ±  0.003             0.9412          ±  0.006                           MOGLAM                                   0.9650       ±  0.007     0.9650           ±  0.007     0.9566           ±  0.011
  NN                        0.9568         ±  0.007             0.9570          ±  0.007             0.9482          ±  0.011
  BPLSDA                0.8050             ±  0.095             0.7504          ±  0.118             0.6022          ±  0.003
  BSPLSDA              0.7972              ±  0.108             0.7334          ±  0.149             0.5910          ±  0.021
  SMSPL                  0.9610            ±  0.004             0.9610          ±  0.004             0.9484          ±  0.010                          generates clusters with small intra-class scatter and large inter-class
  MOGONET            0.9350                 ±  0.009             0.9354          ±  0.008             0.9318          ±  0.014                         scatter. Even for the BRCA subtype classification task which is difficult
  MoGCN                 0.9474              ±  0.008             0.9476          ±  0.008             0.9338          ±  0.010                         to learn, the dimensionality reduction visualization of the embedding
  MMGL                  0.9648             ±  0.010             0.9650          ±  0.010             0.9562          ±  0.013
  MOGLAM                       0.9650       ±  0.007            0.9650           ±  0.007            0.9566           ±  0.011                         information learned by the proposed MOGLAM can better separate the
                                                                                                                                                       different subtypes than MOGONET in Supplementary Figure S4. These
                                                                                                                                                       results show that the embedding representation obtained by omic-
embedding information of different omics may improve the classifi-                                                                                     specific FSDGCN, multi-omics attention mechanism, omic-integrated
cation performance and robustness of the model to a certain extent.                                                                                    representation learning is highly discriminative and beneficial to the
Further, MOGLAM significantly outperformed GCN-based MOGONET                                                                                           downstream classification task.
and MoGCN methods on all datasets. In particular, compared with                                                                                        4.6. Ablation studies
MOGONET  that  only  integrates  multi-omics  label  information,  the
proposed MOGLAM method achieved relative performance gains over                                                                                              Toverifytheeffectivenessofdynamicgraphconvolutionalnetwork
MOGONET by 6.3%, 9.3% and 11.99% in terms of ACC, F1_weighted                                                                                          with feature selection (FSDGCN), multi-omics attention mechanism
and F1_macro on the BRCA dataset. This result suggests that adap-                                                                                      (MOAM) and omic-integrated representation learning (OIRL) in multi-
tive graph learning may obtain omics embedding information that is                                                                                     omicsclassificationtasks,weconductedextensiveablationexperiments
more beneficial for downstream classification tasks. Compared with                                                                                     for the proposed method, in which four variants of MOGLAM were
simple concatenation-based XGBoost and NN methods, MOGLAM also                                                                                         compared. (1) FSNN_MOAM_OIRL: we replaced the FSDGCN part with
obtained  the  best  classification  performance.  Especially,  MOGLAM                                                                                 afullyconnectedneuralnetwork(NN)toexploretheeffectofconsider-
achieved ACC of 0.8380, F1_weighted of 0.8456, and F1_macro of                                                                                         ingsamplesimilarityandperformingaggregationlearninginthegraph
0.8124 in the difficult task of BRCA subtype classification, outperform-                                                                               manner on prediction performance. For a fairer comparative analysis,
ing the most competitive performance of these two methods, which                                                                                       thefullyconnectedneuralnetworkalsoaddsfeatureindicatormatrices
demonstrates that MOGLAM has good multi-omics integration and                                                                                          with inner product regularization constraint. (2) FSGCN_MOAM_OIRL:
representation learning ability. Interestingly, machine learning-based                                                                                 in order to verify the necessity of adaptive graph learning in FSDGCN,
XGBoostandSMSPLmethodsachievedgoodpredictionperformancein                                                                                              dynamic graph convolutional network (DGCN) is used to replace FS-
easytaskssuchaskidneycancertypeclassificationandpan-cancerclas-                                                                                        DGCN. Similarly, the adopted DGCN not only has feature indicator
sification, even better than some deep learning-based methods, which                                                                                   matrices with inner product regularization constraint, but also has the
showsthatdeeplearning-basedmethodshavestrongnonlinearcapture                                                                                           same number of layers, hidden layer size and activation function as
capabilities,butawell-designedintegrationframeworkformulti-omics                                                                                       theFSDGCNpartofthemodel.(3)FSDGCN_MOAM_concat:wedirectly
data is required to achieve superior classification performance. How-                                                                                  concatenated the learned multi-omics embedding information instead
ever, we also observed that the performance of XGBoost and SMSPL                                                                                       of using omic-integrated representation learning in order to explore
methods  is  poor  on  the  BRCA  dataset,  while  deep  learning-based                                                                                the impact of exploiting the common and complementary information
MOGLAM,MMGLandMoGCNhavegoodpredictionperformance.This                                                                                                  between different omics data types on the model classification perfor-
meansthatmachinelearning-basedmethodsareeasilyaffectedbydata                                                                                           mance. (4) FSDGCN_OIRL: we only removed the multi-omics attention
quality and less robust than deep learning-based methods.                                                                                              mechanism andkept other modules inMOGLAM unchangedto explore
                                                                                                                                                       whether adaptively weighting the embedding information of different
4.5. Visualization of the embedding representation                                                                                                     omics can improve the classification performance.
                                                                                                                                                             Tables3and4show the results of ablation experiments applying
      To evaluate the learning ability of the methods more intuitively,                                                                                the different modules of MOGLAM on BRCA and KIPAN datasets, from
we utilized t-SNE [38] to perform dimensionality reduction visualiza-                                                                                  which we can see that the proposed MOGLAM method outperformed
tion analysis for KIPAN, SCC and BRCA datasets, respectively. Note                                                                                     the four variant methods. To be more specific, FSDGCN_MOAM_concat
that we applied the embedding information for final classification to                                                                                  had the worst classification performance, which means that simply
perform visualization analysis. As shown inFig.5, compared with the                                                                                    connecting the learned embedding information of different omics can-
GCN-based MOGONET method, our proposed MOGLAM method can                                                                                               not effectively integrate multi-omics and capture the common and
cluster well into several clusters corresponding to the classes on three                                                                               complementary information between omics. Furthermore, the classi-
different datasets. Especially, the embedding representation learned by                                                                                fication performance of FSNN_MOAM_OIRL was lower than that of
MOGLAM not only preserves the original class distribution well on                                                                                      MOGLAM and FSGCN_MOAM_OIRL, which indicates that the aggre-
SCC and KIPAN datasets (seeFig.5(a)–(b) andFig.5(c)–(d)), but also                                                                                     gation learning of sample similarity networks in the graph manner

D. Ouyang et al.
Fig. 5.     Visualization of embedding representations of MOGONET and MOGLAM methods on different datasets. (a)–(b) Visualization results of SCC dataset. (c)–(d) Visualization
results of KIPAN dataset.
can effectively exploit the similarity between samples and improve the                                                                       Table 5
classification performance of the model. Further, MOGLAM achieved                                                                            The top 10 biomarkers of each omic were selected by MOGLAM on BRCA dataset.
betterperformancethanFSGCN_MOAM_OIRL,whichdemonstratesthat                                                                                     Omics data type                Biomarkers
adaptivelylearninggraphsthatarebeneficialtodownstreamclassifica-                                                                               mRNA expression              ESR1, SOX11, DEK, FABP7, ABCC11, C1orf106,
tion tasks can effectively improve classification performance compared                                                                                                                  DNMBP, ANKRD30A, AGR3, SPDEF
tofixedgraphs.Inaddition,comparedwiththeproposedMOGLAM,the                                                                                     DNA methylation              ACSM2A, NLRP8, TKTL2, SNORA42, PIK3C2A,
                                                                                                                                                                                        ATP8B1, KSR1, C1orf110, MIR128-1, ZNF516
prediction performance of FSDGCN_OIRL without multi-omics atten-                                                                               miRNA expression             hsa-mir-375, hsa-mir-187, hsa-mir-190b, hsa-mir-29a,
tion mechanism was also slightly decreased, indicating that adaptively                                                                                                                  hsa-mir-135b, hsa-mir-25, hsa-mir-9-1, hsa-mir-577,
weightingtheembeddinginformationofdifferentomicsthroughmulti-                                                                                                                           hsa-mir-149, hsa-mir-183
omics attention mechanism can improvethe prediction performance of
the model to a certain extent.
                                                                                                                                             reviewed previousbiomedicalstudies to verifywhether these biomark-
5. Important biomarkers identified by MOGLAM                                                                                                 ers affect the occurrence and development of breast cancer. Next,
                                                                                                                                             the following several representative studies were introduced. Holst
      The  important  application  of  multi-omic  analysis  is  to  identify                                                                𝑒𝑡𝑎𝑙. [39] found that estrogen receptor alpha (ESR1) gene amplifi-
biomarkers for early diagnosis and potential drug targets for treat-                                                                         cation is common in breast cancer. According to the study of mRNA
ments. To this end, we designed a feature indicator matrix with inner                                                                        expression, DEK is the oncogene of breast cancer and has been demon-
product regularization constraint in the FSDGCN module of MOGLAM,                                                                            strated to be highly expressed in breast cancer cells [40]. Shepherd
andoptimizedthefeatureindicatormatrixinanend-to-endmannerfor                                                                                 𝑒𝑡𝑎𝑙. [41] demonstrated that SOX11 is a key regulator of the growth
the discovery of important biomarkers. Then, we ranked the obtained                                                                          and invasion of basal-like breast cancer, and also found that high
feature importance scores and reported the top 10 important biomark-                                                                         expression of SOX11 is closely related to poor survival in women with
ers for different types of omics data. Finally, we performed subtype                                                                         breast cancer. ABCC11 is highly expressed in aggressive breast cancer
classification on the BRCA dataset and pan-cancer classification on                                                                          subtypes, and the expression of ABCC11 in tumors is associated with
the SCC dataset to analyze the biological significance of the identified                                                                     poor prognosis [42]. KSR1 inhibits breast cancer growth by regulating
important biomarkers.                                                                                                                        BRCA1 degradation [43]. Zehentmayr            𝑒𝑡𝑎𝑙. [44] found that hsa-mir-
                                                                                                                                             375isapredictoroflocalcontrolinpatientswithearlybreastcancer.In
                                                                                                                                             addition,hsa-mir-187hasbeenshowntobeanindependentprognostic
5.1. Breast cancer analysis based on BRCA dataset                                                                                            factor for breast cancer [45]. Dai         𝑒𝑡𝑎𝑙. [46] identified hsa-mir-190b
                                                                                                                                             regulating cell progression as a potential biomarker of breast cancer.
      For the BRCA subtype classification dataset, the top 10 important                                                                            To further verify that the biomarkers selected by our proposed
biomarkers in the three omics data are shown inTable5. Further, we                                                                           MOGLAM method are biologically meaningful, we used STRING [47],

D. Ouyang et al.
survival R package, ToppGene Suite [                    48 ] and Metascape [          49 ] to per-
form gene-gene interaction construction, patient survival curve anal-
ysis, gene-gene function and gene-disease enrichment analysis on the
BRCA dataset, respectively. On the one hand, we found that sev-
eral GO terms associated with breast cancer significantly enriched
genes identified by mRNA expression data, including epithelial cell
differentiation(GO:0030855,p=3.909E                         − 4),cellularresponsetowort-
mannin (GO:1904568, p = 2.700E                        − 3) and branch elongation involved
in  mammary  gland  duct  branching  (GO:0060751,  p  =  2.315E                                          − 3).
For example, cell cycle proteins in epithelial cell differentiation may
affect the progression of breast cancer [                   50 ]. Moreover, Yun    𝑒𝑡𝑎𝑙. [51 ]
demonstratedthatWortmannincaninhibittheproliferationandinduce
apoptosis of MCF-7 breast cancer cells. In addition, the morphology
of ductal branches is very significant in the development of mammary
gland [   52 ]. Meanwhile, macrophage colony-stimulating factor produc-
tion (GO:0036301, p = 2.363E                    − 3) was significantly enriched by genes
selected from DNA methylation data. Richardsen            𝑒𝑡𝑎𝑙. [53 ] revealed
that macrophage colony-stimulating factor is correlated with breast
cancer progression and poor prognosis. On the other hand, Metascape
is used to further explore the enrichment analysis of selected genes                                                                   Fig. 6.     For mRNA expression data, enrichment analysis of selected biomarkers and
and diseases based on mRNA expression data. As shown in                                   Fig.  6, we                                  diseases based on the BRCA dataset.
observed that these genes are significantly enriched in several breast
cancer-related diseases. For example, six and four genes were enriched
in invasive carcinoma of breast and basal-Like breast carcinoma, re-
spectively. Further, we constructed an interaction network of genes
selected by mRNA expression and DNA methylation data through uti-
lizingSTRING.SupplementaryFigureS5showstheinteractionnetwork
between these genes, from which we can see that the mRNA genes
ESR1, SOX11, DEK, FABP7, AGR3 and the DNA methylation genes
TKTL2,PIK3C2A,KSR1areinthesameinteractionnetworkandassoci-
ated with breast cancer-related frequently altered genes. For instance,
the expression of FKBP4 is an important prognostic indicator and a
potential drug target for the treatment of luminal A subtype breast
cancer [    54 ]. The increased expression of PELP1 is positively associated
with the markers of poor prognosis in invasive breast cancer [                               55 ].
Next, through using survival R package to draw survival curves for
the genes selected by mRNA expression and DNA methylation data.
Fig.  7, Supplementary Figures S6 and S7 indicate that the expression
levels of selected some genes based on mRNA expression and DNA                                                                         Fig. 7.     Effect of FABP7 expression level on patient survival time based on mRNA
methylation data can affect the survival time of breast cancer patients,                                                               expression data from the BRCA dataset.
which further demonstrates that the expression levels of selected genes
have clinical significance and may be related to the prognosis of breast
cancer [    43 ,56 ]. Finally, we calculated the cosine similarity between                                                             cancer can be used to target drugs for the prevention and treatment
selected genes from mRNA expression, DNA methylation and miRNA                                                                         of human cancer. Meanwhile, aberrant patterns of histone modifica-
expression data, respectively. From Supplementary Figure S8, we ob-                                                                    tions are also associated with human malignancies [                           61 ]. Furthermore,
served that there are high similarities between the identified features,                                                               transcription factor 21 (TCF21) plays a crucial role in a wide range
especially the genes obtained based on mRNA expression data, which                                                                     of  biological  processes,  and  its  dysregulation  is  closely  related  to
further verifies that the use of inner product regularization is able to                                                               cancer [    62 ]. Yan  𝑒𝑡𝑎𝑙. [63 ] revealed that TBX5-AS1 may be one of
consider the correlation between different features.                                                                                   thestrongestcandidatesforpredictingtheprognosisofmalesquamous
5.2. Pan-cancer analysis based on SCC dataset                                                                                          cell carcinoma patients. UTX has also been shown to act as the first
                                                                                                                                       mutatedhistonedemethylasegeneassociatedwithhumancancer[                                      64 ].
                                                                                                                                       For genes identified by DNA methylation data, NADP biosynthetic
      For SCC pan-cancer classification dataset,                     Table    6 shows the top 10                                       process (GO:0006741, p = 1.736E                      − 3) was significantly enriched by
important biomarkers identified by MOGLAM on three different omics                                                                     these genes. Rather     𝑒𝑡𝑎𝑙. [65 ] demonstrated that NADP is converted
data.ForgenesidentifiedbymRNAexpressiondata,wefoundthatsev-                                                                            to NADPH and overexpressed in tumor cells, which provides an op-
eralcancer-relatedGOtermsweresignificantlyenrichedthroughusing                                                                         portunity to target NADPH synthesis for cancer treatment. Based on
ToppGene Suite, including histone lysine demethylation (GO:0070076,                                                                    the above analysis, we observed that biological pathways enriched
p=6.76-8E       − 5),histonemodification(GO:0016570,p=1.483E                               − 3)and                                     by these genes are generally associated with human-related cancers,
intracellularsteroidhormonereceptorsignalingpathway(GO:0030518,                                                                        which indicates that the genes identified by using pan-cancer data
p=1.451E       − 3).Forexample,thedysregulationofhistonelysinemethyl-                                                                  may be more extensive cancer genes. In addition, some studies have
transferases  and  demethylases  contributes  to  cancer  initiation  and                                                              also demonstrated that some of the selected genes are also closely
progression [      57 ]. Moreover, Højfeldt     𝑒𝑡𝑎𝑙. [58 ] demonstrated that                                                          associated with one of the three cancers on the SCC dataset. For
histone lysine demethylases can be targeted for anticancer therapy. In                                                                 instance, Nikolaidis    𝑒𝑡𝑎𝑙. [66 ] uncovered that PAX1 methylation can
addition, the activation change of steroid hormone receptors (SHRs) is                                                                 be used as an auxiliary biomarker for cervical cancer screening. The
also closely related to the occurrence and development of cancer [                                  59 ].                              overexpression of GGA2 leads to accumulation of EGFR protein and
Ahmad   𝑒𝑡𝑎𝑙.  [60 ]  found  that  dysregulation  of  SHRs  signaling  in                                                              increased EGFR signaling, which drives lung tumor progression [                                   67 ].

D. Ouyang et al.
Table 6                                                                                                                                                  Declaration of competing interest
The top 10 biomarkers of each omic were selected by MOGLAM on SCC dataset.
  Omics data type                Biomarkers                                                                                                                     The authors declare that they have no competing interests.
  mRNA expression              TCF21, TBX5-AS1, PAX1, SFTPA2, LHX8, KDM5D,
                                              UTY, EMX2, FOXG1, NRAP                                                                                     Data availability
  DNA methylation              GGA2, DPP9, FBXO15, OR10A6, UQCR10, TMEM39B,
                                              SPRR3, IDH2, TMEM60, MAP3K3
  miRNA expression             hsa-mir-206, hsa-mir-10a, hsa-mir-10b, hsa-mir-196b,                                                                             The implemented code and experimental dataset are available on-
                                              hsa-mir-1269a, hsa-mir-424, hsa-mir-145, hsa-mir-511,                                                      line athttps://github.com/Ouyang-Dong/MOGLAM.
                                              hsa-mir-29a, hsa-mir-1247
                                                                                                                                                         Appendix A. Supplementary data
Wang  𝑒𝑡𝑎𝑙. [68] showed that hsa-mir-1269a can be exploited as a                                                                                                Supplementary material related to this article can be found online
diagnostic marker and play an oncogenic role in non-small cell lung                                                                                      athttps://doi.org/10.1016/j.compbiomed.2023.107303.
cancer.Thehsa-mir-196bhasalsobeenfoundtodirectlytargetANXA1
in head and neck squamous cell carcinoma [69]. Moreover, we also                                                                                         References
constructed an interaction network of the 20 top-ranked genes from
mRNA expression and DNA methylation identified by the proposed                                                                                              [1]B.  Fürtig,  C.  Richter,  J.  Wöhnert,  H.  Schwalbe,  NMR  spectroscopy  of  RNA,
MOGLAM method. As shown in Supplementary Figure S9, the mRNA                                                                                                        ChemBioChem 4 (10) (2003) 936–962.
genes FOXG1, EMX2, LHX8 and the DNA methylation genes GGA2,                                                                                                 [2]J.R. Edwards, H. Ruparel, J. Ju, Mass-spectrometry DNA sequencing, Mutat. Res.
                                                                                                                                                                    Fund. Mol. Mech. Mut. 573 (1–2) (2005) 3–12.
IDH2 are in the same interaction network. In addition, these genes are                                                                                      [3]A.H.  Van  Vliet,  Next  generation  sequencing  of  microbial  transcriptomes:
also associated with genes that cause human cancer. Jiang              𝑒𝑡𝑎𝑙. [70]                                                                                   challenges and opportunities, FEMS Microbiol. Lett. 302 (1) (2010) 1–7.
demonstrated that the overexpression of GGA3 promotes non-small                                                                                             [4]M.A. Van De Wiel, T.G. Lien, W. Verlaat, W.N. van Wieringen, S.M. Wilting,
cell lung cancer proliferation by regulating TrkA receptor. Finally, we                                                                                             Better prediction by use of co-data: adaptive group-regularized ridge regression,
calculated the cosine similarity between biomarkers selected from the                                                                                               Stat. Med. 35 (3) (2016) 368–381.
three omics data on the SCC dataset. The results of Supplementary                                                                                           [5]A. Singh, C.P. Shannon, B. Gautier, F. Rohart, M. Vacher, S.J. Tebbutt, K.-A.
                                                                                                                                                                    Lê Cao, DIABLO: an integrative approach for identifying key molecular drivers
Figure S10 indicate that these features have high similarity, which                                                                                                 from multi-omics assays, Bioinformatics 35 (17) (2019) 3055–3062.
further verifies the effectiveness of inner product regularization.                                                                                         [6]C. Park, J. Ha, S. Park, Prediction of Alzheimer’s disease based on deep neural
                                                                                                                                                                    network by integrating gene expression and DNA methylation dataset, Expert
6. Discussion and conclusion                                                                                                                                        Syst. Appl. 140 (2020) 112873.
                                                                                                                                                            [7]M. Taron, R. Rosell, E. Felip, P. Mendez, J. Souglakos, M.S. Ronco, C. Queralt,
      In this paper, we propose an end-to-end supervised multi-omics in-                                                                                            J. Majo, J.M. Sanchez, J.J. Sanchez, et al., BRCA1 mRNA expression levels as
                                                                                                                                                                    an indicator of chemoresistance in lung cancer, Hum. Mol. Gen. 13 (20) (2004)
tegration method named MOGLAM for biomedical classification tasks,                                                                                                  2443–2449.
which mainly includes FSDGCN, multi-omics attention mechanism and                                                                                           [8]P.M. Das, R. Singal, DNA methylation and cancer, J. Clin. Oncol. 22 (22) (2004)
omic-integratedrepresentationlearning.ComparedwithstandardGCN,                                                                                                      4632–4642.
FSDGCN can adaptively adjust the graph structure to learn the sample                                                                                        [9]N. Ashour, J.C. Angulo, G. Andrés, R. Alelú, A. González-Corpas, M.V. Toledo,
similarity network that is beneficial to classification tasks while select-                                                                                         J.M.  Rodríguez-Barbero,  J.I.  López,  M.  Sánchez-Chapado,  S.  Ropero,  A  DNA
ing important biomarkers. Meanwhile, omic-integrated representation                                                                                                 hypermethylation profile reveals new potential biomarkers for prostate cancer
                                                                                                                                                                    diagnosis and prognosis, Prostate 74 (12) (2014) 1171–1182.
learning can effectively capture complex common and complementary                                                                                         [10]C. Urbich, A. Kuehbacher, S. Dimmeler, Role of microRNAs in vascular diseases,
informationbetweenomicsduringmulti-omicsintegration.Inaddition,                                                                                                     inflammation, and angiogenesis, Cardiovasc. Res. 79 (4) (2008) 581–588.
weighting the embedding representations of different omics through                                                                                        [11]S. Bandyopadhyay, R. Mitra, U. Maulik, M.Q. Zhang, Development of the human
multi-omics attention mechanism can obtain more reasonable omics                                                                                                    cancer microRNA network, Silence 1 (1) (2010) 1–14.
integration information and improve classification performance. Fur-                                                                                      [12]S.  Huang,  K.  Chaudhary,  L.X.  Garmire,  More  is  better:  recent  progress  in
                                                                                                                                                                    multi-omics data integration methods, Front. Genet. 8 (2017) 84.
thermore, the dimensionality reduction visualization of the embedding                                                                                     [13]S. Zhang, C.-C. Liu, W. Li, H. Shen, P.W. Laird, X.J. Zhou, Discovery of multi-
representations learned by MOGLAM can well distinguish different                                                                                                    dimensional modules by integrative analysis of cancer genomic data, Nucleic
categories, which indicates that MOGLAM has good learning capa-                                                                                                     Acids Res. 40 (19) (2012) 9379–9391.
bilities of multi-omics integration and embedding representation. In                                                                                      [14]G. Tini, L. Marchetti, C. Priami, M.-P. Scott-Boyer, Multi-omics integration—a
addition,experimentalresultsondifferentdatasetsshowthatMOGLAM                                                                                                       comparison of unsupervised clustering methodologies, Brief. Bioinform. 20 (4)
achievesbetterclassificationperformanceandrobustnessthantheeight                                                                                                    (2019) 1269–1279.
                                                                                                                                                          [15]O.P. Günther, V. Chen, G.C. Freue, R.F. Balshaw, S.J. Tebbutt, Z. Hollander, M.
comparison methods. Moreover, the results of ablation experiments                                                                                                   Takhar, W.R. McMaster, B.M. McManus, P.A. Keown, et al., A computational
also further validate the effectiveness of FSDGCN, multi-omics atten-                                                                                               pipeline for the development of multi-marker bio-signature panels and ensemble
tion mechanism and omic-integrated representation learning, where                                                                                                   classifiers, BMC Bioinformatics 13 (1) (2012) 1–18.
omic-integrated representation learning may play a more important                                                                                         [16]Z. Yang, N. Wu, Y. Liang, H. Zhang, Y. Ren, Smspl: Robust multimodal approach
role in multi-omics integration classification tasks. Last but not least,                                                                                           to integrative analysis of multiomics data, IEEE Trans. Cybern.  (2020).
                                                                                                                                                          [17]Z. Huang, X. Zhan, S. Xiang, T.S. Johnson, B. Helm, C.Y. Yu, J. Zhang, P. Salama,
MOGLAM can also efficiently identify meaningful potential biomarkers                                                                                                M. Rizkalla, Z. Han, et al., SALMON: survival analysis learning with multi-omics
foreachomicsdatainanend-to-endmannerwithoutadditionalfeature                                                                                                        neural networks on breast cancer, Front. Genet. 10 (2019) 166.
selectionmethods.Tosumup,MOGLAMisanend-to-endinterpretable                                                                                                [18]M. Kang, E. Ko, T.B. Mersha, A roadmap for multi-omics data integration using
multi-omics integration method with superior performance and good                                                                                                   deep learning, Brief. Bioinform. 23 (1) (2022) bbab454.
interpretability.                                                                                                                                         [19]X. Li, J. Ma, L. Leng, M. Han, M. Li, F. He, Y. Zhu, MoGCN: A multi-omics
                                                                                                                                                                    integration method based on graph convolutional network for cancer subtype
Funding                                                                                                                                                             analysis, Front. Genet. (2022) 127.
                                                                                                                                                          [20]T. Wang, W. Shao, Z. Huang, H. Tang, J. Zhang, Z. Ding, K. Huang, MOGONET
                                                                                                                                                                    integrates multi-omics data using graph convolutional networks allowing patient
      This  work  was  supported  in  part  by  the  major  key  project  of                                                                                        classification and biomarker identification, Nature Commun. 12 (1) (2021) 1–13.
PengChengLaboratoryundergrantPCL2023AS1-2,theMacauScience                                                                                                 [21]M. Welling, T.N. Kipf, Semi-supervised classification with graph convolutional
and Technology Development Funds Grands No. 0056/2020/AFJ from                                                                                                      networks,  in:  J.  International  Conference  on  Learning  Representations,  ICLR
the Macau Special Administrative Region of the People’s Republic                                                                                                    2017, 2016.
of China, and the Key Project for University of Educational Com-                                                                                          [22]A. Colaprico, T.C. Silva, C. Olsen, L. Garofano, C. Cava, D. Garolini, T.S. Sabedot,
                                                                                                                                                                    T.M. Malta, S.M. Pagnotta, I. Castiglioni, et al., TCGAbiolinks: an R/Bioconductor
mission of Guangdong Province of China Funds (Natural, Grant No.                                                                                                    package for integrative analysis of TCGA data, Nucleic Acids Res. 44 (8) (2016)
2019GZDXM005).                                                                                                                                                      e71.

D. Ouyang et al.
[23]L. Franceschi, M. Niepert, M. Pontil, X. He, Learning discrete structures for graph                                                                                    [47]D. Szklarczyk, A.L. Gable, D. Lyon, A. Junge, S. Wyder, J. Huerta-Cepas, M.
            neural networks, in: International Conference on Machine Learning, PMLR, 2019,                                                                                            Simonovic, N.T. Doncheva, J.H. Morris, P. Bork, et al., STRING v11: protein–
            pp. 1972–1982.                                                                                                                                                            protein  association  networks  with  increased  coverage,  supporting  functional
[24]Q. Zhu, B. Du, P. Yan, Multi-hop convolutions on weighted graphs, 2019, arXiv                                                                                                     discovery in genome-wide experimental datasets, Nucleic Acids Res. 47 (D1)
            preprintarXiv:1911.04978.                                                                                                                                                 (2019) D607–D613.
[25]Y. Chen, L. Wu, M. Zaki, Iterative deep graph learning for graph neural networks:                                                                                      [48]J. Chen, E.E. Bardes, B.J. Aronow, A.G. Jegga, ToppGene suite for gene list
            Better and robust node embeddings, Adv. Neural Inf. Process. Syst. 33 (2020)                                                                                              enrichment analysis and candidate gene prioritization, Nucleic Acids Res. 37
            19314–19326.                                                                                                                                                              (suppl_2) (2009) W305–W311.
[26]S. Zheng, Z. Zhu, Z. Liu, Z. Guo, Y. Liu, Y. Yang, Y. Zhao, Multi-modal graph                                                                                          [49]Y. Zhou, B. Zhou, L. Pache, M. Chang, A.H. Khodabakhshi, O. Tanaseichuk, C.
            learning for disease prediction, IEEE Trans. Med. Imaging (2022).                                                                                                         Benner, S.K. Chanda, Metascape provides a biologist-oriented resource for the
[27]B. Jiang, Z. Zhang, D. Lin, J. Tang, B. Luo, Semi-supervised learning with graph                                                                                                  analysis of systems-level datasets, Nat. Commun. 10 (1) (2019) 1–10.
            learning-convolutional networks, in: Proceedings of the IEEE/CVF Conference on                                                                                 [50]C.E. Caldon, R.L. Sutherland, E.A. Musgrove, Cell cycle proteins in epithelial
            Computer Vision and Pattern Recognition, 2019, pp. 11313–11320.                                                                                                           cell  differentiation:  implications  for  breast  cancer,  Cell  Cycle  9  (10)  (2010)
[28]Z. Lin, M. Luo, Z. Peng, J. Li, Q. Zheng, Nonlinear feature selection on attributed                                                                                               1918–1928.
            networks, Neurocomputing 410 (2020) 161–173.                                                                                                                   [51]J. Yun, Y. Lv, Q. Yao, L. Wang, Y. Li, J. Yi, Wortmannin inhibits proliferation
[29]H.  Li,  X.  Shi,  X.  Zhu,  S.  Wang,  Z.  Zhang,  FSNet:  Dual  interpretable  graph                                                                                            and induces apoptosis of MCF-7 breast cancer cells, Eur. J. Gynaecol. Oncol. 33
            convolutional network for Alzheimer’s disease analysis, IEEE Trans. Emerg. Top.                                                                                           (4) (2012) 367–369.
            Comput. Intell. (2022).                                                                                                                                        [52]M.D.  Sternlicht,  Key  stages  in  mammary  gland  development:  the  cues  that
[30]M. Qi, T. Wang, F. Liu, B. Zhang, J. Wang, Y. Yi, Unsupervised feature selection                                                                                                  regulate ductal branching morphogenesis, Breast Cancer Res. 8 (1) (2005) 1–11.
            by regularized matrix factorization, Neurocomputing 273 (2018) 593–610.                                                                                        [53]E. Richardsen, R.D. Uglehus, S.H. Johnsen, L.-T. Busund, Macrophage-colony
[31]J.  Hu,  L.  Shen,  G.  Sun,  Squeeze-and-excitation  networks,  in:  Proceedings  of                                                                                             stimulating  factor  (CSF1)  predicts  breast  cancer  progression  and  mortality,
            the IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp.                                                                                                 Anticancer Res. 35 (2) (2015) 865–874.
            7132–7141.                                                                                                                                                     [54]H. Xiong, Z. Chen, W. Zheng, J. Sun, Q. Fu, R. Teng, J. Chen, S. Xie, L. Wang,
[32]C.  Lee,  M.  van  der  Schaar,  A  variational  information  bottleneck  approach                                                                                                X.-F. Yu, et al., FKBP4 is a malignant indicator in luminal A subtype of breast
            to  multi-omics  data  integration,  in:  International  Conference  on  Artificial                                                                                       cancer, J. Cancer 11 (7) (2020) 1727.
            Intelligence and Statistics, PMLR, 2021, pp. 1513–1521.                                                                                                        [55]H.O. Habashy, D.G. Powe, E.A. Rakha, G. Ball, R.D. Macmillan, A.R. Green, I.O.
[33]M. Picard, M.-P. Scott-Boyer, A. Bodein, O. Périn, A. Droit, Integration strategies                                                                                               Ellis, The prognostic significance of PELP1 expression in invasive breast cancer
            of multi-omics data for machine learning analysis, Comput. Struct. Biotechnol.                                                                                            with emphasis on the ER-positive luminal-like subtype, Breast Cancer Res. Treat.
            J. 19 (2021) 3735–3746.                                                                                                                                                   120 (3) (2010) 603–612.
[34]A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser,                                                                                      [56]Q. Xie, Y.-s. Xiao, S.-c. Jia, J.-x. Zheng, Z.-c. Du, Y.-c. Chen, M.-t. Chen, Y.-k.
            I. Polosukhin, Attention is all you need, Adv. Neural Inf. Process. Syst. 30 (2017).                                                                                      Liang, H.-y. Lin, D. Zeng, FABP7 is a potential biomarker to predict response
[35]D.P. Kingma, J. Ba, Adam: A method for stochastic optimization, 2014, arXiv                                                                                                       to neoadjuvant chemotherapy for breast cancer, Cancer Cell Int. 20 (1) (2020)
            preprintarXiv:1412.6980.                                                                                                                                                  1–14.
[36]Y.  LeCun,  Y.  Bengio,  G.  Hinton,  Deep  learning,  Nature  521  (7553)  (2015)                                                                                     [57]R.A.  Varier,  H.M.  Timmers,  Histone  lysine  methylation  and  demethylation
            436–444.                                                                                                                                                                  pathways in cancer, Biochim Biophys Acta (BBA)-Rev. Cancer 1815 (1) (2011)
[37]T. Chen, C. Guestrin, Xgboost: A scalable tree boosting system, in: Proceedings                                                                                                   75–89.
            of the 22nd Acm Sigkdd International Conference on Knowledge Discovery and                                                                                     [58]J.W. Hø jfeldt, K. Agger, K. Helin, Histone lysine demethylases as targets for
            Data Mining, 2016, pp. 785–794.                                                                                                                                           anticancer therapy, Nat. Rev. Drug Discov. 12 (12) (2013) 917–930.
[38]L. Van der Maaten, G. Hinton, Visualizing data using t-SNE, J. Mach. Learn. Res.                                                                                       [59]R.R. Singh, R. Kumar, Steroid hormone receptor signaling in tumorigenesis, J.
            9 (11) (2008).                                                                                                                                                            Cell Biochem. 96 (3) (2005) 490–505.
[39]F. Holst, P.R. Stahl, C. Ruiz, O. Hellwinkel, Z. Jehan, M. Wendland, A. Lebeau,                                                                                        [60]N. Ahmad, R. Kumar, Steroid hormone receptors in cancer development: a target
            L. Terracciano, K. Al-Kuraya, F. Jänicke, et al., Estrogen receptor alpha (ESR1)                                                                                          for cancer therapeutics, Cancer Lett. 300 (1) (2011) 1–9.
            gene amplification is frequent in breast cancer, Nature Genet. 39 (5) (2007)                                                                                   [61]J.E. Audia, R.M. Campbell, Histone modifications and cancer, Cold Spring Harbor
            655–660.                                                                                                                                                                  Perspect. Biol. 8 (4) (2016) a019521.
[40]L. Privette Vinnedge, R. McClaine, P.K. Wagh, K.A. Wikenheiser-Brokamp, S.E.                                                                                           [62]X. Ao, W. Ding, Y. Zhang, D. Ding, Y. Liu, TCF21: a critical transcription factor
            Waltz,  S.I.  Wells,  The  human  DEK  oncogene  stimulates              𝛽-catenin  signaling,                                                                            in health and cancer, J. Mol. Med. 98 (8) (2020) 1055–1068.
            invasion and mammosphere formation in breast cancer, Oncogene 30 (24) (2011)                                                                                   [63]T. Yan, K. Wang, Q. Zhao, J. Zhuang, H. Shen, G. Ma, L. Cong, J. Du, Gender
            2741–2752.                                                                                                                                                                specific eRNA TBX5-AS1 as the immunological biomarker for male patients with
[41]J.H. Shepherd, I.P. Uray, A. Mazumdar, A. Tsimelzon, M. Savage, S.G. Hilsen-                                                                                                      lung squamous cell carcinoma in pan-cancer screening, PeerJ 9 (2021) e12536.
            beck,  P.H.  Brown,  The  SOX11  transcription  factor  is  a  critical  regulator                                                                             [64]G. Van Haaften, G.L. Dalgliesh, H. Davies, L. Chen, G. Bignell, C. Greenman,
            of  basal-like  breast  cancer  growth,  invasion,  and  basal-like  gene  expression,                                                                                    S. Edkins, C. Hardy, S. O’meara, J. Teague, et al., Somatic mutations of the
            Oncotarget 7 (11) (2016) 13106.                                                                                                                                           histone H3k27 demethylase gene UTX in human cancer, Nat. Genet. 41 (5)
[42]A. Yamada, T. Ishikawa, I. Ota, M. Kimura, D. Shimizu, M. Tanabe, T. Chishima,                                                                                                    (2009) 521–523.
            T. Sasaki, Y. Ichikawa, S. Morita, et al., High expression of ATP-binding cassette                                                                             [65]G.M. Rather, A.A. Pramono, Z. Szekely, J.R. Bertino, P.M. Tedeschi, In cancer,
            transporter ABCC11 in breast tumors is associated with aggressive subtypes and                                                                                            all roads lead to NADPH, Pharmacol. Ther. 226 (2021) 107864.
            low disease-free survival, Breast Cancer Res. Treat. 137 (3) (2013) 773–782.                                                                                   [66]C. Nikolaidis, E. Nena, M. Panagopoulou, I. Balgkouranidou, M. Karaglani, E.
[43]J. Stebbing, H. Zhang, Y. Xu, L.C. Lit, A. Green, A. Grothey, Y. Lombardo, M.                                                                                                     Chatzaki, T. Agorastos, T.C. Constantinidis, PAX1 methylation as an auxiliary
            Periyasamy, K. Blighe, W. Zhang, et al., KSR1 regulates BRCA1 degradation and                                                                                             biomarker for cervical cancer screening: a meta-analysis, Cancer Epidemiol. 39
            inhibits breast cancer growth, Oncogene 34 (16) (2015) 2103–2114.                                                                                                         (5) (2015) 682–686.
[44]F. Zehentmayr, C. Hauser-Kronberger, B. Zellinger, F. Hlubek, C. Schuster, U.                                                                                          [67]H. O’Farrell, B. Harbourne, Z. Kurlawala, Y. Inoue, A.L. Nagelberg, V.D. Martinez,
            Bodenhofer, G. Fastner, H. Deutschmann, P. Steininger, R. Reitsamer, et al.,                                                                                              D. Lu, M.H. Oh, B.P. Coe, K.L. Thu, et al., Integrative genomic analyses identifies
            Hsa-mir-375 is a predictor of local control in early stage breast cancer, Clin.                                                                                           GGA2 as a cooperative driver of EGFR-mediated lung tumorigenesis, J. Thorac.
            Epigenetics 8 (1) (2016) 1–13.                                                                                                                                            Oncol. 14 (4) (2019) 656–671.
[45]L. Mulrane, S.F. Madden, D.J. Brennan, G. Gremel, S.F. McGee, S. McNally, F.                                                                                           [68]X. Wang, X. Jiang, J. Li, J. Wang, H. Binang, S. Shi, W. Duan, Y. Zhao, Y.
            Martin, J.P. Crown, K. Jirström, D.G. Higgins, et al., Mir-187 is an independent                                                                                          Zhang, Serum exosomal miR-1269a serves as a diagnostic marker and plays an
            prognostic factor in breast cancer and confers increased invasive potential in                                                                                            oncogenic role in non-small cell lung cancer, Thorac. Cancer 11 (12) (2020)
            vitromir-187 and breast cancer, Clin. Cancer Res. 18 (24) (2012) 6702–6713.                                                                                               3436–3447.
[46]W. Dai, J. He, L. Zheng, M. Bi, F. Hu, M. Chen, H. Niu, J. Yang, Y. Luo, W.                                                                                            [69]S. Álvarez-Teijeiro, S.T. Menéndez, M. Villaronga, E. Pena-Alonso, J.P. Rodrigo,
            Tang, et al., miR-148b-3p, miR-190b, and miR-429 regulate cell progression and                                                                                            R.O. Morgan, R. Granda-Díaz, C. Salom, M.P. Fernandez, J.M. García-Pedrero,
            act as potential biomarkers for breast cancer, J. Breast Cancer 22 (2) (2019)                                                                                             Annexin  A1  down-regulation  in  head  and  neck  squamous  cell  carcinoma  is
            219–236.                                                                                                                                                                  mediated via transcriptional control with direct involvement of miR-196a/b, Sci.
                                                                                                                                                                                      Rep. 7 (1) (2017) 1–12.
                                                                                                                                                                           [70]B.-G. Jiang, Y.-R. Zhou, Up-regulated GGA3 promotes non-small cell lung cancer
                                                                                                                                                                                      proliferation by regulating TrkA receptor, Transl. Cancer Res. 8 (7) (2019) 2543.

