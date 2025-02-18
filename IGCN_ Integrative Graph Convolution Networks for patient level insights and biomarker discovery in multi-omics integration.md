   IGCN: Integrative Graph Convolution Networks
 for patient level insights and biomarker discovery
                                     in multi-omics integration
    Cagri Ozdemir              1,3,4 ,  Yashu Vashishath                 1,3,4 ,  Serdar Bozdag             1,2,3,4  ,
            and for the Alzheimer’s Disease Neuroimaging Initiative                                                  †
1Department of Computer Science and Engineering, University of North Texas,
                                                           Denton, TX, USA.
  2Department of Mathematics, University of North Texas, Denton, TX, USA.
        3BioDiscovery Institute, University of North Texas, Denton, TX, USA.
     4Center for Computational Life Sciences USA, University of North Texas,
                                                           Denton, TX, USA.
                            Contributing authors:cagri.ozdemir@unt.edu;
                      yashu.vashishath@unt.edu;serdar.bozdag@unt.edu;
† Data used in preparation of this article were obtained from the Alzheimer’s Disease
          Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the
investigators within the ADNI contributed to the design and implementation of ADNI
 and/or provided data but did not participate in analysis or writing of this report. A
   complete listing of ADNI investigators can be found at:https://adni.loni.usc.edu/
               wp-content/uploads/how                         to   apply/ADNI               Acknowledgement                   List.pdf
                                                                    Abstract
     Developing computational tools for integrative analysis across multiple types of
     omics data has been of immense importance in cancer molecular biology and
     precision medicine research. While recent advancements have yielded integrative
     prediction solutions for multi-omics data, these methods lack a comprehensive
     and cohesive understanding of the rationale behind their specific predictions.
     To shed light on personalized medicine and unravel previously unknown char-
     acteristics within integrative analysis of multi-omics data, we introduce a novel
     integrative neural network approach for cancer molecular subtype and biomedi-
     cal classification applications, named Integrative Graph Convolutional Networks
     (IGCN). IGCN can identify which types of omics receive more emphasis for each
     patient to predict a certain class. Additionally, IGCN has the capability to pin-
     point significant biomarkers from a range of omics data types. To demonstrate
     the superiority of IGCN, we compare its performance with other state-of-the-art
     approaches across different cancer subtype and biomedical classification tasks.
                                                                            1

1 Introduction
Due to advancements in biotechnology, innovative omics technologies are constantly
emerging, allowing researchers to access multi-layered information from genome-wide
data. These multi-layered information can be obtained for the same set of samples,
leading to generation of multi-omics datasets. Cancer molecular subtype prediction is
a crucial area of research focused on classifying cancers into distinct subtypes based
on their molecular characteristics. These subtypes can provide valuable insights into
the biology of the cancer, predict patient outcomes, and guide personalized treatment
strategies[1,2].Predictingcancermolecularsubtypesthroughmulti-omicsintegration
may reveal complex interactions within biological systems and shed light on molecu-
lar mechanisms that contribute to cancer development and progression, which might
be missed when examining a single type of omics data alone [3–7]. Moreover, a mul-
titude of approaches have shown that integration of multi-omics data can contribute
to the precision medicine efforts in diseases such as Alzheimer’s Disease (AD). [8–10].
Whiledeepneuralnetworks(NN)-basedmethodshavebeenintroducedasmulti-omics
integrative tools for cancer subtype prediction and diverse biomedical classification
tasks [1–7], graph neural network (GNN)-based multi-omics integration approaches
have shown promising results [11–14].
      Multi-omics graph convolutional networks (MOGONET) has been introduced as a
supervisedmulti-omicsintegrationframeworkforawiderangeofbiomedicalclassifica-
tionapplications,whichusesseparategraphconvolutionalnetworks(GCN)forpatient
similarity networks based on mRNA expression, DNA methylation, and microRNA
(miRNA) expression data types [11]. MOGONET also utilizes View Correlation Dis-
covery Network (VCDN) to explore the cross-omics correlations at the label space
for effective multi-omics integration. Another computational tool named SUPREME,
a subtype prediction methodology, integrates multiple types of omics data using
GCN[12].Toobtainembeddings,SUPREMEconcatenatesthefeaturesfromallomics
data types in the input space and utilizes them as node attributes in each GCN
module to derive embeddings. Subsequently, SUPREME integrates these embeddings
and conducts comprehensive evaluations of all possible combinations. Most recently,
Trusted Multi-Omics integration framework based on hypergraph convolutional net-
works(calledHyperTMO)hasbeendeveloped[15].HyperTMOconstructshypergraph
structures to represent the associations between samples in single-omics data. Follow-
ingthat,featureextractionisconductedutilizingahypergraphconvolutionalnetwork,
while multi-omics integration occurs during the late stages of analysis.
      BesidestheseGNN-basedintegrativetoolsappliedformulti-omicsdatasets,several
GNN-based approaches have been introduced as more general integrative compu-
tational tools for multi-modal datasets. Relational Graph Convolutional Networks
(RGCN)[16]providesrelation-specifictransformations,i.e.dependingonthetypeand
directionofanedge,forlarge-scaleandmulti-modaldata.HeterogeneousGraphAtten-
tion Network (HAN) [17] generates meta-path-based networks from a multi-modal
network. The concept of meta-path can be applied to learn a sequence of relations
defined between different objects in a multi-modal graph [18]. After generating meta-
paths, HAN takes node-level attention (for each node using its meta-path-based
neighborhood) and association-level attention (for each meta-path) into consideration
                                                                              2

simultaneously. HAN employs a multi-layer perceptron (MLP) module to compute
class probabilities rather than taking advantage of graph topologies, which could
potentially limit the ability of the model to capture complementary information
from graph topologies to make predictions. Apart from HAN, in the context of han-
dling graph heterogeneity, Heterogeneous Graph Transformer (HGT) [19] has been
developed to maintain representations dependent on node and edge types.
      As outlined in [20], it has been argued that vanilla GCN and GAT methods could
outperform existing integrative approaches after making some modifications to the
networks. This underscores the necessity for more advanced computational method-
ologies in the integrative analysis of multi-omics data. In addition, the existing tools
have some limitations. Omics data, by their nature, do not inherently exhibit a graph
structure, thus a graph construction procedure is needed. However, constructing a
graph from omics data could suffer from data noise due to many possible factors
such as measurement inaccuracies, missing values, or inherent fluctuations within the
dataset. These factors can adversely affect tasks like clustering, classification, or link
prediction. Additionally, in the context of biomedical classification, different types of
omics data have the potential to reveal unique characteristics at the label space. In
other words, some omics types may demonstrate superior performance when predict-
ing one disease label, while others might excel in predicting a different disease type.
Therefore,directlyfusingdifferenttypesofomicsdatawithoutconsideringthesample
level importance of different omics networks may be liable to make wrong predictions
and cause some level of performance degradation. Furthermore, multitudes of existing
multi-omics integration tools do not explain how and why their models came to the
prediction. In predictive modeling, a crucial trade-off arises: Do we merely desire the
prediction, or are we interested in understanding the rationale behind it? [21]. Since
each of multi-omics type captures a different part of the underlying biology, under-
standing the ’why’ can contribute to a deeper comprehension of the problem and
advance the road toward precision medicine.
      To address these limitations, we introduce a novel supervised integrative graph
convolutional networks (IGCN) architecture that operates on multi-omics data struc-
tures.InIGCN,amulti-GCNmoduleisinitiallyemployedtoextractnodeembeddings
from each network. A personalized attention module is then proposed to fuse the
multiple node embeddings into a weighted form. Unlike previous multi-omics integra-
tion studies, the attention mechanism assigns different attention coefficients to each
node/sample for each data modality to help identify which data modality receives
more emphasis to predict a certain class type. This feature makes IGCN interpretable
in terms of understanding the rationale behind the prediction at the sample level.
Furthermore, IGCN has the capability to assign attention coefficients to features for
eachsamplefromarangeofomicsdatatypes,whichwouldfacilitateidentifyingomics
biomakersassociatedwithphenotypesofinterest.Tothebestofourknowledge,IGCN
stands out as the first supervised integrative approach that provides patient level
insights and biomarkers in multi-omics integration.
      We presented our experimental results on four classification tasks: breast invasive
carcinomaandglioblastoma(GBM)molecularsubtypeclassificationusingTheCancer
Genome Atlas dataset [22]; Alzheimer’s Disease (AD) patients vs. cognitively normal
                                                                             3

(CN) classification using The Religious Orders Study and Memory and Aging Project
(ROSMAP) cohort [23]; and AD, Mild Cognitive Impairment (MCI), and CN classi-
fication task using Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset [24].
Our experimental results show that our proposed model outperforms the state-of-the-
art and baseline methods. IGCN identifies which types of omics data receive more
emphasisforeachpatientwhenpredictingaspecific class.Additionally,IGCNhasthe
capability to pinpoint significant biomarkers from a range of omics data types.
2 Results
2.1 Overview of IGCN architecture
IGCNintegratesmulti-omicsdataandrevealspatientlevelinsightsregardingboththe
keyomicstypesandfeaturesforbiomedicalclassificationtasks.TheoverviewofIGCN
architecture is illustrated in            Fig. (1    ). We first construct graphs for each omics type
(Eq. (2) and Eq. (3)). Subsequently, IGCN utilizes GCN modules on graphs to learn
the node embeddings. As depicted in Algorithm1, the hyperparameter                                                    ϵ determines
a threshold for correlation in graph construction. In graph construction, it is common
to encounter data noise from various sources, such as measurement inaccuracies, miss-
ing values, or inherent dataset fluctuations. To alleviate this noise, the Normalized
Temperature-scaled Cross Entropy loss (NT-Xent loss) (Eq. (6)) is utilized for each
GCN module. Following this, we introduce a personalized attention module to merge
the multiple node embeddings into a weighted representation (Eq. (7) and Eq. (8)).
Differingfrompriorresearch,thisattentionmechanismassignsdistinctattentioncoef-
ficients to each patient, facilitating the identification which data modality is more
influential in predicting a particular class type. Similar to the attention mechanism
module, the feature ranking module also offers personalized insights regarding feature
importance (Eq. (12)). To our knowledge, IGCN is the first supervised integrative
approach that offers patient-level insights and biomarkers in multi-omics integration,
making it a significant advancement in the field.
2.2 Biomedical classification tasks
To perform breast invasive carcinoma (BRCA) subtype classification task, mRNA
expression,DNAmethylation,andmiRNAexpressiondatawerecollectedfromTCGA
project  data  portal.  PAM50  labels  of  the  tumor  samples  were  obtained  as  the
ground truth class labels [25]. Specifically, we have five different class labels: Basal-
like, HER2-Enriched, Luminal-A, Luminal-B, and Normal-like. For GBM molecular
subtype classification task, we acquired mRNA expression and miRNA expression
data from the the Broad Institute’s Firehose data portal [26] (available athttp:
//firebrowse.org/). DNA methylation data was not used because most of the samples
available for other data modalities were not available for DNA methylation, leading
to a small sample size. We utilized the ground truth labels provided in [27], which
identify four molecular subtypes based on gene expression, namely Proneural, Neural,
Mesenchymal, and Classical.
                                                                             4

      In addition to the cancer molecular subtype classification tasks, we also conducted
biomedical classification analyses for AD. mRNA expression, DNA methylation, and
miRNA expression data were collected from ROSMAP cohort [23] to classify between
ADandCNindividuals.Wealsocollecteddiverseomicstypes,namelysinglenucleotide
polymorphisms (SNPs), lipidomics, and bileomics, from ADNI dataset [24] and con-
duct  a  classification  task  to  predict  AD,  MCI,  and  CN  individuals.  A  detailed
description of the datasets can be found in                     Table S1      .
Fig. 1  Overview of IGCN architecture on three similarity networks.                                               IGCN employs an
integration module to fuse the node embeddings. Simultaneously, IGCN assigns attention coefficients
to features for individual samples across diverse omics data types.
                                                                                   5

2.3 IGCN demonstrated superior performance in various
          cancer molecular subtype prediction and biomedical
          classification tasks
We compared the performance of IGCN with the state-of-the-art methods (i.e., GCN,
GAT, HAN, HGT, RGCN, MOGONET, SUPREME, and HyperTMO) as well as
baseline methods, namely MLP, Random Forest (RF), and Support Vector Machine
(SVM). We evaluated their performance based on four metrics: accuracy, macro F1
score, weighted F1 score, and Matthew’s correlation coefficient (MCC). We evaluated
all the methods on ten different randomly generated training, validation, and test
splits. We selected 80% of the samples as the training set and 20% of the samples
as the test set in a stratified fashion. We also used 25% of the training set as the
validation set to tune the hyperparameters (e.g., hidden layer size and learning rate)
and perform early stopping.
      The quantitative results of our comparative experiments, presented in                                    Table1         ,
show that IGCN achieved the best performance across all metrics and datasets. Since
GCN, GAT, MLP, SVM, and RF are not integrative tools, we evaluated each data
modality separately and presented the best results for each method. These results
highlight the importance of multi-omics integration because even the top performance
achieved is still not superior to that of IGCN.
      In [20], it was shown that given proper inputs, simple homogeneous GNN-based
integration approaches, such as GCN and GAT, may surpass the performance of all
existing integrative tools across various scenarios. Similarly, our experimental results
show that GCN delivers the second-best performance on GBM dataset, outperform-
ing integrative methods such as HAN, HGT, RGCN, MOGONET, SUPREME, and
HyperTMO. Furthermore, both GCN and GAT outperformed both MOGONET and
HGT on ADNI dataset. However, IGCN demonstrated the best classification perfor-
mance across all dataset, representing a more sophisticated and advanced integrative
tool.
      In  Fig.2(a)            and2     (b)  , the boxplots show the distribution of macro F1 scores of
tenrunsonGBMandADNIdatasets.                       Table1          presentsthemeansandstandarddevi-
ations of these runs. We also calculated Wilcoxon rank-sum test p-values to compare
thedistributionoftheboxplotsbetweenIGCNandothermethods.ForTCGA-GBM,
ADNI,andTCGA-BRCAdatasets,asshownin                              Fig.2        and   Fig.S1(b)      ,IGCNoutper-
formsallothermethodssignificantly(p-value                        <  0.05).ForROSMAPdataset,asshown
in  Fig. S1(b)      , we observed statistically significant difference with all the methods
(p-value     <  0.05) except for HyperTWO (p-value                       >  0.05).
2.4 Attention-driven Interpretability in IGCN
The personalized attention mechanism in IGCN is proposed to integrate multi-omics
data embeddings into a weighted form. As this module assigns specific attention coef-
ficients to each sample, we can observe unique characteristic pattern of each omics
type at the label space. The significance of different embeddings derived from various
types of omics data networks varies from sample to sample, depending on the can-
cer molecular type or specific diagnosis group                      Fig.3(a)           and3     (b)   show the attention
                                                                            6

                                         Table 1       Classification results on TCGA-BRCA, TCGA-GBM, ROSMAP, and ADNI datasets. The
                                         reported values represent the averages along with standard deviations, based on ten runs, for four
                                         performance measures, namely: accuracy, macro F1, weighted F1, and Matthew’s correlation
    Dataset                 Method       coefficient (MCC). The best values for each dataset are shown in bold. The underline is used toAccuracyWeightedF1MacroF1MCC
                               MLP       signify the second-best performance.0.761  ± 0.0120.752  ± 0.015 0.704  ± 0.022             0.648  ± 0.020
                               SVM                0.774  ± 0.022              0.771  ± 0.022              0.736  ± 0.027             0.668  ± 0.033
                                 RF               0.714  ± 0.020              0.690  ± 0.023              0.594  ± 0.036             0.565  ± 0.032
                               GCN                0.787  ± 0.012              0.782  ± 0.016              0.743  ± 0.0247            0.685  ± 0.020
                               GAT                0.789  ± 0.015              0.785  ± 0.017              0.747  ± 0.022             0.688  ± 0.024
TCGA-BRCA                      HAN                0.781  ± 0.025              0.772  ± 0.036              0.714  ± 0.069             0.675  ± 0.039
                               HGT                0.795  ± 0.028              0.789  ± 0.030              0.739  ± 0.044             0.697  ± 0.042
                              RGCN                0.825  ± 0.017              0.824  ± 0.020              0.791  ± 0.032             0.744  ± 0.027
                          MOGONET                 0.813  ± 0.013              0.813  ± 0.014              0.765  ± 0.027             0.727  ± 0.019
                          SUPREME                 0.821  ± 0.020              0.822  ± 0.022              0.783  ± 0.032             0.742  ± 0.031
                          HyperTMO                0.838  ± 0.015              0.841  ± 0.016              0.813  ± 0.025             0.768  ± 0.022
                               IGCN              0.874  ± 0.011              0.878  ± 0.010              0.852  ± 0.017             0.821  ± 0.014
                               MLP                0.793  ± 0.048              0.791  ± 0.050              0.783  ± 0.050             0.725  ± 0.063
                               SVM                0.779  ± 0.057              0.777  ± 0.056              0.767  ± 0.058             0.706  ± 0.075
                                 RF               0.818  ± 0.049              0.811  ± 0.055              0.801  ± 0.059             0.761  ± 0.063
                               GCN                0.880  ± 0.017              0.879  ± 0.017              0.873  ± 0.018             0.839  ± 0.023
                               GAT                0.861  ± 0.018              0.859  ± 0.019              0.852  ± 0.021             0.814  ± 0.025
TCGA-GBM                       HAN                0.858  ± 0.038              0.856  ± 0.041              0.850  ± 0.044             0.809  ± 0.051
                               HGT                0.840  ± 0.033              0.840  ± 0.034              0.839  ± 0.036             0.788  ± 0.045
                              RGCN                0.851  ± 0.043              0.849  ± 0.045              0.845  ± 0.049             0.801  ± 0.057
                          MOGONET                 0.854  ± 0.019              0.854  ± 0.020              0.851  ± 0.023             0.805  ± 0.026
                          SUPREME                 0.818  ± 0.030              0.816  ± 0.034              0.808  ± 0.041             0.756  ± 0.041
                          HyperTMO                0.837  ± 0.026              0.836  ± 0.026              0.832  ± 0.029             0.781  ± 0.034
                               IGCN              0.903  ± 0.014              0.902  ± 0.014              0.898  ± 0.013             0.870  ± 0.019
                               MLP                0.657  ± 0.056              0.650  ± 0.059              0.650  ± 0.059             0.335  ± 0.111
                               SVM                0.645  ± 0.063              0.623  ± 0.079              0.621  ± 0.081             0.308  ± 0.131
                                 RF               0.692  ± 0.066              0.691  ± 0.065              0.691  ± 0.066             0.387  ± 0.136
                               GCN                0.701  ± 0.042              0.700  ± 0.042              0.699  ± 0.042             0.405  ± 0.085
                               GAT                0.670  ± 0.032              0.669  ± 0.033              0.669  ± 0.033             0.347  ± 0.064
   ROSMAP                      HAN                0.775  ± 0.025              0.775  ± 0.025              0.774  ± 0.025             0.550  ± 0.052
                               HGT                0.758  ± 0.022              0.756  ± 0.022              0.756  ± 0.022             0.527  ± 0.041
                              RGCN                0.744  ± 0.024              0.741  ± 0.024              0.740  ± 0.024             0.503  ± 0.049
                          MOGONET                 0.782  ± 0.019              0.781  ± 0.019              0.781  ± 0.019             0.571  ± 0.037
                          SUPREME                 0.782  ± 0.028              0.781  ± 0.028              0.781  ± 0.028             0.575  ± 0.056
                          HyperTMO                0.796  ± 0.035              0.795  ± 0.035              0.795  ± 0.035             0.596  ± 0.071
                               IGCN              0.824  ± 0.034              0.823  ± 0.034              0.823  ± 0.034             0.659  ± 0.069
                               MLP                0.774  ± 0.028              0.770  ± 0.029              0.770  ± 0.029             0.660  ± 0.041
                               SVM                0.766  ± 0.045              0.763  ± 0.048              0.762  ± 0.048             0.651  ± 0.065
                                 RF               0.791  ± 0.047              0.788  ± 0.048              0.788  ± 0.049             0.689  ± 0.071
                               GCN                0.783  ± 0.037              0.782  ± 0.038              0.785  ± 0.037             0.677  ± 0.054
                               GAT                0.760  ± 0.036              0.759  ± 0.035              0.763  ± 0.034             0.642  ± 0.055
      ADNI                     HAN                0.799  ± 0.032              0.800  ± 0.031              0.802  ± 0.029             0.702  ± 0.047
                               HGT                0.758  ± 0.039              0.757  ± 0.041              0.762  ± 0.041             0.642  ± 0.056
                              RGCN                0.807  ± 0.034              0.806  ± 0.034              0.808  ± 0.033             0.713  ± 0.050
                          MOGONET                 0.733  ± 0.033              0.732  ± 0.034              0.735  ± 0.035             0.601  ± 0.049
                          SUPREME                 0.803  ± 0.036              0.803  ± 0.038              0.806  ± 0.037             0.709  ± 0.052
                          HyperTMO                0.794  ± 0.027              0.793  ± 0.025              0.796  ± 0.025             0.695  ± 0.042
                               IGCN              0.840  ± 0.026              0.840  ± 0.026              0.842  ± 0.026             0.762  ± 0.039

Fig. 2     The boxplots show the distribution of macro F1 scores of ten different runs on                                       (a)   TCGA-
GBM and        (b)   ADNI datasets for all methods. The means and standard deviations of these runs
are shown in       Table1          . Wilcoxon rank-sum test p-values were computed between IGCN and other
methods to compare the distribution of box plots, representing p-value                                                   <   0.001 by ***, else if            <   0.01
by **, and else if           <   0.05 by *.
coefficients computed for 50 correctly predicted test samples in TCGA-BRCA and
ROSMAP datasets, respectively.
      For TCGA-BRCA dataset, mRNA expression data had the main contribution
toward the prediction of Basal-like, HER2-enriched and Luminal A breast cancer sub-
types, which is expected as PAM50 subtypes are based on mRNA expression data.
Interestingly, however, miRNA expression data had the main contribution to predict-
ing Normal-like breast cancer subtype. It is also notable that the attention level of
DNA methylation data is slightly higher compared to the attention level of mRNA
expression data in the Luminal B samples. Concerning to ROSMAP dataset, mRNA
expression plays a primary role in both CN and AD samples. However, the attention
                                                                                    8

level given to mRNA expression data is slightly higher for AD samples compared to
CN samples. The attention level of each omics type varies significantly across differ-
ent samples. It is apparent that various types of omics data are integrated according
to a patient-specific golden ratio. This feature makes IGCN interpretable in terms of
understanding the rationale behind the prediction at the sample level. Moreover, it
has the potential to pave the way for a new research direction in analyzing different
omics data types on different cancer subtype samples.
      On the other hand, the feature ranking layer shown in                            Fig.1         assigns attention
coefficientstoinputfeaturesacrossallomicstypes.Thefeaturerankinglayerisunique
such that it assigns attention values customized to each individual sample. This indi-
cates that rather than employing a global attention mechanism that treats all samples
uniformly, the module personalizes the attention scores for each individual sample.
As illustrated in        Fig. S2(a)      , the attention values for mRNA expressions within the
TCGA-GBM dataset exhibit variability across samples. We enforce the normalization
of attention values (given in Eq. (12)), ensuring that their cumulative sum amounts to
1 for each sample. Considering the attention scores across the features demonstrated
Fig. 3       Attention coefficients of 50 test samples of                    (a)   TCGA-BRCA and               (b)   ROSMAP datasets.
IGCN provides an attention mechanism, which computes a specific attention coefficient for each node
embedding. This speciality might allow us to investigate which feature is most informative for each
sample in different node type prediction.
                                                                                 9

in,itcanbeinferredthatthegeneswithrelativelyhigherattentionscoresholdgreater
significance compared to others. We ranked omics features based on their average
attention across all samples and reported the top ten biomarkers in                                 Table2         . Recent
studies have provided evidence supporting the involvement of these biomarkers in the
pathogenesis of GBM and AD [28–34].
      In AD, beta-amyloid (A              β ) peptides can aggregate and accumulate, forming insol-
uble plaques in the brain. These plaques are a hallmark pathological feature of AD
and are believed to contribute to the progressive neurodegeneration and cognitive
decline seen in the disease. KIF5A gene is a protein-coding gene that belongs to the
kinesin family. Some studies have suggested that KIF5A protein expression correlated
inversely with the levels of soluble A                 β  in AD brains [28,29]. Research studies have
suggested that CSRP1, HOPX, HMGN2, TF, and CDK2AP1 may play a role in the
pathogenesis of AD through their involvement in gene regulation processes [30–34].
Moreover, hsa-miR-27a-3p, hsa-miR-16-5p, hsa-miR-142-3p, hsa-miR-199a-5p, hsa-
miR-107,and hsa-miR-1248miRNAs have beenidentified ascandidatebiomarkersfor
AD [35–42].
      The genes SERPINA3, PTPRZ1, and CST3 have been extensively researched
withinthecontextofGBM.SerineproteaseinhibitorcladeA,member3(SERPINA3)
is a protein that influences GBM by modulating actions that promote tumor growth.
SERPINA3 expression is higher in the peritumoral brain zone (PBZ) of GBM com-
paredtothetumorcore[43].TheincreasedlevelsofSERPINA3inthePBZcontribute
to the aggressive behavior of GBM by facilitating the infiltration of tumor cells into
surrounding brain tissue. This infiltration is a critical factor in the recurrence of the
tumor, as the PBZ is often the site where tumor cells invade and spread into adjacent
areas[44]. Elevated levels of PTPRZ1 are also associated with increased cell migration
and invasion in GBM. Studies have demonstrated that downregulation of PTPRZ1
leads to reduced migration and invasion capabilities of glioma cells, suggesting that
PTPRZ1 facilitates these processes, which are critical for tumor spread and recur-
rence [45–47]. CST3 has been investigated as a potential prognostic marker in GBM.
Elevated levels of CST3 may correlate with more aggressive tumor characteristics and
poorer clinical outcomes, making it a candidate for further research in the context of
GBM prognosis and treatment strategies [48,49]. Furthermore, hsa-let-7b, hsa-let-7a,
hsa-let-7c, and hsa-let-7f are members of the let-7 family of miRNAs, which have been
extensively studied in various cancers, including GBM [50,51]. These miRNAs have
been implicated in regulating the expression of genes involved in GBM progression
and may have potential as a therapeutic target [52]. hsa-miR-125b miRNA is part of
a larger family of microRNAs that includes that includes miR-125b-1 and miR-125b-
2 [53]. Its upregulation can contribute to tumor growth and invasion by promoting
pathways that enhance cell survival and proliferation [53]. hsa-miR-21 is known to be
upregulated in GBM and has been implicated in promoting tumor growth, invasion,
and resistance to therapy [54]. hsa-miR-9 has been shown to regulate the mobility
behavior of GBM cells and may have implications in GBM progression [55].
                                                                              10

                                        Table 2       The highest-ranked 10 omics biomarkers identified through IGCN in TCGA-GBM and
                                        ROSMAP datasets.
                                        2.5 Ablation study
                                        We carried out an ablation study to investigate how the attention mechanism and
                                        Normalized Temperature-scaled Cross Entropy Loss (                         L NT  − Xent   ) funtions affect the
                                        modeling ability of IGCN. Particularly, we developed three variants of IGCN: 1) we
                                        disabled the attention mechanism, and computed node embeddings as an average of
                                        node embeddings from each network; 2) we disabled the                           L NT  − Xent    loss functions for
                                        all omics networks (shown in               Fig.1       ); and 3) we used the proposed IGCN architecture
                                        to observe how disabling of different components affect the model performance. We
                                        conducted the experiments on TCGA-BRCA and ROSMAP datasets and reported
                                        the classification performance with macro F1 score measure in                                Table3         . The results
                                        show that both the attention mechanism module and the                            L NT  − Xent    loss functions
                                        are vital in node classification tasks, as the full IGCN setup achieved the best perfor-
                                        mance.Therefore,directlycombiningdifferenttypesofomicsdataembeddingswithout
                                        accountingforthesample-levelimportanceofvariousnetworks,providedbytheatten-
                                        tionmodule,mayleadtoincorrectpredictionsanddegradeperformance.Additionally,
                                        some edges in the graph may not accurately represent the true interactions or rela-
                                        tionships between nodes. These inaccuracies can adversely affect classification tasks.
                                        The results demonstrate that the               L NT  − Xent    loss can alleviate these inaccuracies and
                                        boost the classification performance of IGCN.
                                                                                                                      11
   Dataset                              mRNA                                        miRNA                             DNA methylation
                                           SSP1                                     hsa-let-7b
                                       SPARCL1                                      hsa-let-7a
                                     SERPINA3                                   hsa-miR-125b
                                        PTPRZ1                                    hsa-miR-21
TCGA-GBM                                CHI3L1                                     hsa-miR-9                                        N/A
                                          CST3                                      hsa-let-7c
                                          PMP2                                      hsa-let-7f
                                        CRYAB                                    hsa-miR-29a
                                           HBB                                   hsa-miR-26a
                                          HBA2                                  hsa-miR-1290
                                         QDPR                                  hsa-miR-27a-3p                                 cg02595219
                                        PPDPF                                    hsa-miR-340                                  cg14837165
                                      PLEKHB1                                   hsa-miR-374b                                  cg15775914
                                         KIF5A                                  hsa-miR-16-5p                                 cg01182697
  ROSMAP                                 CSRP1                                   hsa-miR-107                                  cg05382123
                                         HOPX                                  hsa-miR-142-3p                                 cg07546360
                                        HMGN2                                 hsa-miR-199a-5p                                 cg12120741
                                      PLEKHM2                                  hsa-miR-574-3p                                 cg20870559
                                             TF                                hsa-miR-770-5p                                 cg23571857
                                      CDK2AP1                                   hsa-miR-1248                                  cg24322623

                                                 Table 3       The average macro F1 scores of different variants of IGCN on TCGA-BRCA and
                                                 ROSMAP datasets over ten runs. Attn. Mech.: Attention mechanism,                                 L NT  − Xent   : Normalized
                                                 Temperature-scaled Cross Entropy Loss.
                                                 3 Discussion
                                                 Due to the rapid advancements in omics technologies, along with major studies such
                                                 as TCGA, ROSMAP, and ADNI, multi-omics datasets have become prevalent. There-
                                                 fore, creating computational tools for the integrative analysis of various omics data
                                                 typeshasbeencriticallyimportantincancermolecularbiologyandprecisionmedicine
                                                 research. Toward this goal, to advance personalized medicine and uncover previously
                                                 unknown characteristics through integrative analysis of multi-omics data, we present
                                                 IGCN as a framework to integrate multi-omics datasets. To demonstrate the superior-
                                                 ity of IGCN, we not only compare its performance with other multi-omics integrative
                                                 tools but also compare it with recently introduced multi-modal graph representation
                                                 learning methods, which have not been applied for multi-omics integration.
                                                       IGCN utilizes multi-GCN modules to extract node embeddings from each omics
                                                 data. The personalized attention mechanism in IGCN is designed to integrate multi-
                                                 omicsdataembeddingsintoaweightedform.Thismoduleassignsomictypeattention
                                                 values specific for each sample, allowing us to observe the unique characteristic pat-
                                                 terns of each omics type within the label space. Our findings revealed that the
                                                 importanceofdifferentembeddingsfromvariousomicsdatanetworkschangesforeach
                                                 sample (    Fig.3       ). In line with previous research, where mRNA expression data has
                                                 been consistently recognized as a key factor in identifying breast cancer subtypes, our
                                                 analysis of the TCGA-BRCA dataset confirms that mRNA expression significantly
                                                 contributed to the prediction of Basal-like, HER2-enriched, and Luminal A subtypes.
                                                 Thisisexpected,giventhatPAM50subtypesaretraditionallydefinedbasedonmRNA
                                                 expression profiles. Notably, however, our findings also reveal that miRNA expression
                                                 data was the most influential in predicting the Normal-like breast cancer subtype,
                                                 which adds a new dimension to the understanding of breast cancer classification and
                                                 highlightsthepotentialofmiRNAasacomplementarydatatypeinsubtypeprediction.
                                                       Moreover, IGCN’s ability to assign attention coefficients to features within indi-
                                                 vidual samples across various omics data types represents a significant advancement.
                                                 This capability not only aids in identifying key biomarkers in the pathogenesis of dis-
                                                 eases such as BRCA, GBM, and AD but also suggests specific genes as candidates for
                                                 predicting and differentiating cancer subtypes and other biomedical outcomes at the
                                                 molecularlevel(       Table2          and   TableS2      ).Byeffectivelyhighlightingbiomarkersacross
                                                 different omics layers, IGCN enhances our understanding of the underlying molecular
                                                                                                                             12
              Components                                                                         Macro F1
Attn. Mech.                         L NT   −  Xent                      TCGA-BRCA                                       ROSMAP
             ✗                                  ✓                        0 .829   ±   0 .023                        0 .750   ±   0 .021
            ✓                                   ✗                        0 .831   ±   0 .018                        0 .749   ±   0 .035
            ✓                                   ✓                      0 .852    ±   0 .017                       0 .811    ±   0 .032

mechanisms driving disease heterogeneity. This insight supports more accurate sub-
type classification, ultimately leading to more personalized and targeted therapeutic
strategies.
      One challenge encountered in our study was the limited availability of data across
all modalities for some patients, which resulted in a smaller sample size. However, as
more advanced datasets with additional omics layers become available, future stud-
ies could conduct a more comprehensive analysis. Such enriched data would allow for
a deeper exploration of the complex interactions between different biological layers,
ultimately enhancing our ability to uncover meaningful insights and improve predic-
tiveaccuracy.Anotherimportantaspecttoconsiderinourstudyistheuseofdatasets
derivedfrombulktissuesamplesratherthansingle-celldata.Whilebulktissueanalysis
offers valuable insights into overall gene expression patterns, it can obscure the cel-
lular heterogeneity within the tissue. Single-cell omics datasets would provide a more
detailed view of the diversity among individual cells, enabling a deeper understanding
of the specific roles of different cell populations in pathophysiology.
4 Methods
4.1 Customizing GCN for omics-focused learning
IGCNemploysGCNmodulesongraphnetworkstoobtainthenodeembeddings.Each
GCN module in IGCN can be defined as:
                                                  H  i =  σ (D − 1/2i     A iD − 1/2i     X  iW  i),                                      (1)
for  i = 1  ,2,...,p, where     p  is the total number of data modalities (omics types) and
X  i ∈ R m × d is the feature matrix (          m   is the number of nodes and                d is the feature size).
D  i and   W  i are the node degree and the learnable weight matrices, respectively.                                    σ  is
the activation function. We used a single layer GCN to obtain the node embeddings
(H  i)foreachnetworklayer,however,amulti-layerGCNcanbeconsideredasoutlined
in [56].
      In our work, the original adjacency matrix                        A i ∈ R m × m   was constructed by calcu-
lating cosine similarity of each node pair and filtering out edges with cosine similarity
<ϵ  . The adjacency matrix can be defined as:
                                                        a (q,w  )i      =  Ind  (s(x qi,x wi )),                                           (2)
where     x qi  and   x wi  are the node features of the node                  q  and    w   (the   qth  and    w th  row
vectors of     X  i), respectively.       a (q,w  )i      is an element of the matrix               A i corresponding to the
qth  row and      w th  column.      s(x qi,x wi ) =      ⟨x qi,x wi ⟩
                                                                     ||x qi||2||x wi ||2  is the cosine similarity.           Ind  (.) is an
indicator function that maps the input to 1 if the input is greater than or equal to
ϵ, and to 0 otherwise. As shown in                 Algorithm1                   , the threshold        ϵ can be determined
                                                                           13

based on a given parameter                 k  as:
                                                     k =   1      X      Ind  (s(x q
                                                              m                         i,x wi )).                                         (3)
                                                                   q,w
Algorithm 1          Determine threshold             ϵ based on pre-specified parameter                   k
  1:  Input:     pre-specified parameter             k  and initialize       ϵ =1
  2:  Output:       ϵ
  3:  while true do                  P
  4:       counter     =    1m            q,w  Ind  (s(x qi,x wi ))
  5:      if counter >      =  k then
  6:           break
  7:      else
  8:            ϵ ←   ϵ−  10 − 6
  9:      end if
10:  end while
Choosing a proper            k  value depends on the topological structure of the data. The
results shown in        Fig. S3     indicate that IGCN was robust to the change of                           k  value and
outperformed other integrative tools under different                            k  values.
      When building a graph based on features and employing cosine similarity, it is
frequent to come across data noise arising from diverse origins, such as measure-
ment inaccuracies, missing values, or inherent fluctuations within the dataset. These
inaccuracies can have negative impacts on node classification tasks. To alleviate
these inaccuracies, the Normalized Temperature-scaled Cross Entropy loss (NT-Xent
loss) [57] was utilized for each GCN module.
                                                    pos  q =  X         exp  (s(x qi,x ji))/τ,                                         (4)
                                                                 j∈ +
                                                   neg  q =  X          exp  (s(x qi,x ℓi))/τ.                                         (5)
                                                                 ℓ∈−
      In Eq. (4), we calculate the cosine similarity between the anchor sample                                         q  and
its positive pairs, scaled by the temperature parameter                              τ. Similarly, in Eq. (5), we
calculate the cosine similarity between the anchor sample                                q  and its negative pairs,
scaled by the temperature parameter. It is noteworthy that the anchor sample and
its corresponding positive pairs belong to the same class, while the negative pairs
were selected from classes different from that of the anchor sample. Thus, to learn
representations that bring similar data points closer in the embedding space while
pushingdissimilardatapointsfartherapart,theNT-Xentlossforthe                                     ith  GCNmodule
                                                                           14

can be defined as follows:
                                                                          υX                                   !
                                           L iNT  − Xent    =   1              log             pos  q             .                               (6)
                                                                    υ   q=1             pos  q +  neg  q
where     υ  is the total number of samples in the training set.
4.2 Computing attention coefficients and predictions
After computing node embeddings using Eq. (1), IGCN provides an attention mech-
anism to fuse multiple node embeddings into a weighted form by assigning attention
coefficients to node embeddings. Inspired from  [17,58], attention coefficients can be
determined as:
                                        cni  =       exp  (LeakyReLU         (h ni W  a +  b))P  p
                                                       j=1  exp  (LeakyReLU         (h nj W  a +  b)) ,                             (7)
where    h ni ∈ R d  is the   n th  node embedding of the              ith  similarity network and              p  is the
total number of data modalities.                  W  a ∈ R d× 1 and   b ∈ R 1 are learnable weight and bias
parameters, respectively. Attention coefficient                        cni ∈ R 1  represents the importance of
the   n th  node embedding of the               ith  network. Attention coefficients can be computed
for all nodes of the similarity network and represented as a vector. Therefore, we can
fuse the multiple node embeddings using element-wise multiplication, as follows:
                                                                           pX
                                                                Z  =            ⃗ci⊗  H  i,                                                 (8)
                                                                         i=1
where “   ⊗ ” denotes element-wise multiplication.                      H  i  is the node embedding matrix
corresponding to          ith  similarity network and              ⃗ci is the attention coefficient vector for
the nodes in the         ith  similarity network. It conveys that, although all embeddings are
derived from a particular network, individual node embeddings may have distinct
coefficient values. For example, consider the vector                           ⃗c1 as a column vector of size              m  :
                                                                                c11c21...
                                                                                   ,
                                                                     ⃗c1 =
                                                                                 cm1
and let    H  1 be a matrix of size          m  ×  d:
                                                              h (1 ,1)                                   
                                                            1       h (1 ,2)1       ... h (1 ,d)1     .
                                                H  1 =         h (2 ,1)1       h (2 ,2)1       ... h (2 ,d)1
                                                                   ...         ...     ...     ...
                                                              h (m,  1)1       h (m,  2)1       ... h (m,d  )1
                                                                             15

We also note that           n th  row of    H  1 can also be represented as a row vector                   h n1 :
                                                             h                                          i
                                                  h n1  =      h (n, 1)1      h (n, 2)1      ... h (n,d )1 .
The element-wise multiplication of each row of                           ⃗c1 by the corresponding row of                H  1 can
be represented as follows:
                                                       c11 ·h (1 ,1)                                                     
                                                            1        c11 ·h (1 ,2)1       ...  c11 ·h (1 ,d)1         .
                                ⃗c1 ⊗  H  1 =           c21 ·h (2 ,1)1        c21 ·h (2 ,2)1       ...  c21 ·h (2 ,d)1
                                                               ...                ...         ...         ...
                                                      cm1  ·h (m,  1)1       cm1  ·h (m,  2)1       ... cm1  ·h (m,d  )1
      The weighted form of embeddings computed using Eq. (8) is utilized on a neural
network to obtain the node label predictions. Thus, it can be written as:
                                                                ˆY  =  σ (Z  ¯W  1) ¯W  2,                                                (9)
where      ¯W  1 and     ¯W  2 are learnable weight matrices.
      Besides the NT-Xent loss given in Eq. (6), we also used the cross entropy loss as
follows:                                                           vX                                  !
                                                   L CE   =             − log        Pe⟨ˆy j,y j⟩         ,                                    (10)
                                                                 j=1                     k e ˆy (j,k )
where     ˆy j ∈ R d is the   jth  row in     ˆY , which is the predicted label distribution of the                       jth
training sample. ˆ       y (j,k ) is the   k th  element in       ˆy j.y j is the one-hot encoded vector of the
ground truth label of the             jth  training sample.       ⟨ˆy j,y j⟩ represents the inner product of
the vector       ˆy j  and the vector        y j. To determine all learnable weights and biases, the
total lost function can be written as:
                                                                                    pX
                                                 L IGCN     =  L CE   +                  L iNT  − Xent  ,                                   (11)
                                                                                  i=1
where     p is the total number of data modalities.
      Adam optimization [59] was used as the state-of-the-art for stochastic gradient
descent algorithm and 0.5 dropout was added for each GCN layer. Early stopping was
usedwiththepatienceof30forcedtohaveatleast200epochs,whichweredetermined
empirically.
4.3 Uncovering significant biomarkers during the prediction
          process
AnothercomponentofIGCNisthefeaturerankingmodulewhichidentifiesnoteworthy
biomarkers across various omics datasets. Similar to the attention mechanism module
                                                                             16

giveninSec.4.2,thefeaturerankingmodulealsoofferspersonalizedinsightsregarding
feature importance. The attention values for each feature can be computed as follows:
                                                                  LeakyReLU         (x (j,k )                                 !
                               exp       LeakyReLU                                             i      ˜W  i) ˆW  i +  ˆbi
          r(j,k )i      =P  di                                         LeakyReLU         (x (j,k )                                 ! ,        (12)
                              k=1  exp        LeakyReLU                                              i      ˜W  i) ˆW  i +  ˆbi
where    r(j,k )i    ∈ R 1 representstheattentionvalueofthe                   k th  featureinthe       ith  omicstype
correspondingtothe            jth  sample.     x (j,k )i    ∈ R 1 isthe   k th  rowfeatureinthe          ith  omicstype
corresponding to the            jth  sample.      ˜W  i ∈ R 1× ω  and    ˆW  i ∈ R ω× 1 are the learnable weight
matrices.      ˆbi ∈ R 1 is the bias parameter. As the input                   x (j,k )i      is a scalar,      ˜W  i ∈ R 1× ω  and
 ˆW  i ∈ R ω× 1  were employed as an expansion and a compression units, respectively.                                       di
is the feature size of the           ith  omics type and          ω  is the latent space dimension. Hence, we
assign a feature rank for each feature used as an input. Moreover, since each sample
was evaluated individually, the significance of each feature may differ for each patient.
5 Acknowledgements
This work was supported by the National Institute of General Medical Sciences of
the National Institutes of Health under Award Number R35GM133657. Data col-
lection and sharing for this project was funded by ADNI (National Institutes of
Health Grant U01 AG024904) and DOD ADNI (Department of Defense award num-
ber W81XWH-12-2-0012). ADNI is funded by the National Institute on Aging, the
National Institute of Biomedical Imaging and Bioengineering, and through gener-
ous contributions from the following: AbbVie, Alzheimer’s Association; Alzheimer’s
Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers
Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.;
Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated
company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer
Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical
Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale
Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharma-
ceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical
Company; and Transition Therapeutics. The Canadian Institutes of Health Research
is providing funds to support ADNI clinical sites in Canada. Private sector con-
tributions are facilitated by the Foundation for the National Institutes of Health
(www.fnih.org). The grantee organization is the Northern California Institute for
Research and Education, and the study is coordinated by the Alzheimer’s Thera-
peutic Research Institute at the University of Southern California. ADNI data are
disseminated by the Laboratory for Neuro Imaging at the University of Southern
California.
                                                                            17

6 Code availability
The source code and documentation is available athttps://github.com/bozdaglab/
IGCN.
References
  [1]Chaudhary, K., Poirion, O.B., Lu, L., Garmire, L.X.: Deep learning–based multi-
        omics  integration  robustly  predicts  survival  in  liver  cancer.  Clinical  Cancer
        Research      24 (6), 1248–1259 (2018)
  [2]Poirion, O.B., Chaudhary, K., Garmire, L.X.: Deep learning data integration
        for  better  risk  stratification  models  of  bladder  cancer.  AMIA  Summits  on
        Translational Science Proceedings                  2018   , 197 (2018)
  [3]Sharifi-Noghabi,  H.,  Zolotareva,  O.,  Collins,  C.C.,  Ester,  M.:  Moli:  multi-
        omics late integration with deep neural networks for drug response prediction.
        Bioinformatics        35 (14), 501–509 (2019)
  [4]Huang, Z., Zhan, X., Xiang, S., Johnson, T.S., Helm, B., Yu, C.Y., Zhang, J.,
        Salama, P., Rizkalla, M., Han, Z.,                   et al. : Salmon: survival analysis learning with
        multi-omicsneuralnetworksonbreastcancer.Frontiersingenetics                                 10 ,166(2019)
  [5]Choi, J.M., Chae, H.: mobrca-net: a breast cancer subtype classification frame-
        workbasedonmulti-omicsattentionneuralnetworks.BMCbioinformatics                                        24 (1),
        169 (2023)
  [6]Khadirnaikar, S., Shukla, S., Prasanna, S.: Machine learning based combination
        of multi-omics data for subgroup identification in non-small cell lung cancer.
        Scientific Reports         13 (1), 4636 (2023)
  [7]Gong, P., Cheng, L., Zhang, Z., Meng, A., Li, E., Chen, J., Zhang, L.: Multi-
        omicsintegrationmethodbasedonattentiondeeplearningnetworkforbiomedical
        dataclassification.ComputerMethodsandProgramsinBiomedicine                                     231  ,107377
        (2023)
  [8]Abbas, Z., Tayara, H., Chong, K.T.: Alzheimer’s disease prediction based on con-
        tinuous feature representation using multi-omics data integration. Chemometrics
        and Intelligent Laboratory Systems                   223  , 104536 (2022)
  [9]Shigemizu, D., Akiyama, S., Higaki, S., Sugimoto, T., Sakurai, T., Boroevich,
        K.A., Sharma, A., Tsunoda, T., Ochiya, T., Niida, S.,                               et al. : Prognosis prediction
        model for conversion from mild cognitive impairment to alzheimer’s disease cre-
        ated by integrative analysis of multi-omics data. Alzheimer’s research & therapy
        12 , 1–12 (2020)
[10]Li, Z., Jiang, X., Wang, Y., Kim, Y.: Applied machine learning in alzheimer’s
                                                                            18

        diseaseresearch:omics,imaging,andclinicaldata.Emergingtopicsinlifesciences
        5(6), 765–777 (2021)
[11]Wang,T.,Shao,W.,Huang,Z.,Tang,H.,Zhang,J.,Ding,Z.,Huang,K.:Mogonet
        integrates multi-omics data using graph convolutional networks allowing patient
        classification and biomarker identification. Nature Communications                                   12 (1), 3445
        (2021)
[12]Kesimoglu,  Z.N.,  Bozdag,  S.:  SUPREME:  multiomics  data  integration  using
        graph  convolutional  networks.  NAR  Genomics  and  Bioinformatics                                      5(2),  063
        (2023)
[13]Yin, C., Cao, Y., Sun, P., Zhang, H., Li, Z., Xu, Y., Sun, H.: Molecular subtyping
        ofcancerbasedonrobustgraphneuralnetworkandmulti-omicsdataintegration.
        Frontiers in Genetics           13 , 884028 (2022)
[14]Xiao, S., Lin, H., Wang, C., Wang, S., Rajapakse, J.C.: Graph neural networks
        with multiple prior knowledge for multi-omics data analysis. IEEE Journal of
        Biomedical and Health Informatics (2023)
[15]Wang, H., Lin, K., Zhang, Q., Shi, J., Song, X., Wu, J., Zhao, C., He, K.:
        Hypertmo: a trusted multi-omics integration framework based on hypergraph
        convolutionalnetworkforpatientclassification.Bioinformatics                              40 (4),159(2024)
[16]Schlichtkrull,M.,Kipf,T.N.,Bloem,P.,VanDenBerg,R.,Titov,I.,Welling,M.:
        Modeling relational data with graph convolutional networks. In: The Semantic
        Web:15thInternationalConference,ESWC 2018,Heraklion,Crete,Greece,June
        3–7, 2018, Proceedings 15, pp. 593–607 (2018). Springer
[17]Wang,X.,Ji,H.,Shi,C.,Wang,B.,Ye,Y.,Cui,P.,Yu,P.S.:Heterogeneousgraph
        attention network. In: The World Wide Web Conference, pp. 2022–2032 (2019)
[18]Sun, Y., Han, J., Yan, X., Yu, P.S., Wu, T.: Pathsim: Meta path-based top-
        k similarity search in heterogeneous information networks. Proceedings of the
        VLDB Endowment               4(11), 992–1003 (2011)
[19]Hu, Z., Dong, Y., Wang, K., Sun, Y.: Heterogeneous graph transformer. In:
        Proceedings of the Web Conference 2020, pp. 2704–2710 (2020)
[20]Lv, Q., Ding, M., Liu, Q., Chen, Y., Feng, W., He, S., Zhou, C., Jiang, J., Dong,
        Y., Tang, J.: Are we really making much progress? revisiting, benchmarking and
        refining heterogeneous graph neural networks. In: Proceedings of the 27th ACM
        SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 1150–1160
        (2021)
[21]Doshi-Velez, F., Kim, B.: Towards a rigorous science of interpretable machine
        learning. arXiv preprint arXiv:1702.08608 (2017)
                                                                            19

[22]Colaprico,  A.,  Silva,  T.C.,  Olsen,  C.,  Garofano,  L.,  Cava,  C.,  Garolini,  D.,
        Sabedot, T.S., Malta, T.M., Pagnotta, S.M., Castiglioni, I.,                                 et al. : Tcgabiolinks:
        an r/bioconductor package for integrative analysis of tcga data. Nucleic acids
        research     44 (8), 71–71 (2016)
[23]Hodes, R.J., Buckholtz, N.: Accelerating medicines partnership: Alzheimer’s dis-
        ease (amp-ad) knowledge portal aids alzheimer’s drug discovery through open
        data sharing. Expert opinion on therapeutic targets                          20 (4), 389–391 (2016)
[24]Petersen, R.C., Aisen, P.S., Beckett, L.A., Donohue, M.C., Gamst, A.C., Harvey,
        D.J., Jack Jr, C., Jagust, W.J., Shaw, L.M., Toga, A.W.,                                  et al. : Alzheimer’s
        disease neuroimaging initiative (adni) clinical characterization. Neurology                                   74 (3),
        201–209 (2010)
[25]Parker,J.S.,Mullins,M.,Cheang,M.C.,Leung,S.,Voduc,D.,Vickery,T.,Davies,
        S., Fauron, C., He, X., Hu, Z.,                 et al. : Supervised risk predictor of breast cancer
        based on intrinsic subtypes. Journal of clinical oncology                           27 (8), 1160 (2009)
[26]Deng, M., Br¨agelmann, J., Kryukov, I., Saraiva-Agostinho, N., Perner, S.: Fire-
        browser: an r client to the broad institute’s firehose pipeline. Database                                2017   , 160
        (2017)
[27]Verhaak, R.G., Hoadley, K.A., Purdom, E., Wang, V., Qi, Y., Wilkerson, M.D.,
        Miller,  C.R.,  Ding,  L.,  Golub,  T.,  Mesirov,  J.P.,                           et  al. :  Integrated  genomic
        analysis identifies clinically relevant subtypes of glioblastoma characterized by
        abnormalities in pdgfra, idh1, egfr, and nf1. Cancer cell                          17 (1), 98–110 (2010)
[28]Hares, K., Miners, J.S., Cook, A.J., Rice, C., Scolding, N., Love, S., Wilkins,
        A.: Overexpression of kinesin superfamily motor proteins in alzheimer’s disease.
        Journal of Alzheimer’s Disease                60 (4), 1511–1524 (2017)
[29]Hares, K., Miners, S., Scolding, N., Love, S., Wilkins, A.: Kif5a and klc1 expres-
        sion in alzheimer’s disease: relationship and genetic influences. AMRC Open
        Research      1, 1 (2019)
[30]Rocchio, F., Tapella, L., Manfredi, M., Chisari, M., Ronco, F., Ruffinatti, F.A.,
        Conte, E., Canonico, P.L., Sortino, M.A., Grilli, M.,                             et al. : Gene expression, pro-
        teome and calcium signaling alterations in immortalized hippocampal astrocytes
        from an alzheimer’s disease mouse model. Cell Death & Disease                                 10 (1), 24 (2019)
[31]Liu, Y., Bilen, M., McNicoll, M.-M., Harris, R.A., Fong, B.C., Iqbal, M.A., Paul,
        S.,Mayne,J.,Walker,K.,Wang,J.,                     et al. :Earlypostnataldefectsinneurogenesis
        in the 3xtg mouse model of alzheimer’s disease. Cell Death & Disease                                   14 (2), 138
        (2023)
[32]Hermkens, D.M., Stam, O.C., Wit, N.M., Fontijn, R.D., Jongejan, A., Moer-
        land,  P.D.,  Mackaaij,  C.,  Waas,  I.S.,  Daemen,  M.J.,  Vries,  H.E.:  Profiling
                                                                             20

        the unique protective properties of intracranial arterial endothelial cells. Acta
        neuropathologica communications                    7, 1–16 (2019)
[33]Li,Q.S.,DeMuynck,L.:Differentiallyexpressedgenesinalzheimer’sdiseasehigh-
        lighting the roles of microglia genes including olr1 and astrocyte gene cdk2ap1.
        Brain, Behavior, & Immunity-Health                      13 , 100227 (2021)
[34]Li, H., Wood, C.L., Getchell, T.V., Getchell, M.L., Stromberg, A.J.: Analysis of
        oligonucleotide array experiments with repeated measures using mixed models.
        BMC bioinformatics             5, 1–12 (2004)
[35]Sala Frigerio, C., Lau, P., Salta, E., Tournoy, J., Bossers, K., Vandenberghe, R.,
        Wallin, A., Bjerke, M., Zetterberg, H., Blennow, K.,                              et al. : Reduced expression
        of hsa-mir-27a-3p in csf of patients with alzheimer disease. Neurology                                  81 (24),
        2103–2106 (2013)
[36]Wang, L., Zhen, H., Sun, Y., Rong, S., Li, B., Song, Z., Liu, Z., Li, Z., Ding,
        J., Yang, H.,        et al. : Plasma exo-mirnas correlated with ad-related factors of chi-
        nese individuals involved in a               β  accumulation and cognition decline. Molecular
        Neurobiology        59 (11), 6790–6804 (2022)
[37]Harati,R.,Hammad,S.,Tlili,A.,Mahfood,M.,Mabondzo,A.,Hamoudi,R.:mir-
        27a-3pregulatesexpressionofintercellularjunctionsatthebrainendotheliumand
        controls the endothelial barrier permeability. PLoS One                            17 (1), 0262152 (2022)
[38]Swarbrick, S., Wragg, N., Ghosh, S., Stolzing, A.: Systematic review of mirna as
        biomarkers in alzheimer’s disease. Molecular neurobiology                              56 , 6156–6167 (2019)
[39]Herrera-Espejo, S., Santos-Zorrozua, B.,                             ´Alvarez-Gonz´alez, P., Lopez-Lopez, E.,
        Garcia-Orad,          ´A.: A systematic review of microrna expression as biomarker of
        late-onset alzheimer’s disease. Molecular Neurobiology                            56 , 8376–8391 (2019)
[40]Kumar,  P.,  Dezso,  Z.,  MacKenzie,  C.,  Oestreicher,  J.,  Agoulnik,  S.,  Byrne,
        M., Bernier, F., Yanagimachi, M., Aoshima, K., Oda, Y.: Circulating mirna
        biomarkers for alzheimer’s disease. PloS one                      8(7), 69807 (2013)
[41]Gattuso, G., Falzone, L., Costa, C., Giamb`o, F., Teodoro, M., Vivarelli, S.,
        Libra, M., Fenga, C.: Chronic pesticide exposure in farm workers is associ-
        ated with the epigenetic modulation of hsa-mir-199a-5p. International Journal of
        Environmental Research and Public Health                         19 (12), 7018 (2022)
[42]Hewel,C.,Kaiser,J.,Wierczeiko,A.,Linke,J.,Reinhardt,C.,Endres,K.,Gerber,
        S.: Common mirna patterns of alzheimer’s disease and parkinson’s disease and
        theirputativeimpactoncommensalgutmicrobiota.Frontiersinneuroscience                                        13 ,
        113 (2019)
[43]Giambra,M.,DiCristofori,A.,Valtorta,S.,Manfrellotti,R.,Bigiogera,V.,Basso,
                                                                            21

        G., Moresco, R.M., Giussani, C., Bentivegna, A.: The peritumoral brain zone
        in glioblastoma: where we are and where we are going. Journal of Neuroscience
        Research      101  (2), 199–216 (2023)
[44]Nimbalkar, V.P., Kruthika, B.S., Sravya, P., Rao, S., Sugur, H.S., Verma, B.K.,
        Chickabasaviah, Y.T., Arivazhagan, A., Kondaiah, P., Santosh, V.: Differential
        gene expression in peritumoral brain zone of glioblastoma: role of serpina3 in
        promoting invasion, stemness and radioresistance of glioma cells and association
        with poor patient prognosis and recurrence. Journal of Neuro-Oncology                                      152  ,
        55–65 (2021)
[45]Papadimitriou, E., Kanellopoulou, V.K.: Protein tyrosine phosphatase receptor
        zeta1asapotentialtargetincancertherapyanddiagnosis.InternationalJournal
        of Molecular Sciences           24 (9), 8093 (2023)
[46]Xia, Z., Ouyang, D., Li, Q., Li, M., Zou, Q., Li, L., Yi, W., Zhou, E.: The
        expression, functions, interactions and prognostic values of ptprz1: a review and
        bioinformatic analysis. Journal of Cancer                     10 (7), 1663 (2019)
[47]Zeng,  A.,  Yan,  W.,  Liu,  Y.,  Wang,  Z.,  Hu,  Q.,  Nie,  E.,  Zhou,  X.,  Li,  R.,
        Wang, X., Jiang, T.,             et al. : Tumour exosomes from cells harbouring ptprz1–met
        fusion contribute to a malignant phenotype and temozolomide chemoresistance
        in glioblastoma. Oncogene               36 (38), 5369–5381 (2017)
[48]Cheng, X., Ren, Z., Liu, Z., Sun, X., Qian, R., Cao, C., Liu, B., Wang, J., Wang,
        H., Guo, Y.,        et al. : Cysteine cathepsin c: a novel potential biomarker for the
        diagnosis and prognosis of glioma. Cancer Cell International                              22 (1), 53 (2022)
[49]Konduri, S.D., Yanamandra, N., Siddique, K., Joseph, A., Dinh, D.H., Olivero,
        W.C., Gujrati, M., Kouraklis, G., Swaroop, A., Kyritsis, A.P.,                                   et al. : Modulation
        of cystatin c expression impairs the invasive and tumorigenic potential of human
        glioblastoma cells. Oncogene               21 (57), 8705–8712 (2002)
[50]Li,  Y.,  Zhang,  X.,  Chen,  D.,  Ma,  C.:  Retracted  article:  Let-7a  suppresses
        glioma cell proliferation and invasion through tgf-                         β /smad3 signaling pathway by
        targeting hmga2. Tumor Biology                   37 (6), 8107–8119 (2016)
[51]Lee, S.-T., Chu, K., Oh, H.-J., Im, W.-S., Lim, J.-Y., Kim, S.-K., Park, C.-K.,
        Jung, K.-H., Lee, S.K., Kim, M.,                   et al. : Let-7 microrna inhibits the proliferation
        of human glioblastoma cells. Journal of neuro-oncology                            102  , 19–24 (2011)
[52]Xi, X., Chu, Y., Liu, N., Wang, Q., Yin, Z., Lu, Y., Chen, Y.: Joint bioinformat-
        ics analysis of underlying potential functions of hsa-let-7b-5p and core genes in
        human glioma. Journal of translational medicine                          17 , 1–16 (2019)
[53]Wang, Y., Zeng, G., Jiang, Y.: The emerging roles of mir-125b in cancers. Cancer
        management and research, 1079–1088 (2020)
                                                                             22

[54]Akers,J.C.,Ramakrishnan,V.,Kim,R.,Skog,J.,Nakano,I.,Pingle,S.,Kalinina,
        J., Hua, W., Kesari, S., Mao, Y.,                  et al. : Mir-21 in the extracellular vesicles (evs)
        of cerebrospinal fluid (csf): a platform for glioblastoma biomarker development.
        PloS one      8(10), 78115 (2013)
[55]Ben-Hamo,R.,Zilberberg,A.,Cohen,H.,Efroni,S.:hsa-mir-9controlsthemobil-
        ity behavior of glioblastoma cells via regulation of mapk14 signaling elements.
        Oncotarget       7(17), 23170 (2016)
[56]Kipf, T.N., Welling, M.: Semi-supervised classification with graph convolutional
        networks. arXiv preprint arXiv:1609.02907 (2016)
[57]Sohn, K.: Improved deep metric learning with multi-class n-pair loss objective.
        Advances in neural information processing systems                           29   (2016)
[58]Veliˇckovi´c,P.,Cucurull,G.,Casanova,A.,Romero,A.,Lio,P.,Bengio,Y.:Graph
        attention networks. arXiv preprint arXiv:1710.10903 (2017)
[59]Kingma,D.P.,Ba,J.:Adam:Amethodforstochasticoptimization.arXivpreprint
        arXiv:1412.6980 (2014)
                                                                             23

