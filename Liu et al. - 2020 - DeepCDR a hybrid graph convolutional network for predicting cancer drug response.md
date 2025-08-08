Bioinformatics, 36(26), 2020, i911–i918


doi: 10.1093/bioinformatics/btaa822

ECCB2020

## Data
# DeepCDR: a hybrid graph convolutional network for predicting cancer drug response


Qiao Liu [1,2], Zhiqiang Hu [2,3], Rui Jiang [1,2,] - and Mu Zhou [4,] 

1 Ministry of Education Key Laboratory of Bioinformatics, Research Department of Bioinformatics, Beijing National Research Center,
Information Science and Technology, Center for Synthetic and Systems Biology, [2] Department of Automation, Tsinghua University,
Beijing 100084, China, [3] SenseTime Research, Shanghai 200233, China and [4] SenseBrain Research, San Jose, CA 95131, USA


*To whom correspondence should be addressed.


Abstract


Motivation: Accurate prediction of cancer drug response (CDR) is challenging due to the uncertainty of drug efficacy
and heterogeneity of cancer patients. Strong evidences have implicated the high dependence of CDR on tumor genomic and transcriptomic profiles of individual patients. Precise identification of CDR is crucial in both guiding anticancer drug design and understanding cancer biology.
Results: In this study, we present DeepCDR which integrates multi-omics profiles of cancer cells and explores intrinsic chemical structures of drugs for predicting CDR. Specifically, DeepCDR is a hybrid graph convolutional network
consisting of a uniform graph convolutional network and multiple subnetworks. Unlike prior studies modeling handcrafted features of drugs, DeepCDR automatically learns the latent representation of topological structures among
atoms and bonds of drugs. Extensive experiments showed that DeepCDR outperformed state-of-the-art methods in
both classification and regression settings under various data settings. We also evaluated the contribution of different types of omics profiles for assessing drug response. Furthermore, we provided an exploratory strategy for identifying potential cancer-associated genes concerning specific cancer types. Our results highlighted the predictive
power of DeepCDR and its potential translational value in guiding disease-specific drug design.
[Availability and implementation: DeepCDR is freely available at https://github.com/kimmo1019/DeepCDR.](https://github.com/kimmo1019/DeepCDR)
Contact: ruijiang@tsinghua.edu.cn or muzhou@sensebrain.site
[Supplementary information: Supplementary data are available at Bioinformatics online.](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)



1 Introduction


Designing novel drugs with desired efficacy for cancer patients is of
great clinical significance in pharmaceutical industry (Lee et al.,
2018). However, the intra- and inter-tumoral heterogeneity results
in diverse anti-cancer drug responses among patients (Kohane,
2015; Rubin, 2015), highlighting the complexity of genomics and
molecular backgrounds. Recent advances in high-throughput
sequencing technologies have deepened our understanding of cancer
phenotypes from the aggregated amounts of cancer omics profiles
(Gagan and Van Allen, 2015). For example, the pharmacogenomics
(Daly, 2017; Musa et al., 2017) is evolving rapidly by addressing the
interactions between genetic makeup and drug response sensitivity.
Precise identification of cancer drug response (CDR) has become
a crucial problem in guiding anti-cancer drug design and understanding cancer biology. Particularly, cancer cell lines (permanently
established in vitro cell cultures) play an important role in pharmacogenomics research as they reveal the landscape of environment
involved in cellular models of cancer (Iorio et al., 2016). Databases
such as Cancer Cell Line Encyclopedia (CCLE) (Barretina et al.,
2012) provide large-scale cancer profiles including genomic (e.g.
genomic mutation), transcriptomic (e.g. gene expression) and epigenomic data (e.g. DNA methylation). Also, the Genomics of Drug



Sensitivity in Cancer (GDSC) (Iorio et al., 2016) has been carried
out for investigating the drug response to numerous cancer cell lines.
For example, the half-maximal inhibitory concentration (IC 50 ) is a
common indicator reflecting drug response across cancer cell lines.
Mining these cancer-associated profiles and their interactions will
help characterize cancer molecular signatures with therapeutic impact, leading to accurate anti-cancer drug discovery. However, due
to the complexity of omics profiles, the translational potential of
identifying molecular signatures that determines drug response has
not been fully explored.
So far, a handful of computational models have been proposed
for predicting CDR which can be divided into two major categories.
The first type is the network-driven methods which analyze the information extracted from drug–drug similarities and cancer cell line
similarities. The core idea is to construct a similarity-based model
and assign the sensitivity profile of a known drug to a new drug if
there are structurally similar. For example, (Zhang et al., 2015)
established a dual similarity network based on the gene expression
of cancer cell lines and chemical structures of drugs to predict CDR.
(Turki and Wei, 2017) proposed a link-filtering algorithm on cancer
cell line network followed by a linear regression for predicting the
CDR. HNMDRP (Zhang et al., 2018) is a heterogeneous network
that integrates multiple networks, including cell line similarity,



V C The Author(s) 2020. Published by Oxford University Press. All rights reserved. For permissions, please e-mail: journals.permissions@oup.com i911


i912 Q.Liu et al.



drug similarity and drug target similarity. An information flow algorithm was proposed for predicting novel cancer drug associations.
Notably, network-driven methods tend to show poor scalability and
low computational efficiency. Machine learning methods are another type of computational analysis directly exploring profiles from
large-scale drugs and cancer cell lines. Typical approaches include
logistic regression (Geeleher et al., 2014), Support Vector Machines
(SVM) (Dong et al., 2015), random forest (Daemen et al., 2013) and
neural networks (Chang et al., 2018; Liu et al., 2019; Manica et al.,
2019; Sharifi-Noghabi et al., 2019). Most machine learning methods used single omics data from cancer cell lines, such as genomic
mutation or gene expression. For example, CDRscan (Chang et al.,
2018) used the molecular fingerprints for drug representation and
genomic mutation as cancer cell profile. They were fed to an ensemble CNN model for CDR prediction. tCNNs (Liu et al., 2019) takes
SMILES sequence for drug representation and genomic mutation as
cancer cell profile, which will be fed to a twin convolutional neural
network (CNN) as inputs. We summarized the major limitations of
prior studies as follows.


- Conventional feature extractions are unable to capture intrinsic
chemical structures of drugs. For example, engineered features of
compounds only consider chemical descriptors and molecular
fingerprints (Chang et al., 2018; Liu et al., 2018a; Wei et al.,
2019). Although they have been applied to drug discovery and
compound similarity search (Cereto-Massague´ et al., 2015), such
features are sparse and computationally expensive for drug representation. Also, string-based (e.g. SMILES) representation of
drugs (Guimaraes et al., 2017; Liu et al., 2019; Popova et al.,
2018; Segler et al., 2018) is quite brittle as small changes in the
string can lead to completely different molecules (Kusner et al.,
2017).

- Despite the emergence of multi-omics profiles, the vast majority
of previous studies merely focused on the analysis of single type
of omics data, such as genomic or transcriptomic profiles of cancer cells. The synergy of omics profiles and their interplay has
not been fully explored. In addition, the epigenomic data (e.g.
DNA methylation), proven to be highly related to cancer occurrence (Klutstein et al., 2016), is largely ignored.


Considering the above limitations, we proposed a hybrid graph
convolutional network for predicting CDR (Fig. 1). DeepCDR consists of a uniform graph convolutional network (UGCN) for drug
representation based on the chemical structure of drugs.



Additionally, DeepCDR contains several subnetworks for feature
extraction of multi-omics profiles from genomics, transcriptomics
and epigenomics inputs. The high-level features of drugs and multiomics data were then concatenated together and fed into a 1-D
CNN. DeepCDR enables prediction of the IC 50 sensitivity value of a
drug with regard to a cancer cell line in a regression task or claiming
the drug to be sensitive or resistant in a classification task.
Conceptually, DeepCDR can be regarded as a multimodal deep
learning solution for CDR prediction. We summarized our contributions as follows.


- We proposed a UGCN for novel feature extraction of drugs.
Compared to hand-crafted features (e.g. molecular fingerprints)
or string-based features (e.g. SMILES), the novel design of
UGCN architecture can automatically capture drug structures by
considering the interactions among atoms within a compound.

- We discovered that the synergy of multi-omics profiles from cancer cell lines can significantly improve the performance of CDR
prediction and epigenomics profiles are particularly helpful
according to our analysis.

- We designed extensive experiments to reveal the superiority of
our model. DeepCDR achieves state-of-the-art performance in
both classification and regression settings, highlighting the strong
predictive power of UGCN architecture and multimodal learning

strategy.


2 Materials and methods


2.1 Overview of DeepCDR framework
DeepCDR is constructed by a hybrid graph convolutional network
for CDR prediction, which integrates both drug-level and multiomics features (Fig. 1). The output of DeepCDR is measured by the
IC 50, which denotes the effectiveness of a drug in inhibiting the
growth of a specific cancer cell line. For example, small IC 50 value
reveals a high degree of drug efficacy, implying that the drug is sensitive to the corresponding cancer cell line.
DeepCDR consists of a UGCN and several subnetworks for
extracting drug and cancer cell line information, respectively (see
[detailed hyperparameters in Supplementary Tables S1 and S2). On](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
the one hand, the UGCN takes the adjacent information of atoms in
a drug into consideration by aggregating the features of neighboring
atoms together. On the other hand, the subnetworks extract highlevel features of cancer omics profiles from a certain cancer cell line



Fig. 1. The overview framework of DeepCDR. DeepCDR contains a UGCN and three subnetworks for processing drug structures and cancer cell profiles (genomic mutation,
gene expression and DNA methylation data) respectively. DeepCDR takes a pair of drug and cancer cell profiles as inputs and predicts the drug sensitivity (IC 50 ) (regression)
or claims the drug to be sensitive or resistant (classification). The drug will be represented as a graph based on the chemical structure before transformed into a high-level latent
representation by the UGCN. Omics featured learned by subnetworks will be concatenated to the drug feature


DeepCDR predicts cancer drug response i913



(i.e. genomic data, transcriptomic data and epigenomic data). Then
the high-level features of drug and multiple omics data were concatenated and fed to a 1D CNN. To alleviate potential overfitting in
the training process, we used Batch normalization (Ioffe and
Szegedy, 2015) and Dropout (Srivastava et al., 2014) after each convolutional layer. We used Adam as the optimizer for updating the
parameters of DeepCDR in the back-propagation process. Similar to
Liu et al. (2018b), the DeepCDR classification model takes a sigmoid layer for prediction and cross-entropy (CE) as loss function,
while the DeepCDR regression model directly uses a linear layer
without an activation function and takes mean square error (MSE)
as loss function.


2.2 Drug feature representation
Each drug has its unique chemical structure which can be naturally
represented as a graph where the vertices and edges denote chemical
atoms and bonds, respectively. Suppose we have M drugs in our
study, the graph representation of these drugs can be described as
fG i ¼ ðX i ; A i Þj [M] i¼1 [g][ where][ X] [i] [ 2][ R] [N] [i] [�][C] [ and][ A] [i] [ 2][ R] [N] [i] [�][N] [i] [ are the fea-]
ture matrix and adjacent matrix of the ith drug, respectively. N i is
the number of atoms in the ith drug and C is the number of feature
channels. Each row of feature matrix corresponds to the attributes
of an atom. Following the description in (Ramsundar et al., 2019),
the attributes of each atom in a compound were represented as a 75dimensional feature vector (C ¼ 75), including chemical and topological properties such as atom type, degree and hybridization. We
downloaded the structural files (.MOL) of all drugs (M ¼ 223) from
PubChem library (Kim et al., 2019) of which the number of atoms
N i varies from 5 to 96.


2.3 Uniform graph convolutional network (GCN)
We seek to achieve graph-level classification as each input of drug
represents a unique graph structure while the original GCN (Kipf
and Welling, 2017) aims at node classification within a single graph.
To address this issue, we extended the original GCN architecture
and presented a UGCN for processing drugs with variable sizes and
structures. The core idea of UGCN is to introduce an additional
complementary graph to the original graph of each drug to ensure
the consistent size of feature matrix and adjacent matrix. Given
the original graph representation of M drugs fG i ¼ ðX i ; A i Þj [M] i¼1 [g][, the]
complementary graphs of drugs can be represented as fG [c] i [¼]
ðX [c] i [;][ A] [c] i [Þj] [M] i¼1 [g][, where][ X] [c] i [2][ R] [ð][N][�][N] [i] [Þ�][C] [ and][ A] [c] i [2][ R] [ð][N][�][N] [i] [Þ�ð][N][�][N] [i] [Þ] [.][ N][ is]
a fixed number which is set to 100. Thus, the consistent representation of a drug is designed as follows:



� BA [T] ii AB [c] ii �; X [0] i [¼] � XX [c] ii



c
H i [ð][1][;][b][Þ] ¼ rððð c D [~] i [þ][ D] [B] i [T] [Þ] ~~[�]~~ c 2 [1] B [T] i c [ð][D][ ~] [i] [ þ][ D] [B] i [Þ] ~~[�]~~ 2 [1] X i þ (4)
ðD [~] i [þ][ D] [B] i [T] [Þ] ~~[�]~~ 2 [1] ~A i [ð][D][ ~] i [þ][ D] [B] i [T] [Þ] ~~[�]~~ 2 [1] X [c] i [Þ][H] [ð][0][Þ] [Þ][;]


where D [~] i and D [~] ci [are the degree matrix of][ ~][A] [i] [ and][ ~][A] ci [and][ D] [B] i [and]
D [B] i [T] are two diagonal matrix for describing row sum and column
sum of B i . D [B] i [½][j][;][ j][�¼][ P] k [B] [i] [½][j][;][ k][�] [and][ D] [B] i [T] [½][j][;][ j][�¼][ P] k [B] [T] i [½][j][;][ k][�][.]
With mathematical induction, it can be inferred that the general
layer-wise propagation rule of UGCN can be represented by the following two equations:


H i [ð][l][þ][1][;][a][Þ] ¼ rðððD [~] i þ D [B] i [Þ] ~~[�]~~ 2 [1] A~ i ðD ~ i þ D [B] i [Þ] ~~[�]~~ 2 [1] H i [ð][l][;][a][Þ] þ

c
ðD [~] i þ D [B] i [Þ] ~~[�]~~ 2 [1] B i ðD ~ i [þ][ D] [B] i [T] [Þ] ~~[�]~~ 2 [1] H i [ð][l][;][b][Þ] ÞH [ð][l][Þ] Þ;


c
H i [ð][l][þ][1][;][b][Þ] ¼ rððð c D [~] i [þ][ D] [B] i [T] [Þ] ~ ~~[�]~~ c 2 [1] B [T] i c [ð][D][ ~] [i] [ þ][ D] [B] i [Þ] ~~[�]~~ 2 [1] H [ð] i [l][;][a][Þ] þ
ðD [~] i [þ][ D] [B] i [T] [Þ] ~~[�]~~ 2 [1] A i [ð][D][ ~] i [þ][ D] [B] i [T] [Þ] ~~[�]~~ 2 [1] H i [ð][l][;][b][Þ] ÞH [ð][l][Þ] Þ:


We further consider a special case where the complementary
graphs have no connection to the original graphs (B i ¼ 0), so the
layer-wise propagation rule of UGCN will be simplified as


H i [ð][l][þ][1][Þ] ¼ ½ rðD ~ ~~�~~ i [1] 2 A~ i ~D ~~�~~ i [1] 2 [H] i [ð][l][;][a][Þ] H [ð][l][Þ] ÞrðD [~] ci ~~[�]~~ 2 [1] A~ ci [D][~] ci [�] 2 [1] H i [ð][l][;][b][Þ] H [ð][l][Þ] Þ� [:] (5)


Overall, we showed that the convolution on the original graph
and the corresponding complementary graph is independent in
UGCN given the conjunction matrix B i ¼ 0. At last, we applied a
global max pooling over the graph nodes in A [0] i [to ensure that drugs]
with different size will be embedded into a fixed dimensional vector
(default dimension: 100). In our study, we set B i ¼ 0; X [c] i [¼][ 0 and]
l ¼ 3 as the default settings for initializing DeepCDR. We also
explored another initialization strategy in the discussion section.


2.4 Omics-specific subnetworks
We designed omics-specific subnetworks to integrate the information of multi-omics profiles. We used the late-integration fashion in
which each subnetwork will first learn a representation of a specific
omics data in a latent space and then be concatenated together. The
three subnetworks can be represented as fy g ¼ f g ðx g Þ; y t ¼ f t ðx t Þ;
y e ¼ f e ðx e Þg for processing genomic, transcriptomic and epigenomic
data per sample, respectively. Similar to Chang et al. (2018), we
used a 1 D convolutional network for processing genomic data as
the mutation positions are distributed linearly along the chromosome. For transcriptomic and epigenomic data, we directly used
fully connected networks for feature representation. (see detailed
[hyperparameters of subnetworks in Supplementary Table S2).](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)



8><>:



f g : x g 2 R [1][�][d] [g] 7!y g 2 R [1][�][d]

f t : x t 2 R [1][�][d] [t] 7!y t 2 R [1][�][d] :

f e : x e 2 R [1][�][d] [e] 7!y e 2 R [1][�][d]



A [0] i [¼] � BA [T] ii AB [c] ii



� XX [c] ii �; (1)



where B i 2 R [N] [i] [�ð][N][�][N] [i] [Þ] is a conjunction matrix which represents the
connection between the ith original graph and complementary
graph. A [0] i [2][ R] [N][�][N] [ and][ X] [0] i [2][ R] [N][�][C] [ are the consistent adjacent ma-]
trix and feature matrix. The UGCN applied to ith drug is defined as
f ðX [0] i [;][ A] [0] i [Þ][ with a layer-wise operation as]


0 ~~�~~ [1] 0 0 ~~�~~ [1]
H i [ð][l][þ][1][Þ] ¼ rðD [~] i 2 A [~] i [D][~] i 2 H [ð] i [l][Þ] [H] [ð][l][Þ] [Þ][;] (2)

connections,whereP k [A][~] [i] 0 ½jA;~ k 0i � [¼] . H [ A] ði D~ l [0] iÞ [þ] 0i and [ I] is the degree matrix of [N] His the adjacent matrix with added self- [ð][l][Þ] are the convolved signal and filter parame-A~ 0i which D~ 0i [½][j][;][ j][�¼]
ters of the lth layer. rð�Þ is the activation function, which is set to
ReLuð�Þ ¼ maxð0; �Þ.We further denote the first N i rows of H [ð] i [l][Þ] as
H i [ð][l][;][a][Þ] and the remaining ðN � N i Þ rows as H i [ð][l][;][b][Þ] . For the first graph
layer where l ¼ 0, we initialized the first layer as H [ð] i [0][Þ] ¼ X [0] i [and sub-]
stituted formula (1) into (2), we can derive the propagation rule of
first layer of UGCN as the following:


H [ð] i [1][;][a][Þ] ¼ rðððD [~] i þ D [B] i [Þ] ~~[�]~~ 2 [1] ~A ic ðD ~ i þ D [B] i [Þ] ~~[�]~~ 2 [1] X i þ (3)
ðD [~] i þ D [B] i [Þ] ~~[�]~~ 2 [1] B i ðD ~ i [þ][ D] [B] i [T] [Þ] ~~[�]~~ 2 [1] X [c] i [Þ][H] [ð][0][Þ] [Þ][;]



The dimension of latent space d is set to 100 in our experiments
by default.


2.5 Data preparation
We integrated three public databases in our study including GDSC
(Iorio et al., 2016), CCLE (Barretina et al., 2012) and TCGA patient
data (Weinstein et al., 2013). GDSC database provides IC 50 values
for a large-scale drug screening data, of which each IC 50 value corresponds to a drug and a cancer cell line interaction pair. CCLE database provides genomic, transcriptomic and epigenomic profiles for
more than a thousand cancer cell lines. For the three omics data, we
focused on genomic mutation data, gene expression data and DNA
methylation data, which can be easily accessed and downloaded
[using DeMap portal (https://depmap.org). TCGA patient data pro-](https://depmap.org)
vide both genetic profiles of patients and clinic annotation after
drug treatment. We used TCGA dataset for an external validation.

We downloaded IC 50 values (natural log-transformed) across
hundreds of drugs and cancer cell lines from GDSC database as the
ground truth of drug sensitivity profiles for measuring CDR. We


i914 Q.Liu et al.



excluded drug samples without PubChem ID in GDSC database and
removed cancer cell lines in which any type of omics data was missing. Note that several drugs with different GDSC ids may share the
same PubChem ids due to the different screening condition. We
treated them as individual drugs in our study. We finally collected a
dataset containing 107 446 instances across 561 cancer cell lines
and 238 drugs. Considering all the 561 � 238 ¼ 133 518 drug and
cell line interaction pairs, approximately 19.5% (26 072) of the IC 50
values were missing. The corresponding drug and cancer cell line
[datasets used in this study are summarized in Supplementary Tables](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
[S3 and S4. Each instance corresponds to a drug and cancer cell line](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
interaction pair. Each cell line was annotated with a cancer type
defined in The Cancer Genome Atlas (TCGA) studies and we only
considered TCGA cancer types suggested by (Chang et al., 2018) in
the downstream analysis.
For multi-omics profiles of cancer cell lines, we only consider
data related to 697 genes from COSMIC Cancer Gene Census
[(https://cancer.sanger.ac.uk/census). For genomic mutation data,](https://cancer.sanger.ac.uk/census)
34 673 unique mutation positions including SNPs and Indels within
the above genes were collected. The genomic mutation of each cancer cell line was represented as a binary feature vector in which ’1’
denotes a mutated position and ‘0’ denotes a non-mutated position
(d g ¼34 673). For gene expression data, the TPM value of gene expression was log 2 transformed and quantile normalized. Then the
gene expression of each cell line can be represented as a 697-dimensional feature vector (d t ¼697). The DNA methylation data was directly obtained from the processed Bisulfite sequencing data of
promoter 1 kb upstream TSS region. Then we applied a median
value interpolation to the data as there were a minority of missing
values. The methylation of each cell line is finally represented by a
808-dimensional feature vector (d e ¼808). The three types of omics
data were finally transformed into a latent space where the
embedded dimension was fixed to 100 (d ¼ 100).

For TCGA patient data, we chose the patients with cervical
squamous cell carcinoma and endocervical adenocarcinoma (CESC)
disease with two criterions. (i) The gene mutation, gene expression
and DNA methylation data are available. (ii) The clinic annotation
of drug response was also available. We finally created an external
data source with 54 records across 31 patients and 12 drugs. The
genetic profiles were preprocessed the same way as cell line data and
we took records with ‘Complete Response’ clinic annotation as posi[tive examples (see details in Supplementary Table S5).](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)


2.6 Baseline methods
The following competing methods were considered. The best or default parameters of each method were used for model comparison.


- Ridge regression is a linear regression model with L 2 penalty. We
first concatenated genomic mutation features and molecular fingerprints of drugs together and then fed to ridge regression
model. The ridge regression model was implemented using
sklearn library (Pedregosa et al., 2011). Basically, we found no
significant changes in results as we tried different settings of the
L 2 penalty coefficient from f0.1,0.5,1.0,5.0g. We finally chose
the default coefficient parameter 1.0 provided by sklearn library
in the comparing experiments.

- Random forest is a tree-based regressor in which the input is the
same as the ridge regression model. The random forest was also
implemented with sklearn library (Pedregosa et al., 2011). We
set the number of trees in the forest from f10,100,200,500g and
chose the best parameter in the comparing experiments.

- CDRscan (Chang et al., 2018) applies an ensemble CNNs model
for predicting CDR using molecular fingerprints of drugs and
genomic mutation data of cancer cell line.

- tCNNs (Liu et al., 2019) applies an CNN for predicting CDR
using SMILES sequences of drugs and genomic mutation data of
cancer cell line. SMILES sequences of drugs will first be encoded
into one-hot representation and fed to the neural network.




- MOLI (Sharifi-Noghabi et al., 2019) is one of the few studies
that considers multi-omics profiles (genomic mutation, copy
number and gene expression) with encoder neural networks.
Specifically, MOLI is a drug-specific model where each model is
trained for a specific drug.


2.7 Model evaluation
In the regression experiments for predicting natural log-transformed
IC 50 values given the profiles of drugs and cancer cell lines, we used
three common metrics for measuring the statistical correlation between observed values and predicted IC 50 values, including
Pearson’s correlation coefficient (PCC), Spearman’s correlation coefficient (SCC) and root mean squared error (RMSE). PCC measures
the linear correlation between observed and predicted IC 50 values
while SCC is a non-parametric measure of rank correlation of
observed and predicted IC 50 values. RMSE directly measures the difference of observed and predicted IC 50 values.
For classification experiments, we chose the area under the receiver operating characteristic curve (AUC) and area under the precision–recall curve (auPR) as the two commonly used measurements
of a classifier.
To comprehensively evaluate the performance of our model
DeepCDR, we demonstrated results under various data settings. We
briefly summarized these different data settings in the following:


- Rediscovering known CDRs. Based on the known drug–cell line
interactions across 561 cancer cell lines and 238 drugs, we randomly selected 95% of instances of each TCGA cancer type as
the training set and the remaining 5% of the instances as the testing set for model evaluation. The five-fold cross-validation was

conducted.

- Predicting unknown CDRs. We trained DeepCDR model with
all the known drug–cell line interaction pairs and predicted the
missing pairs in GDSC database (approximately 19.5% of all
pairs across 561 cancer cell lines and 238 drugs).

- Blind test for both drugs and cell lines. In order to evaluate the
predictive power of DeepCDR when given a new drug or new
cell line that is not included in the training data. We randomly
split the data into 80% training set and 20% test set on the cell
line or drug level. The five-fold cross-validation using leave-drugout and leave-cell-line-out strategy was conducted.

- External validation with patient data. To evaluate whether
DeepCDR trained with in vitro cell line data can be generalized
to in vivo patient data. We trained DeepCDR classification
model with cell line data and tested on TCGA patient data.


3 Results


3.1 DeepCDR recovers continuous degree of drug
sensitivity
We first designed a series of experiments to see whether DeepCDR
can help recover continuous degree of drug sensitivity. For this objective, we created datasets of the drug and cancer cell lines profiles
from GDSC (Iorio et al., 2016) and CCLE (Barretina et al., 2012)
database, respectively. We then evaluated the regression performance of DeepCDR and five comparing methods based on the
observed IC 50 values and predicted IC 50 values. Three common regression evaluation metrics, including Pearson’s correlation coefficient (PCC), Spearman’s correlation coefficient (SCC) and root
mean square error (RMSE), were considered.
First, we evaluated the ability of DeepCDR and competing methods by rediscoverying CDR across multiple drugs and cell lines. We
observed that DeepCDR demonstrated superior predictive performance of drug response in the regression experiments by achieving the
highest Pearson’s correlation and Spearman’s correlation and lowest


DeepCDR predicts cancer drug response i915


Table 1. Regression experiments of IC 50 values with DeepCDR and five comparing methods


Methods Pearson’s correlation Spearman’s correlation RMSE


Ridge regression 0.780 0.731 2.368
Random forest 0.809 0.767 2.270

MOLI 0.813 6 0.007 0.782 6 0.005 2.282 6 0.008

CDRscan 0.871 6 0.004 0.852 6 0.003 1.982 6 0.005

tCNNs 0.885 6 0.008 0.862 6 0.006 1.782 6 0.006

DeepCDR 0.923 6 0.006 0.903 6 0.004 1.058 6 0.006


Note: Three different measurements, including Pearson’s correlation, Spearman’s correlation and root mean square error (RMSE), were illustrated. We trained
neural network-based models from scratch for five times and the standard deviations of each method were also calculated for evaluating the model robustness.
DeepCDR demonstrates a consistent highest performance in all measurements comparing to other methods. Pearson’s correlation improved by 3.8% to the best
comparing method. Spearman correlation improved by 4.1% to the best comparing method. RMSE decreased by 0.724.


Fig. 2. Performance of DeepCDR in CDR prediction under different experiment settings. (A) and (B) highlighted the scatter plots in two TCGA cancer types with the best
(MM) and worst (LAML) performance. (C) and (D) showed the scatter plots in two drugs with the best (Belinostat) and worst (Pazopanib) performance. (E) The predicted
IC 50 values of missing data in GDSC database grouped by drugs. Drugs were sort according to the average predicted IC 50 value in missing cell lines. The number of missing
cell lines for each drug is also denoted below/above the violin plot. Each violin plot corresponds to a specific drug response in all missing cell lines. The blue and red violin plots
denote the top-10 drugs with the highest and the lowest efficacy. (F) The performance of DeepCDR and tCNNs in blind test for drugs. The x-axis and y-axis of each dot represent the Pearson’s correlation of tCNNs and DeepCDR, respectively. The dot fallen into the left upper side denotes the case where DeepCDR outperforms tCNNs. (G) The performance of DeepCDR and tCNNs in blind test for cell lines. (H) and (I) show the receiver operating characteristic (ROC) and precision–recall (PR) curve of the four
comparing methods, respectively. (J) and (K) show the violin plots of the area under ROC curve (AUC) and area under PR curve (auPRs) across TCGA cancer types. Note that
each dot within a violin plot represents the average AUC or auPR score within one TCGA cancer type. Additionally, one-sided Mann-Whitney U tests between DeepCDR and
tCNNs were conducted. *P-value ¼ 1.01�10 [�][5], **P-value ¼ 0.062



RMSE comparing to five competing methods (Table 1). Generally,
deep neural network models significantly outperformed other baselines as linear or tree-based model may not well capture the intrinsic
structural information within drugs. Among the four deep learning
models, DeepCDR outperforms three other deep learning methods
with a relatively large margin by achieving a Pearson’s correlation of
0.923 as compared to 0.813 of MOLI, 0.871 of CDRscan and 0.885
of tCNNs. This conclusion is also consistent considering other metrics such as Spearman’s correlation and root mean square error
(RMSE). In addition, we also showed the model variance by independent training for five times.



Next, we illustrated several prediction cases across multiple
TCGA cancer types or different drug compounds. Among the 30 different TCGA cancer types, DeepCDR reveals a consistently high
performance by achieving a Pearson’s correlation ranging from
0.889 to 0.941. The best prediction case in multiple myeloma cancer
type and the worst prediction case in acute myeloid leukemia cancer
type were shown in Figure 2A and B, respectively. In the perspective
of drug, we also evaluated the regression performance with respect
to a specific drug. We observed that the DeepCDR illustrates a relatively more dynamic regression performance by achieving a
Pearson’s correlation ranging from 0.328 to 0.938 (Fig. 2C and D),


i916 Q.Liu et al.



which may due to the drug similarity diversity. We then validated
this conclusion by measuring the drug similarity among training
set and found that Belinostat has a significantly higher drug
[similarity score compared to Pazopanib (Supplementary Fig. S1,](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
P-value ¼ 2.38�10 [�][37] ). The distribution of correlation across
[TCGA cancer types and drugs were provided in Supplementary](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
[Figure S2.](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
Next, we applied DeepCDR to predicting unknown CDRs in the
GDSC database. Towards this goal, DeepCDR was trained on all
known drug cell line interaction pairs across 561 cell lines and 238
drugs, then it was used for predicting the missing pairs in GDSC
database (approximately 19.5% of all pairs). Figure 2E illustrates
the distributions of predicted IC 50 values in GDSC database grouped
by drugs. Note that drugs were sorted by the median predicted IC 50
value across all missing cell lines. We provided the predicted IC 50
[values of top-10 drugs and related cancer cell lines in Supplementary](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
[Table S6. Interestingly, Bortezomib was the drug with highest effi-](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
cacy in our prediction which has been proved to be a proteasome inhibitor that has activity in multiple cancer cell lines (Richardson
et al., 2003). Specifically, the predicted IC 50 of Bortezomib with an
oesophagus cell line KYSE-510 is 7:45 � 10 [�][5] which implies a
strong therapeutic effect. This prediction was supported by the findings in Lioni et al. (2008), which highlighted the robust activity of
Bortezomib in esophageal squamous cells. Phenformin and AICA
ribonucleotide are predicted to have the lowest efficacy. The former
was used for treating type 2 diabetes mellitus by inhibiting complex
I (Marchetti et al., 1987). The latter is capable of stimulating AMPdependent protein kinase (AMPK) activity (Corton et al., 1995).
The anti-cancer of the two drugs might be not their main function
but the side effect.
At last, we designed a series of blind tests for both drugs and cell
lines. The task becomes much more challenging as the drugs or cell
lines in the test data were unseen during the training process. The
drug sensitivity data from GDSC database were split into training
and test sets on the drug or cell line level. We compared our model
DeepCDR to the best baseline model tCNNs in the previous experiments. In the blind test for drugs, the performance of both methods
largely decreased compared to previous experiments. However,
DeepCDR still achieves an average Pearson’s correlation of 0.503,
compared to 0.429 of tCNNs (Fig. 2F, P-value < 1.51�10 [�][6],
[Supplementary Table S7). In the blind test for cell lines, DeepCDR](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
again outperforms tCNNs by a quite large margin by achieving an
average Pearson’s correlation of 0.889, compared to 0.865 of
tCNNs (Fig. 2G, P-value < 2.2�10 [�][16] [, Supplementary Table S8).](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)


3.2 DeepCDR predicts binary drug sensitivity status
In this section, we binarized IC 50 according to the threshold of each
drug provided by Iorio et al. (2016). After filtering drug samples
without a binary threshold, we collected a dataset with 7488 positive instances in which drugs are sensitive to the corresponding cancer cell line and 52 210 negative instances where drugs are resistant
to cancer cell lines. Similar to the regression experiment settings, we
first compared DeepCDR to three other neural network models by
rediscovering the CDR status. Despite of the unbalanced dataset
(around 1:7), DeepCDR outperforms three other methods by a large


Table 3. Top-5 cancer-associated genes prioritized by DeepCDR



margin by achieving a significantly higher AUC and auPR score of
0.841 and 0.502 (Fig. 2H and I), reaffirming the advance of
DeepCDR in capturing the interaction information of drug and cancer cells. As seen in Figure 2J and K, we grouped the test instances
(each instance denotes a drug and cancer cell line pair) according to
the TCGA cancer types, then we calculated the AUCs and auPRs of
the two methods under different cancer type groups. We observed
that DeepCDR achieves higher AUC score and auPR score than
tCNNs with respect to every TCGA cancer type. In the blind test for
both drugs and cell lines, DeepCDR achieves a consistently better
performance than the best baseline tCNNs with average AUC of
[0.737 (Supplementary Figs S3 and S4). Besides, statistical hypothesis](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
tests, including binomial exact test and Mann-Whitney U test, were
additionally conducted in both blind test experiments for drugs and
[cell lines (Supplementary Tables S9 and S10).](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)
Last, we introduced TCGA patient data as an external validation. We trained DeepCDR model on in vitro cell line data
described above, and tested on in vivo patient data. Note that the
external dataset even contains more than 40% of drugs that were
not included in the cell line data. DeepCDR still achieves a performance with AUC 0.688, compared to 0.618 of tCNNs
[(Supplementary Fig. S5).](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btaa822#supplementary-data)


3.3 Model ablation analysis
Since most early studies only considered single type of omics data, it
is necessary for us to evaluate the contribution of different types of
omics data. For each type of omics data, we discarded other types of
omics data and trained DeepCDR regression model from scratch for
model ablation analysis. When using single omic data, the Pearson’s
correlation of DeepCDR ranges from 0.878 to 0.890, indicating the
usefulness of all individual omics profiles (Table 2). In particular,
the epigenomics data (DNA methylation) contributes the most
among different omics profiles. Notably, DeepCDR still achieved a
higher Pearson’s correlation than tCNNs even only genomic data
was used in both methods (0.889 versus 0.885). Furthermore, to verify the effectiveness of graph convolution, we first eliminated the adjacent information of atoms within drugs by setting adjacent


Table 2. Model ablation studies with different experimental
settings


Experimental setting Pearson’s correlation


Single genomics 0.889
Single transcriptomics 0.878
Single epigenomics 0.890
Multi-omics without adjacent info 0.886
Multi-omics with adjacent info 0.923


Note: We showed both the contribution of each omic profile and the contribution of graph convolution module. Mult-omic imporves Pearson’s correlation by at least 3.3%. Note that the results were calculated based on five
independent runs, it is hard to do statistical test due to the small sample size

(5).



Drug Cell line TCGA type Top-5 cancer-associated genes ln(IC 50 )


Observed Predicted


Erlotinib A3/KAW DLBC EGFR, ALK, BCL10, CREB3L1, STAG1 1.110 1.206
Lapatinib BT-474 BRCA ERBB2, MDS2, FOXL2, EGFR, MNX1 �1.028 �0.879
Bleomycin A-375 SKCM ACKR3, ASXL 1, MTCP1, FOXL2, SALL4 �1.514 �1.428
Nilotinib BHT-101 THCA CBLC, ABI1, POU5F1, KLF4, ZNF198 �0.630 �0.714
Salubrinal SUP-B15 ALL JAK3, EIF1AX, NUMA1, PRDM1, IL21R 1.781 1.471


Note: We proposed a simple gradient-based strategy for prioritizing all the genes when making a prediction of a specific drug and cancer cell line pair. Many
top-ranked genes have been verified to be highly associated with cancers by existing literatures.


DeepCDR predicts cancer drug response i917



matrices to identity matrices (A [~] i ¼ I N i ). Then the DeepCDR model
without adjacent information achieved a reasonable Pearson’s correlation of 0.886. We concluded that the regression performance
can be significantly boosted with the powerful representation
inferred from adjacent information by the proposed UGCN architecture (0.923 versus 0.886).


3.4 DeepCDR helps prioritize cancer-associated genes
To deepen the understanding of biological knowledge revealed by
DeepCDR, we further proposed an exploratory strategy for prioritizing cancer-associated genes given an input drug and cancer cell
line pair, where we prioritized the involved genes by assigning each
gene with an associated score. In detail, to obtain the associated
scores for 697 genes involved in COSMIC Cancer Gene Census, we
considered the absolute gradient of the predicted outcome from
DeepCDR regression model with respect to the each gene’s expression. We highlighted several cases where the drugs were shown sensitive to the corresponding cancer cell lines (Table 3). Importantly,
we found that many top-ranked genes have been verified to be associated with cancers by existing literature. For example, Erlotinib
and Lapatinib, two known drugs for treating lung cancer, have been
proven to be EGFR inhibitors (Sayar et al., 2014), EGFR gene ranks
first and fourth from DeepCDR prioritizer in A3/KAW and BT-474
cell lines, respectively. Also, Nilotinib is a potential drug treatment
for chronic myelogenous leukemia (Kantarjian et al., 2011).
Interestingly, in our predictive task in a BHT-101 cell line, ABI1
ranked as the second cancer-associated gene, which has been previously proved to have specific expression patterns in leukemia cell
lines (Shibuya et al., 2001). Taken together, these evidences support
that DeepDCR could reveal potential therapeutic targets for anticancer drugs and help discover hypothetical cancer-associated genes
for additional clinical testing.


4 Discussion


In this study, we have proposed DeepCDR as an end-to-end deep
learning model for precise anti-cancer drug response prediction. We
found that GCNs were extremely helpful for capturing structural information of drugs according to our analysis. To the best of our
knowledge, DeepCDR is the first work to apply GCN in CDR problem. In addition, we demonstrated that the combination of multiomics profiles and intrinsic graph-based representation of drugs are
appealing for assessing drug response sensitivity. Extensive experiments highlighted the predictive power of DeepCDR and its potential translational value in guiding disease-specific drug design.
We provide two future directions for improving our method. (i)
The proposed UGCN can be utilized for data augmentation when
training instances were not adequate enough or extremely unbalanced by randomly sampling multiple complementary graphs for
each drug. In the classification experiment where the training data is
unbalanced, if we randomized the feature matrix and gave random
connections of complementary graphs and augmented the positive
training instances by five times, the average AUC can be further
improved by 0.8%. Augmentation with UGCN can potentially further improve prediction performance. (ii) DeepCDR can be leveraged in combination with molecule generation tasks. Current
molecule generation models based on RNN language models (Segler
et al., 2018), generative adversarial networks (GANs) (Guimaraes
et al., 2017) and deep reinforcement learning (Popova et al., 2018)
focused on generating general compounds and ignored profiles of
targeted cancer cell. Methods focused on cancer-specific or diseasespecific novel drug design can be proposed by using CDR predicted
by DeepCDR as a prior knowledge or a reward score for guiding
molecule generation.
To sum up, we introduced DeepCDR that can be served as an
application for exploring drug sensitivity with large-scale cancer
multi-omics profiles. DeepCDR outperforms multiple baselines and
our analysis illustrates how our method can help prioritize therapeutic targets for anti-cancer drug discovery. In future work, we
plan to expand data inclusion for a large-scale omics data profiled



both before and after treatment to assess how their molecular profiles respond to perturbation by the testing drugs.


Acknowledgements


The authors thank Fengling Chen and Zhana Duren for their helpful
discussion.


Funding


This work was supported by the National Key Research and Development
Program of China [2018YFC0910404], the National Natural Science
Foundation of China [61873141, 61721003, 61573207], the TsinghuaFuzhou Institute for Data Technology and Shanghai Municipal Science and
Technology Major Project [2017SHZDZX01]. R.J. is also supported by a
RONG professorship at the Institute for Data Science of Tsinghua University.


Conflict of Interest: none declared.


References


Barretina,J. et al. (2012) The cancer cell line encyclopedia enables predictive
modelling of anticancer drug sensitivity. Nature, 483, 603–607.
Cereto-Massague´,A. et al. (2015) Molecular fingerprint similarity search in
virtual screening. Methods, 71, 58–63.
Chang,Y. et al. (2018) Cancer drug response profile scan (CDRscan): a deep
learning model that predicts drug effectiveness from cancer genomic signature. Sci. Rep., 8, 8857.
Corton,J.M. et al. (1995) 5-Aminoimidazole-4-carboxamide ribonucleoside: a
specific method for activating amp-activated protein kinase in intact cells?
Eur. J. Biochem., 229, 558–565.
Daemen,A. et al. (2013) Modeling precision treatment of breast cancer.
Genome Biol., 14, R110.
Daly,A.K. (2017) Pharmacogenetics: a general review on progress to date. Br.
Med. Bull., 124, 1–79.
Dong,Z. et al. (2015) Anticancer drug sensitivity prediction in cell lines from
baseline gene expression through recursive feature selection. BMC Cancer,
15, 489.
Gagan,J. and Van Allen,E.M. (2015) Next-generation sequencing to guide
cancer therapy. Genome Med., 7, 80.
Geeleher,P. et al. (2014) Clinical drug response can be predicted using baseline
gene expression levels and in vitro drug sensitivity in cell lines. Genome
Biol., 15, R47.
Guimaraes,G.L. et al. (2017) Objective-reinforced generative adversarial networks (organ) for sequence generation models. arXiv preprint arXiv:

1705.10843.

Ioffe,S. and Szegedy,C. (2015) Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of
Machine Learning Research, Vol 37, pp. 448–456.
Iorio,F. et al. (2016) A landscape of pharmacogenomic interactions in cancer.
Cell, 166, 740–754.
Kantarjian,H.M. et al. (2011) Nilotinib is effective in patients with chronic
myeloid leukemia in chronic phase after imatinib resistance or intolerance:
24-month follow-up results. Blood, 117, 1141–1145.
Kim,S. et al. (2019) Pubchem 2019 update: improved access to chemical data.
Nucleic Acids Res., 47, D1102–D1109.
Kipf,T.N. and Welling,M. (2017) Semi-supervised classification with graph
convolutional networks. In International Conference on Learning
Representations (ICLR), Toulon, France.
Klutstein,M. et al. (2016) DNA methylation in cancer and aging. Cancer Res.,
76, 3446–3450.
Kohane,I.S. (2015) Ten things we have to do to achieve precision medicine.
Science, 349, 37–38.
Kusner,M.J. et al. (2017) Grammar variational autoencoder. In Proceedings
of the 34th International Conference on Machine Learning-Volume 70.
JMLR.org, PMLR, Sydney, Australia, pp. 1945–1954.
Lee,J.-K. et al. (2018) Pharmacogenomic landscape of patient-derived tumor
cells informs precision oncology therapy. Nat. Genet., 50, 1399–1411.
Lioni,M. et al. (2008) Bortezomib induces apoptosis in esophageal squamous
cell carcinoma cells through activation of the p38 mitogen-activated protein
kinase pathway. Mol. Cancer Therap., 7, 2866–2875.


i918 Q.Liu et al.



Liu,H. et al. (2018a) Anti-cancer drug response prediction using neighborbased collaborative filtering with global effect removal. Mol. Therapy
Nucleic Acids, 13, 303–311.
Liu,Q. et al. (2018b) Chromatin accessibility prediction via a hybrid deep convolutional neural network. Bioinformatics, 34, 732–738.
Liu,P. et al. (2019) Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional network. BMC Bioinformatics, 20,
408.

Manica,M. et al. (2019) Toward explainable anticancer compound sensitivity
prediction via multimodal attention-based convolutional encoders. Mol.
Pharm., 16, 4797–4806.
Marchetti,P. et al. (1987) Plasma biguanide levels are correlated with metabolic effects in diabetic patients. Clin. Pharmacol. Therap., 41, 450–454.
Musa,A. et al. (2017) A review of connectivity map and computational
approaches in pharmacogenomics. Brief. Bioinf., 19, 506–523.
Pedregosa,F. et al. (2011) Scikit-learn: machine learning in Python. J. Mach.
Learn. Res., 12, 2825–2830.
Popova,M. et al. (2018) Deep reinforcement learning for de novo drug design.
Sci. Adv., 4, eaap7885.
Ramsundar,B. et al. (2019) Deep Learning for the Life Sciences: Applying
Deep Learning to Genomics, Microscopy, Drug Discovery, and More.
O’Reilly Media, Inc., Sebastopol, California.
Richardson,P.G. et al. (2003) A phase 2 study of bortezomib in relapsed, refractory myeloma. N. Engl. J. Med., 348, 2609–2617.
Rubin,M.A. (2015) Health: make precision medicine work for cancer care.
Nat. News, 520, 290–291.



Sayar,B.S. et al. (2014) EGFR inhibitors erlotinib and lapatinib ameliorate epidermal blistering in pemphigus vulgaris in a non-linear, v-shaped relationship. Exp. Dermatol., 23, 33–38.
Segler,M.H. et al. (2018) Generating focused molecule libraries for drug discovery with recurrent neural networks. ACS Central Sci., 4, 120–131.
Sharifi-Noghabi,H. et al. (2019) MOLI: multi-omics late integration with
deep neural networks for drug response prediction. Bioinformatics, 35,
i501–i509.

Shibuya,N. et al. (2001) t(10;11)-acute leukemias with MLL-AF10 and
MLL-ABI1 chimeric transcripts: specific expression patterns of abi1 gene in
leukemia and solid tumor cell lines. Genes, Chromosomes Cancer, 32, 1–10.
Srivastava,N. et al. (2014) Dropout: a simple way to prevent neural networks
from overfitting. J. Mach. Learn. Res., 15, 1929–1958.
Turki,T. and Wei,Z. (2017) A link prediction approach to cancer drug sensitivity prediction. BMC Syst. Biol., 11, 94.
Wei,D. et al. (2019) Comprehensive anticancer drug response prediction based
on a simple cell line-drug complex network model. BMC Bioinformatics,
20, 44.
Weinstein,J.N. et al.; The Cancer Genome Atlas Research Network. (2013)
The cancer genome atlas pan-cancer analysis project. Nat. Genet., 45,
1113–1120.

Zhang,F. et al. (2018) A novel heterogeneous network-based method for drug
response prediction in cancer cell lines. Sci. Rep., 8, 3355.
Zhang,N. et al. (2015) Predicting anticancer drug responses using a dual-layer
integrated cell line-drug network model. PLoS Comput. Biol., 11,
e1004498.


