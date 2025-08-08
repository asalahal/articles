Bioinformatics, 38(2), 2022, 461–468


doi: 10.1093/bioinformatics/btab650


Advance Access Publication Date: 24 September 2021


Original Paper

## Systems biology
# TGSA: protein–protein association-based twin graph neural networks for drug response prediction with similarity augmentation


Yiheng Zhu 1,†, Zhenqiu Ouyang 2,†, Wenbo Chen 2, Ruiwei Feng 1,
Danny Z. Chen 3, Ji Cao 4, - and Jian Wu 5, 

1 College of Computer Science and Technology, Zhejiang University, Hangzhou 310000, China, 2 Polytechnic Institute, Zhejiang
University, Hangzhou 310000, China, [3] Department of Computer Science and Engineering, University of Notre Dame, Notre Dame, IN
46556, USA, [4] College of Pharmaceutical Sciences, Zhejiang University, Hangzhou 310000, China and [5] Department of Ophthalmology of
the Second Affiliated Hospital School of Medicine, and School of Public Health, Zhejiang University, Hangzhou 310000, China


*To whom correspondence should be addressed.

- The authors wish it to be known that, in their opinion, the first two authors should be regarded as Joint First Authors.

Associate Editor: Pier Luigi Martelli


Received on June 17, 2021; revised on August 16, 2021; editorial decision on September 1, 2021; accepted on September 24, 2021


Abstract


Motivation: Drug response prediction (DRP) plays an important role in precision medicine (e.g. for cancer analysis
and treatment). Recent advances in deep learning algorithms make it possible to predict drug responses accurately
based on genetic profiles. However, existing methods ignore the potential relationships among genes. In addition,
similarity among cell lines/drugs was rarely considered explicitly.
Results: We propose a novel DRP framework, called TGSA, to make better use of prior domain knowledge. TGSA
consists of Twin Graph neural networks for Drug Response Prediction (TGDRP) and a Similarity Augmentation (SA)
module to fuse fine-grained and coarse-grained information. Specifically, TGDRP abstracts cell lines as graphs based
on STRING protein–protein association networks and uses Graph Neural Networks (GNNs) for representation learning. SA views DRP as an edge regression problem on a heterogeneous graph and utilizes GNNs to smooth the representations of similar cell lines/drugs. Besides, we introduce an auxiliary pre-training strategy to remedy the identified
limitations of scarce data and poor out-of-distribution generalization. Extensive experiments on the GDSC2 dataset
demonstrate that our TGSA consistently outperforms all the state-of-the-art baselines under various experimental
settings. We further evaluate the effectiveness and contributions of each component of TGSA via ablation experiments. The promising performance of TGSA shows enormous potential for clinical applications in precision
medicine.

[Availability and implementation: The source code is available at https://github.com/violet-sto/TGSA.](https://github.com/violet-sto/TGSA)
Contact: wujian2000@zju.edu.cn or caoji88@zju.edu.cn
[Supplementary information: Supplementary data are available at Bioinformatics online.](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data)



1 Introduction


Precision medicine aims to tailor therapeutic regimens for each patient based on individual genes, lifestyle and environment (Hodson,
2016). Although genomics tends to be synonymous with precision
medicine in cancer analysis, it is still challenging to fulfill precision
medicine based on genetic information, since the exact relationships
between genotype and phenotype are not yet clear (Friedman et al.,
2015). Such challenges have prompted researchers to undertake
large-scale anti-cancer drug screens and propose computational



methods to explore the potential impacts of patients’ genetic profiles
on drug response (Baptista et al., 2021).
Recently, the development of high-throughput screening technologies has facilitated screening hundreds of anti-cancer drugs against
a panel of cancer cell lines. Several large-scale cancer genetic projects
have been launched and systematized into public repositories,
including Cancer Cell Line Encyclopedia (CCLE) (Barretina et al.,
2012) and Genetics of Drug Sensitivity in Cancer (GDSC) (Yang
et al., 2013). Given that these datasets provide not only abundant
anti-cancer drug response information but also genetic profiles of



V C The Author(s) 2021. Published by Oxford University Press. All rights reserved. For permissions, please e-mail: journals.permissions@oup.com 461


462 Y.Zhu et al.



cancer cell lines, they are of great help in spurring the development
of drug response prediction (DRP) techniques.
In the past decade, two main categories of computational DRP
methods have been proposed. (i) Matrix factorization: These methods generally decompose a drug response matrix into two low-rank
matrices indicating the representations of cell lines and drugs. By
extending matrix factorization, pathway-drug response associations
(Ammad-ud din et al., 2016) and cell line/drug similarity (Suphavilai
et al., 2018; Wang et al., 2017) are learned to improve the performance. While these early attempts have made remarkable progress in
DRP, they did not explore the features of drugs and cell lines and
showed poor extrapolation when inferring drug response on unseen
drugs or cell lines. (ii) Machine learning: many studies attempted to
resolve the aforementioned issues with machine learning algorithms,
e.g. elastic net (Barretina et al., 2012), support vector machine
(Dong et al., 2015) and random forest (Iorio et al., 2016). While
these methods showed considerable effect, they still have major
drawbacks. First, commonly used features such as molecular fingerprints of drugs and genetic profiles of cell lines are high-dimensional
data, which may cause the methods to become severely overfitting
(Teschendorff, 2019). Second, feature selection is indispensable but
requires deep insight into the involved biological processes (Mayr
et al., 2016). Third, the relationships in biological data are too intricate to be modeled completely by these methods with limited
capacity.
With an unprecedented power of representing high-dimensional
data (e.g. images, videos and text), deep learning methods have attracted
a multitude of attention. Recently, deep learning also found its way into
DRP and outperformed traditional machine learning algorithms
(Baptista et al., 2021). According to the training modes, deep learningbased DRP methods can be categorized into (i) two-stage frameworks
and (ii) end-to-end frameworks. Two-stage frameworks generally first
learn compressed representations via autoencoders, and then conduct
DRP based on the compressed representations (Chiu et al., 2019; Ding
et al., 2018). End-to-end frameworks learn high-level representations
from raw data, and infer drug responses simultaneously. Specifically,
MOLI (Sharifi-Noghabi et al., 2019) used three type-specific sub-networks to learn representations from multi-omics data, and the learned
representations were concatenated as input to the subsequent subnetwork for DRP. CDRscan (Chang et al., 2018) used Convolutional
Neural Networks (CNNs) to learn representations of drugs and cell
lines from molecular fingerprints and somatic mutation, respectively.
Besides, CNNs have been utilized to process the SMILES (Liu et al.,
2019) and Kekule structures (Corte´s-Ciriano and Bender, 2019) of
drugs.
Exploiting the 2D topology of molecules, drugs can be explicitly
represented as molecular graphs, whose nodes and edges denote
atoms and chemical bonds, respectively. As deep learning-based
methods have been developed for graph problems, GNNs have been
applied to molecule representation learning and become state-of-theart approaches for drug discovery (Sun et al., 2020). More recently,
attempts of DRP also paid close attention to processing drugs with
GNNs. DeepCDR (Liu et al., 2020) proposed a uniform graph convolutional network (UGCN) to capture the intrinsic chemical structures of drugs. However, few studies have been conducted on
applying GNNs to learn the representations of cell lines.
We summarize the major limitations which existing deep
learning-based DRP methods still suffer:


- Existing methods are not yet able to capture the relationships
among genes which are essential for accurately representing cell
lines. For example, CNNs have become a leading architecture in
computer vision due to their ability of exploring spatial topology
of images (Lecun et al., 1998); but, CNNs are still unsuited for
dealing with genomic profiles without readily exploitable spatial

information.

- Existing methods have not taken advantage of the similarity
among cell lines/drugs well enough. Note that cell lines with similar genetic profiles and drugs with similar chemical structures



generally have similar drug responses, as discussed in previous
studies (Wang et al., 2017; Zhang et al., 2015).

- Existing methods still face difficulties in generalization to unseen
cell lines and drugs in blind tests, which means that these methods are far insufficient for precision medicine in clinical cancer
applications.


To address the limitations above, we propose a novel DRP
framework, called Twin Graph neural networks with Similarity
Augmentation (TGSA) (see Fig. 1 for an overview). TGSA is composed of Twin Graph neural networks for Drug Response Prediction
(TGDRP) and a Similarity Augmentation (SA) module, incorporating prior domain knowledge to fuse fine-grained (gene-level and
atom-level) and coarse-grained (sample-level) information. TGDRP
utilizes two GNN encoders to learn the representations of drugs and
cell lines based on molecular graphs and STRING protein–protein
association networks (Szklarczyk et al., 2019), respectively, and then
the learned representations are concatenated and fed to a fully connected network (FCN) to make the final prediction. In the SA module, we incorporate similarity among cell lines/drugs by viewing
DRP as an edge regression problem on a heterogeneous graph in
which each node denotes either a drug or a cell line. SA utilizes
GNNs to smooth the representations of similar cell lines/drugs. In
addition, we introduce an auxiliary pre-training strategy for
TGDRP’s drug encoder to alleviate the issues of insufficient data and
poor generalization on out-of-distribution molecules. Our main contributions are as follows.


- Better use of prior domain knowledge. We propose TGDRP to
incorporate relationships among genes, which are paramount for
capturing complex patterns of cell lines. We also propose SA to
incorporate similarity among drugs and cell lines. To the best of
our knowledge, we are the first to utilize GNNs in DRP for cell
line representation learning.

- Information fusion at different levels of granularity. TGDRP and
SA are coupled into TGSA to fuse fine-grained and coarsegrained information.

- A strategy for the dilemma of limited data. We adapt an auxiliary
pre-training strategy to further improve the generalization, especially on out-of-distribution drugs.

- Promising performance. Our comprehensive experiments show
that TGSA outperforms all the baselines on the GDSC2 dataset
under various experimental settings empirically, demonstrating
the superior predictive ability of TGSA.


2 Materials and methods


2.1 Problem definition
In this work, we formulate DRP as a regression problem for the lognormalized half maximal inhibitory concentration (lnðIC50Þ) values
of corresponding drug–cell line pairs with a mapping function
f : D �C ! Y, where D ¼ fd 1 ; d 2 ; . . . ; d n g and C ¼ fc 1 ; c 2 ; . . . ; c m g
are the sets of drugs and cell lines, respectively, and Y 2 R [m][�][n] is the
drug response matrix, in which each entry Y ij is the lnðIC50Þ value
of c i and d j .


2.2 Data preparation
In this section, we describe how to integrate three public datasets,
GDSC2, CCLE and COSMIC (Tate et al., 2019), into our experimental dataset. Moreover, we show how to represent both drugs
and cell lines as graphs.
Data integration. GDSC2 provides lnðIC50Þ values across hundreds of cell lines and drugs, and CCLE provides elaborate genetic
profiles for cell lines, among which we take gene expression (EXP),
somatic mutation (MU) and copy number variation (CNV) into account. Besides, 706 cancer-related genes are selected according to


TGSA 463


Fig. 1. An overview of our proposed TGSA framework. (a) TGDRP uses two GNNs to learn the drug and cell line representations, and then applies an FCN to predict
lnðIC50Þ. (b) TGSA uses TGDRP’s GNN d and GNN c to generate the initialized features of the nodes in a heterogeneous graph. In the heterogeneous graph, the blue and orange
nodes denote drugs and cell lines, respectively (the colors of the nodes changed from light to dark indicate an evolution of the node representations), and the solid and dashed
edges denote similarity and drug-cell line responses, respectively



COSMIC (Tate et al., 2019) as feature selection. Note that those
drugs without PubChem ID and the cell lines missing any types of
aforementioned omics data are removed. Consequently, we obtain
82833 available lnðIC50Þ across 580 cell lines and 170 drugs (see
[Supplementary Tables S1–S3), while approximately 16% (15767) of](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data)
the lnðIC50Þ are unknown. A profile of the basic statistics of GDSC2
[is shown in Supplementary Figure S1.](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data)
Graph construction. Generally, a graph G ¼ ðV; EÞ can be
denoted as (A, F), where V is a set of N nodes, E is a set of edges,
A 2 f0; 1g [N][�][N] denotes the adjacency matrix, and F 2 R [N][�][K] denotes
matrix of node feature vectors. In our work, each molecular graph
G [d] is constructed following (Liu et al., 2020), and the details of all
the atom features are presented in Table 1. For each cell line graph
G [c] ¼ ðV [c] ; E [c] Þ; V [c] and E [c] denote the sets of genes and interactions
among them, respectively. F [c] 2 R [706][�][3] are determined by the aforementioned gene-wise multi-omics data. Unlike chemical bonds in
molecules, there are no explicit relational structures among genes in
cell lines. Thus, we introduce a gene interaction network as the prior
knowledge to define A [c] 2 R [706][�][706] . STRING is a curated dataset
containing interactions among genes (Szklarczyk et al., 2019).
Formally, A [c] ij [¼][ 1 if the combined interaction score between gene][ i]


Table 1. Atom features


Feature Size Description


Atom type 43 C, N, O, S, F, etc. (one-hot)
Degree 11 0–10 (one-hot)
Implicit valence 7 0–6 (one-hot)
Formal charge 1 Formal charge number (integer)
Radical electrons 1 Number of radical electrons (integer)
Hybridization 5 SP, SP2, SP3, SP3D, SP3D2 (one-hot or null)
Aromatic 1 Whether the atom is in an aromatic system
(binary)
Hydrogens 5 0–4 (one-hot)
Ring 1 Whether the atom is in ring (binary)
Chirality 2 R, S (one-hot or null)



and gene j in STRING is larger than a predefined threshold s, set to
0.95 by default, and otherwise A [c] ij [¼][ 0. Statistically,][ G] [c] [ is a discon-]
nected graph with the largest connected component of size 408,
sparsity of 0.006 and average degree of 4.46.


2.3 Our proposed method
Without loss of generality, we first briefly review Message Passing
Neural Networks (MPNNs) (Gilmer et al., 2017) as a typical framework, to introduce the basic concepts of GNNs. Next, we present
TGDRP and SA (Fig. 1). Finally, we discuss how to combine
TGDRP and SA into TGSA.

MPNNs. The main idea of MPNNs is that each node can recursively receive messages from its neighbor nodes in the message passing stage. In the readout stage, MPNNs utilize a permutationinvariant readout function to summarize all node representations
into fixed-length graph-level representations, which can be used for
various downstream tasks. Formally, given a graph G, its node representations h [l] v [2][ R] [n] [l] [ for node][ v][ and graph-level representations][ z] [l] [ in]
layer l can be computed as follows:



h [l] v [¼][ U] [l][�][1] [ð][h] [l] v [�][1] ; m [l] v [Þ][;] (2)


z [l] ¼ Rðfh [l] v [j][ v][ 2][ G][gÞ][;] (3)


where m [l] v [denotes the summed message of node][ v’][s neighbors in]
layer l—1, N(v) denotes the set of neighbors of node v, e vw denotes
the features of the edge between nodes v and w, and M, U and R are
message function, update function and readout function,
respectively.
TGDRP. In order to explore and exploit the topology of cell line
graphs and better learn cell line representations, we propose the
GNN-based TGDRP. As illustrated in Figure 1, TGDRP is a twobranch network that takes a molecular graph G [d] and a cell line



m [l] v [¼] X M l�1 ðh [l] v [�][1] ; h [l] w [�][1] [;][ e] [vw] [Þ][;] (1)

w2NðvÞ


464 Y.Zhu et al.



graph G [c] as input and outputs the predicted lnðIC50Þ. TGDRP comprises of the following three components.
Drug branch. Given a molecular graph G [d], we use the Graph
Isomorphism Network (GIN) (Xu et al., 2019) as GNN d to update
atom features. GINs yielded state-of-the-art performance on a host
of molecule-related tasks (Hu et al., 2020). Inspired by Jumping
Knowledge Network to preserve information at different scales (Xu
et al., 2018), we concatenate graph-level representations across
layers which are read out by global max pooling, followed by FCN d
to learn the drug representations z d 2 R [256] .
Cell line branch. Given a cell line graph G [c], we use the Graph
Attention Network (GAT) (Velickovic et al., 2018) as GNN c to update gene features. During the message passing phase, GATs assign
different weights to different neighbor nodes with a self-attention
mechanism. Cell line graphs contain hierarchical information on
genes’ interactions (Erwin and Davidson, 2009). But, vanilla GAT
and global pooling are inherently flat and unable to capture such information (Ying et al., 2018). Hence, we use Graclus (Dhillon et al.,
2007) to coarsen the graph gradually by clustering two nodes into
one ‘super node’ after every layer of GAT. Based on spectral clustering and kernel k-means, Graclus is an efficient and model-free graph
clustering algorithm (Bianchi et al., 2020). A previous study demonstrated the effectiveness of Graclus in grouping the genes strongly
connected (Dhillon et al., 2007). With this operation, not only the
hierarchical structure of cell line graphs is captured, but also the
advantages of pooling operations in CNNs are preserved. There are
two beneficial characteristics of cell line graphs. (i) All the cell line
graphs share the same topology. In practice, we only need to run
Graclus once before training, to avoid duplicated computation. (ii)
Cell line graphs are of larger scale compared to molecular graphs.
We straightly concatenate all the super node representations to read
out the graph-level representations, rather than global pooling which
preserves only first-order statistics and loses much useful information. With the graph-level representations, FCN c subsequently outputs the cell line representations z c 2 R [256] .
Prediction FCN. The drug and cell line representations are concatenated, and fed to the prediction FCN, to produce the final predicted lnðIC50Þ.
Pre-training strategy. Pre-training strategies have been studied
extensively in computer vision (He et al., 2019) and natural language processing (Kenton et al., 2019), but their effect in DRP has
not been well explored. Training effective GNNs to capture the
characteristics of drugs can be a great challenge due to the following
two issues. (i) limited data: there are only 170 drugs in the whole
dataset. (ii) Out-of-distribution samples: drug graphs in the test set
are structurally quite different from those in the training set, leading
to poor out-of-distribution generalization (Hu et al., 2020). To alleviate these issues, we pre-train GNN d in large-scale molecule datasets and transfer the domain knowledge of chemistry to our task.
In this study, we follow the pre-training strategy in (Hu et al.,
2020), whose key idea is to capture the domain-specific semantics at
both the node and graph levels. Specifically, we first conduct nodelevel self-supervised pre-training on the ZINC15 dataset (Sterling
and Irwin, 2015) with Deep Graph Infomax (DGI) (Velickovic et al.,
2019). Then, we conduct graph-level multi-task supervised pretraining on the ChEMBL dataset (Gaulton et al., 2012), which contains 456k molecules with 1310 biochemical properties. During pretraining, GNN d accumulates chemical knowledge of molecules (e.g.
valency and biochemical properties). Unlike (Hu et al., 2020), we
randomly hold out a validation set for model selection.
Similarity augmentation. As in the discussion of the second limitation of the known deep learning-based DRP methods in Section 1,
similar cell lines/drugs tend to have similar representations, eliciting
similar responses. Prior knowledge of cell line/drug similarity has
been commonly considered as regularization terms to improve the
DRP performance in matrix factorization methods (Guan et al.,
2019; Wang et al., 2017). However, such regularization terms show
poor extrapolation, and only attempt to reduce the difference of representations of similar cell lines/drugs, but ignore the extra relationship among similar cell lines/drugs. GNNs as low-pass filters can
make the adjacent nodes in a graph tend to share similar



representations (Wu et al., 2019) and pass message between adjacent
nodes (Gilmer et al., 2017). Therefore, we take advantage of the
similarity among cell lines/drugs explicitly and sidestep above issues
by resorting to GNNs. Specifically, we formulate DRP as an edge regression problem on a heterogeneous graph in which each node
denotes either a drug or a cell line, and utilize GNNs to smooth the
cell line/drug node representations. This is our Similarity
Augmentation (SA) module.
The aforementioned heterogeneous graph is constructed by connecting two homogeneous k-nearest-neighbor graph of drugs and
cell lines via known drug-cell line response. Formally, we define the
heterogeneous graph as G ¼ ðV; EÞ with V ¼ D [ C and
E ¼ E IC [ E d [ E c . E d and E c are formed so that each drug node or
cell line node is connected to its k (set to 5 by default) most similar
neighbors according to the Jaccard similarity of Extended
Connectivity Fingerprints (ECFP) (Rogers and Hahn, 2010) and
Pearson correlation coefficient of gene expression, respectively.
While E IC denotes edges between drug nodes and cell line nodes with
labels of corresponding lnðIC50Þ.
For each cell line/drug, type-specific GraphSAGE (Hamilton
et al., 2017) is applied to the heterogeneous graph G to update its
representations with the message from its neighbors. In this way, informative similarity-aware node representations are obtained, and
the obtained representations of adjacent nodes (similar cell lines/
drugs) tend to be similar. The computing process of SA can be formulated as follows:


Z [c] ¼ GraphSAGE c ðX [c] ; E c Þ (4)


Z [d] ¼ GraphSAGE d ðX [d] ; E d Þ (5)


where GraphSAGE c ð�Þ and GraphSAGE d ð�Þ are type-specific
GraphSAGE networks (the detailed formulas are shown in
[Supplementary Material) for cell lines and drugs, respectively. X](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data) [c] 2
R [m][�][dim] [c] and X [d] 2 R [n][�][dim] [d] are the initial feature matrices of cell lines
and drugs, with each row corresponding to a cell line/drug. Z [c] 2
R [m][�][256] and Z [d] 2 R [n][�][256] are the output representation matrices of
cell lines and drugs. After obtaining the cell line representations Z [c] i
and the drug representations Z [d] j [, the prediction step is the same as]
that of TGDRP.
TGSA. Because the nodes in TGDRP and in SA denote objects at
different levels of granularity (i.e. genes versus cell lines and atoms
versus drugs), TGDRP captures gene-level and atom-level information, while SA captures sample-level information. To fuse such finegrained and coarse-grained information, we seek to couple TGDRP
and SA via fine-tuning.
First, we train TGDRP end-to-end. Then in SA, X [d] and X [c] are
generated by applying the corresponding trained GNN d and GNN c
to the sets of molecular graphs and cell line graphs. To avoid additional parameters, the hidden dimensions and number of hidden
layers of the type-specific GraphSAGEs and prediction FCN are kept
the same as the corresponding FCNs in TGDRP. Finally, for finetuning, the parameters of GraphSAGEs and the prediction FCN are
initialized by the corresponding trained FCNs in TGDRP. Through
these processes, TGDRP can be viewed as a special case of TGSA
with E d [ E c ¼ 1.


2.4 Experimental setup
To comprehensively demonstrate the effectiveness of our method,
we aim to examine the following three questions:


- Q1: Could TGSA outperform state-of-the-art baselines under diverse experimental settings?

- Q2: Does each component of TGSA work well?

- Q3: Could TGSA capture any biological meaning?


As suggested in (Baptista et al., 2021), we evaluate our method
under two experimental settings. (i) For rediscovering known drug–
cell line responses: The whole dataset is split into training/validation/test sets with a ratio of 8:1:1 by stratified sampling, and each


TGSA 465



experiment is repeated with 10 random seeds. (ii) For blind test
(leave-drug/cell-line-out): The whole dataset is split on the cell line/
drug level to guarantee that the test set only includes unseen cell
lines/drugs in the training stage, and 5-fold cross validation is conducted, where three folds are regarded as training set, and the other
two folds are regarded as validation set and test set. This more rigorous scenario is in line with the clinical applications of drug repositioning and recommendation. Given that the data distribution of the
test set may be quite different from that of the training set, cross validation can estimate the true generalization errors more accurately.
Three representative state-of-the-art methods: CDRscan (Chang
et al., 2018), MOLI (Sharifi-Noghabi et al., 2019) and tCNNS (Liu
et al., 2019) and two recent GNN-based methods: GraphDRP
(Nguyen et al., 2021) and DeepCDR (Liu et al., 2020) are considered as our competing baselines. To explore the effectiveness of the
cell line graph and GNN c, we make some minor adjustments to
DeepCDR for fair comparison. Specifically, UGCN of DeepCDR is
replaced by our drug branch, and the multi-omics data of DeepCDR
is replaced by ours as well. We denote this variant as DeepCDR [�] .
Three widely used metrics for regression tasks are adopted to measure the performance: Root Mean Square Error (RMSE), Mean
Absolute Error (MAE) and Pearson correlation coefficient (r).
To train our model, we use the mean square error as the loss
function. All the baselines and our model are implemented in
PyTorch (Paszke et al., 2019) and PyTorch Geometric (Fey and
Lenssen, 2019). The implementation details are shown in
[Supplementary Material. The hyper-parameters are tuned by grid](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data)
[search on the validation set (see Supplementary Table S4). We report](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data)
the mean and standard deviation values, and conduct Student’s t-test
to analyze whether the performance difference is statistically significant between our method and the baselines.


3 Experimental results


3.1 Comparison with baselines
Table 2 compares the architectures and performance of our method
with the state-of-the-art baselines. Our method significantly outperforms all the baselines on all the three metrics, providing positive
answers to Q1. We discuss the results for diverse experiment settings
as follows.
Rediscovering known drug-cell line responses. tCNNS and
GraphDRP attain comparable performances, while GraphDRP is
less robust with higher standard deviation values. This phenomenon
is inconsistent with the observation that GNNs tend to outperform
traditional methods on many molecule-related tasks (Sun et al.,
2020). We argue that it is difficult for GNNs to learn expressive and
robust representations with such limited drugs. DeepCDR [�] surpasses
other baselines significantly. We think that DRP benefits from multiomics data, which is a main difference between DeepCDR [�] and
GraphDRP. Thus, we suggest using as much related omics data as
possible to represent cell lines, since such data can stably improve
performance without increasing too much computation costs.



The performance improvement of TGDRP w=o pre over
DeepCDR [�] can be simply attributed to obtaining better cell line representations, which demonstrates that cell line graphs provide richer
information than the raw multi-omics profiles and GNNs are capable of learning sophisticated patterns from cell line graphs.
Compared with DeepCDR [�], TGSA attains a decrease of 0.037 in
RMSE and 0.028 in MAE, which is about 4.1% relative improvement. Figure 2 shows the prediction results of TGSA. Specifically,
Figure 2c illustrates the relationship between the observed lnðIC50Þ
and predicted lnðIC50Þ, which is highly linearly correlated.
Figure 2a and b visualizes the distributions of the test RMSE on the
cell line/drug level, which are approximate normal distributions.
Compared with cell lines, the generalization on several drugs is particularly poor (RMSE > 3). This phenomenon might be due to the
out-of-distribution test drugs in the enormous chemical space.
Blind test. As presented in Figure 2e and f, even though all the
methods do not perform well in the blind test scenarios compared to
the above scenario, especially in the leave-drug-out scenario, our
method still outperforms the best baseline DeepCDR [�] . Note that we
do not compare the performances of all the variants, because pretraining and SA only explicitly affect drug and cell line representations, respectively (as discussed in Section 3.2).
In leave-drug-out, the r of TGDRP w=o pre is higher than
DeepCDR [�] by 0.033 (0.493–0.46), and TGDRP achieves the best r
of 0.527. Decreased performance indicates that our method still
faces difficulties in generalizing to structurally different drugs. The
pre-training strategy alleviates this defect by utilizing large-scaled
molecules; nevertheless, the improvement is still unsatisfactory. We
argue that this is because anti-cancer drugs are a very special kind of
molecule and general domain knowledge seems insufficient. In
leave-cell-line-out, TGDRP and TGSA are slightly better than
DeepCDR (0.874–0.872).
Our experiments demonstrate that TGSA shows potential in the
leave-cell-line-out scenario compared to leave-drug-out, indicating
that the TGSA predictions could be taken as a reference for drug
recommendation.


3.2 Ablation experiments
To answer Q2, we conduct ablation experiments from three aspects:
gene features, the topology of cell line graphs and framework components. Note that we also pre-train DeepCDR [�] (DeepCDR [�] pre [) for a]
fair comparison.
Gene features. For three types of omics data, namely, gene expression (EXP), somatic mutation (MU) and copy number variation
(CNV), are used as gene features, it is of great necessity to evaluate
the contributions of different types of omics data individually.
Specifically, for each single-omics data, we exclude other types of
omics data from the cell line graphs (F [c] 2 R [706][�][1] ). As shown in the
upper half of Table 3 (the used single-omics data are denoted with
subscript), TGDRP CNV achieves the best performance among three
single-omics variants, and TGDRP can even outperform



Table 2. Comparison of the model architectures and performances in the scenario of rediscovering known drug-cell line responses


Method Input feature format Feature encoder RMSE MAE r


Cell line Drug Cell line Drug


MOLI Sequence – FCN – 1.130 6 0.027 0.875 6 0.023 0.931 6 0.005
CDRscan Sequence Fingerprint CNN CNN 0.993 6 0.013 0.737 6 0.011 0.937 6 0.002
tCNNS Sequence SMILES CNN CNN 0.951 6 0.009 0.700 6 0.008 0.942 6 0.001
GraphDRP Sequence Molecular graph CNN GNN 0.953 6 0.020 0.702 6 0.017 0.942 6 0.003
DeepCDR [�] Sequence Molecular graph FCN GNN 0.914 6 0.019 0.674 6 0.016 0.946 6 0.002
TGDRP w=o pre Graph Molecular graph GNN GNN 0.894 6 0.013 0.657 6 0.010 0.949 6 0.002
TGDRP Graph Molecular graph GNN GNN 0.885 6 0.009 0.650 6 0.006 0.950 6 0.001
TGSA Graph Molecular graph GNN GNN 0.877 6 0.008 0.646 6 0.006 0.951 6 0.001


a We report the standard deviation values after ‘6’. The best performance is highlighted in bold.
b The comparison of the model architectures is based on the perspectives of input feature format and feature encoder.


466 Y.Zhu et al.


Fig. 2. Performance and comparison results. (a, b) The histograms illustrate the distribution of RMSE for each cell line/drug in the test set. (c) The scatter plot presents the
observed lnðIC50Þ and predicted lnðIC50Þ in the test set. (d) The box plot presents the distribution of the predicted lnðIC50Þ of the missing drug-cell line pairs grouped by drugs.
(e, f) The violin plots show performances in the leave-drug/cell-line-out scenarios. (g, h) The t-SNE plots present 2D representations of cell lines colored by corresponding tissue
labels without/with SA



Table 3. Ablation studies


Method RMSE MAE r


TGDRP EXP 0.891 6 0.009 0.655 6 0.008 0.949 6 0.001

TGDRP MU 0.900 6 0.018 0.661 6 0.015 0.948 6 0.002

TGDRP CNV 0.889 6 0.008 0.653 6 0.007 0.950 6 0.001
DeepCDR [�] pre 0.904 6 0.010 0.665 6 0.008 0.948 6 0.001
TGDRP Erdo€s�Re�nyi 0.905 6 0.008 0.665 6 0.008 0.948 6 0.001
TGDRP permutation 0.900 6 0.013 0.662 6 0.012 0.948 6 0.002
TGDRP pearson 0.891 6 0.015 0.656 6 0.014 0.949 6 0.002
TGDRP 0.885 6 0.009 0.650 6 0.006 0.950 6 0.001


a We report the standard deviation values after ‘6’. The best performance is
highlighted in bold.


DeepCDR [�] pre [with any single-omics data, indicating the effects of all]
single-omics data and the powerful capability of TGDRP.
Topology. To assess the effects of the STRING-based topology
of cell line graphs, we evaluate the performance of TGDRP with diverse topological structures: random counterpart generated by
Erdo¨s-Re´nyi model (Erdos et al., 1960), permutational counterpart
generated by permuting A [c], and statistic counterpart generated by
replacing combined interaction score in STRING with Pearson correlation coefficient of gene expression in the process of graph construction. Note that the average degree of each counterpart is set to
be the same as that of STRING-based topology for a fair comparison. As shown in the lower half of Table 3 (the used topology is
denoted with subscript), TGDRP Erdo€s�Re�nyi (TGDRP with meaningless topology) achieves comparable performance with DeepCDR [�] pre [,]
and the STRING-based topology outperforms all counterparts, demonstrating that the improvement over DeepCDR [�] pre [comes from the]
beneficial prior knowledge contained in STRING.
Framework component. Pre-training strategies and SA have rarely been considered in previous DRP methods. To verify their effectiveness in our method, we conduct ablation experiments to compare
two variants of TGSA: TGDRP w=o pre (i.e. TGDRP without using the
pre-training strategy) and TGDRP. Their performances are shown in
the lower half of Table 2.
Considering our SA module, we find that the most decrease in
RMSE (0.008) is attained only with cell line similarity augmentation, which indicates that drug similarity augmentation seems useless. This phenomenon is aligning with some recent research (Wang
et al., 2017), while contrasted with Zhang et al. (2015). We doubt



that the drug response similarity is too complicated to be measured
by Jaccard similarity of ECFP. Figure 2g and h visualize the t-SNE
(Van der Maaten and Hinton, 2008) 2D representations of the cell
lines from the top 5 most frequent tissue categories (without and
with SA), which confirm the effect of SA since SA makes the cell
lines from different tissues easier to distinguish. Likewise, visualization results colored by TCGA cancer type are shown in
[Supplementary Figure S2.](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data)
On our pre-training strategy, TGDRP’s RMSE decreases by
0.009. We point out an unexpected finding that considering only
node-level pre-training can yield better performance than combining
with graph-level pre-training. We speculate that such negative transfer may be caused by inconsistency between drug responses and the
numerous and fuzzy biochemical assays in ChEMBL.


3.3 Biological meanings
A successful DRP method ought to not only make accurate predictions, but also capture important biological meanings. Hence, we
further utilize a post hoc exploratory strategy and literature-based
case studies to exhibit the capability of our method in discovering
unknown drug-cell line responses and finding drug-gene
interactions.
Discovering unknown drug-cell line responses. Among all the
predicted lnðIC50Þ of the missing drug-cell line pairs (see
[Supplementary Table S5), the (Daporinad, EU-3) pair has the lowest](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btab650#supplementary-data)
value (-8.05) where EU-3 is a B-cell precursor leukemia cell line,
indicating that Daporinad has excellent biological activities against
hematologic malignant cells. This prediction is supported by
Nahimana et al. (2009), which reported that most hematologic cancer cells are sensitive to low concentrations of Daporinad, on account of NAD depletion resulted from that Daporinad treatment
elicits tumor cell death. Figure 2d illustrates the distributions of the
predicted lnðIC50Þ values grouped by drugs which are ordered by
the median predicted lnðIC50Þ, and Bortezomib has the lowest median. This observation is aligned with the experimental results that
Bortezomib has therapeutic effects on multiple cancers (Tan et al.,
2019) such as relapsed multiple myeloma (Richardson et al., 2003)
and breast cancer (Periyasamy-Thandavan et al., 2010).
Drug–gene interaction. Given a drug-cell line pair, we modify
Grad-CAM (Selvaraju et al., 2017) to estimate the node-level gene
importance since genes are represented as nodes in our method.
Specifically, ReLU function is replaced with absolute function to be
suited for our regression problem which is not only interest in the
features that have a positive influence on lnðIC50Þ, as the vanilla


TGSA 467



Grad-CAM was proposed for classification problem. Accordingly,
the formula of Grad-CAM can be rewritten as:



N
h [l] k;n X
k n¼1



learn more sophisticated cell line representations. Finally, the performances of the blind test scenarios are still not satisfactory, especially the leave-drug-out scenario. Further efforts are needed to
improve the out-of-distribution generalization of GNNs, and investigating pharmacophore-aware self-supervised learning appears promising for it.
To sum up, our TGSA integrates chemical domain knowledge
into deep learning, and the promising performance of TGSA shows
enormous potential for clinical applications in precision medicine.


Acknowledgements


The authors thank Minshan Lai and Yufeng Xie for their helpful discussions.


Funding


This work was supported by the National Research and Development
Program of China [2019YFC0118802]; and the Key R & D Program of
Zhejiang Province [2020C03010]; and the National Natural Science
Foundation of China [61672453]; and the Zhejiang University Education
Foundation under grants [K18-511120-004, K17-511120-017 and K17518051-02]; and the Zhejiang public welfare technology research project

[LGF20F020013]; and the Medical and Health Research Project of Zhejiang
Province of China [2019KY667]; and the Wenzhou Bureau of Science and
Technology of China [Y2020082]; and the Key Laboratory of Medical
Neurobiology of Zhejiang Province. D.Z.C.’s research was supported in part
by the National Science Foundation [CCF-1617735].


Conflict of Interest: none declared.


References


Ammad-Ud Din,M. et al. (2016) Drug response prediction by inferring pathway–response associations with kernelized Bayesian matrix factorization.
Bioinformatics, 32, i455–i463.
Baptista,D. et al. (2021) Deep learning for drug response prediction in cancer.
Brief. Bioinf., 22, 360–379.
Barretina,J. et al. (2012) The cancer cell line encyclopedia enables predictive
modelling of anticancer drug sensitivity. Nature, 483, 603–607.
Bianchi,F.M. et al. (2020) Spectral clustering with graph neural networks for
graph pooling. In: ICML. PMLR, Vienna, Austria, pp. 874–883.
Chang,Y. et al. (2018) Cancer drug response profile scan (CDRscan): a deep
learning model that predicts drug effectiveness from cancer genomic signature. Sci. Rep., 8, 8857–8811.
Chiu,Y.-C. et al. (2019) Predicting drug response of tumors from integrated
genomic profiles by deep neural networks. BMC Med. Genomics, 12,

119–155.

Corte´s-Ciriano,I. and Bender,A. (2019) KekuleScope: prediction of cancer cell
line sensitivity and compound potency using convolutional neural networks
trained on compound images. J. Cheminf., 11, 41–16.
Dhillon,I.S. et al. (2007) Weighted graph cuts without eigenvectors a multilevel approach. IEEE Trans. Pattern Anal. Mach. Intell., 29, 1944–1957.
Ding,M.Q. et al. (2018) Precision oncology beyond targeted therapy: combining omics data with machine learning matches the majority of cancer cells to
effective therapeutics. Mol. Cancer Res., 16, 269–278.
Dong,Z. et al. (2015) Anticancer drug sensitivity prediction in cell lines from
baseline gene expression through recursive feature selection. BMC Cancer,
15, 489–412.
Erdos,P. et al. (1960) On the evolution of random graphs. Publ. Math. Inst.
Hung. Acad. Sci., 5, 17–60.
Erwin,D.H. and Davidson,E.H. (2009) The evolution of hierarchical gene
regulatory networks. Nat. Rev. Genet., 10, 141–148.
Fey,M. and Lenssen,J.E. (2019) Fast graph representation learning with
PyTorch geometric. In: ICLR Workshop, New Orleans, Louisiana, United

States.

Friedman,A.A. et al. (2015) Precision medicine for cancer with
next-generation functional diagnostics. Nat. Rev. Cancer, 15, 747–756.
Gaulton,A. et al. (2012) ChEMBL: a large-scale bioactivity database for drug
discovery. Nucleic Acids Res., 40, D1100–D1107.
Gilmer,J. et al. (2017) Neural message passing for quantum chemistry. In:
ICML. PMLR, Sydney, Australia, pp. 1263–1272.



L½l; n�¼ j X



n¼1



@y
j (6)
@h [l] k;n



where j �j calculates the absolute value element-wise, y is the predicted lnðIC50Þ; h [l] k;n [is the][ k] [0] [th feature of the graph convolutional]
activations for node n in layer l, and L½l; n� represents the importance
score of node n in the corresponding graph.
According to the computed gene importance scores, the top 5 important genes of several cases are shown in Table 4. Remarkably,
we notice that many top-ranked genes thus determined have been
verified to be associated with the mechanism of action of the corresponding drugs. For instance, PAFAH1B2 is a target gene of HIF �
1a (Ma et al., 2018), the latter of which has been reported to be
inhibited by Bortezomib thus inhibiting tumor adaptation to hypoxia (Shin et al., 2008). PAFAH1B2 ranks first with respect to the
(Bortezomib, 8MGBA) pair. Moreover, Irinotecan is a derivative of
Camptothecin targeting DNA replication pathway (Pommier, 2006).
And in their sensitivity to KMH2 cell line, the target gene TOP1 and
a DNA replication-related gene POLQ (DNA polymerase h) rank
fourth and third, respectively. Also, BRD4 has been reported to be a
potential target of AZD5153 by regulating transcriptional programs
(Rhyasen et al., 2016) and was estimated by our method to be important in AZD5153 sensitivity to NCIH526 cell line.
Conclusively, these literature-based case studies exemplify that
our TGSA could be applied to discover new therapeutic effects as
well as potential therapeutic targets of anti-cancer drugs.


4 Discussion


In this article, we proposed a novel DRP framework, called TGSA,
which incorporates prior domain knowledge to fuse fine-grained
and coarse-grained information. To explore relationships among
genes, we incorporated STRING protein–protein association networks to construct cell line graphs. To capture similarity among cell
lines/drugs, we treated DRP as an edge regression problem on a heterogeneous graph, and utilized GNNs to smooth the representations
of similar cell lines/drugs. Besides, we introduced an auxiliary pretraining strategy to alleviate the dilemma of limited data and improve the out-of-distribution generalization for drugs. Extensive
experiments on the GDSC2 dataset demonstrated that our TGSA
outperforms several state-of-the-art DRP methods under different
experimental settings. Ablation study verified the effectiveness and
contributions of each component of TGSA. Moreover, biological
case analysis demonstrated that TGSA can, to some extent, enlighten
experts to further explore related domain knowledge.
Despite the above promising performances, our method still
needs to address several limitations and also reveals insight for future research. First, the topology of cell line graphs is constructed
simply based on the combined interaction score of STRING. To determine graph topology dedicatedly, taking more abundant domain
knowledge into account is a fruitful avenue. Second, Graclus only
considers the graph topology while ignoring the node features
(Bianchi et al., 2020), and thus cell line graphs cannot be adaptively
coarsened based on a specific DRP task. It is worth considering to
apply feature-based hierarchical pooling algorithms to help GNNs


Table 4. Top 5 important genes with respect to drug-cell line pairs


Drug Cell line Top 5 important genes


Bortezomib 8MGBA PAFAH1B2, DDX10, FAT3, FOXR1, KCNJ5
Camptothecin KMH2 NUTM2B, FKBP9, POLQ, TOP1, CCX6C
Irinotecan KMH2 FKBP9, NUTM2B, POLQ, TOP1, CCX6C
AZD5153 NCIH526 ELK4, HERPUD1, KLF6, RGS7, BRD4


Note: Gene importance is estimated based on the modified Grad-CAM

method.


468 Y.Zhu et al.



Guan,N.-N. et al. (2019) Anticancer drug response prediction in cell lines
using weighted graph regularized matrix factorization. Mol. Therapy
Nucleic Acids, 17, 164–174.
Hamilton,W.L. et al. (2017) nductive representation learning on large graphs.
In: NeurIPS, Long Beach, CA, United States, pp. 1024–1034.
He,K. et al. (2019) Rethinking ImageNet pre-training. In: ICCV, Seoul, Korea.
Hodson,R. (2016) Precision medicine. Nature, 537, S49.
Hu,W. et al. (2020) Strategies for pre-training graph neural networks. In:
ICLR, Addis Ababa, Ethiopia.
Iorio,F. et al. (2016) A landscape of pharmacogenomic interactions in cancer.
Cell, 166, 740–754.
Kenton,J. et al. (2019) BERT: pre-training of deep bidirectional transformers
for language understanding. In: NAACL-HLT, Minneapolis, Minnesota,

4171–4186.

Lecun,Y. et al. (1998) Gradient-based learning applied to document recognition. Proc. IEEE, 86, 2278–2324.
Liu,P. et al. (2019) Improving prediction of phenotypic drug response on cancer cell lines using deep convolutional network. BMC Bioinformatics, 20,

408–414.

Liu,Q. et al. (2020) DeepCDR: a hybrid graph convolutional network for predicting cancer drug response. Bioinformatics, 36, i911–i918.
Ma,C. et al. (2018) PAFAH1B2 is a HIF1a target gene and promotes metastasis in pancreatic cancer. Biochem. Biophys. Res. Commun., 501, 654–660.
Mayr,A. et al. (2016) DeepTox: toxicity prediction using deep learning. Front.
Environ. Sci., 3, 80.
Nahimana,A. et al. (2009) The NAD biosynthesis inhibitor APO866 has potent antitumor activity against hematologic malignancies. Blood, 113,

3276–3286.

Nguyen,T.-T. et al. (2021) Graph convolutional networks for drug response
prediction. IEEE/ACM Trans. Comput. Biol. Bioinf.
Paszke,A. et al. (2019) PyTorch: an imperative style, high-performance deep
learning library. In: NeurIPS, Vancouver, Canada, pp. 8024–8035.
Periyasamy-Thandavan,S. et al. (2010) Bortezomib blocks the catabolic process of autophagy via a cathepsin-dependent mechanism, affects endoplasmic reticulum stress, and induces caspase-dependent cell death in
antiestrogen–sensitive and resistant ERþ breast cancer cells. Autophagy, 6,

19–35.

Pommier,Y. (2006) Topoisomerase i inhibitors: camptothecins and beyond.
Nat. Rev. Cancer, 6, 789–802.
Rhyasen,G.W. et al. (2016) Azd5153: a novel bivalent bet bromodomain inhibitor highly active against hematologic malignancies. Mol. Cancer
Therap., 15, 2563–2574.
Richardson,P.G. et al. (2003) A phase 2 study of bortezomib in relapsed, refractory myeloma. N. Engl. J. Med., 348, 2609–2617.
Rogers,D. and Hahn,M. (2010) Extended-connectivity fingerprints. J. Chem.
Inf. Model., 50, 742–754.



Selvaraju,R.R. et al. (2017) Grad-cam: visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE
International Conference on Computer Vision, Venice, Italy, pp. 618–626.
Sharifi-Noghabi,H. et al. (2019) MOLI: multi-omics late integration with deep
neural networks for drug response prediction. Bioinformatics, 35,
i501–i509.

Shin,D.H. et al. (2008) Bortezomib inhibits tumor adaptation to hypoxia by
stimulating the FIH-mediated repression of hypoxia-inducible factor-1.
Blood. J. Am. Soc. Hematol., 111, 3131–3136.
Sterling,T. and Irwin,J.J. (2015) ZINC 15–ligand discovery for everyone. J.
Chem. Inf. Model., 55, 2324–2337.
Sun,M. et al. (2020) Graph convolutional networks for computational drug
development and discovery. Brief. Bioinf., 21, 919–935.
Suphavilai,C. et al. (2018) Predicting cancer drug response using a recommender system. Bioinformatics, 34, 3907–3914.
Szklarczyk,D. et al. (2019) STRING v11: protein–protein association networks with increased coverage, supporting functional discovery in
genome-wide experimental datasets. Nucleic Acids Res., 47, D607–D613.
Tan,C.R.C. et al. (2019) Clinical pharmacokinetics and pharmacodynamics of
bortezomib. Clin. Pharmacokinet., 58, 157–168.
Tate,J.G. et al. (2019) COSMIC: the catalogue of somatic mutations in cancer.
Nucleic Acids Res., 47, D941–D947.
Teschendorff,A.E. (2019) Avoiding common pitfalls in machine learning omic
data science. Nat. Mater., 18, 422–427.
Van der Maaten,L. and Hinton,G. (2008) Visualizing data using t-SNE. J.
Mach. Learn. Res., 9, 2579–2605.
Velickovic,P. et al. (2018) Graph attention networks. In: ICLR, Vancouver,
Canada.

Velickovic,P. et al. (2019) Deep graph infomax. In: ICLR, New Orleans,
Louisiana, United States.
Wang,L. et al. (2017) Improved anticancer drug response prediction in cell
lines using matrix factorization with similarity regularization. BMC Cancer,
17, 513–512.
Wu,F. et al. (2019) Simplifying graph convolutional networks. In ICML.
PMLR, Long Beach, California, pages 6861–6871.
Xu,K. et al. (2018) Representation learning on graphs with jumping knowledge networks. In: ICML. PMLR, Stockholm, Sweden, pp. 5453–5462.
Xu,K. et al. (2019) How powerful are graph neural networks? In: ICLR, New
Orleans, Louisiana, United States.
Yang,W. et al. (2013) Genomics of drug sensitivity in cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids
Res., 41, D955–D961.
Ying,Z. et al. (2018) Hierarchical graph representation learning with differentiable pooling. In: NeurIPS, Montre´al, Canada.
Zhang,N. et al. (2015) Predicting anticancer drug responses using a dual-layer
integrated cell line-drug network model. PLoS Comput. Biol., 11,
e1004498.


