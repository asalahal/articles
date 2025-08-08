Pattern Recognition 128 (2022) 108659


Contents lists available at [ScienceDirect](http://www.ScienceDirect.com)

# ~~Pattern Recognition~~


journal homepage: [www.elsevier.com/locate/patcog](http://www.elsevier.com/locate/patcog)

## Molecular substructure graph attention network for molecular property identification in drug discovery


Xian-bin Ye [a][,][1], Quanlong Guan [a][,][b][,][1], Weiqi Luo [a][,][b], Liangda Fang [a], Zhao-Rong Lai [c][,][∗],
Jun Wang [d]


a _Department_ _of_ _Computer_ _Science,_ _College_ _of_ _Information_ _Science_ _and_ _Technology,_ _Jinan_ _University,_ _Guangzhou_ _510632,_ _China_
b _Guangdong_ _Institute_ _of_ _Smart_ _Education,_ _Jinan_ _University,_ _Guangzhou_ _510632,_ _China_
c _Department_ _of_ _Mathematics,_ _College_ _of_ _Information_ _Science_ _and_ _Technology,_ _Jinan_ _University,_ _Guangzhou_ _510632,_ _China_
d _Ping_ _An_ _Healthcare_ _Technology,_ _Chaoyang,_ _Beijing_ _100027,_ _China_



a r t i c l e i n f o


_Article_ _history:_
Received 7 July 2021
Revised 27 January 2022
Accepted 16 March 2022
Available online 18 March 2022


_Keywords:_
Molecular substructure

Graph attention
Molecular property identification


**1.** **Introduction**



a b s t r a c t


Molecular machine learning based on graph neural network has a broad prospect in molecular property identification in drug discovery. Molecules contain many types of substructures that may affect their
properties. However, conventional methods based on graph neural networks only consider the interaction information between nodes, which may lead to the oversmoothing problem in the multi-hop operations. These methods may not efficiently express the interacting information between molecular substructures. Hence, We develop a Molecular SubStructure Graph ATtention (MSSGAT) network to capture
the interacting substructural information, which constructs a composite molecular representation with
multi-substructural feature extraction and processes such features effectively with a nested convolution
plus readout scheme. We evaluate the performance of our model on 13 benchmark data sets, in which 9
data sets are from the ChEMBL data base and 4 are the SIDER, BBBP, BACE, and HIV data sets. Extensive
experimental results show that MSSGAT achieves the best results on most of the data sets compared with
other state-of-the-art methods.


© 2022 Elsevier Ltd. All rights reserved.



Drug discovery is time-consuming, labor intensive, and expensive. It usually starts with experimental discoveries of molecules
and targets (i.e., de novo drug design) and the validations with in
vitro experiments on cell lines and animals before moving to clinical tests [1]. The entire process from the discovery to the regulatory approval of a new drug can take as long as 12 years and cost
upwards of US 2.8 billion. Furthermore, each drug developing stage
has a very low success rate of about 1 _/_ 5000.
Drug discovery is equipped with statistical learning since the
rise of computational chemistry. In order to increase the speed
of drug screening and reduce costs, researchers in cheminformatics have been building quantitative structure activity relationships
(QSAR) via machine learning methods [2,3]. In recent years, with
increasing biochemistry data volumes and advanced computing


∗ Corresponding author.
_E-mail_ _addresses:_ [yexianbin@stu2019.jnu.edu.cn](mailto:yexianbin@stu2019.jnu.edu.cn) (X.-b. Ye), gql@jnu.edu.cn
(Q. Guan), [lwq@jnu.edu.cn](mailto:lwq@jnu.edu.cn) (W. Luo), [fangld@jnu.edu.cn](mailto:fangld@jnu.edu.cn) (L. Fang), laizhr@jnu.edu.cn
(Z.-R. Lai), [junwang.deeplearning@gmail.com](mailto:junwang.deeplearning@gmail.com) (J. Wang).

1 These two authors contribute equally to this article.


[https://doi.org/10.1016/j.patcog.2022.108659](https://doi.org/10.1016/j.patcog.2022.108659)
0031-3203/© 2022 Elsevier Ltd. All rights reserved.



machines (e.g., Graphics Processing Unit, GPU), a large number
of deep learning methods are applied to drug discovery because
of their powerful capability of feature extraction and flexibility
of model structures compared with conventional machine learning methods [4,5]. Due to the particularity of compound structures
and the limitations of early-era feature engineering (e.g., molecular
fingerprints, descriptors, and Simplified Molecular-Input Line-Entry
System strings, SMILES [6]), it is difficult for conventional neural
networks to extract compound substructural information from raw
molecules.

The emergence of graph convolutional networks (GCN) brings in
a new breakthrough in drug-related tasks [7]. Niepert et al. [8] propose a general method to extract local information from graph
data and apply it to the activity prediction of compound molecules.
Structural representation of compound molecules is encoded as a
molecular fingerprint, a high dimensional vector of binary digits.
Duvenaud et al. [9] use a GCN to obtain molecular fingerprints and
apply it to molecular property prediction. Kearnes et al. [10] develop a GCN called “weave module”, which can aggregate the atom
and bond information as node features, and apply it to activity prediction. Zhang et al. [11] propose a graph neural network based
on the graph structure GSCN, which balances between the impor

_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Fig.** **1.** Entire structure of MSSGAT.



tance of graph structural information and the node neighboring information. Since molecular property prediction is on the level of
the entire compound structure, Herr et al. [12] propose the entire
graph-level representation learning, which is shown to be effective
by the experiments. Ding et al. [13] learn the graph-level representation by combining the depth-first-search algorithm with their
node selection strategy on the features of local structures. Fang
et al. [14] introduce a structured multi-head self-attention mechanism to obtain the graph-level representation of the fused graph
structural information.

Although there are a large number of GNNs and GCNs handling molecular structures, these conventional methods only consider the interaction information between nodes, which may suffer from the oversmoothing problem of multi-hop operations. They
seldom take molecular substructures into consideration, but the interacting information between substructures is crucial to molecular properties. Consequently, the molecular substructural information is not fully utilized, especially for biomacromolecules containing polycyclic structures. To fill this gap, we propose a Molecular SubStructure Graph ATtention (MSSGAT) network, whose entire
structure is shown in Fig. 1. The main contributions can be summarized as follows: **1.** We propose to use a structural feature extraction scheme including 3 types of features (raw + tree decomposition + ECFP): raw molecular graphs, molecular structural features via tree decomposition [15], and Extended-Connectivity FingerPrints (ECFP) [16]. **2.** We design a framework including several
graph attention convolutional (GAC) blocks and deep neural network (DNN) blocks to process the above structural features. We
also improve the GAC blocks to relieve the gradient vanishing or
exploding problem. **3.** We design a readout block based on gated
recurrent units (GRU) [17]. The readout blocks collaborate with the
GAC blocks in a nested architecture to obtain molecular embed
dings. **4.** We visualize the molecules and mark the most important atoms with the attention scores produced by MSSGAT, which
can be a good reference for subsequent researches by medicinal
chemists. We evaluate the performance of MSSGAT on 13 benchmark data sets, in which 9 are from the ChEMBL data base [18] and
4 are the SIDER [3], BBBP [3], BACE [3], and HIV [3] data sets. Extensive experimental results show that MSSGAT achieves the best
results on most of the data sets compared with other state-of-theart methods.


**2.** **Related** **works**


Due to the establishment of drug data bases, methods based
on deep learning have caught more attention in the pharmaceutical industry. First, DNN has been widely used in the quantitative
structure activity relationship (QSAR). Ma et al. [19] use experiments to verify that QSAR models based on DNNs are better than



some traditional machine learning models (the random forest and
the support vector machine). You et al. [20] show that DNNs are
effective in predicting drug-target pairs and can be used for drug
repurposing. Li et al. [21] use a multi-task DNNs model to predict
human cytochrome P450 inhibitors, and the results show that the
multi-task model has a better predictive effect than several traditional machine learning models (SVM, KNN, the decision tree, and
the logistic regression).
Accelerating the speed of virtual screening and accurately capturing compounds that interact with the targets have been hot
spots in drug research in recent years. The emergence of Generative Adversarial Networks (GAN) [22] provides new ideas for
speeding up the research of virtual drug screening. Kadurin et al.

[23] adopt the anti-autoencoding (AAE) network structure, and use
NCI-60 cell line assay data for 6252 compounds to train the network. The output of the network is used to search the pubchem
data base and screen out candidates with anticancer activities.
AAEs can be used to generate new molecular fingerprints that have
specific molecular characteristics.
Recently, some GCNs have been applied to the property prediction for small molecules. They mainly consider the interacting
information between nodes, which is indicated by the adjacency
matrix of the molecular graph. However, traditional GCNs may neglect the fact that chemical bonds (edges) in different molecules
can be similar if the interatomic distances are similar. To address

this problem, Shang et al. [24] develop an Edge–Aware multi-view
spectral GCN (EAGCN) approach to enhance the property prediction for small molecules.

Nevertheless, existing graph-based models may neglect the interacting information between molecular substructures, which also
influences the molecular property based on the knowledge of
chemistry. Zhang et al. [25] develop a fragment-oriented GAT (FraGAT) to boost the interaction between fragments of molecular
graphs, which may retain functional groups. FraGAT also aggregates
the atom-level features to represent the molecular graph. However, if most rings are partitioned into the same fragment, FraGAT may deteriorate in macromolecules like polycyclic molecules
(i.e., molecules containing no less than 5 rings), because the topological information between rings is not fully utilized. Similarly,
the RNN-based MSGG model [26] transforms a molecule into a
substructure-based graph. Then this graph is expanded into threechannel sequences for the input of a Bi-GRU model. However,
MSGG pays less attention to the interacting and topological information between atoms in the original molecular graph. ECFP
represents many molecular substructures via sparse binary vectors
but neglects the topological information compared with the graphbased method. Thus ECFP may not catch the interacting information of the atoms. To better exploit the useful fine-grained fragments of ECFP, we adopt it as one of the features in the proposed
MSSGAT.



2


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Table** **1**

Summary of 13 benchmark data sets.


Data set Name Data type Number of compounds


CHEMBL203 Epidermal growth factor receptor erbB1 SMILES 1794
CHEMBL267 Tyrosine-protein kinase SRC SMILES 1251
CHEMBL279 Vascular endothelial growth factor receptor 2 SMILES 3266
CHEMBL325 Histone deacetylase 1 SMILES 517
CHEMBL340 Cytochrome P450 3A4 SMILES 3542
CHEMBL333 Matrix metalloproteinase-2 SMILES 321
CHEMBL2971 Tyrosine-protein kinase JAK2 SMILES 1582
CHEMBL2842 Serine/threonine-protein kinase mTOR SMILES 2455
CHEMBL4005 PI3-kinase p110-alpha subunit SMILES 2232
HIV Human immuno-deficiency virus SMILES 41 _,_ 913
BBBP Blood-brain barrier penetration SMILES 2053
BACE Human _β_ -secretase 1 SMILES 1522
SIDER Side Effect Resource SMILES 1427



**3.** **Dataset** **preparation**


_3.1._ _Anti-cancer_ _data_ _sets_ _from_ _ChEMBL_


The anti-cancer active molecules are collected from the ChEMBL

data base [18], which includes some common variables like the
IC50 value, the EC50 value, Inhibition, and the Ki value. The data
base uses pChEMBL values to record the relative activity of the
compounds, which allows for a number of measurements (i.e.,
half-maximal response concentration/epotency/affinity) to be compared in a negative logarithmic scale. According to Lenselink et al.

[27], pChEMBL = 6 _._ 5 (approximately 300 _nM_ ) is chosen as the decision boundary. It indicates that a compound with pChEMBL ⩾ 6 _._ 5
is an inhibitor, otherwise it is a non-inhibitor. In addition, some
compounds have multiple legal activity test records, so we average all the legal pChEMBL values for the same compound as a
relatively reasonable result. To demonstrate the superiority of our
model for biomacromolecules containing polycyclic structures, we
retain molecules containing no less than 5 ring structures in the
data set.


_3.2._ _Other_ _benchmark_ _data_ _sets_


_3.2.1._ _HIV_

The HIV data set is introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, which tests the abilities of 41,913
compounds to inhibit HIV replication. Original results are divided
into three categories: inactive, active, and moderately active. Wu
et al. [3] combine the latter two classes, making it a binary classification task and propose a scaffold splitting for this data set to
discover new structures of HIV inhibitors.


_3.2.2._ _BACE_

The BACE data set contains the experimental values collected
from the scientific literature over the past decade. It provides binding results (binary labels) for the set of inhibitors of BACE-1 [3].


_3.2.3._ _BBBP_

The Blood-brain barrier penetration (BBBP) data set is collected
from the study of modeling and predicting the barrier permeability.


_3.2.4._ _SIDER_

The Side Effect Resource (SIDER) is a data set collected from
marketed drugs with adverse drug reactions. This data set includes
12 binary-classification tasks.
All the above data sets are summarized in Table 1. We use a

scaffold split [3] to divide a data set into three parts: a training set, a validation set and a test set (the ratio is 8 : 1 : 1). The
scaffold split attempts to discriminate between different molecular



structures in the train/validation/test sets, which offers a greater
challenge and demands a higher level of generalization ability for
deep learning models than the random split. In addition, ROC-AUC
is used for model evaluation. Anti-cancer data sets of 9 targets are
mentioned with the following “ChEMBL” IDs.


**4.** **MSSGAT**


_4.1._ _Structural_ _feature_ _extraction_ _for_ _anti-cancer_ _inhibitors_


Traditional machine learning methods usually use molecular
descriptors (e.g., molecular weight and Alogp) as inputs, but pharmacologists usually analyze molecular structures instead of molecular descriptors. Besides, molecular descriptors may easily neglect
the local structural information of molecules. Hence molecular descriptors may not provide sufficient classification information. On
the other hand, a molecular fingerprint is high-dimensional and
sparse. The valid substructure bits in the fingerprint vector are
sparse, and it is difficult to obtain the effective correlation information between the substructures. In recent years, although many
GNN models come out, their input features are just local information of molecular graphs.
In order to extract structural features for anti-cancer inhibitors,
we propose a composite feature scheme “raw + tree decomposition + ECFP” as follows.

_Raw_ _molecular_ _graph_ _and_ _its_ _descriptors_ . The raw molecular
graph is a basic structure of atomic relationships, where each node
represents an atom. Each atom has 9 atomic features, which are
summarized in Table 2. The number of charges and the number of
free radicals are encoded as integers, while other features are encoded as one-hot vectors. Such raw features are acquired by the
open-source chemical information calculation library RDkit [28].
The distributions of atom numbers and pChEMBL values of the
CHEMBL340 data set are shown in Fig. 2.
_Structural_ _features_ _via_ _tree_ _decomposition_ . In order to extract
global structural features, we adopt the tree decomposition algorithm for molecular graphs and generate multiple effective substructures [15,29,30]. Such a tree-like structure could represent
the substructural components and the connections between these
components, then we could use the connection trees formed by
these substructures to represent the molecules. Substructures are
regarded as nodes and their connections are regarded as edges.
All substructures corresponding to SMILES (namely, token) form
the vocabulary, and the substructures mapping dictionaries are defined for each data set. The tree decomposition process is shown
in Fig. 3. Word embeddings are initialized by summing up the
atom embedding vectors in each substructure of the raw molecular graph from the “raw” branch. The substructural embeddings
are represented by concatenating word embeddings and one-hot



3


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Table** **2**

Atomic descriptors for raw molecular graph: initialization of atomic representations of
molecules.


Atom feature Feature size Description


Atom 16 [B, C, N, O, F, Si, P, S, Cl, As, Se, Br, Te, I, At, metal]
Degree 11 Number of covalent bonds [0,1,2,3,4,5,6,7,8,9,10]
Formal charge 1 Electrical charge (integer)
Radical electrons 1 Number of radical electrons (integer)
Hybridization 6 [sp, sp2, sp3, sp3d, sp3d2, other]
Aromaticity 1 Aromatic system (0/1)
Hydrogens 5 Number of connected hydrogens [0,1,2,3,4]
Chirality 1 Chiral center (0/1)
Chirality type 2 R/S


**Fig.** **2.** Distributions of atom numbers and pChEMBL values of CHEMBL340 data set.



embeddings, and the entire connection tree is formed by a matrix
of these substructural embeddings.
_Extended-connectivity_ _FingerPrints_ _(ECFP)_ . It is better to represent chemical molecules by structural descriptors (e.g., atom-pair

[31] and topological torsion [32]) besides global descriptors (e.g.,
molecular weight, polar surface area, and logP). Molecular fingerprints provide structural molecular characteristics and improve
stability and generalization of MSSGAT. We use the ExtendedConnectivity FingerPrints (ECFP) [16] for MSSGAT, and design some
particular network blocks to process these features (Section 4.2.3).
ECFP splits the molecule into structural identifiers by the traversal
substructures within a distance from each atom. Then the identifiers are hashed to a vector with a fixed size (See Fig. 4). The RDkit
can be used to calculate ECFPs, and the effective diameter and the
length of the representation vectors are set as 2 and 512 according
to Rogers and Hahn [16], respectively.


_4.2._ _Key_ _modules_ _for_ _MSSGAT_


Now we have 3 types of features: “raw + tree decomposition + ECFP”, denoted by { **a** [[] _[l]_ []] } _[L]_ _l_ =1 [,] [{] **[b]** [[] _[l]_ []] [}] _[L]_ _l_ =1 [and] [{] **[c]** [[] _[l]_ []] [}] _[L]_ _l_ =1 [(] _[L]_ [is] [the]
number of samples in a batch), respectively. Then they will be processed by MSSGAT to make classifications. MSSGAT mainly consists
of four modules: several GAC blocks for { **a** [[] _[l]_ []] } _[L]_ _l_ =1 [and] [{] **[b]** [[] _[l]_ []] [}] _[L]_ _l_ =1 [,] [a]

DNN block for { **c** [[] _[l]_ []] } _[L]_ _l_ =1 [,] [a] [readout] [block] [based] [on] [GRU,] [and] [a] [classi-]
fier based on a multilayer perceptron. After receiving and processing the above features, the GAC, DNN and readout blocks output



graph embedding vectors, which are further concatenated as the
final embedding vector. This final vector is fed into the classifier
to get the classification result. The whole framework of the entire
MSSGAT is shown in Fig. 1.


_4.2.1._ _Graph_ _attention_ _convolutional_ _block_
The existing GCN [33] assigns the same weight to all the neighboring nodes of the central node, which is not suitable for representing molecular structures, because the contributions of different
atoms or clusters to the central atom are different. For example,
the benzene ring has a different effect from the hydroxyl group on
the atom C of the carboxyl group in the benzoic acid. Inspired by

[34], we propose a kind of GAC block to address such different effects of different molecular parts. It consists of 3 steps:


_•_ Calculate the attention coefficient _α_ _ij_ [[] _[l]_ []] [.]

_•_ Compute the weighted feature summation **h** [[] _i,_ _[l]_ _(_ []] [′] _K_ _)_ [.]

_•_ Implement several post-processing operations to obtain the updated hidden states.


Given the initial input of the _l_ th sample **z** _[(]_ [0] _[)]_ _[,]_ [[] _[l]_ []] containing
vertices { _._ _._ _._ _,_ **h** [[] _i_ _[l]_ []] _[,]_ _[.]_ _[.]_ _[.]_ _[,]_ **[h]** [[] _j_ _[l]_ []] _[,]_ _[.]_ _[.]_ _[.]_ [}] [,] [the] [attention] [coefficient] [is] [calcu-]
lated with a concatenation operator and a single-layer feedforward

map:

_e_ [[] _ij_ _[l]_ []] [=] _[f]_ �� **Wh** [[] _i_ _[l]_ []] [∥] **[Wh]** [[] _j_ _[l]_ []] �� _,_ _j_ ∈ _N_ _i_ [[] _[l]_ []] _[,]_ (1)



4


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Fig.** **3.** Tree decomposition process for molecules. The upper black box indicates the initialization of substructural embeddings.


fore, we add a ReLU layer and a Batch Normalization (BN) layer
after each convolution kernel in each GAC block


_[l]_ []] _[l]_ []] [′]
**z** [[] _i,_ _(_ _k_ _)_ [=] _[ReLU]_ _[(]_ **[h]** [[] _i,_ _(_ _k_ _)_ _[)]_ _[,]_ (4)


_[l]_ []] [′] _[l]_ []] _[l]_ []]
**z** [[] _i,_ _(_ _k_ _)_ [=] _[BN]_ _[(]_ **[z]** [[] _i,_ _(_ _k_ _)_ _[,]_ [{] **[z]** [[] _i,_ _(_ _k_ _)_ [}] _[L]_ _l_ =1 _[)]_ _[.]_ (5)


Then we concatenate these new features and implement a full connection _FC_ 0 to obtain the updated hidden state **z** _i_ _[(]_ _,_ [1] _(_ _[)]_ _K_ _[,]_ _)_ [[] _[l]_ []] [:]


_[,]_ [[] _[l]_ []] _[l]_ []] [′]
**z** _i_ _[(]_ _,_ [1] _(_ _[)]_ _K_ _)_ [=] _[F][C]_ [0] _[(]_ [∥] _[K]_ _k_ =1 **[z]** [[] _i,_ _(_ _k_ _)_ _[)]_ _[.]_ (6)


The architectures of the Graph Attention kernels (named as GA
below) and the GAC blocks are shown in Fig. 5. Furthermore, multiple GAC blocks can be stacked to constitute a deeper network that
can process molecular parts with more nodes:


**z** _[(]_ _[n]_ [+][1] _[)]_ _[,]_ [[] _[l]_ []] = _GAC_ _(_ **z** _[(]_ _[n]_ _[)]_ _[,]_ [[] _[l]_ []] _)_ _,_ _n_ = 0 _,_ 1 _,_ _._ _._ _._ _,_ _N,_ (7)



**Fig.** **4.** Taking the aspirin as an example, if the pre-defined substructures exist, the
corresponding positions of the ECFP vector are set as 1.




_[l]_ []] exp � LeakyReLU � _e_ [[] _ij_ _[l]_ []] ��
_α_ [[]
_ij_ [=]



where the subscripts _i_ and _(_ _K_ _)_ can be omitted since they do not
disturb the operator _GAC_ . Since the raw molecular graph **a** [[] _[l]_ []] usually has much more nodes than its structural features via tree decomposition **b** [[] _[l]_ []], we use _N_ GAC blocks for **a** [[] _[l]_ []] while only 1 GAC
block for **b** [[] _[l]_ []] . Then the number of stacked blocks is consistent with
the number of nodes, which is beneficial to the convolution performance. The deployments are combined with the readout block in
the next subsubsection.


_4.2.2._ _Readout_ _block_ _based_ _on_ _gated_ _recurrent_ _units_
Readout operation is similar to the global pooling of CNN,
which performs an aggregation operation on the features of all
nodes to output a global representation of the graph. Inspired by
GRU [17] (a variant of an LSTM [35] recurrent network unit), we
design a readout block that can synthesize molecular embeddings
according to the order of the GAC blocks. Suppose the hidden state

_[)]_ _[,]_ [[] _[l]_ []]
of the _i_ th node after the _n_ th GAC block for the _l_ th sample is **z** _i_ _[(]_ _,_ _[n]_ _(_ _K_ _)_ [,]

then the graph embedding is **g** _[(]_ _[n]_ _[)]_ _[,]_ [[] _[l]_ []] :

**g** _[(]_ _[n]_ _[)]_ _[,]_ [[] _[l]_ []] = _Mean_ � **z** _i_ _[(]_ _,_ _[n]_ _(_ _K_ _[)]_ _[,]_ _)_ [[] _[l]_ []] [|] [∀] _[v]_ [[] _i_ _[l]_ []] [∈] _[V]_ [[] _[l]_ []] [�] _,_ (8)


where _V_ [[] _[l]_ []] is the vertex set of the _l_ th sample. Denote **G** _[(]_ _[n]_ [+][1] _[)]_ _[,]_ [[] _[l]_ []] as
the molecular embedding after the _n_ th GAC block and _GRU_ _[(]_ _[n]_ _[)]_ as
the update function at iteration _n_, then

**G** _[(]_ [0] _[)]_ _[,]_ [[] _[l]_ []] = _Mean_ � _f_ _[Lin]_ _(_ **z** _i_ _[(]_ _,_ [0] _(_ _[)]_ _K_ _[,]_ _)_ [[] _[l]_ []] _[)]_ [|] [∀] _[v]_ [[] _i_ _[l]_ []] [∈] _[V]_ [[] _[l]_ []] [�] _,_ (9)



~~�~~



_,_ (2)
_m_ ∈ _N_ _i_ [[] _[l]_ []] [exp] ~~�~~ LeakyReLU ~~�~~ _e_ [[] _im_ _[l]_ []] ~~��~~



where _N_ _i_ [[] _[l]_ []] [is] [the] [neighboring] [node] [set] [of] [vertex] **[h]** [[] _i_ _[l]_ []] [,] **[W]** [is] [a] [shared]
parameter of the linear map _f_ that adjusts the features of the
vertices { _._ _._ _._ _,_ **h** [[] _i_ _[l]_ []] _[,]_ _[.]_ _[.]_ _[.]_ _[,]_ **[h]** [[] _j_ _[l]_ []] _[,]_ _[.]_ _[.]_ _[.]_ [}] [,] [and] [∥] [is] [the] [concatenation] [operator]
that concatenates the dominated terms. Eq. (1) calculates a kind of
correlation between vertices **h** [[] _i_ _[l]_ []] [and] **[h]** [[] _j_ _[l]_ []] [,] [and] [(2)] [normalizes] [this]
correlation with the softmax function. Next, the attention coefficient _α_ _ij_ [[] _[l]_ []] [is] [used] [to] [adjust] [the] [importance] [of] [the] [neighboring] [node.]
Moreover, we use a multi-head attention mechanism that includes
_K_ convolution kernels to calculate _K_ new features **h** [[] _[l]_ []] [′]
_i,_ _(_ _k_ _)_

**h** [[] _i,_ _[l]_ _(_ []] [′] _k_ _)_ [=] � _α_ _ij_ [[] _[l]_ _,_ []] _(_ _k_ _)_ **[W]** _[(]_ _[k]_ _[)]_ **[h]** [[] _j_ _[l]_ []] _[.]_ (3)

_j_ ∈ _N_ _i_ [[] _[l]_ []]


The eigenvalues produced by the convolution operation easily
deviate from the normal distribution, thus they have an adverse
effect on the convergence of the network (gradient disappearance
or gradient explosion) and worsen the model performance. There


5


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Fig.** **5.** Architectures of Graph Attention Kernel and Graph Attention Convolutional Block.



**G** _[(]_ _[n]_ [+][1] _[)]_ _[,]_ [[] _[l]_ []] = _GRU_ _[(]_ _[n]_ _[)]_ [�] **g** _[(]_ _[n]_ _[)]_ _[,]_ [[] _[l]_ []] _,_ **G** _[(]_ _[n]_ _[)]_ _[,]_ [[] _[l]_ []] [�] _,_ _n_ = 0 _,_ 1 _,_ _._ _._ _._ _,_ _N,_ (10)


where _f_ _[Lin]_ is a linear transform for initialization.
The architecture of the readout block is shown in Fig. 6. As for
the raw molecular graph **a** [[] _[l]_ []] and its structural features via tree decomposition **b** [[] _[l]_ []], the readout deployments are:


_[,]_ [[] _[l]_ []] _[l]_ []] _[l]_ []] [′] _[,]_ [[] _[l]_ []]
**z** _[(]_ [0] _[)]_ ← **a** [[] _,_ **a** [[] ← **G** _[(]_ _[N]_ [+][1] _[)]_ ; (11)


_[,]_ [[] _[l]_ []] _[l]_ []] _[l]_ []] [′] _[,]_ [[] _[l]_ []]
**z** _[(]_ [0] _[)]_ ← **b** [[] _,_ **b** [[] ← **G** _[(]_ [2] _[)]_ _._ (12)



Note that the notations **G** _[(]_ _[N]_ [+][1] _[)]_ _[,]_ [[] _[l]_ []] and **G** _[(]_ [2] _[)]_ _[,]_ [[] _[l]_ []] here go through different readout progresses because their inputs are different ( **a** [[] _[l]_ []] and
**b** [[] _[l]_ []] ).
If we zoom in the GRU operator (10) and omit the superscript

[ _l_ ] without confusion, **g** _[(]_ _[n]_ _[)]_ and **G** _[(]_ _[n]_ _[)]_ first go through the update
gate **u** _[(]_ _[n]_ _[)]_ and the reset gate **r** _[(]_ _[n]_ _[)]_ :


**u** _[(]_ _[n]_ _[)]_ = _σ_ [�] **W** **u** _(_ _n_ _)_ **g** _[(]_ _[n]_ _[)]_ + **X** **u** _(_ _n_ _)_ **G** _[(]_ _[n]_ _[)]_ [�] _,_ (13)


**r** _[(]_ _[n]_ _[)]_ = _σ_ [�] **W** **r** _(_ _n_ _)_ **g** _[(]_ _[n]_ _[)]_ + **X** **r** _(_ _n_ _)_ **G** _[(]_ _[n]_ _[)]_ [�] _,_ (14)


where **W** **u** _(_ _n_ _)_, **X** **u** _(_ _n_ _)_, **W** **r** _(_ _n_ _)_ and **X** **r** _(_ _n_ _)_ are linear transforms to be
trained for **u** _[(]_ _[n]_ _[)]_ and **r** _[(]_ _[n]_ _[)]_, respectively. Then the hidden state **G** [˜] _[(]_ _[n]_ _[)]_



6


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Fig.** **6.** Readout block based on GRU for MSSGAT.


is computed by:



**G** ˜ _[(]_ _[n]_ _[)]_ = tanh [�] **W** **G** _(_ _n_ _)_ **g** _[(]_ _[n]_ _[)]_ + **X** **G** _(_ _n_ _)_ _(_ **r** _[(]_ _[n]_ _[)]_ � **G** _[(]_ _[n]_ _[)]_ _)_ [�] _,_ (15)


where **W** **G** _(_ _n_ _)_ and **X** **G** _(_ _n_ _)_ are linear transforms to be trained, and � is
the element-wise multiplication. When **r** _[(]_ _[n]_ _[)]_ is close to **0**, the current state **G** _[(]_ _[n]_ _[)]_ would be highly forgotten. The 1st recurrent state is
an element-wise linear interpolation


**G** _[(]_ _[n][,]_ [1] _[)]_ = _(_ **1** − **u** _[(]_ _[n]_ _[)]_ _)_ � **G** _[(]_ _[n]_ _[)]_ + **u** _[(]_ _[n]_ _[)]_ � **G** [˜] _[(]_ _[n]_ _[)]_ _,_ (16)


where the update gate **u** _[(]_ _[n]_ _[)]_ controls the update strength from the
current embedding **G** _[(]_ _[n]_ _[)]_ to the hidden state **G** [˜] _[(]_ _[n]_ _[)]_ .
We let **g** _[(]_ _[n]_ _[)]_ and **G** _[(]_ _[n][,]_ [1] _[)]_ go through the recurrence (13)–(16) and
obtain **G** _[(]_ _[n][,]_ [2] _[)]_, then **g** _[(]_ _[n]_ _[)]_ and **G** _[(]_ _[n][,]_ [2] _[)]_ go through the recurrence... The
next molecular embedding is **G** _[(]_ _[n]_ [+][1] _[)]_ ≜ **G** _[(]_ _[n][,][M]_ _[)]_, where _M_ is the number of recurrences. The weights of these two module are updated
within one backward pass. We will conduct experiments to compare the readout modules of LSTM and Concat + FC (concatenating
the 3 types of features and using a fully connected layer) to prove
the superiority of our GRU readout module in Section 5.4.


_4.2.3._ _Deep_ _neural_ _network_ _block_ _for_ _ECFPs_
In order to fully consider the structural molecular characteristics and improve stability and generalization of MSSGAT, molecular fingerprints are necessary as input features, because it is more
suitable to represent chemical molecules by structural descriptors (e.g., atom-pair [31] and topological torsion [32]) rather than
global descriptors (e.g., molecular weight, polar surface area, and
logP). The ECFPs are good choices, but they are high-dimensional
and sparse vectors where each bit is binary, which will cause the
dimensionality disaster. Therefore, we introduce a pyramid-form
DNN block where the number of neurons is gradually reduced by
one-half per layer from the input layer to the output layer, to further extract lower-dimensional features from the ECFPs.


**c** [[] _[l]_ []] [′] = _DNN_ _(_ **c** [[] _[l]_ []] _)_ _._ (17)


_4.2.4._ _Classification_ _block_
We propose a three-layer feedforward classification block for
MSSGAT. The above GAC and DNN blocks produce concatenated
embedding features **d** _l_ ≜ [ **a** [[] _[l]_ []] [′] [⊤] ; **b** [[] _l_ _[l]_ []] [′] [⊤] ; **c** [[] _l_ _[l]_ []] [′] [⊤] ] [⊤], which are then fed
into two parallel modules: the fully connected layer 1 (FC1) and
the wide fully connected layer (FCwide). FCwide provides a feature
extraction channel by one fully connected layer directly connecting
with high-level features.


**d** _[BN]_ _l_ [=] _[BN]_ _[(]_ **[d]** _[l]_ _[,]_ [{] **[d]** _[l]_ [}] _[L]_ _l_ =1 _[)]_ _[,]_ (18)


**d** _[F]_ _l_ _[C]_ [1] = _FC_ 1 _(_ **d** _[BN]_ _l_ _[)]_ _[,]_ (19)



where _Y_ = { _y_ _l_ } _[L]_ _l_ =1 [and] _[P]_ [=] [{] _[p]_ _[l]_ [}] _[L]_ _l_ =1 [are] [the] [true] [probabilities] [and]
the estimated probabilities by MSSGAT for a batch of observations being inhibitors, respectively. The diagram of the classification block is shown in Fig. 7.


_4.3._ _Model_ _summary_ _for_ _MSSGAT_


The model structure and training details for MSSGAT are summarized as follows:



**d** _[F]_ _l_ _[Cwide]_ = _FCwide_ _(_ **d** _[BN]_ _l_ _[)]_ _[.]_ (20)


The FC1 features would further go through the fully connected
layer 2 (FC2) before being concatenated with the FCwide fea
tures.


**d** _[F]_ _l_ _[C]_ [1] _[,][BN]_ = _BN_ _(_ **d** _[F]_ _l_ _[C]_ [1] _[,]_ [{] **[d]** _[F]_ _l_ _[C]_ [1] [}] _[L]_ _l_ =1 _[)]_ _[,]_ (21)


**d** _[F]_ _l_ _[C]_ [2] = _FC_ 2 _(_ **d** _[F]_ _l_ _[C]_ [1] _[,][BN]_ _)_ _,_ (22)


**d** _l_ _[f in]_ [=] [[] **[d]** _[F]_ _l_ _[Cwide]_ [⊤] ; **d** _[F]_ _l_ _[C]_ [2][⊤] ] [⊤] _._ (23)


With more FC layers, MSSGAT could gather low-level features to
form higher-level features for classification. The reasons to concatenate FC2 and FCwide features are:


_•_ High-level features could capture global information. Simultaneously using low-level and high-level features could improve
the generalization ability of MSSGAT.

_•_ The backpropagation of the error terms could get smoother and
the gradient vanishing problem could be relieved to some ex
tent.


We use the ReLU activation and the dropout technique with
dropout rate 0.1 in FC1, FC2 and FCwide layers, and use the softmax function for the output layer:


1
_p_ _l_ = 1 + _e_ [−] _[θ]_ [ ⊤] **[d]** _l_ _[f in]_ _,_ (24)


where _θ_ is the regression coefficient vector that would be trained
in the network training, and _p_ _l_ could be seen as the _l_ th sample probability being an anti-cancer inhibitor. Last, MSSGAT can be
trained by maximizing the log-likelihood of the observations:



_l_ _(_ _Y,_ _P_ _)_ =



_L_
� _(_ _y_ _l_ log _(_ _p_ _l_ _)_ + _(_ 1 − _y_ _l_ _)_ log _(_ 1 − _p_ _l_ _))_ _,_ (25)


_l_ =1



7


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Fig.** **8.** Average ROC-AUCs on 9 ChEMBL data sets for different models.



**Fig.** **7.** Diagram of classification block for MSSGAT.


_•_ For the raw molecular features _**α**_ [[] _[l]_ []], we use _N_ = 3 GAC blocks
and _K_ = 4 graph attention kernels for each GAC block. For the

_[l]_ []]
structural features via tree decomposition _**b**_ [[], we set _N_ = 2 and
_k_ = 4. The sizes of _**α**_ [[] _[l]_ []] and _**b**_ [[] _[l]_ []] are 44 and 128, respectively. We
set 4 as a moderate number of heads for the multi-head atten
tion mechanism.


_•_ The DNN processing ECFPs consists of 1 input layer with 512
neurons, 1 output layer with 128 neurons, and 2 hidden layers
with 256 and 128 neurons, respectively.

_•_ The number of recurrences for the GRU operator is _M_ = 2. The
feature sizes of the current graph embedding **g** _[(]_ _[n]_ _[)]_ and the current state **G** _[(]_ _[n]_ _[)]_ are both 128.

_•_ The classification block consists of 1 input layer with 384 neurons, 1 **FC1** layer with 64 neurons, 1 **FC2** layer with 192 neurons, 1 **FCwide** layer with 192 neurons, 1 output layer with 64
neurons, and 1 prediction layer with 2 neurons (to compute the
probabilities of being an inhibitor and a non-inhibitor, respectively).

_•_ The dropout rate for the fully connected layers in the DNN and
the classification blocks is set as 0.1.

_•_ The batch size _L_ is set as 256.

_•_ The initial global learning rate is set as _η_ 0 = 0 _._ 001. In addition,
to reduce the oscillation of the loss function in the later stage
of training and make the network converge better, we use an
exponential decay scheme of learning rate. Hence the learning
rate for the _t_ th epoch is _η_ _t_ = _η_ 0 _γ_ _[t]_, where _γ_ is set as 0.9.

_•_ The maximum number of epochs for training is set as 300, but
we use an early stopping scheme to avoid overfitting and save
training computation. If the performance of MSSGAT on the validation set does not improve during a certain number of epochs
(called the “patience”), the training will be terminated in advance and the resulted model will be saved. We set the patience as 10 for MSSGAT in our experiments.


**5.** **Experimental** **results**


We conduct extensive experiments to verify the performance
of MSSGAT, including ablation studies that analyze the effectiveness of each module in MSSGAT. Each data set is scaffold-split into



three sets (training, validation, and test). We use ROC-AUC scores
to evaluate MSSGAT and other competitors, including EAGCN [24],
FraGAT [25], MSGG [26], AttentiveFP [36], weave [10], MPNN [37],
NF [9], GCN, and GAT. We use three different random seeds in the
experiments and average the final results. The hardware platform
for this work is a Ubuntu 16.04 workstation equipped with an Intel
Core i9-9820X CPU @ 3.30 GHz × 20, a 64 GB RAM, and an NVIDIA
Geforce RTX 2080 Ti card. The entire MSSGAT is implemented with
the PyTorch [2] and the Deep Graph Library [3] frameworks. The model
parameters are initialized by the Xavier scheme [38]. The Adam
optimizer [39] is used for optimization.


_5.1._ _Comparison_ _results_ _on_ _ChEMBL_ _data_ _sets_ _from_ _ChEMBL_


Experimental results on 9 ChEMBL benchmarks are presented
in Table 3 and the average results are shown in Fig. 8. In brief,
MSSGAT achieves the best average result on all the anti-cancer
molecule data sets. It significantly outperforms all the competitors
with an average ROC-AUC score of 0.8586. Hence the improved features of “raw + tree decomposition + ECFP” provide sufficient and
useful structural information ranging from single atom features to
substructural features, and finally to cluster features. Besides, the
GAC, DNN and readout blocks effectively process and integrate the
improved structural features to achieve better performance. Thus
MSSGAT as a multi-level substructural feature extraction method
can significantly improve the classification performance of biological macromolecules containing polycyclic structures.
To examine the extendability of MSSGAT, we also train MSSGAT by the HIV data set and test it on the ZINC data base. [4] Fig. 9
shows the histogram of ring numbers of the molecules on HIV. It
indicates that about 15% are polycyclic molecules (i.e. molecules
containing no less than 5 rings) and 85% are oligocyclic molecules
(i.e. molecules containing less than 5 rings). As for the test set,
we randomly sample 500 polycyclic molecules for each of the 10
most common scaffolds from ZINC, resulting in 5000 polycyclic
molecules labelled with 10 different scaffolds. Then we use the

Uniform Manifold Approximation and Projection (UMAP) [40] to
visualize the embeddings of these 5000 polycyclic molecules from
the last embedding layer of MSSGAT, shown in Fig. 10(a). Although
there are only a small proportion of training samples are polycyclic
molecules, the embeddings of the polycyclic molecules in the test


2 [https://pytorch.org/.](https://pytorch.org/)

3 [https://www.dgl.ai/.](https://www.dgl.ai/)

4 [http://zinc.docking.org.](http://zinc.docking.org)



8


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Table** **3**

Prediction results on 9 ChEMBL data sets for various models. All the models have been tested for 3 times on each test set

and the average results are presented. The best result on each data set is bold and the second best result is underlined.


ModelROC-AUCData Set 267 203 340 279 2842 325 333 4005 2971


NF 0.7403 0.6737 0.5884 0.7110 0.7899 0.6697 0.5827 0.6618 0.6091

GAT 0.8005 **0.8811** 0.8283 0.6686 0.8232 0.6234 0.9154 0.8629 0.6434

GCN 0.7667 0.8238 0.8232 **0.8418** 0.8214 0.8252 0.8566 **0.9009** 0.804

MPNN 0.6728 0.7471 0.7923 0.8043 0.8055 0.6411 0.8194 0.8157 0.7716

Weave 0.7939 0.8116 **0.9269** 0.7136 0.7612 0.7350 0.8802 0.8750 0.7273

AttentiveFP 0.7252 0.7806 0.7949 0.7561 0.7901 0.6986 **0.9412** 0.8707 0.7351

EAGCN 0.7576 0.8285 0.8607 0.8021 0.8474 0.8443 0.8297 0.8480 0.8277

MSGG 0.7230 0.7904 0.8353 0.7746 0.7784 0.7186 0.9113 0.8169 0.7510

FraGAT 0.7310 0.8236 0.8164 0.7735 0.8661 0.7808 0.9167 0.8074 0.7970

MSSGAT **0.8125** 0.8345 0.8948 0.8162 **0.8687** **0.9080** 0.8915 0.8418 **0.8592**


**Fig.** **9.** Histogram of ring numbers of molecules on the HIV data set. There are 6333
polycyclic (containing no less than 5 rings) and 34794 oligocyclic (containing less
than 5 rings) molecules, respectively.


set are well discriminated with a high Silhouette index [41]. On the
other hand, we also sample 5000 oligocyclic molecules from ZINC
and visualize their embeddings in the same way as the polycyclic
molecules, shown in Fig. 10(b). MSSGAT is relatively less effective
in oligocyclic molecules with a smaller Silhouette index, since they
may not take good advantage of the multi-level molecular substructures of MSSGAT.


_5.2._ _Comparison_ _results_ _on_ _4_ _benchmark_ _data_ _sets_ _from_ _MoleculeNet_



We use 4 more benchmark data sets (with scaffold split) from
MoleculeNet [3] to verify the generalization ability of MSSGAT,
shown in Table 4. MSSGAT achieves the best results on 3 data sets

and ranks the third on BBBP. It indicates that MSSGAT can effec
tively process molecular substructural features and has a good generalization ability, since the scaffold split is challenging to the generalization ability of a model.


_5.3._ _Training_ _process_ _for_ _MSSGAT_


To analyze the training process of MSSGAT, we show it on the
training and validation sets of BACE and BBBP in Fig. 11. The loss
curves of MSSGAT on the training and validation sets tend to be
smooth after training for about 50 epochs and 15 epochs, respectively. The ROC-AUC curves of MSSGAT on both BACE and BBBP are
close to 1.0, thus MSSGAT can be efficiently trained.
To further validate the prediction performance of MSSGAT, we
use the UMAP to visualize the latent spaces of MSSGAT and MPNN



**Fig.** **10.** Visualization of the molecular embeddings of MSSGAT on the ZINC data
base. MSSGAT is trained by the HIV data set beforehand. A higher Silhouette index
indicates a better discrimination. (a) and (b) represent the embeddings of polycyclic
molecules and oligocyclic molecules, respectively.


on BACE, shown in Fig. 12. We can see that MSSGAT learns discriminative embeddings for inhibitor identification, while MPNN could
not separate the two classes well.


_5.4._ _Ablation_ _experiments_ _for_ _MSSGAT_


In order to analyze the contribution of each module to the
whole MSSGAT, we conduct ablation studies on the HIV and
the 9 ChEMBL data sets. The HIV data set contains about as



9


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Fig.** **11.** Training Processes of MSSGAT on BACE and BBBP.



**Table** **4**

Prediction results on 4 benchmark data sets (with scaffold split) for various models. All the models have been tested for 3 times on each test set

and the average results are presented. The best result on each data set is
bold and the second best result is underlined. **OOM** : Out of Memory.


ModelROC-AUCData Set BACE SIDER BBBP HIV


NF 0.6099 0.5173 0.6333 0.6971

GAT 0.6704 0.5435 0.6583 0.7733

GCN 0.6132 0.5713 0.6836 0.7770

MPNN 0.6870 0.5235 0.6723 0.7181

Weave 0.6440 0.5351 0.6596 0.7457

AttentiveFP 0.6587 0.5619 0.659971 0.7503

MSGG 0.8740 0.5278 0.7530 OOM

EAGCN 0.8337 0.6063 **0.8399** 0.7497

FraGAT 0.7896 0.5788 0.6913 0.7341

MSSGAT **0.8805** **0.6170** 0.7264 **0.7870**


many as 40 thousand samples, thus it is reliable for ablation experiments. The results in Table 5 show that MSSGAT with the
whole “raw + tree decomposition + ECFP” features outperforms
the single ECFP module, the single tree decomposition module
and the single raw molecular graph module on the validation
and the test sets. Next, we retain our “raw + tree decomposition + ECFP” features but try different readout modules (GRU,



**Table** **5**

Ablation experiments on input features and the readout module
for MSSGAT on the HIV data set. ROC-AUC scores are used in

evaluation.


Model Validation Test


MSSGAT **0** _**.**_ **8209** **± 0** _**.**_ **025** **0** _**.**_ **7828** **± 0** _**.**_ **020**

Tree-only 0 _._ 8034 ± 0 _._ 003 0 _._ 7540 ± 0 _._ 004
Raw-only 0 _._ 8038 ± 0 _._ 007 0 _._ 7663 ± 0 _._ 010
ECFP-only 0 _._ 7598 ± 0 _._ 001 0 _._ 7184 ± 0 _._ 002
MSSGAT(GRU) **0** _**.**_ **8209** **± 0** _**.**_ **025** **0** _**.**_ **7828** **± 0** _**.**_ **020**
MSSGAT(LSTM) 0 _._ 8197 ± 0 _._ 011 0 _._ 7547 ± 0 _._ 021
MSSGAT(Concat + FC) 0 _._ 7915 ± 0 _._ 017 0 _._ 7451 ± 0 _._ 021


LSTM, and Concat + FC). The results in Table 5 indicate that our
GRU readout module outperforms other readout modules to some

extent.

To further examine whether tree decomposition is effective
in extracting substructural features from molecules or other extraction methods could be better, we adopt a common fragmentation algorithm _rdkit.Chem.Fragmentonbonds_ [28] in RDkit for
ablation experiments. Similar to the fragmentation in FraGAT

[25], we retain all ring structures and break all acyclic single
bonds to obtain the corresponding fragments by the function _rd-_
_kit.Chem.Fragmentonbonds_, and substitute these fragments for the



10


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Table** **6**

Ablation experiments on feature extraction methods for MSSGAT on 9 ChEMBL data sets. ROC-AUC scores are used in evaluation. The
best result on each data set is bold and the second best result is underlined. “AVG” indicates the average ROC-AUC score of a model on
9 ChEMBL data sets. MSSGAT _[a]_ : MSSGAT with tree decomposition features. MSSGAT _[b]_ : MSSGAT with common fragmentation features.


ModelROC-AUCData Set 267 203 340 279 2842 325 333 4005 2971 AVG


MSSGAT _[a]_ (Ours) 0.8125 **0.8345** **0.8948** 0.8162 **0.8687** **0.9080** 0.8915 0.8418 **0.8592** **0.8586**
MSSGAT _[b]_ 0.8079 0.8169 0.8682 **0.8266** 0.8518 0.8387 0.9192 **0.8523** 0.8153 0.8441

Tree-only 0.8054 0.7766 0.8865 0.7723 0.8342 0.7845 0.9135 0.833 0.7540 0.8178
Raw-only **0.8135** 0.8138 0.8698 0.7567 0.8041 0.7448 **0.9549** 0.7695 0.7877 0.8128
ECFP-only 0.7543 0.7955 0.8295 0.8157 0.8477 0.7759 0.855 0.8451 0.7911 0.8122


tree decomposition features in MSSGAT. We denote our default
MSSGAT with tree decomposition features and the altered MSSGAT with the common fragmentation features as MSSGAT _[a]_ and
MSSGAT _[b]_, respectively. The results on the 9 ChEMBL data sets in
Table 6 show that MSSGAT _[a]_ outperforms MSSGAT _[b]_ in most cases.
We also visualize the fragment features of MSSGAT _[a]_ and MSSGAT _[b]_

in Fig. 13, which indicate that MSSGAT _[a]_ provides more fragments
and finer segmentations than MSSGAT _[b]_ . This may be the reason
why MSSGAT _[a]_ is better than MSSGAT _[b]_ . To summarize this subsection, MSSGAT is effective in expoiting multi-level molecular substructures from the proposed “raw + tree decomposition + ECFP”
features according to the above ablation experiments.


_5.5._ _Important_ _structure_ _visualization_


To further explore what information MSSGAT can provide on
molecular structures, we visualize some molecules on the BACE
data set and label the most important structures according to the
attention scores (weights) in the prediction step of MSSGAT. Specifically, We extract the attention scores from the last GAC block of
the “tree decomposition” branch of the well-trained MSSGAT. Then
we sort the attention scores of all the tree nodes, and visualized
the largest one (colored orange in Fig. 14). It indicates that MSSGAT allocates major attention to some common structures (e.g.,
carbon-oxygen double bonds, fluorine atoms and structures with
ammonia), which may be an interesting reference for drug designers from a different perspective of machine learning.


**6.** **Conclusion**



**Fig.** **12.** Latent space visualizations via UMAP for MSSGAT (Upper) and MPNN
(Lower) on BACE. A higher Silhouette index indicates a better discrimination.



In this work, we develop a novel Molecular SubStructure Graph
ATtention (MSSGAT) network to capture substructural interacting information with structural feature extraction including raw
molecular graphs, tree decomposition features, and ExtendedConnectivity FingerPrints (ECFP). MSSGAT consists of several GAC,
DNN and readout blocks that could effectively process molecular structural features and exploit the relationships between different molecular cliques of tree decomposition. Furthermore, MSSGAT uses both low-level and high-level features in classification to
improve generalization ability, and adopts the dropout technique
to relieve the gradient vanishing problem. Experimental results
show that MSSGAT outperforms other state-of-the-art competitors in most cases. MSSGAT could also reveal important molecular
structures by examining the attention scores, which gives a reference for drug designers from the perspective of machine learning.
In summary, MSSGAT is an effective tool for molecular property
identification and worth further investigations. Since MSSGAT is
designed mainly for large and polycyclic molecules, it is relatively
less effective in oligocyclic molecules. Future works could be designing different models for molecules with different sizes or finding more general molecular features for molecules with different
substructures.



11


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_


**Fig.** **13.** Fragment features for: (a) MSSGAT _[a]_ . (b) MSSGAT _[b]_ .


**Fig.** **14.** Important structure visualization on the BACE data set. The atoms in orange represent the most important components indicated by MSSGAT. (For interpretation of
the references to colour in this figure legend, the reader is referred to the web version of this article.)



**Declaration** **of** **Competing** **Interest**


The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.


**Acknowledgments**


This work is supported in part by the National Natural Science Foundation of China under Grants 62176103, 61703182,
in part by the Science and Technology Planning Project of
Guangzhou, China under Grants 201902010041, 202102021173,
202102080307, in part by the Guangdong Basic and Applied Basic
Research Foundation under Grant 2020A1515011476, in part by
the Science and Technology [Planning](https://doi.org/10.13039/501100012245) Project of Guangdong under
Grants 2021B0101420003, 2020B0909030005, 2020B1212030003,
2020ZDZX3013, 2019B1515120010, 2019A101002015,
2019KTSCX010, 2018KTSCX016, 2021A1515011873, in part by
the Project of Guangdong Key Lab of Traditional Chinese Medicine
Information Technology under Grant 2021B1212040007, in part by
the Project of Guangxi Key Laboratory of Trusted Software under



Grant kx202007, and in part by the High Performance Public
Computing Service Platform of Jinan University.


**Supplementary** **materials**


Supplementary data associated with this article can be found,
in the online version, at [https://github.com/leaves520/MSSGAT.](https://github.com/leaves520/MSSGAT)


**References**


[1] R.G. Hill, D. [Richards,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0001) Drug Discovery and Development E-Book: Technology in
Transition, Elsevier Health Sciences, 2021.

[2] A. [Varnek,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0002) I. [Baskin,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0002) Machine learning methods for property prediction in
chemoinformatics: quo vadis? J. Chem. Inf. Model. 52 (6) (2012) 1413–1437.

[3] Z. [Wu,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003) B. [Ramsundar,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003) E.N. [Feinberg,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003) J. [Gomes,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003) C. [Geniesse,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003) A.S. [Pappu,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003)
K. [Leswing,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003) V. [Pande,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0003) Moleculenet: a benchmark for molecular machine learning, Chem. Sci. 9 (2) (2018) 513–530.

[4] T. [Ching,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) D.S. [Himmelstein,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) B.K. [Beaulieu-Jones,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) A.A. [Kalinin,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) B.T. Do, G.P. [Way,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004)
E. [Ferrero,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) P.-M. [Agapow,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) M. [Zietz,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) M.M. [Hoffman, Opportunities](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) and obstacles
for deep learning in biology and [medicine,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0004) J. R. Soc. Interface 15 (141) (2018)
20170387.

[5] H. [Chen,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0005) O. [Engkvist,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0005) Y. [Wang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0005) M. [Olivecrona,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0005) T. [Blaschke,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0005) The rise of deep
learning in drug discovery, Drug Discov. Today 23 (6) (2018) 1241–1250.



12


_X.-b._ _Ye,_ _Q._ _Guan,_ _W._ _Luo_ _et_ _al._ _Pattern_ _Recognition_ _128_ _(2022)_ _108659_




[6] D. [Weininger,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0006) SMILES, a chemical language and information system. 1. Introduction to methodology and [encoding](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0006) rules, J. Chem. Inf. Comput. Sci. 28 (1)
(1988) 31–36.

[7] M. [Sun,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0007) S. [Zhao,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0007) C. [Gilvary,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0007) O. [Elemento,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0007) J. [Zhou,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0007) F. [Wang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0007) Graph convolutional
networks for computational drug [development](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0007) and discovery, Brief. Bioinform.
21 (3) (2020) 919–935.

[8] M. [Niepert,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0008) M. [Ahmed,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0008) K. [Kutzkov,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0008) Learning convolutional neural networks
for graphs, in: Proceedings of The [33rd](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0008) International Conference on Machine
Learning, vol. 48, 2016, pp. 2014–2023.

[9] D. [Duvenaud,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0009) D. [Maclaurin,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0009) J. [Aguilera-Iparraguirre,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0009) R. [Gómez-Bombarelli,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0009)
T. [Hirzel,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0009) A. [Aspuru-Guzik,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0009) R.P. [Adams,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0009) Convolutional networks on graphs for
learning molecular fingerprints, in: Proceedings of the 28th International Conference on Neural Information Processing Systems, vol. 2, 2015, pp. 2224–2232.

[10] S. [Kearnes,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0010) K. [McCloskey,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0010) M. [Berndl,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0010) V. [Pande,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0010) P. [Riley,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0010) Molecular graph convolutions: moving beyond [fingerprints,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0010) J. Computer-Aided Mol. Des. 30 (8) (2016)
595–608.

[11] Q. Zhang, J. Chang, G. Meng, S. Xu, S. Xiang, C. Pan, Learning graph structure
via graph convolutional networks, Pattern Recognit. 95 (2019) 308–318, doi:10.
1016/j.patcog.2019.06.012.

[12] J. Li, D. Cai, X. He, Learning graph-level representation for drug discovery, arXiv
preprint [arXiv:1709.03741(2017).](http://arxiv.org/abs/1709.03741)

[13] J. Ding, R. Cheng, J. Song, X. Zhang, L. Jiao, J. Wu, Graph label prediction based
on local structure characteristics representation, Pattern Recognit. 125 (2022)
108525, [doi:10.1016/j.patcog.2022.108525.](https://doi.org/10.1016/j.patcog.2022.108525)

[14] X. Fan, M. Gong, Y. Xie, F. Jiang, H. Li, Structured self-attention architecture
for graph-level representation learning, Pattern Recognit. 100 (2020) 107084,
[doi:10.1016/j.patcog.2019.107084.](https://doi.org/10.1016/j.patcog.2019.107084)

[15] W. Jin, R. [Barzilay,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0015) T. [Jaakkola, Junction](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0015) tree variational autoencoder for molecular graph generation, in: [Proceedings](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0015) of the 35th International Conference on
Machine Learning, vol. 80, 2018, pp. 2323–2332.

[16] D. [Rogers,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0016) M. [Hahn,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0016) Extended-connectivity fingerprints, J. Chem. Inf. Model. 50
(5) (2010) 742–754.

[17] J. Chung, C. Gulcehre, K. Cho, Y. Bengio, Empirical evaluation of gated recurrent
neural networks on sequence modeling, [1412.3555(2014).](http://arxiv.org/abs/1412.3555)

[18] A. [Gaulton,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) L.J. [Bellis,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) A.P. [Bento,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) J. [Chambers,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) M. [Davies,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) A. [Hersey,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) Y. [Light,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018)
S. [McGlinchey,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) D. [Michalovich,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) B. [Al-Lazikani,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) J.P. [Overington,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0018) Chembl: a
large-scale bioactivity database for drug discovery, Nucleic Acids Res. 40 (D1)
(2011) D1100–D1107.

[19] J. [Ma,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0019) R.P. [Sheridan,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0019) A. [Liaw,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0019) G.E. Dahl, V. [Svetnik, Deep](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0019) neural nets as a
method for quantitative [structure–activity](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0019) relationships, J. Chem. Inf. Model.
55 (2) (2015) 263–274.

[20] J. [You,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0020) R.D. [McLeod,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0020) P. [Hu,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0020) Predicting drug-target interaction network using
deep learning model, Comput. Biol. Chem. 80 (2019) 90–101.

[21] X. Li, Y. [Xu,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0021) L. [Lai,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0021) J. [Pei,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0021) Prediction of human cytochrome p450 inhibition using a multitask deep autoencoder [neural](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0021) network, Mol. Pharm. 15 (10) (2018)
4336–4345.

[22] I.J. [Goodfellow,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022) J. [Pouget-Abadie,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022) M. [Mirza,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022) B. Xu, D. [Warde-Farley,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022) S. [Ozair,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022)
A. [Courville,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022) Y. [Bengio,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022) Generative adversarial nets, in: Proceedings of the
27th International Conference on [Neural](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0022) Information Processing Systems, in:
NIPS’14, vol. 2, 2014, pp. 2672–2680.

[23] A. [Kadurin,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023) A. [Aliper,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023) A. [Kazennov,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023) P. [Mamoshina,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023) Q. [Vanhaelen,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023) K. [Khrabrov,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023)
A. [Zhavoronkov,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023) The cornucopia of meaningful leads: applying deep adversarial autoencoders for new molecule [development](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0023) in oncology, Oncotarget 8 (7)
(2017) 10883.

[24] C. [Shang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0024) Q. [Liu,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0024) Q. [Tong,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0024) J. [Sun,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0024) M. [Song,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0024) J. Bi, Multi-view spectral graph convolution with consistent edge [attention](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0024) for molecular modeling, Neurocomputing 445 (2021) 12–25.

[25] Z. [Zhang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0025) J. [Guan,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0025) S. [Zhou,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0025) FraGAT: a fragment-oriented multi-scale graph attention model for molecular [property](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0025) prediction, Bioinformatics 37 (18) (2021)
2981–2987.

[26] S. [Wang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0026) Z. Li, S. [Zhang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0026) M. [Jiang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0026) X. [Wang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0026) Z. [Wei,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0026) Molecular property prediction based on a multichannel [substructure](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0026) graph, IEEE Access 8 (2020)
18601–18614.

[27] E.B. [Lenselink,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) N. [Ten](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) Dijke, B. [Bongers,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) G. [Papadatos,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) H.W. [Van](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) Vlijmen,
W. [Kowalczyk,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) A.P. [IJzerman,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) G.J. Van Westen, Beyond the hype: deep neural
networks outperform established [methods](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0027) using a chembl bioactivity benchmark set, J. Cheminform. 9 (1) (2017) 1–14.

[28] G. Landrum, et al., Rdkit: Open-source cheminformatics software, 3(04) (2006)
2012. [http://www.rdkit.org.](http://www.rdkit.org)

[29] M. [Rarey,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0029) J.S. [Dixon,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0029) Feature trees: a new molecular similarity measure based
on tree matching, J. Computer-Aided Mol. Des. 12 (1998) 471–490.

[30] J. [McAuley,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0030) T. [Caetano,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0030) Exploiting within-clique factorizations in junction-tree
algorithms, in: Proceedings of the [Thirteenth](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0030) International Conference on Artificial Intelligence and Statistics, vol. 9, 2010, pp. 525–532.




[31] R.E. [Carhart,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0031) D.H. [Smith,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0031) R. [Venkataraghavan,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0031) Atom pairs as molecular features
in structure-activity studies: definition [and](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0031) applications, J. Chem. Inf. Comput.
Sci. 25 (2) (1985) 64–73.

[32] R. [Nilakantan,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0032) N. [Bauman,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0032) J.S. [Dixon,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0032) R. [Venkataraghavan,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0032) Topological torsion:
a new molecular descriptor for SAR [applications.](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0032) comparison with other descriptors, J. Chem. Inf. Comput. Sci. 27 (2) (1987) 82–85.

[33] T.N. Kipf, M. [Welling,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0033) Semi-supervised classification with graph convolutional
networks, in: International Conference [on](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0033) Learning Representations (ICLR),
2017.

[34] P. [Veliˇckovi´c,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0034) G. [Cucurull,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0034) A. [Casanova,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0034) A. [Romero,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0034) P. [Liò,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0034) Y. [Bengio,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0034) Graph
attention networks, in: International [Conference](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0034) on Learning Representations
(ICLR), 2018.

[35] S. [Hochreiter,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0035) J. [Schmidhuber,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0035) Long short-term memory, Neural Comput. 9 (8)
(1997) 1735–1780.

[36] Z. [Xiong,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) D. [Wang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) X. [Liu,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) F. [Zhong,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) X. [Wan,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) X. Li, Z. Li, X. [Luo,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) K. [Chen,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) H. [Jiang,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036)
M. [Zheng,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) Pushing the boundaries of molecular representation for drug discovery with the graph attention [mechanism,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0036) J. Med. Chem. 63 (16) (2019)
8749–8760.

[37] J. [Gilmer,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0037) S.S. [Schoenholz,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0037) P.F. [Riley,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0037) O. [Vinyals,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0037) G.E. [Dahl,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0037) Neural message passing for quantum chemistry, in: [International](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0037) Conference on Machine Learning,
PMLR, 2017, pp. 1263–1272.

[38] X. [Glorot,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0038) Y. [Bengio,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0038) Understanding the difficulty of training deep feedforward
neural networks, in: Proceedings of the 13th International Conference on Artificial Intelligence and Statistics, JMLR Workshop and Conference Proceedings,
2010, pp. 249–256.

[39] D.P. [Kingma,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0039) J.L. Ba, Adam: A method for stochastic gradient descent, in: International Conference on Learning Representations, 2015, pp. 1–15.

[40] L. [McInnes,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0040) J. [Healy,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0040) N. [Saul,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0040) L. [Grossberger,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0040) Umap: uniform manifold approximation and projection, J. Open Source Softw. 3 (29) (2018) 861.

[41] P.J. [Rousseeuw,](http://refhub.elsevier.com/S0031-3203(22)00140-6/sbref0041) Silhouettes: a graphical aid to the interpretation and validation
of cluster analysis, J. Comput. Appl. Math. 20 (1987) 53–65.


**Xian-Bin** **Ye** received the B.Sc in Pharmaceutical Engineering from the School of
Pharmacy, Guangdong Pharmaceutical University Guangzhou, China in 2018. He is
currently a postgraduate student in College of Information Science and Technology,
Jinan University, Guangzhou, China. His research interests focus on graph neural
network, drug discovery, artificial intelligence and bioinformatics.


**Quanlong** **Guan** is Professor in the faculty of computer science and core member
of the research institute for Guangdong intelligent education at Jinan University,
China. He is directing the Guangdong R&D Institute for the big data of service and
application on education. His research interests include the application of artificial
intelligence, information technology in education, data protection and processing.


**Weiqi** **Luo** received his B.S. degree from Jinan University in 1982 and Ph.D. degree
from South China University of Technology in 2000. Currently, he is a professor
with School of Information Science and Technology in Jinan University, Guangzhou.
His research interests include net work security, big data, artificial intelligence, etc.
He has published more than 100 high-quality papers in international journals and
conferences


**Liangda** **Fang** received the Ph.D. degree from Sun Yat-sen University, Guangzhou, in
computer science, in 2015. He is currently an Assistant Professor with the Department of Computer Science, Jinan University, Guangzhou. His current research interests include artificial intelligence, knowledge representation and reasoning, and
automated planning.


**Zhao-Rong** **Lai** received the B.Sc. in mathematics, M.Sc. in computational science,
Ph.D. in statistics, all from the School of Mathematics, Sun Yat-Sen University,
Guangzhou, China, in 2010, 2012 and 2015,respectively. He is currently an Associate Professor with the Department of Mathematics, Jinan University, Guangzhou,
China. He was an invited Senior Program Committee member of IJCAI 2021 (Session Chair as well) and IJCAI 2020. His research interests include machine learning,
image processing, multivariate statistics, and portfolio optimization.


**Jun** **Wang** received his Ph.D. from Peking University in 2016. He was a Visiting
Scholar in ETH Zurich in 2015. He was working as an engineer at the Institute of
Software, Chinese Academy of Sciences in 2011–2012. He joined IBM Research China
as a Research Scientist during 2016–2018. Since 2018, he is a Senior algorithm researcher in PingAn Technology. His research interests include Deep Learning, Medical Imaging.



13


