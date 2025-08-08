_Briefings in Bioinformatics,_ 23(1), 2022,1–12


**[https://doi.org/10.1093/bib/bbab346](https://doi.org/10.1093/bib/bbab346)**
Problem Solving Protocol

# **Drug–target interaction predication via multi-channel** **graph neural networks**

## Yang Li, Guanyu Qiao, Keqi Wang and Guohua Wang


Corresponding author: Guohua Wang, College of information and Computer Engineering, Northeast Forestry University, 150004, Harbin, China.
E-mail: ghwang@nefu.edu.cn


Abstract


Drug–target interaction (DTI) is an important step in drug discovery. Although there are many methods for predicting drug
targets, these methods have limitations in using discrete or manual feature representations. In recent years, deep learning
methods have been used to predict DTIs to improve these defects. However, most of the existing deep learning methods lack
the fusion of topological structure and semantic information in DPP representation learning process. Besides, when learning
the DPP node representation in the DPP network, the different influences between neighboring nodes are ignored. In this
paper, a new model DTI-MGNN based on multi-channel graph convolutional network and graph attention is proposed for
DTI prediction. We use two independent graph attention networks to learn the different interactions between nodes for the
topology graph and feature graph with different strengths. At the same time, we use a graph convolutional network with
shared weight matrices to learn the common information of the two graphs. The DTI-MGNN model combines topological
structure and semantic features to improve the representation learning ability of DPPs, and obtain the state-of-the-art
results on public datasets. Specifically, DTI-MGNN has achieved a high accuracy in identifying DTIs (the area under the
receiver operating characteristic curve is 0.9665).


**Key words:** drug–target interaction; biologic network; graph neural network; graph attention network.



Introduction


Drug targets are able to combine with drugs and play a special
role in intracellular molecules [7, 30]. Identifying interactions
between known drugs and targets is essential in medicine field,
which can facilitate the discovery and reposition of drugs. However, it is time consuming and expensive to identify drug–target
interactions (DTIs) via _in vitro_, _in vivo_ experiments. Thus, more
and more researchers are exploring computational approaches
to predict DTIs, which can not only reduce the loss in the process
of drug discovery, but also provide guidance to the research



of drug relocation, polypharmacology, drug resistance [43] and
side-effects prediction [23].


An intuitive way to identify new targets for a drug is to compare the candidate proteins with those existing targets of that
drug. Generally speaking, different data or different prediction
methods will affect the prediction performance of DTI. Different
results may be obtained depending on which perspective the
comparison is made with respect to [20]. At present, the methods
of DTI prediction can be summarized into three categories:
structure-based [21, 24, 28], ligand-based methods and machine
learning based methods [26, 40].



**Yang Li** is an associate professor at the College of Information and Computer Engineering, Northeast Forestry University. Her main research interests
include natural language processing, machine learning and bioinformatics.
**Guanyu Qiao** is a master student at the College of Information and Computer Engineering, Northeast Forestry University. His main research interests
include machine learning and computational biology.
**Keqi Wang** is a master student at the College of Information and Computer Engineering, Northeast Forestry University. Her main research interests include
machine learning and computational biology.
**Guohua Wang** is a professor at the College of Information and Computer Engineering, Northeast Forestry University. He is also a Principal Investigator at
Key Laboratory of Tree Genetics and Breeding, Northeast Forestry University. His research interests are bioinformatics, machine learning and algorithms.
**Submitted:** 1 June 2021; **Received (in revised form):** 21 July 2021


© The Author(s) 2021. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


1


2 _Li et al._


In the early days, most prediction methods for DTIs are based
on the structure of drugs and protein targets [3]. Faulon _et al_
used support vector machine (SVM) [36] to predict the probability
of DTIs based on both the chemical (drug chemical structure)
and genomic (protein structure) features. Shaikh _et al_ improved
protein chemometrics (PCM) to predict DTIs by designing a new
method of reliable negative data set generation and fingerprint
based applicability domain analysis. Then the predicted molecular recognition interactions between drug and target pairs were
quantified by reverse molecular docking method. Meng _et al_ proposed a computational method based on protein sequence. The
model combines Bi-gram probabilities (BIGP), Position-Specific
Scoring Matrix (PSSM) and Principal Component Analysis (PCA)

[10] with Relevance Vector Machine (RVM). However, for proteins with unknown structure, the return of structure-based
prediction method is small.
The methods based on ligand similarity utilized existing
active small molecule structures to establish pharmacophore
model or quantitative structure–activity relationships (QSARs).
QSAR is the most common used ligand method, which assumes
that molecules with similar structures have similar biological
activities. Unlike most QSAR models that only predict the activity of one protein target, González-Díaz _et al_ proposed a multitarget QSAR classifier to predict the DTIs and achieved a good
result. Nevertheless, the ligand-based approach is ineffective
when only a few ligands are known to bind to the target.
In addition to the widely used algorithm such as random
forest (RF) [17], Bernoulli Naïve Bayesian (BNB), Decision Tree
(DT) and SVM, many advanced machine learning based methods have been proposed for DTI prediction. Yamanishi _et al_
developed a bipartite graph model where the chemical and
genomic spaces as well as the DPI network are integrated into a
pharmacological space. Zheng _et al_ proposed a semi-supervised
learning method – Laplacian regularized least square (LapRLS) to
utilize both the small amount of available labeled data and the

abundant unlabeled data together in order to give the maximum
generalization ability from the chemical and genomic spaces.
At the same time, pharmacological or phenotypic information,
such as side-effects [2, 22], transcriptional response data [9],
drug–disease associations [38], public gene expression data [29]
and functional data have been incorporated in DTIs to provide
diverse information and a multi-view perspective for predicting
novel DTIs. Luo _et al_ proposed a network integration pipeline
DTINet to integrate heterogeneous data sources (e.g., drugs,
proteins, diseases and side-effects) to extract low-dimensional
features from drug and protein heterogeneous networks, and
then an inductive matrix was used to predict the relationship of
protein and drug [19]. Subsequently, Zeng _et al_ developed a deep
learning methodology deepDTnet for new target identification
and drug repositioning in a heterogeneous drug–gene–disease
network embedding 15 types of chemical, genomic, phenotypic
and cellular network profiles. After learning the feature matrix
for drugs and targets, deepDTnet applies PU-matrix completion
to find the best projection from the drug space onto target
(protein) space and infers new targets for a drug.
Recently, with the great success of deep learning in various fields of bioinformatics, network representation learning
approaches that can learn rich topological information and the
complex interaction relationship of heterogeneous data have
been gradually used in DTI prediction[26, 33, 46, 47]. Peng _et al_
proposed a learning-based method DTI-CNN for DTI prediction
that learn low-dimensional vector representations of features
from heterogeneous networks. DTI-CNN takes the concatenate
representation vector of drug and protein as input, and adopts



convolution neural networks (CNNs) as the classification model.
Nevertheless, DTI-CNN ignores the interactions between drug
and protein pairs (DPPs) in the modeling and learning process.
To incorporate the associations between DPPs into DTI modeling,
Zhao _et al_ built a DPP network based on multiple drugs and proteins in which DPPs are the nodes and the associations between

DPPs are the edges of the network. Then they proposed a model
GCN-DTI for DTI identification. The model first used a graph
convolutional network (GCN) [12] to learn the representation for
each DPP, and applied a deep neural network to predict the final
label based on the representation.
Recently, Graph Attention Network (GAT) [37] has been used
to increase the interpretability of graph neural network. Cheng
_et al_ used multi-attention mechanism to extract amino acid

sequence features of protein and GAT to extract the chemical
structure of drugs in the task of DTI prediction. They solve the
problem that CNN cannot get context-sensitive information, and
avoid the possible influence of noise connection on nodes in the
graph. Li _et al_ proposed a two-level neural attention mechanism
approach IMCHGAN to learn drug and target latent feature representations from the heterogeneous network, respectively [16].
GAT is leveraged to learn drugs (or targets) latent feature representations for a specific metapath. Then the attention mechanism is further employed to integrate different meta-path latent
representations into final latent features. Finally, the learned
latent features are fed into the Inductive Matrix Completion
(IMC) prediction score model and output DTI score via the inner
product of projected drug and target feature representations.
This work also studies the representation of drug and target
separately, without considering the interaction between them
during the graph learning process.
Above all, previous studies mostly focused on learning the
topological structures of DPP networks, while ignoring the fusion
of topological structures and semantic features in the process of
DPP representation learning. At the same time, when learning
the DPP node representation in the DPPs network, the different
influences between neighboring nodes are ignored.
To tackle the above challenges, we propose a Multichannel Graph Neural Network [37] (DTI-MGNN) for DTI
prediction. Inspired by previous work, DTI-MGNN first learns
low-dimensional representations of drugs and proteins from
heterogeneous networks, and constructs a topology graph and
a feature graph for DPPs, respectively. Then a multi-channel
graph neural network is used to learn the representation of
drug–protein pairs and finally the DTIs are identified through a
Multilayer Perception (MLP) neural network. On the one hand, we
use two independent GATs [37] to learn the different interactions
between nodes for the topology graph and feature graph. On
the other hand, we use GCNs [14] with shared weight matrices
to learn the common information of the two graphs. In this
way, DTI-MGNN can enhance the capability of fusing topological
structures and semantic features at the same time. Experiments
on public datasets show that our model outperforms the
state-of-the-art methods significantly. Specifically, DTI-MGNN
achieves the best results of 0.9665 and 0.9683 in AUROC and

AUPR, respectively. Furthermore, the introduction of attention
mechanism also provides interpretability for DTI prediction.


Method


In this section, we will introduce our proposed model DTI-MGNN
for DTI prediction in detail. The overall model is shown in
Figure 1, which consists of four parts, namely drug and protein


_Drug–target interaction predication_ 3


**Figure 1.** Flow chart of Multi-channel Graph Neural Network (DTI-MGNN) for DTI prediction. The model is composed of four parts, namely Drug and Protein

Representation Learning (a), DPP Network Construction (b), DPP Network Representation Learning (c) and DTI Prediction (d). First, five heterogeneous networks and two
similarity networks are integrated by Jaccard similarity coefficient and RWR algorithm, and then DAE is used to denoise and reduce the dimension of the integrated
information. Second, the topology graph and feature graph are constructed by the obtained drug and protein information. Then, the DTI-MGNN model is used to obtain
the final representation _Z_ of DPP from multiple channels. Finally, an MLP model is used to predict the DPP score.



representation learning, DPP network construction, DPP network representation learning and DTI prediction. First, we learn
the representation of drugs and proteins in their independent
spaces from heterogeneous networks that characterize the interactions among proteins, drugs, side-effects and diseases. Then,
we obtain the feature vectors of DPPs by combining the feature
vectors of drugs and proteins, and the topology graph and feature graph of DPPs are constructed. For each DPP, DTI-MGNN
aims to learn its representation vector consisting of three parts,
the common representation learned from a GCN, the topology
and feature representation learned through two independent
graph attention networks. Finally, we train an MLP model, the
input of which is a DPP representation vector, and the output
is whether there is a relationship between the drug and the
protein.
The following of this section will provide detailed descriptions of the four parts, respectively.


**Drug and Protein Representation Learning**


The aim of _Drug and Protein Representation Learning_ step is to
learn low-dimensional but informative vector representations



of features for both drugs and proteins. Inspired by previous
work [19], we integrate diverse information from heterogeneous
data sources, including four drug-related networks (drug–drug
relationship network **M** _drug-drug_, drug-related disease network
**M** _drug-disease_, drug and side-effect network **M** _drug-effect_, drug-chemical
structure similarity network **H** _drug-chemical_ ) and three proteinrelated networks (protein-related disease network **M** _protein-disease_,
protein–protein relationship network **M** _protein-protein_ and proteinsequence similarity network **H** _protein-sequence_ ). Details of the dataset
information will be introduced in Section 3.1.


_**Network Integration**_


Among the five heterogeneous data sources and two similarity
networks for drugs and proteins mentioned above, five of heterogeneous data sources ( **M** _drug-drug_, **M** _drug-disease_, **M** _drug-effect_, **M** _protein-disease_
and **M** _protein-protein_ ) are binary matrices, of which each entry equals
0 or 1. For example, in matrix **M** _drug-disease_, each element _M_ [ _i_ ][ _j_ ]
is defined as the associated score between drug _i_ and disease
_j_ if there is a link between drug _i_ and disease _j_ in the network,
otherwise _M_ [ _i_ ][ _j_ ] = 0.


4 _Li et al._


To integrate the discrete and sparse heterogeneous biological
data sources, we use Jaccard similarity coefficient [25] to transform each heterogeneous network into a homogeneous network
of drugs and proteins, respectively. Jaccard similarity coefficient
is a statistic used for comparing the similarity and diversity of
sample sets. The Jaccard coefficient is defined as the size of the
intersection divided by the size of the union of the sample sets.
Here, we take the drug-related disease network as an example.
Given the drug-related disease network **M** _drug-disease_ ∈
R _[N]_ [drug] [×] _[N]_ [disease], where _N_ drug denotes the number of drugs and _N_ disease
denotes the number of diseases in our whole dataset. For each

element _M_ _i_, _j_ ∈ **M** _drug-disease_, it represents the relation between
drug _i_ and the corresponding disease _j_ . At the same time, each
row **M** _i_ ∈ R _[N]_ [disease] can be regarded as a feature vector of drug _i_,
where each dimension corresponds to a disease. We calculate
the Jaccard similarity between the _i-th_ row and _j-th_ row of the
matrix **M** _drug-disease_, respectively, and construct a drug similarity
matrix **H** _drug-disease_, each element _J_ ( **M** _i_, **M** _j_ ) is defined as follows:


� **M** _j_ |
_J_ ( **M** _i_, **M** _j_ ) = [|] | **[M]** **M** _[i]_ _i_ ~~�~~ **M** _j_ | (1)


where **M** _i_ and **M** _j_ represent the disease sets of drug _i_ and drug _j_
in drug-related disease network, | **M** _i_ � **M** _j_ | represents the size of
the intersection of two sets and | **M** _i_ � **M** _j_ | represents the size of
the union of two sets.

In the same way, we transform other networks **M** _drug-drug_,
**M** _drug-effect_, **M** _protein-disease_ and **M** _protein-protein_ into **H** _drug-drug_, **H** _drug-effect_,
**H** _protein-disease_ and **H** _protein-protein_, respectively.
So far, we have transformed all the drug-related networks
into _N_ drug dimensional dense square matrices and all the
protein-related networks into _N_ protein dimensional dense square
matrices, which can be viewed as similarity matrices of drugs
and proteins as shown in Figure 1(a).


_**Representation Learning**_


After obtaining the similarity matrix of heterogeneous networks,
the Random Walk algorithm with Restart (RWR) [34] is applied
to each similarity matrix, which represents a weighted network.
Starting from a certain node in the graph, each step is faced with
two choices, that is, randomly selecting neighboring nodes or
returning to the starting node. The algorithm includes parameters as a restart probability and a probability of moving to
neighboring nodes. The diffusion state of each drug or protein
on each network includes the topological relationship between
each drug or protein and all the other nodes in the network. After
several iterations and convergence, the probability distribution
can be regarded as the distribution influenced by the initial
node. The probability distributions of all nodes are combined to
form the overall representation on the graph. Compared with the
traditional method of calculating the distance of graph (such as
the shortest path, the maximum path, etc.), RWR can capture
the relationship between two nodes and the whole structure
information of the graph.
Then we concatenate the output of **H** _drug-drug_, **H** _drug-effect_,
**H** _drug-disease_ and **M** _drug-chemical_ obtained from the RWR algorithm,
to get a high-dimensional matrix for drugs **V** _drug_ ∈ R _[N]_ [drug] [×][4] _[N]_ [drug] .
Similarly, we concatenate the output of **H** _protein-disease_, **H** _protein-protein_
and **M** _protein-sequence_ obtained from the RWR algorithm, to get a
high-dimensional matrix for proteins **V** _protein_ ∈ R _[N]_ [protein] [×][3] _[N]_ [protein] .
In order to obtain low-dimensional features for drugs and
proteins, we follow Peng _et al_ ’s work, and adopt a Denoising Autoencoder (DAE) model on **V** _drug_ and **V** _protein_ . Specifically, DAE model



adds noise to the input sample **V** _drug_, automatically denoises
the noisy input through an encoder and maps it to the abstract
space **X** _drug_ . Then decodes it through the decoder, and maps the
low-dimensional data of the abstract space **X** _drug_ back to the
original space to reconstruct sample **V** [′] _drug_ . Through DAE, we
obtain the drug representation **X** _drug_ ∈ R _[N]_ [drug] [×] _[d]_ [drug] and the protein
representation **X** _protein_ ∈ R _[N]_ [protein] [×] _[d]_ [protein], as shown in Figure 2.


**DPP Network Construction**


In order to capture the deep and comprehensive relationships
between drugs and proteins, we combine each drug _p_ and protein
_q_ to form a drug and protein pair DPP _i_ . The representation of DPP
node _i_ can be represented as **X** _[i]_ _DPP_ [= {] **[X]** _pdrug_ [;] **[ X]** _qprotein_ [}][, which is the]
concatenation of the drug representation **X** _pdrug_ [and the protein]
representation **X** _qprotein_ [.] **[ X]** _[DPP]_ ∈ R _[N]_ [DPP] [×] _[d]_ [DPP] is the representation
matrix of all the DPP nodes. Inspired by a recent study [39], we
learn the DPP information based on both the topological space
propagation and feature space propagation [39]. Therefore, we
try to construct topology network and feature network for the
DPPs. The former models the structural information of DPPs,

while the latter models the semantic information of DPPs.


_**Topology Graph Construction**_


First, we build the topology graph **G** **t** = ( **A** **t**, **X** _DPP_ ), which can
reflect the structural information between DPPs. We follow the

principle that if two DPPs contain a common drug or a common
protein, there is an edge between them. The adjacency matrix **A** **t**
of the DPPs network:



_f_ ( _DPP_ 1, _DPP_ 1 ) - · · _f_ ( _DPP_ 1, _DPP_ _n_ )
_f_ ( _DPP_ 2, _DPP_ 1 ) - · · _f_ ( _DPP_ 2, _DPP_ _n_ )

... ...                - · · ...

_f_ ( _DPP_ _n_, _DPP_ 1 ) - · · _f_ ( _DPP_ _n_, _DPP_ _n_ )



⎤

= � _DPP_ _ij_ � (2)
⎥⎥⎥⎥⎦



**A** **t** =



⎡

⎢⎢⎢⎢⎣



where the adjacency matrix **A** **t** ∈ R _[N]_ [DPP] [×] _[N]_ [DPP] represents the
relationship of edges between nodes in a graph. The value of
_f_ ( _DPP_ _i_, _DPP_ _j_ ) equals to 1, meaning that the two DPPs share some
common features, and vice versa.


_**Feature Graph Construction**_


We construct the feature graph of DPPs based on the lowdimensional representation matrix of DPPs ( **X** _DPP_ ) from heterogeneous data sources, which can fully express the characteristics
of drugs and proteins.
According to the idea of _K_ -nearest neighbor (KNN) [12], for
each DPP _i_, we calculate the cosine similarity of the representation between it and other DPPs, and then select the top _K_
nearest DPP nodes as its adjacent nodes [1] . For example, if DPP
_j_ and DPP _k_ are adjacent nodes of DPP _i_, then we can define in
the adjacent matrix **A** **f**, the element in _i_ -th row and _j_ -th column
and the element in _i_ -th row and _k_ -th column are 1. Except that
the diagonal element of the matrix are 1, all other cases are
0. Therefore, we can construct a feature graph of DPPs **G** **f** =
( **A** **f**, **X** _DPP_ ), where **A** **f** is the adjacency matrix.


**DPP Network Representation Learning**


After establishing the topology graph **G** **t** = ( **A** **t**, **X** _DPP_ ) and the
feature graph **G** **f** = ( **A** **f**, **X** _DPP_ ) of DPPs, we use multi-channel graph
neural networks to learn the representation of DPPs. Specifically,
we use two parallel GATs to model the topology graph and the


_Drug–target interaction predication_ 5


**Figure 2.** In the flow chart of DAE, by adding noise to the original input **V** _drug_ or **V** _protein_, the encoder denoises the input to obtain the low-dimensional and low-noise

data, and maps the input to representation space **X** _drug_ ∈ R _[N]_ [drug] [×] _[d]_ [drug] or **X** _protein_ ∈ R _[N]_ [protein] [×] _[d]_ [protein] . In this work, we set _d_ drug = 100 and _d_ protein = 400. Finally, the
decoder can restore data from feature space to the original input **V** [′] _drug_ or **V** [′] _protein_ .



feature graph, and learn contextual representation and semantic
representation of DPPs. At the same time, we use a GCN to learn
the common representation of DPPs.


_**Topology and Feature Graph Modeling with GAT**_


The idea of GAT is to compute the hidden representations of each
node in the graph, by attending over its neighbors, following a
self-attention strategy.
Given the topology graph **G** **t** = ( **A** **t**, **X** _DPP_ ) as input, for each DPP
node _i_, GAT calculates the similarity coefficient _e_ _ij_ between node
_i_ and its neighbor node _j_ ( _j_ ∈ _N_ _i_ ) as follows:


_e_ _ij_ = **u** ([ **W** _[i]_ **t** **[X]** _[i]_ [||] **[W]** _j_ **t** **[X]** _[j]_ []][),] _[ j]_ [ ∈] _[N]_ _[i]_ (3)


where **W** ( **t** ∗) ∈ R _[d]_ [h] [×] _[d]_ [DPP] is a learnable weight matrix, **u** is a 2 _d_ h
dimensional transformation vector and _d_ h represents the hidden
dimension of GAT, and _d_ DPP denotes the dimension of DPP vectors
as the input of GAT. **X** (·) is a _d_ DPP -dimensional input vector of the
node (·), _N_ _i_ is the neighborhood set of the central node _i_ and ·||·
is the concatenation of the transformed features of the node _i_

and _j_ .
Then the obtained coefficients are normalized by softmax
function. Hence, the attention coefficients between nodes can

be defined as follows:



head ensemble representation **Z** _[i]_ **t** [of node] _[ i]_ [:]



**Z** _[i]_ **t** [=] _k_ = ∥ _K_ 1 _σ_ ( _j_ � ∈ _N_ _i_ _α_ _ij_ ( _k_ ) **[W]** ( **t** _k_ ) **[X]** _[j]_ [)] (6)



Similarly, we can obtain the _K_ head ensemble representation
**Z** _[i]_ **f** [of node] _[ i]_ [ from the feature graph] **[ G]** _[f]_ [ =][ (] **[A]** _[f]_ [,] **[ X]** _[DPP]_ [) of DPPs:]



**Z** _[i]_ **f** [=] _k_ = ∥ _K_ 1 _σ_ ( _j_ � ∈ _N_ _i_ _α_ _ij_ ( _k_ ) **[W]** ( **f** _k_ ) **[X]** _[j]_ [)] (7)



_exp_ ( _LeakyReLU_ ( _e_ _ij_ ))
_α_ _ij_ = _softmax_ ( _e_ _ij_ ) = ~~�~~ _k_ ∈ _N_ _i_ _[exp]_ [(] _[LeakyReLU]_ [(] _[e]_ _[ik]_ [))] (4)



**h** _i_ = _σ_ (� _α_ _ij_ **W** **t** **X** _j_ ) (5)

_j_ ∈ _N_ _i_



Through the above process, we learn the contextual representation and semantic representation of DPPs from topology graph
and feature graph by two parallel GAT models, and obtain the
corresponding representation matrices **Z** **t** and **Z** **f** for DPP nodes,
respectively.


_**Common Graph Modeling with GCN**_


Intuitively, besides the ‘domain-specific’ representations
learned in specific networks, DPPs should also have shared
common-sense representations. Therefore, in addition to the
independent learning of topological network and feature
network representation ( **Z** **t** and **Z** **f** ), we also use a GCN with
shared weights to learn the common feature representation of

DPPs.

The reason we use the GCN with shared weights is that the
parameter-sharing convolution module can learn the common
features of two similar spaces. Let **Z** **ct** and **Z** **cf** denote the common
representation learned from the topology graph and from the
feature graph, respectively; the output of the _l-th_ layer of GCN
model is as follows. **topology space:**



− **t** 2 **Z** ( **ctl** − **1** ) **W** ( _cl_ ) [)] (8)



where **h** _i_ is the weighted and aggregated hidden representations
for all the neighbor nodes of _i_ learned from GAT. In this way, GAT
can learn not only all the information of neighboring nodes, but
also the importance of different neighboring nodes.
In order to enhance the performance, we use the multi-head
attention mechanism to learn different meanings of DPP nodes
from different representation sub-spaces, and finally get the _K_



**Z** ( **ctl** ) [=] _[ ReLU]_ [(] **[D]** [�] − **t** [1] 2



− [1] � � − [1]

**t** 2 **A** **t** **D** **t** 2



**feature space:**



**Z** ( **cfl** ) [=] _[ ReLU]_ [(] **[D]** [�] − **f** [1] 2



− [1] � � − [1]

**f** 2 **A** **f** **D** **f** 2



− **f** 2 **Z** ( **cfl** − **1** ) **W** ( _cl_ ) [)] (9)


6 _Li et al._


where **A** [�] (·) = **A** (·) + **I**, we add an identity matrix **I** to the adjacency
matrix **A** to indicate the node itself, and **D** [�] (·) is the diagonal degree

matrix of **A** [�] (·) . **W** ( _cl_ ) [is the shared weight matrix of] _[ l-th]_ [ layer of GCN.]

We define the input of the first layer **Z** ( **c0** (·)) [=] **[ X]** _[DPP]_ [, and the]
output of GCN learned in the topology space and the feature
space is the common embedding **Z** **ct** and **Z** **cf** . Finally, we take
the average of the two representations to obtain the common
representation of DPP nodes, which is denoted as **Z** **c** :



**Z** **c** = **[Z]** **[cf]** [ +] **[ Z]** **[ct]** (10)

2



Experiments


In this section, we perform experiments to answer the following
research questions: RQ1. Is it feasible and effective to predict
the DTIs based on the proposed Multi-channel graph neural
network? RQ2. Is it useful to integrate multi-channel network
representation learning into the framework? If so, which one
is more effective? RQ3. Can attention mechanism improve the
representation of DPPs and provide meaningful interpretation of
the final results?


**Datasets**


Inspired by the previous work [19], we construct our datasets as
follows. First, we extract the drug-related information from the
DrugBank database (Version 3.0) [13], including the interactions
between drugs and known drug–target interactions. The known
drug–target interactions are used as our ground-truth data.
Second, we also introduce heterogeneous networks from
multiple resources. The interactions between proteins are
obtained from the human protein reference database HPRD

[11]. We obtain diseases information from the Comparative
Toxicogenomics Database, including the relationship between
diseases and drugs, and the relationship between diseases and
proteins [5]. The side-effects of drugs are from the Side-Effect
Resource SIDER database [15].
Finally, there are four types of nodes and eight types of
interactions constructing totally six heterogeneous information
and two similarity information, which covers 12 015 nodes and
1894 854 edges in total. Among the heterogeneous networks,
Drug–Protein network is used as the ground-truth data, while
the Drug–Drug network, Drug–Disease network, Drug-Side-effect
network, Protein-Disease network and Protein–Protein network
are used to construct the DPP networks and learn DPP representations.

At the same time, we also introduce the similarity information of protein sequence and chemical structure of drugs [38].
The protein sequence can be downloaded from the integrated
medicinal genomic database of Sophic [38], and then the Smith
Waterman algorithm [41] is used to calculate the target similarity. The chemical structures of all drug compounds were downloaded from Drugbank [13]. Then the chemical development kit

[32] and Tanimoto score were used to calculate the chemical
structure similarity. The details of all data sets are summarized
in Table 1.


**Experimental Settings**


_**Parameter settings**_


We refer our method to DTI-MGNN, which is a multi-channel
graph neural network. In this work, we use three different channels to learn different spatial representations of DPP. The three
channels consist of two parallel GATs for DPP topology and
feature graph and one GCN to learn the common representation
of DPPs. All the networks in the three channels have two layers.
The number of nodes in the first layer is 256, and the number
of nodes in the second layer is 64. We use dropout to avoid over
fitting during the training process of the model [31]. The dropout
is set to 0.2 and weight decay is 1e-3. We first randomly divide
the positive instances into 10 subsets. In each run, we use nine
subsets as training data and the 10th subset for testing. We make
sure that there is no intersection between the training set and
the test set. This is repeated 10 times and we report the average
performance.



**DTI Prediction**


In the previous sections, we learned the topological representation, feature representation and common representation of
DPP nodes by using a framework of multi-channel graph neural
networks. To further enhance the interpretability of the model,
we used an attention mechanism to fuse all the representations.
For a DPP node _i_, the three representation of node _i_ can be
denoted by **Z** _[i]_ **t** [,] **[ Z]** _[i]_ **f** [and] **[ Z]** **c** _[i]_ [. Let vector] _**[ α]**_ _[i]_ [ represent their attention]
vector, which can be obtained by the attention function _Att_
( **Z** **t**, **Z** **f**, **Z** **c** ):


_**α**_ _[i]_ ( _α_ **t** _[i]_ [,] _[ α]_ **f** _[i]_ [,] _[ α]_ **c** _[i]_ [)][ =] _[ Att]_ [(] **[Z]** _[i]_ **t** [,] **[ Z]** _[i]_ **f** [,] **[ Z]** **c** _[i]_ [)] (11)


where _α_ **t** _[i]_ [,] _[ α]_ **f** _[i]_ [and] _[ α]_ **c** _[i]_ [are the corresponding attention scores of] **[ Z]** _[i]_ **t** [,] **[ Z]** _[i]_ **f**
and **Z** _[i]_ **c** [.]
Next, the realization of _Att_ function is introduced. We first
compute a weight score _w_ _[i]_ **t** [for] **[ Z]** _[i]_ **t** [as follows:]


_w_ _[i]_ **t** [=] **[ q]** [⊺] [·] _[ tanh]_ [(] **[W]** [ ·][ (] **[Z]** _[i]_ **t** [)] [⊺] [+] **[ b]** [)] (12)


where **W** ∈ R _[d]_ _[l]_ [×] _[d]_ [h] is a weight matrix, **b** ∈ R _[d]_ _[l]_ [×][1] is a bias vector
and **q** ∈ R _[d]_ _[l]_ [×][1] is a shared attention vector. _d_ _l_ is the dimension
of attention layer. _d_ h is the dimension of final representation
of a DPP node learned from GAT and GCN. Through the same
operation, we can also obtain the values _w_ _[i]_ **f** [and] _[ w]_ **c** _[i]_ [of node] _[ i]_ [ for]
**Z** _[i]_ **f** [and] **[ Z]** **c** _[i]_ [.]
Then, the obtained weight values are normalized by _softmax_
function as attention scores.

Consequently, we can use the following equation to compute
_α_ **t** _[i]_ [:]


_exp_ ( _w_ _[i]_ **t** [)]
_α_ **t** _[i]_ [=] _[ softmax]_ [(] _[w]_ _[i]_ **t** [)][ =] (13)
_exp_ ( _w_ _[i]_ **t** [)][ +] _[ exp]_ [(] _[w]_ _[i]_ **f** [)][ +] _[ exp]_ [(] _[w]_ **c** _[i]_ [)]


Similarly, we can calculate _α_ **f** _[i]_ [=] _[ softmax]_ [(] _[w]_ _[i]_ **f** [) and] _[ α]_ **c** _[i]_ [=] _[ softmax]_ [(] _[w]_ _[i]_ **c** [).]
Then we can obtain the final representation of a DPP node _i_
by weighted summation of **Z** _[i]_ **t** [,] **[ Z]** _[i]_ **f** [and] **[ Z]** **c** _[i]_ [:]


**Z** _[i]_ = _α_ **t** _[i]_ [·] **[ Z]** _[i]_ **t** [+] _[ α]_ **f** _[i]_ [·] **[ Z]** _[i]_ **f** [+] _[ α]_ **c** _[i]_ [·] **[ Z]** _[i]_ **c** (14)


Finally, we model the task as a binary classification, and use
an MLP model shown in Figure 1 as our classification model,
which is composed of input layer, hidden layer and output
layer. We trained an MLP model of which the input is a DPP
representation vector **Z** _[i]_ and the output is the probability that
the drug and protein pair is related.


_Drug–target interaction predication_ 7


**Table 1.** Statistics of the data sets


Database Nodes Related nodes Relations Address


DrugBank 708 drugs 708 drugs 10 036 relations [https://go.drugbank.com/](https://go.drugbank.com/)
708 drugs 1,512 proteins 1332 relations
Comparative 5603 diseases 708 drugs 199 214 relations [http://ctdbase.com/](http://ctdbase.com/)
Toxicogenomics

5603 diseases 1512 proteins 1596 745 relations
SIDER 4192 side-effects 708 drugs 80 164 relations [http://sideeffects.embl.de/](http://sideeffects.embl.de/)
HPRD 1512 proteins 1512 proteins 7363 relations [http://www.hprd.org/](http://www.hprd.org/)



_**Baselines**_


For comparison, we use the following competitive methods for
DTI prediction as baselines:


  - **NRLMF** [18] Neighborhood Regularized Logistic Matrix Factorization. NRLMF uses matrix decomposition methods and
low-dimensional vectors to represent drugs and targets. By
studying the local structure of the interaction data and
using the neighborhood influence from the most similar
drug and the most similar target, the accuracy of DTI prediction is further improved.

  - **DTINet** [19] By collecting heterogeneous network information, the low-dimensional representation of drug and target
is obtained, and then the optimal projection from drug
space to target space is found. DTINet makes it possible to
predict the new DTIs according to the geometric proximity
of mapping vector in a unified space.

  - **DTI-CNN** [26] DTI-CNN learns low-dimensional vector
representations of features from heterogeneous networks,
and adopting CNNs as classification model. DTI-CNN
contains three components, named as heterogeneousnetwork-based feature extractor, denoising-autoencoderbased feature selector and CNN-based interaction predictor.
Different from our method, the DPP network is not
constructed in this work, and the drug and target were
directly concatenated as the input of the model.

  - **GCN-DTI** [47] GCN for DTI prediction. The DPP network
is constructed based on the known drug and target relationship, and the same weight is set between the directly
related and indirectly related DPP pairs. The neighbor node
information is gathered through GCN, and then the DNN
is used to complete the prediction task. Different from
our method, GCN-DTI does not notice that the correlation
between DPP features may affect the prediction accuracy of
DTI. Our method uses multi-channel graph neural network
to deal with the topology structure and semantic features
of DPP network.

  - **IMCHGAN** [16] Inductive Matrix Completion with Heterogeneous GAT. IMCHGAN adopts a two-level neural attention
mechanism approach to learn drug and target latent feature
representations from the DTI heterogeneous network separately, without considering the interaction between drug
and target during the graph learning process.


For all the comparison methods mentioned above, we carry
out experiments on the same data set with our model and follow
the best parameter settings in their papers.


_**Evaluation metrics**_


In this paper, we use Area Under the Receiver Operating Characteristic curve ( **AUROC)** and Area Under the Precision-Recall
curve **(AUPR)** scores as the evaluation metrics, which is similar
to previous work [26].



**Table 2.** The AUROC and AUPR results of DTI prediction from
different methods


Method AUROC AUPR


NRLMF 0.8792 0.6874

DTINet 0.9030 0.9187

DTI-CNN 0.9404 0.9467

GCN-DTI 0.9391 0.9507

IMCHGAN 0.9544 0.9203

**DTI-MGNN** **0.9665** **0.9683**


**Figure 3.** ROC curves of all the baseline methods and our model DTI-MGNN.


**AUROC** curve is a probability curve used to evaluate the correction rate of the model. The horizontal ordinate of the curve is

the false positive rate (FPR), and the longitudinal coordinate is

FP TP
the true positive rate (TPR), where FPR = TN+FP [and TPR][ =] TP+FN [.]
By plotting the relationship between true positive rate and false
positive rate, specificity and sensitivity of the methods can be
accurately reflected.
**AUPR** is the area under Precision-Recall curve. The horizontal

ordinate of the curve is the Recall, and the longitudinal coor
TP
dinate is the Precision, where Recall = TP+FN [and Precision][ =]
TP

TP+FP [.]


**Experimental Results**


_**Performance Comparison**_


We compare our model and the baselines on the task of DTI
prediction, the AUROC and AUPR results of different methods
are shown in Table 2.

From Table 2, we can draw the following: ( _i_ ) Deep learning based models including DTI-CNN, GCN-DTI and DTI-MGNN


8 _Li et al._


**Table 3.** AUROC and AUPR results from variants of our method


Method AUROC AUPR


Without common module 0.9125 0.9173

Without topology module 0.8370 0.8495
Without feature module 0.8970 0.9041

Without attention module 0.9494 0.9502

**DTI-MGNN full model** **0.9665** **0.9683**


**Figure 4.** AUROC and AUPR results on different number of GCN layers.


**Figure 5.** AUROC and AUPR results on different number of GAT heads.


significantly outperform the matrix factorization based method
NRLMF and DTINet. This demonstrates that deep neural networks are effective in graph representation learning. ( _ii_ ) Compared with NRLMF method, DTINet is greatly improved by fusing
heterogeneous network information, especially in AUPR, which
is increased by about 30%. ( _iii_ ) Our model DTI-MGNN achieves the
best performance in the task of DTI prediction. Compared with
the other three graph neural network based models DTI-CNN,
GCN-DTI and IMCHGAN, DTI-MGNN gives an improvement of at
least 1% in both AUROC and AUPR.



**Table 4.** Prediction results of three drugs Olanzapine, Clozapine and
Aripiprazole


Drug Related Protein Prediction Result


DB00334- P11229-CHRM1 True

Olanzapine P08913-ADRA2A True

P08908-HTR1A True

P46098-HTR3A True

P21917-DRD4 True

P28223-HTR2A True

P35348-ADRA1A True

P08172-CHRM2 True

P34969-HTR7 True

P14416-DRD2 True

P28222-HTR1B False


**Accuracy** **90.9%**


DB00363- P11229-CHRM1 True

Clozapine P08913-ADRA2A True

P08908-HTR1A True

P46098-HTR3A True

P21917-DRD4 True

P35348-ADRA1A True

P34969-HTR7 True

Q9NYX4-CALY True

P14416-DRD2 True

P28222-HTR1B True

P50406-HTR6 True

Q13131-PRKAA1 True

Q92569-PIK3R3 True


**Accuracy** **100%**


DB01238- P21728-DRD1 True

Aripiprazole P11229-CHRM1 True

P08913-ADRA2A True

P08908-HTR1A True

P46098-HTR3A True

P21917-DRD4 True

P28223-HTR2A True

P35348-ADRA1A True

P08172-CHRM2 True

P28222-HTR1B True

P04424-ASL False


**Accuracy** **90.9%**


We also plot the ROC curves in Figure 3, and the area under
these curves also prove that our method is more effective than
the competitive baselines. We innovatively use multi-channel
graph neural networks to learn different DPP representation and
introduce the attention mechanism. We believe that these inno
vations significantly improved the performance of the method,
which will be verified in later experiments.


_**Ablation Experiment**_


To test the effectiveness of different parts in our model, we
conduct the ablation study. Specifically, we denote our method
as the full model and perform the leave-one-out validation on
each part of the model to test which part is the most useful. First,
we explore the influence of different channels of our model.
Then, we test the effectiveness of the attention mechanism for
the representation learned from the three channels. ‘ **Without**
**common module** ’ denotes our method without the common


_Drug–target interaction predication_ 9


**Figure 6.** Example of the attention heat map in GAT model. From the attention map in Figure 6(a), we can observe that the neighbors of the central node have different
importance. The row represents the central nodes, and the column represents the importance of different neighbor nodes to the central node. The darker the color is,
the more important the neighbor node is. As shown in Figure 6(b), DPP 509 and its neighbor node DPP 507 have a large attention coefficient in our model, because they

share the same drug imatinib.



GCN-based graph modeling, ‘ **Without topology module** ’ represents our method without the topology GAT-based graph modeling, ‘ **Without feature module** ’ denotes our method without the
feature GAT-based graph modeling, and ‘ **Without attention mod-**
**ule** ’ represents our method without the attention mechanism of
different channels.

From Table 3, we can draw the following conclusions: ( _i_ )
Topology graph modeling is the most important part in our
model. After removing it, both the AUROC and AUPR values get
a decrease of about 13%. ( _ii_ ) Either without topology, feature
or common graph modeling, the performance of the model
decrease. The full model with three channels achieves the best

AUROC performance of 0.9665, and the best AUPR performance
of 0.9683, which is much better than that of using any single
channel. ( _iii_ ) After removing the attention mechanism of the
three channels, it was found that the results also dropped obviously. The overall results demonstrate that the representations
learned from the three channels are complementary to each
other, and the combination can improve the prediction results.


**Parameter Sensitivity Analysis**


In our model, DPP network representation learning is mainly
based on two types of graph neural networks: GCN and GAT.
In order to evaluate the influence of layer number in GCN and
the number of attention heads in GAT, we conducted parameter
sensitivity experiments to show the robustness of our proposed
method.


_**Number of GCN Layers**_


GCN aggregates one-hop of neighbor information for each additional layer. To analyze the impact of number of layers in GCN
model, we try to vary the number of layers in DTI-MGNN. The
experimental results are shown in Figure 4. Each point of a curve
represents different number of GCN layers, ranging from 1 to 5.



From Figure 4, we can see that when the number of GCN
layers increases to 2, the performance of the model is the best.
However, when the number of layers becomes larger than 2, the
performance of the model gradually decreases. The reason may
be that more layers the GCN model has, the closer the representation of each node is, because more common neighboring nodes
are used. This will bring disadvantages to the subsequent node
classification tasks.


_**Number of Heads in GAT**_


At the same time, we test the influence of different heads of
attention in GAT model. Figure 5 shows the results when we vary
the number of attention heads _M_ from 1 to 5.

We can observe that with the increase of the number of

attention heads, the AUROC and AUPR of our model are on the
rise. When the number of attention heads reaches 4, the model
shows the best performance. The results show that increasing
the number of attention heads can improve the performance of
the GAT model in a certain range.


**Qualitative Analysis**


In this part, we conduct the qualitative analysis on our model.
First, we use the attention heat map to qualitatively analyze the
attention mechanism and analyze the interpretability brought
by the attention mechanism. Then, we conduct error analysis of
our results.


_**Analysis of Attention Mechanism in GAT**_


We use the attention mechanism in the GAT model to learn

the different importance of different neighboring DPP nodes
to the central DPP node. We try to show that the GAT model
with attention mechanism can provide some effective explanations for topology modeling and feature modeling through


10 _Li et al._


qualitative analysis. We extract the attention coefficients of GAT
used in topology graph modeling and feature graph modeling,
respectively.
Taking the attention coefficient matrix of the GAT model
in feature graph modeling as an example, we anonymously
process and number the DPP nodes, and draw the attention heat
map in Figure 6(a). For the DPP node DPP 507, we find that its
neighbor DPP node DPP 509 has a high attention score over it.
After our verification, the two pairs of DPP are ‘ _Imatinib-KIT_ ’ and
‘ _Imatinib-ABL1_ ’, respectively, sharing the same drug _Imatinib_ [27].
Furthermore, the protein ‘ _KIT_ ’and ‘ _ABL1_ ’ have mutual relations
in human protein reference database HPRD. It shows that these
two DPP nodes are indeed related, which is consistent with the
high attention.
At the same time, in the attention coefficient of the GAT
model in feature graph modeling, we also find that although
the DPP node ‘ _Terbinafine-PECR_ ’ DPP 1385 and ‘ _Nortriptyline-TACR1_ ’
DPP 2424 do not share the same drug or protein, but there is a high
attention score between them. A recent research [35] has proved
that there is a competitive inhibition relationship between _Nor-_
_triptyline_ and _Terbinafine_, the nortriptyline intoxication is possibly
due to a pharmacokinetic interaction between terbinafine and
nortriptyline, which is consistent with the conclusion of our
attention heat map.
In conclusion, the attention mechanism in the GAT model
gives higher weight to similar DPPs to a certain extent, so as
to fuse more accurate neighbor information when learning the
network representation of DPP nodes, which not only improves
the prediction results of our model in DTI tasks, but also provides
interpretability.


_**Error Analysis**_


We also conduct the error analysis of the results. We found that
the DPP node DPP 2123 ‘ _Morphine-LY96_ ’, which is composed of
drug ‘ _Morphine_ ’ and protein ‘ _LY96_ ’. For the this DPP, our model
gives a ‘True’ prediction result in the task of DTI prediction,
which is yet ‘False’ in the ground truth dataset. However, we
found that a recent study [1] verified the interaction relationship
between ‘ _Morphine_ ’ and protein ‘ _LY96_ ’, which is consistent with
our predicted results.


_**Analysis of Prediction Results**_


In this section, we analyze the accuracy of DTI prediction results
of some common drugs and proteins. We randomly selected
drugs with more than 10 related proteins from the test data
for observation, and analyzed the prediction accuracy of these
drugs. In Table 4, we list the related proteins predicted by our
model for the three common drugs _Olanzapine_, _Clozapine_ and
_Aripiprazole_, and compare the results with the ground-truth data.
Specifically, for drug _Clozapine_, the accuracy is 100%. For drugs
_Olanzapine_ and _Aripiprazole_, our models can also achieve more
than 90% accuracy. The average prediction accuracy of all diseases is 92.8 %.

At the same time, we randomly selected proteins with more
than 20 related drugs from the test set, and analyzed the prediction accuracy of these proteins. In Table 5, we list the prediction results of three proteins including _HTR2A_ and _ADRA1A_ .
For protein _HTR2A_, our model correctly predicted all the DTI
relationships. For all the proteins, the average accuracy of our
model is about 92.4 %.



**Table 5.** Prediction results of proteins ADRA1A and HTR2A


Protein Related Drug Prediction Result


P35348- DB00211-Midodrine True

ADRA1A DB00334-Olanzapine True

DB00346-Alfuzosin True

DB00363-Clozapine True
DB00388-Phenylephrine True
DB00449-Dipivefrin True
DB00450-Droperidol True
DB00543-Amoxapine True

DB00656-Trazodone True

DB00679-Thioridazine True

DB00696-Ergotamine True
DB00734-Risperidone True

DB00745-Modafinil True

DB00964-Apraclonidine False

DB01136-Carvedilol True

DB01149-Nefazodone True

DB01151-Desipramine True
DB01186-Pergolide True
DB01224-Quetiapine True
DB01238-Aripiprazole True
DB06216-Asenapine True
DB06711-Naphazoline True


**Accuracy** **95.5%**


P28223-HTR2A DB00247-Methysergide True
DB00248-Cabergoline True
DB00268-Ropinirole True
DB00334-Olanzapine True
DB00370-Mirtazapine True
DB00408-Loxapine True
DB00413-Pramipexole True
DB00434-Cyproheptadine True
DB00502-Haloperidol True
DB00714-Apomorphine True
DB00734-Risperidone True
DB00843-Donepezil True

DB01149-Nefazodone True

DB01151-Desipramine True
DB01186-Pergolide True
DB01200-Bromocriptine True
DB01238-Aripiprazole True
DB01242-Clomipramine True

DB01618-Molindone True

DB06216-Asenapine True

DB00324-Fluorometholone True


**Accuracy** **100%**


Conclusion


In this paper, we proposed a multi-channel graph neural network
DTI-MGNN to predict the interaction between drug and target. DTI-MGNN first learns low-dimensional representations of
drugs and proteins from heterogeneous networks, and construct
a topology graph and a feature graph for DPPs, respectively.
Then a multi-channel graph neural network is used to learn
the representation of drug–protein pairs and finally the DTIs
are identified through an MLP neural network. We conducted
a series of experiments to evaluate our model against several state-of-the-art models and found that our method outperforms the baselines significantly. This shows that the fusion of


multiple network representation of drug–protein pairs is effective for predicting DTI. Through ablation experiments, we found
that topological representation, feature representation and common representation are useful and complementary, and the
combination of them can improve the DTI prediction results. In
addition, we also conducted visual experiments, which proved
that the attention mechanism in our model can accurately learn
the relationship between drug–protein pairs and provide interpretability for the model. Finally, our model DTI-MGNN can be
extended to other similar tasks, especially the tasks that need
to integrate multiple network representation learning, in which
GCN and GAT modules can be replaced by other models.
There are several directions that we want to explore in the
future. First, we will add more heterogeneous data sources,
such as protein or drug molecular structure information, so as
to enhance the ability of the model to learn the relationship
between drugs and proteins. Second, we will further explore an
end-to-end DTI prediction framework to improve our current
model. All these issues will be left as our future works.


**Key Points**


   - By combining the topological structure and semantic
features, the DTI-MGNN model improved the representation learning ability of DPPs.

   - We conducted experiments on a public dataset and
compare our method to some competitive baselines. Experimental results show that our model outperforms the state-of-the-art methods significantly.
Specifically, DTI-MGNN achieves the results of 0.9665
and 0.9683 in AUROC and AUPR, respectively.

   - Furthermore, the introduction of attention mechanism also provided interpretability for DTI prediction.
To the best of our knowledge, this is the first work on
multi-channel graph neural network combined with
attention mechanism for DTI prediction.


Acknowledgments


We thank the anonymous reviewers for their constructive
suggestions. This work was supported by the National Natural Science Foundation of China [61806049,61771165,62072095]
and the Heilongjiang Postdoctoral Science Foundation [LBHZ20104]. Corresponding author: Guohua Wang, E-mail:
ghwang@nefu.edu.cn.


Data availability


Experimental data sets and experimental codes can be found
[in https://github.com/catly/drug-target.](https://github.com/catly/drug-target)


References


1. Hutchinson MR, Lewis SS, Coats BD, et al. Possible invsolvement of toll-like receptor 4/myeloid differentiation factor-2
activity of opioid inactive isomers causes spinal proinflammation and related behavioral consequences. _Neuroscience_
2010; **167** (3): 880–93.
2. Campillos M, Kuhn M, Gavin A-C. Lars Juhl Jensen, and Peer
Bork. Drug target identification using side-effect similarity.
_Science_ 2008; **321** (5886): 263–6.



_Drug–target interaction predication_ 11


3. Cheng F, Liu C, Jiang J, et al. Prediction of drug-target interactions and drug repositioning via network-based inference.
_PLoS Comput Biol_ 2012; **8** (5):e1002503.
4. Cheng Z, Cheng Y, Wu F, et al. Drug-target interaction prediction using multi-head self-attention and graph attention
network. _IEEE/ACM Trans Comput Biol Bioinform_ 2021.
5. Davis AP, Grondin CJ, Johnson RJ, et al. The comparative toxicogenomics database: update 2017. _Nucleic Acids Res_ 2017;
**45** (D1): D972–8.
6. Faulon J-L, Misra M, Martin S, et al. Genome scale enzyme–
metabolite and drug–target interaction predictions using
the signature molecular descriptor. _Bioinformatics_ 2008; **24** (2):

225–33.

7. Feng Y, Wang Q, Wang T. Drug target protein-protein interaction networks: a systematic perspective. _Biomed Res Int_ 2017;

**2017** .

8. González-Díaz H, Prado-Prado F, García-Mera X, et al. Mindbest: Web server for drugs and target discovery; design, synthesis, and assay of mao-b inhibitors and theoretical- experimental study of g3pdh protein from trichomonas gallinae. _J_
_Proteome Res_ 2011; **10** (4): 1698–718.
9. Iorio F, Bosotti R, Scacheri E, et al. Discovery of drug
mode of action and drug repositioning from transcriptional
responses. _Proc Natl Acad Sci_ 2010; **107** (33): 14621–6.
10. Jolliffe IT. Principal component analysis. _J Market Res_ 2002;
**87** (4): 513.
11. Prasad TTSK, Goel R, Kandasamy K, et al. Human protein reference database-2009 update. _Nucleic Acids Res_ 2009;
**37** (suppl_1): D767–72.
12. Kipf TN, Welling M. Semi-supervised classification
with graph convolutional networks arXiv preprint
arXiv:1609.02907. 2016.

13. Knox C, Law V, Jewison T, et al. Drugbank 3.0: a comprehensive resource for ‘omics’ research on drugs. _Nucleic Acids Res_
2010; **39** (suppl_1): D1035–41.
14. Krizhevsky A, Sutskever I, Hinton G. Imagenet classification
with deep convolutional neural networks. _Advances in neural_
_information processing systems_ 2012; **25** (2).
15. Kuhn M,Campillos M,Letunic I,et al.A side effect resource to
capture phenotypic effects of drugs. _Mol Syst Biol_ 2010; **6** (1):

343.

16. Li J, Wang J, Lv H, et al. Imchgan: Inductive matrix completion with heterogeneous graph attention networks for drugtarget interactions prediction. _IEEE/ACM Trans Comput Biol_
_Bioinform_ 2021.
17. Liaw A, Wiener M. Classification and regression by randomforest. _R News_ 2002; **23** (23).
18. Liu Y, Wu M, Miao C, et al. Neighborhood regularized logistic
matrix factorization for drug-target interaction prediction.
_PLoS Comput Biol_ 2016; **12** (2):e1004760.
19. Luo Y, Zhao X, Zhou J, et al. A network integration approach
for drug-target interaction prediction and computational
drug repositioning from heterogeneous information. _Nat_
_Commun_ 2017; **8** (1): 1–13.
20. Mei J-P, Kwoh C-K, Yang P, et al. Drug–target interaction
prediction by learning from local information and neighbors.
_Bioinformatics_ 2013; **29** (2): 238–45.
21. Meng F-R, You Z-H, Chen X, et al. Prediction of drug–
target interaction networks from the integration of protein
sequences and drug chemical structures. _Molecules_ 2017;
**22** (7): 1119.
22. Mizutani S, Pauwels E, Stoven V, et al. Relating drug–protein
interaction network with drug side effects. _Bioinformatics_
2012; **28** (18): i522–8.


12 _Li et al._


23. Mongia A, Majumdar A. Drug-target interaction prediction
using multi graph regularized nuclear norm minimization.
_Plos one_ 2020; **15** (1):e0226484.
24. Morris GM, Huey R, Lindstrom W, et al. Autodock4 and
autodocktools4: Automated docking with selective receptor
flexibility. _J Comput Chem_ 2009; **30** (16): 2785–91.
25. Niwattanakul S, Singthongchai J, Naenudorn E, et al. Using
of jaccard coefficient for keywords similarity. In: _Proceedings_
_of the international multiconference of engineers and computer_
_scientists_, Vol. **1**, 2013, 380–4.
26. Peng J, Li J, Shang X. A learning-based method for drugtarget interaction prediction based on feature representation learning and deep neural network. _BMC bioinformatics_
2020; **21** (13): 1–13.
27. Rausch JL, Boichuk S, Ali AA, et al. Opposing roles of kit and
abl1 in the therapeutic response of gastrointestinal stromal
tumor (gist) cells to imatinib mesylate. _Oncotarget_ 2017; **8** (3):

4471–83.

28. Shaikh N, Sharma M, Garg P. An improved approach for
predicting drug–target interaction: proteochemometrics to
molecular docking. _Mol Biosyst_ 2016; **12** (3): 1006–14.
29. Sirota M, Dudley JT, Kim J, et al. Discovery and preclinical
validation of drug indications using compendia of public
gene expression data. _Sci Transl Med_ 2011; **3** (96): 96ra77–7.
30. Skwarczynska M, Ottmann C. Protein–protein interactions
as drug targets. _Future Med Chem_ 2015; **7** (16): 2195–219.
31. Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: a simple
way to prevent neural networks from overfitting. _The journal_
_of machine learning research_ 2014; **15** (1): 1929–58.
32. Steinbeck C, Han Y, Kuhn S, et al. The chemistry development kit (cdk): an open-source java library for chemo- and
bioinformatics. _Chem_ 2003; **34** (2): 493–500.
33. Sun M, Zhao S, Gilvary C, et al. Graph convolutional networks
for computational drug development and discovery. _Brief_
_Bioinform_ 2020; **21** (3): 919–35.
34. Tong H, Faloutsos C, Pan J-Y. Random walk with restart:
fast solutions and applications. _Knowledge and Information_
_Systems_ 2008; **14** (3): 327–46.



35. Van PH, Van RW, Vanmolkot LM. Pharmacokinetic interaction between nortriptyline and terbinafine. _Annals of Phar-_
_macotherapy_ 2002; **36** (11): 1712–4.
36. VAPNIK and VLADIMIR. _Nature of statistical learning theory, the_,

1995.

37. Velickovi´c P, Cucurull G, Casanova A, et al. Graph attentionˇ
networks. 2018.

38. Wang W, Yang S, Zhang X, et al. Drug repositioning by integrating target information through a heterogeneous network model. _Bioinformatics_ 2014; **30** (20):

2923–30.

39. Wang X, Zhu M, Bo D, et al. Am-gcn: Adaptive multi-channel
graph convolutional networks. In: _Proceedings of the 26th ACM_
_SIGKDD International Conference on Knowledge Discovery & Data_
_Mining_, 2020, 1243–53.
40. Wen M, Zhang Z, Niu S, et al. Deep-learning-based drug–
target interaction prediction. _J Proteome Res_ 2017; **16** (4):

1401–9.

41. Willy N. _Smith-Waterman Algorithm_ Smith-Waterman Algorithm, 2012.
42. Zheng X, Wu LY, Wong ZST. Semi-supervised drugprotein interaction prediction from heterogeneous biological spaces. _BMC Syst Biol_ 2010.
43. Xue H, Li J, Xie H, et al. Review of drug repositioning
approaches and resources. _Int J Biol Sci_ 2018; **14** (10): 1232.
44. Yamanishi Y, Kotera M, Kanehisa M, et al. Drug-target interaction prediction from chemical, genomic and pharmacological data in an integrated framework. _Bioinformatics_ 2010;
**26** (12): i246–54.
45. Zeng X, Zhu S, Lu W, et al. Target identification among
known drugs by deep learning from heterogeneous networks. _Chem Sci_ 2020; **11** .
46. Zeng X, Zhu S, Lu W, et al. Target identification among
known drugs by deep learning from heterogeneous networks. _Chem Sci_ 2020; **11** (7): 1775–97.
47. Zhao T, Hu Y, Valsdottir LR, et al. Identifying drug–target
interactions based on graph convolutional network and
deep neural network. _Brief Bioinform_ 2020.


