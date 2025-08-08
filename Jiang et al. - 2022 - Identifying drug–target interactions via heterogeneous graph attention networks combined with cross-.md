_Briefings in Bioinformatics_, 2022, **23** ( **2** ), 1–11


**https://doi.org/10.1093/bib/bbac016**

**Problem Solving Protocol**

# **Identifying drug–target interactions via heterogeneous** **graph attention networks combined with cross-modal** **similarities**


Lu Jiang, Jiahao Sun, Yue Wang, Qiao Ning, Na Luo and Minghao Yin
Corresponding authors: Qiao Ning, Department of Information Science and Technology, Dalian Maritime University, Lingshui Street, 116026, Dalian, China; Email:
ningq669@dlmu.edu.cn, Minghao Yin, School of Information Science and Technology, Northeast Normal University, Jingyue Street, 130117, Changchun, China;
E-mail: ymh@nenu.edu.cn


Abstract


Accurate identification of drug–target interactions (DTIs) plays a crucial role in drug discovery. Compared with traditional experimental methods that are labor-intensive and time-consuming, computational methods are more and more popular in recent years.
Conventional computational methods almost simply view heterogeneous networks which integrate diverse drug-related and targetrelated dataset instead of fully exploring drug and target similarities. In this paper, we propose a new method, named DTIHNC,
for **D** rug– **T** arget **I** nteraction identification, which integrates **H** eterogeneous **N** etworks and **C** ross-modal similarities calculated by
relations between drugs, proteins, diseases and side effects. Firstly, the low-dimensional features of drugs, proteins, diseases and
side effects are obtained from original features by a denoising autoencoder. Then, we construct a heterogeneous network across
drug, protein, disease and side-effect nodes. In heterogeneous network, we exploit the heterogeneous graph attention operations
to update the embedding of a node based on information in its 1-hop neighbors, and for multi-hop neighbor information, we
propose random walk with restart aware graph attention to integrate more information through a larger neighborhood region.
Next, we calculate cross-modal drug and protein similarities from cross-scale relations between drugs, proteins, diseases and side
effects. Finally, a multiple-layer convolutional neural network deeply integrates similarity information of drugs and proteins with
the embedding features obtained from heterogeneous graph attention network. Experiments have demonstrated its effectiveness
and better performance than state-of-the-art methods. Datasets and a stand-alone package are provided on Github with website
https://github.com/ningq669/DTIHNC.


Keywords: DTIs prediction, heterogeneous graph attention network, random walk with restart, cross-modal similarity, convolutional
neural network



Introduction


The therapeutic effect of a drug on disease stems from
its action on the target protein and its influence on its
expression [1]. Study of drug–target interactions (DTIs) is
helpful for drug repositioning, which means that existing drugs can be utilized to treat diseases other than
those originally developed for. On the one hand, drug
repositioning could highly reduce the cost of drug discovery and accelerate the research process. On the other
hand, a large number of small molecule compounds
have not yet been used as drugs, the majority of whose



interaction with drugs are still not clear. Therefore, DTIs
identification plays a key role in drug discovery. However, rapid increase in the number of drugs and target
proteins makes traditional biological experiments timeconsuming and expensive [2]. Computational methods
have received more attention for accurate DTI prediction

[3–5].

Computational methods for DTIs identification can
be mainly divided into two categories: (1) feature-based
methods [6, 7] and (2) graph-based methods [8, 9].
In feature-based methods, various types of features



**Lu Jiang** received a B.S. degree in computer science and technology from the Northeast Normal University, Changchun, China, in 2015, where she is currently
pursuing a Ph.D. degree. Her current research interests include bioinformatics, data mining and machine learning.
**Jiahao Sun** received a B.S. degree in internet of things engineering from the Xi‘an University, Xi’an, China, in 2020. He is a postgraduate student digital
information, Northeast Normal University. Her research interests include drug prediction, data mining and machine learning in bioinformatics.
**Yue Wang** received a B.S. degree in computer science and technology from Shandong Agricultural University, Tai’an, China, in 2020. She is a postgraduate student
information science and technology, Dalian Maritime University. Her research interests include disease and noncoding RNAs, protein sites prediction and machine
learning in bioinformatics.
**Qiao Ning** received a B.S. and a PhD degree from School of information science and technology, Northeast Normal University, Changchun, China, in 2019. She is
currently a Lecturer in information science and technology, Dalian Maritime University, Dalian. Her research interests include machine learning and
bioinformatics.

**Na Luo** received a Ph.D. degree in computer science from Jilin University in 2009. She is currently an Assistant Professor in information science and technology,
Northeast Normal University, Changchun. Her research interests include bioinformatics, recommender system and data mining.
**Minghao Yin** received a Ph.D. degree in computer science from JiLin University, China, in 2008. He has been the President of college since 2021. He is the author of
two books, and more than 100 articles. His research interests include data mining, swarm intelligence, automated reasoning, automated planning and algorithms.
**Received:** November 3, 2021. **Revised:** January 4, 2022. **Accepted:** January 13, 2022
© The Author(s) 2022. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


2 | _Jiang_ et al.


extraction strategies are utilized to extract biological
features for drugs and proteins [10–13]. Feature-based
methods transform DTIs prediction into a binary classification problem and machine learning methods, such
as support vector machine (SVM) and random forest,
are used as classifiers [7, 14]. For instance, bipartite
local models was built based on SVM and drug–drug,
target–target similarity that were calculated by chemical
and genomic data [15]. MvGRLP, a multi-view graph
regularized link propagation model, predicted DTIs by
fusing complementary information between different
views [16]. Ding _et al._ established a method to predict DTIs
via Dual Laplacian Regularized Least Squares model with
Hilbert–Schmidt Independence Criterion-based Multiple
Kernel Learning [17]. Graph-based methods, which
describe the complex interactions between different
types of biological entities, are build on the assumption
that nodes connected to similar ones tend to have more

interactions [18–20].In graph-based methods,similarities
between drugs and targets are calculated based on
local or global topological information [9, 21, 22]. For
example, Ezzat _et al._ constructed drug–protein networks
and established prediction models based on singular
value matrix decomposition [23]. NRLMF method focuses
on modeling the probability that a drug would interact
with a target by logistic matrix factorization, where the
properties of drugs and targets are represented by drugspecific and target-specific latent vectors [24]. However,
those DTI prediction methods are shallow-learning
ones which cannot fully extract deep and complex
associations between drugs and proteins.

In recent years, heterogeneous network combined
with deep learning algorithms have been developed for
DTI prediction by integrating diverse drug-related and
target-related information. Different from homogeneous
networks, heterogeneous networks cover multiple types
of entities and complex interactions between different
types of entities [25]. For example, DTINet is a method
which focuses on learning a low-dimensional vector
representation of features that accurately explains
the topological properties of individual nodes in the
heterogeneous network [18]. NeoDTI integrated diverse
information from heterogeneous network data and automatically learned topology-preserving representations
of drugs and targets to facilitate DTI prediction [26].
Zheng _et al._ developed a method for DTIs prediction by
combining Long Short-Term Memory (LSTM) networks
and convolutional neural network (CNN) [27]. Compared
with CNN, graph convolutional networks can extract
graph features effectively. Manoochehri _et al._ aggregated
a heterogeneous graph of drug–target with graph convolutional network to predict DTIs [28], which aggregate
the embeddings from different kinds of interactions
with independent weights. Liu _et al._ proposed a graph
autoencoder approach, named GADTI, for DTI prediction using a heterogeneous network, which combines
graph convolutional network, matrix factorization and
random walk [29]. EEG-DTI built a graph convolutional



networks-based model to learn the low-dimensional

feature representation of drugs and targets and optimize
the model by end-to-end learning [8]. Jin _et al._ proposed
a multi-resolutional collaborative heterogeneous graph
convolutional autoencoder which fully exploited the
local and global topological structure of heterogeneous
drug–target networks by collaboratively aggregating the
embeddings from different types of links with independent convolution kernels in each graph convolutional
layer [30].

However, different neighbor nodes in heterogeneous
graphs have different importance to DTIs, while GCN
treats all neighbor nodes equally in convolution and cannot assign different weights according to the importance
of nodes. Besides, previous methods almost simply view
heterogeneous networks without fully exploring drug
and protein similarities.

Inspired by a previous study, we elaborate a new
method for DTIs prediction named DTIHNC, which
integrates heterogeneous graph attention networks and
multi-modal similarities of drugs and proteins.

The major contributions are summarized below.


1) A new feature encoding scheme associating drugs,
proteins, diseases and side effects is proposed,
which employs denoising autoencoder (DAE) to
extract informative features.

2) In heterogeneous network, which include drug, protein, disease and side-effect nodes, a heterogeneous
graph attention operations is exploited to update the
embedding of a node based on information in its 1hop neighbors, and for multi-hop neighbor information, we propose random walk with restart (RWR)
aware graph attention to integrate more information through a larger neighborhood region.
3) We calculate cross-modal drug similarities and
protein similarities which integrate drug–drug interactions, drug–protein interactions, drug–disease
associations, drug–side-effect associations, protein–
drug interactions, protein–protein interactions
and protein–disease associations. Then, an attention mechanism method was proposed to obtain
attribute embedding from similarities vectors.
4) A multiple-layer CNN is designed to deeply integrated similarity information of drugs and proteins
with the embedding features obtained from heterogeneous network.
5) We conduct extensive experiments to show the
effectiveness of our proposed framework on a
benchmark heterogeneous dataset.


Materials and methods


In this section, we first introduce the materials. Then,
we present the framework overview of the proposed
method. Finally, we introduce the core architecture of our
method. Some materials information are summarized in

Table 1.


**Table 1.** Sources of datasets and statistical information


**Node** **Number**


Drug 708
Target 1512
Disease 5603

Side effect 4192

**Relation** **Number**

Drug–target interaction 1923
Drug–drug interaction 10 036
Protein–protein interaction 7363
Drug–disease association 199 214
Drug–side-effect association 80 164
Protein–disease association 1596 745


**Materials**


We obtain the dataset from the previous study [31]. There
are four types of nodes and six types of relations. Nodes
consist of drugs, diseases, side effects and targets which
refer to the proteins in this work. Relations consist of
DTIs, drug–drug interactions, drug–disease associations,
drug–side-effects associations, protein–protein interactions and protein–disease associations.

The drug nodes, the known DTIs and drug–drug
interactions were extracted from DrugBank3.0 [32].
The protein nodes and the protein–protein interactions
were downloaded from the HPRD9.0 [33]. The disease
nodes,the drug–disease and protein–disease associations
were extracted from the Comparative Toxicogenomics
Database. The side-effect nodes and the drug–sideeffect associations were obtained from the SIDER2.0. [34].
Finally, isolated nodes were deleted from the dataset; in
other words, only nodes which had at least one edge (see
below) in the network were considered.

Table 1 shows the sources and statistics of these data.

All weights of edges in the networks are nonnegative, and
the binary values indicate whether there is an interaction
or association between nodes or not.


**Framework overview**


Figure 1 shows the overview of our proposed framework.
The framework includes four key components: (1) feature
representation and encoding for drug, protein, disease
and side effect. We extract original features from associations of drugs, proteins, diseases and side effects, and
a DAE is utilized to obtain low-dimensional features.

(2) neighbor topology representation learning from heterogeneous information network. 1-hop neighbor topology information and the low-dimensional features of
nodes in the heterogeneous network are integrated by
a graph attention network, and multi-hop neighborhood
information is aggregated by RWR. (3) cross-modal similarities integration. We calculate cross-modal similarities of drugs and proteins and a hierarchical attention
mechanism is used to obtain attribute embedding of
proteins and drugs. (4) CNN prediction model. We use
the CNN prediction method to calculate the interaction



_Heterogeneous GAT_ | 3


score between drugs and targets based on the heterogeneous network neighbor topology representation integrating cross-modal similarities, and optimize the model
by cross-entropy.


**Representation learning for drugs, proteins,**
**diseases and side effects**
_Original feature representation of different entities_

Taking drugs as an example, we construct a synthesis
matrix. Given matrix _G_ _[drug]_ [−] _[drug]_ ∈ _R_ _[Nr]_ [×] _[Nr]_, values in this
matrix represent whether two drugs may have interactions or not, where value is 1 or 0. Matrix _G_ _[drug]_ [−] _[protein]_ ∈
_R_ _[Nr]_ [×] _[Np]_ indicates interactions between _N_ _r_ drugs and _N_ _p_
proteins. And the _i_ th row represents the feature vector
of drug _r_ _i_ at the drug–protein interaction level. The associations between _N_ _r_ drugs and _N_ _d_ diseases are represented by matrix _G_ _[drug]_ [−] _[disease]_ ∈ _R_ _[Nr]_ [×] _[Nd]_, where the value
of _G_ _[drug]_ _ij_ [−] _[disease]_ corresponds to the associated value where
the row of drug _r_ _i_ and the column of disease _d_ _j_ locates. In
summary, _G_ _[drug]_ [−] _[drug]_, _G_ _[drug]_ [−] _[protein]_ and _G_ _[drug]_ [−] _[disease]_ are feature
matrices of _N_ _r_ drugs from three diverse perspectives,
in which the _i_ th row feature vectors of three matrices

denote different levels of feature representation. Therefore, their concatenation matrix _X_ _G_ ∈ _R_ _[Nr]_ [×] _[(][Nr]_ [+] _[Np]_ [+] _[Nd][)]_ is
the drug feature matrix and the _i_ th row of represents
the feature vector of _r_ _i_ . In the same way, we can get
the feature matrix of proteins and diseases. _P_ _[protein]_ [−] _[drug]_,
_P_ _[protein]_ [−] _[protein]_ and _P_ _[protein]_ [−] _[disease]_ are concatenated together
to get a protein matrix _X_ _P_ ∈ _R_ _[Np]_ [×] _[(][Nr]_ [+] _[Np]_ [+] _[Nd][)]_ . And a new
disease feature matrix _X_ _D_ ∈ _R_ _[Nd]_ [×] _[(][Nr]_ [+] _[Np][)]_ is obtained by
joining _D_ _[disease]_ [−] _[drug]_ and _D_ _[disease]_ [−] _[protein]_ . For side effect, _X_ _E_ ∈
_R_ _[Ns]_ [×] _[Nd]_ is the same as _S_ _[sideeffect]_ [−] _[drug]_, the association matrix
between side effects and drugs.


_DAE model_


The original feature embedding obtained in the previous step is high-dimensional. In order to obtain a highquality feature representation, we adopt a DAE model
based on an autoencoder for data manipulation [36]. The
DAE can reconstruct the input data with noise.

For instance, the dimensions of original entity feature
are as follows:



Figure 1(a) shows that DAE involves two steps: encode
and decode. The encode step projects the original feature
representation with superimposed noise to the objective



⎧
⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎨

⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎩



_X_ _G_ : _N_ _r_ × _(N_ _r_ + _N_ _p_ + _N_ _d_ + _N_ _e_ _)_



= 708 × _(_ 708 + 1512 + 5603 + 4192 _)_


= 708 × 12015



_X_ _P_ : _N_ _p_ × _(N_ _r_ + _N_ _p_ + _N_ _d_ _)_ =



1512 × _(_ 708 + 1512 + 5603 _)_ = 1512 × 7823



(1)



_X_ _D_ : _N_ _d_ × _(N_ _r_ + _N_ _p_ _)_



= 5603 × _(_ 708 + 1512 _)_ = 5603 × 2220



_X_ _E_ : _N_ _e_ × _(N_ _r_ _)_ = 4192 × 708


4 | _Jiang_ et al.


Figure 1. Framework of our model.


feature space, while the decode step recovers the lowdimensional feature vector to a reconstruction space
remove superimposed noise. The final learned features
are almost the same as those learned from the data

without superimposed noise, but the obtained features
are more robust. We use the softplus [37] and RMSProp
functions to optimize the mean square error (MSE), and
the DAE model is trained by the backpropagation(BP)
algorithm.


**Heterogeneous graph attention networks**
_Construct the heterogeneous network_

As shown in Figure 1(b), we construct the heterogeneous
network graph _G_ = _(V_, _E)_, and node set _V_ = { _V_ _[r]_,
_V_ _[d]_, _V_ _[p]_, _V_ _[e]_ } is composed of drug nodes _V_ _[r]_, disease
nodes _V_ _[d]_, protein nodes _V_ _[p]_ and side-effect nodes _V_ _[e]_ .
The edge set _E_ = { _E_ _[r]_ [−] _[p]_, _E_ _[r]_ [−] _[e]_, _E_ _[r]_ [−] _[d]_, _E_ _[p]_ [−] _[d]_, _E_ _[d]_ [−] _[d]_, _E_ _[p]_ [−] _[p]_ }
consists of drug–protein interactions _E_ _[r]_ [−] _[p]_, drug–side
effects associations _E_ _[r]_ [−] _[e]_, drug–disease associations _E_ _[r]_ [−] _[d]_,
protein–disease associations _E_ _[p]_ [−] _[d]_, drug–drug interaction
_E_ _[d]_ [−] _[d]_ and protein–protein interaction _E_ _[p]_ [−] _[p]_ . Since each
relationship between nodes is interactive, the edges in
the heterogeneous graph are bidirectional.


_Heterogeneous graph attention layers_

For the representation learning in the heterogeneous
network, there are two challenges to be solved: (1) How to
aggregate the 1-hop neighbor information in the graph?



(2) How to aggregate the multi-hop neighbor information
in the graph?

For the first problem, we use Heterogeneous Graph
Attention Operations to aggregate the 1-hop neighbor
information. For the second problem, we propose RWR
aware Graph attention to aggregate the multi-hop neighbor information.


In this paper, we exploit the heterogeneous graph
attention operations update the embedding of a node
based on information in its 1-hop neighbors. For the
feature embedding, we transform the embeddings of
nodes in graph _G_ from **D** ∈ R [|] _[V]_ [|×] _[F]_ into new embedding
matrix **D** [′] ∈ R [|] _[V]_ [|×] _[F]_ [′] .

Given a node _v_ _i_ whose neighbors are _N_ _vi_, heterogeneous graph attention operations transform its embedding from _d_ _vi_ to _d_ ′ _vi_ [. Firstly, it computes the attention]
scores between nodes based on a shared attentional

mechanism _att_ : R _[F]_ [′] × R _[F]_ [′] → R.

For each node _v_ _j_ ∈ _N_ _vi_, the attention coefficient
between node _v_ _i_ and _v_ _j_ is _e_ _ij_ . The attention mechanism
_att_ is a single-layer neural network, and applying the
LeakyReLU.


⃗
_e_ _[(]_ _ij_ _[k][)]_ = _LeakyReLU_ � _a_ _[(][k][)][T]_ [ �] _W_ _[(][k][)]_ _d_ _[(]_ _i_ _[k][)]_ [∥] _[W]_ _[(][k][)]_ _[d]_ _[(]_ _j_ _[k][)]_ �� (2)


_W_ _[(][k][)]_ _d_ _[(]_ _j_ _[k][)]_ is a linear transformation of the lower layer

embedding and _W_ _[(][k][)]_ is a learnable weight matrix. _e_ _[(]_ _ij_ _[k][)]_


Figure 2. Starting from any node in the graph, through random walk a
certain number of steps to get the access sequence of the node. Each step
is faced with two choices: randomly choose adjacent node or select the
starting node.


is the calculated paired unnormalized attention score
between two neighbors _v_ _i_ and _v_ _j_, where || denotes concatenation then takes a dot product of it and a learnable
weight vector ⃗ _a_ _[(][k][)][T]_ . Then, the _softmax_ function is used to
normalize the attention coefficient _e_ _ij_ into the attention
score _β_ _[k]_
_ij_



_exp_ � _e_ _[(]_ _i_ _[k][)]_ _[j]_ �
_β_ _ij_ _[k]_ [=]



, (3)
_e_ _[(][k][)]_
_t_ ∈ _N_ _[i]_ _r_ _[exp]_ ~~�~~ _(it)_ ~~�~~



~~�~~



where _exp_ is an exponential function and _β_ _ij_ _[k]_ [indicates]
the score in the layer attention mechanism. The node’s
new embedding _d_ _[(]_ _i_ _[k]_ [+][1] _[)]_ is computed using the sum on its
neighbors’ features, scaled by the attention scores.



_Heterogeneous GAT_ | 5


Given a heterogeneous graph _G_ and a source node _v_ _i_, RWR
computes a relevance score between _v_ _i_ and each node
using a random neighbor from _v_ _i_ . The restarted random
walk spread step performs as follows:


 - Random Walk. The current node randomly moves to
one of its neighbors with probability 1- _P_ _r_ .

 - Restart. The neighbor goes back to _v_ _i_ with probability

_P_ _r_ .


Assuming the heterogeneous network transition
matrix is _A_ and the restart probability is _P_ _r_, the influence
of node _v_ _i_ on node _v_ _j_ is defined as _A_ _inf_ [38]:


_A_ _inf_ = _P_ _r_ _(I_ − _(_ 1 − _P_ _r_ _)A)_ [−][1], (5)


where _I_ is the identity matrix; according to Equation 5, we
can spread node information over 1-hop to get the node
embedding vector:


_D_ [1] = _P_ _r_ _(I_ − _(_ 1 − _P_ _r_ _)A)_ [−][1] _D_ [0] (6)


_D_ [0] represents the node embedding vector matrix,
which is represented by the aforementioned convolution
operation with attention mechanism gets. Then, we
introduce the iterative form of Equation 6 due to the
large scale of heterogeneous network


_D_ _[k]_ [+][1] = _(_ 1 − _P_ _r_ _)AD_ _[k]_ + _P_ _r_ _D_ [0] (7)


It is easy to prove that


lim _k_ →∞ _D_ _[k]_ = _P_ _r_ _(I_ − _(_ 1 − _P_ _r_ _)A)_ [−][1] _D_ [0] . (8)


Finally, our model will embed the node representation
that aggregates both 1-hop and multi-hop neighbor information in heterogeneous network. Based on the biological premise that similar drugs are more likely to interact
with similar proteins, this paper will also explore the
fusion of heterogeneous network information and crossmodal similar information.


**Cross-modal similarities integration**

First, we calculate multiple drug similarities and protein
similarities which integrates drug–drug interaction,
drug–protein interactions, drug–disease associations,
drug–side effect associations, protein–protein interactions, protein–disease associations and protein–drug
interactions.


For example, if drug _r_ _i_ interacts with proteins consisting _p_ 1, _p_ 2, _p_ 5 and _r_ _j_ interacts with _p_ 3, _p_ 5, _p_ 6, the similarity
between _M_ 1 = { _p_ 1, _p_ 2, _p_ 5 } and _M_ 2 = { _p_ 3, _p_ 5, _p_ 6 } is considered as the similarity between drug _r_ _i_ and drug _r_ _j_, so the
value is 1 \ _(_ 3 × 3 _)_ = 1 \ 9.



_d_ _[(][k]_ [+][1] _[)]_ = _σ_
_i_



⎛



⎝ _j_ ∈ [�] _N_ _[(]_ _i_



_β_ _ij_ _[k]_ _[W]_ _[(][k][)]_ _[d]_ _[(]_ _j_ _[k][)]_
_j_ ∈ [�] _N_ _[(]_ _i)_



⎞

(4)
⎠



We obtained the node feature representation by aggregating node’s 1-hop neighbor information. Further more,
if we want to obtain the node’s multi-hop neighbor information, one of the operation methods is to add multilayer graph attention neural network to transfer the
information. However, the multi-hop neighborhood information aggregation achieved by multi-layer convolution
often leads to excessive smoothness, which makes each
node in the network have a very similar representation.
In order to prevent the loss of node information in heterogeneous networks and overcome excessive smoothing, we propose a new neighbor information aggregation
method when aggregating multi-hop neighbor information instead of simple stacking of convolutional layers,
which will be introduced in detail next.


_RWR aware graph attention network_

In this paper, we propose a novel RWR aware graph
attention network to obtain the multi-hop neighbor information. RWR considers the connection mode of the local

topology and the global topology in the network to utilize
the potential direct and indirect links between nodes.


6 | _Jiang_ et al.


The formula is as follows [39] :



_M_ _[r]_ [−] _[p]_ ∩ _M_ _[r]_ [−] _[p]_
_R_ _[(]_ _ij_ [1] _[)]_ = _Mi_ ~~_[r]_~~ ~~[−]~~ ~~_[p]_~~ ∪ _Mj_ ~~_[r]_~~ ~~[−]~~ ~~_[p]_~~ (9)

_i_ _j_



In the same way, the other three drug similarity matrices _R_ _[(]_ [2] _[)]_, _R_ _[(]_ [3] _[)]_ and _R_ _[(]_ [4] _[)]_ are calculated based on the drug–
drug, drug–side effect and drug–disease, respectively. For
protein, the protein similarity matrices _P_ _[(]_ [1] _[)]_, _P_ _[(]_ [2] _[)]_ and _P_ _[(]_ [3] _[)]_

are calculated based on the protein–drug interaction,
protein–protein and protein–disease association.

In addition, we also used the drug chemical structure
similarity _R_ _[(]_ [5] _[)]_ and protein sequence similarity _P_ _[(]_ [4] _[)]_ in the
previous study. Many types of drug similarities may be
regarded as the cross-modal similarities of drugs, so does
proteins.

Because the drug similarities and protein similarities
are calculated from different correlation, the contribution value to the final attribute embedding of proteins
or drugs may be different. Therefore, we implemented
the attention mechanism on the similarity to obtain the
attribute embedding of the drugs and proteins. Take drug
similarity as an example, _R_ _[(]_ [1] _[)]_, _R_ _[(]_ [2] _[)]_, _R_ _[(]_ [3] _[)]_, _R_ _[(]_ [4] _[)]_ and _R_ _[(]_ [5] _[)]_ are
calculated based on the drug–protein, drug–drug, drug–
side effect, drug–disease and drug chemical structure,
respectively. After the attention layer, the weighted summation of the similarity in different view is performed to
obtain the final drug similarity vector matrix _R_ _[(]_ _sim_ [708][×][708] _[)]_ . In
the same way, we can get the final drug similarity matrix
_P_ _[(]_ _sim_ [1512][×][1512] _[)]_ . The process is as follows:


_S_ _i_ = _Vtanh_ � _W_ _[j]_ _R_ _[(]_ _i_ _[j][)]_ [+] _[ b]_ � (10)


The _R_ _[(]_ _i_ _[j][)]_ represents the similarity between the drug
nodes obtained from the j level for the i-th drug. _W_ _[j]_ is
the weight matrix, _V_ and _b_ are the bias vector and weight
vector, respectively.


_exp(S_ _i_ _)_
_γ_ _i_ = (11)
~~�~~ _j_ =5 _[exp][(][S]_ _[j]_ _[)]_ [,]


where _exp_ is an exponential function and _γ_ _i_ indicates the
score in the layer attention mechanism.


_g_ _i_ = [�] _jε(_ 1,2,3,4,5 _)_ _[γ]_ _[i]_ _[R]_ _[(]_ _i_ _[j][)]_ (12)


_g_ _i_ represents the result of fusing the similarity of drugs
from different angles through attention. We can use the
same method to obtain the protein similarity vector _p_ _i_
from different angles through the attention mechanism
fusion.


**CNN prediction model**

We stitch the similarity information of drugs and
proteins with the heterogeneous information network



Figure 3. CNN prediction model.


learned by GAT with RWR as the input of the CNN, and
the output is a pair of drug–protein interaction scores.
Specifically, we stack up and down _R_ _[(][sim][)]_ and _G_ _[drug]_ [−] _[protein]_

to get three-dimensional matrix _X_ _[(]_ _r_ [708][×][2][×][708] _[)]_ . Similarly,
we stack up and down _G_ _[protein]_ [−] _[drug]_ and _P_ _[(][sim][)]_ to get threedimensional matrix _X_ _p_ _[(]_ [1512][×][2][×][1512] _[)]_ . Then, we connect
_X_ _[(]_ _r_ [708][×][2][×][708] _[)]_ and _X_ _p_ _[(]_ [1512][×][2][×][1512] _[)]_ left and right to get the
paired drug–protein embedding matrix. For example, the
embedding matrix of drug _r_ _i_ and protein _p_ _j_ is _X_ _ij_ _[(]_ [1][×][2][×][2220] _[)]_ .
Finally, we stitch the paired drug _r_ _i_ and protein _p_ _j_ through
the embedding representation learned by the GAT with
RWR to get _S_ _ij_ _[(]_ [1][×][2][×][2920] _[)]_ .
We use the spliced paired drug and protein node feature matrix as the input of CNN. In our model, in order to
learn the edge information of the drug protein feature
matrix, first perform zero padding. The CNN model is
shown in Figure 3. The first convolutional layer is composed of 16 convolution kernels, and the size of each convolution kernel is 2×2. Convolution operations between
different convolution kernels are independent. In the
convolutional layer, the feature map of the previous layer
is convolved with the learnable kernel, and the output
feature is formed through the activation function. The
weight sharing is computed by


_x_ _[(]_ _k_ _[l][)]_ _[(][i]_ [,] _[ j][)]_ [ =] _[ f]_ � _x_ _[(][l]_ [−][1] _[)]_ _(i_, _j)_ ∗ _W_ _k_ _[(][l][)]_ [+] _[ b]_ _[(]_ _k_ _[l][)]_ �, (13)


where _f_ indicates the activation function, _x_ _[(]_ _k_ _[l][)]_ _[(][i]_ [,] _[ j][)]_ [ rep-]
resents the feature map obtained when the upper left
corner of the _k_ th filter on the _l_ − 1 layer is moved to the
_i_ th row and _k_ th column of the feature matrix. ∗ is the _k_ th
convolution kernel _W_ _k_ _[(][l][)]_ [convolution operation on the] _[ l]_ [-th]
layer. _j_ is the kernel size.

The ReLU activation function can effectively simplify
our calculation process and avoid gradient disappearance and explosion. After the convolution operation, the
ReLU activation function is applied to process the feature
map obtained in the last step:


_f_ _(x)_ = _max(_ 0, _x)_ (14)


Next layer is the max-pooling layer, which can extract
the maximum value in the feature area. The max-pooling


layer is equivalent to retaining only the most important information after convolution, thereby reducing the
dimensionality of each feature map. In the neural network model, the use of batch normalization technology
can optimize the efficiency and stability of the neural
network. It can also handle the problem of overfitting
and reduce the risk of gradient loss. After a convolutional
layer and a pooling layer, we added a normalization processing layer. The second convolutional layer and pooling
layer are exactly the same as the first. Dropout regularization is usually used in CNN models to avoid overfitting
by dropping a certain proportion of neurons. After the
second pooling layer, we use dropout 0.5 which means
that each output of the pooling layer will randomly select
a number in the range of 0 to 1, and if the random number
is less than 0.5, the output will be discarded.

We will feed the final output vector to the fully connected layer and the softmax layer to obtain the _r_ _i_ − _p_ _j_
interaction score.


_S(x)_ _i_ = _softmax(Wx_ _i_ + _b)_ (15)


In our model, the cross-entropy between the true distribution of the drug protein and the interaction prediction score is used as a loss function, as shown below:


_Loss_ = − [�] _jεT_ _[s]_ _[j]_ _[logy]_ _[j]_ [,] (16)


where _jεT_ and T is our training sample. _s_ _j_ is the probability score of drug–protein interaction. _y_ _j_ is the label of the
actual interaction between the drug and the protein.


Experiments


In this section, we first introduce the parameter setting
and evaluation metrics. Then, we present the result analysis.


**Parameter setting**

According to the known drug–protein interaction matrix,
we take all known drug–protein interaction data as positive samples, and conduct two rounds of negative sampling at the same time. First, randomly select negative samples with the same number of positive samples
from all negative samples, and finally get 3846 samples,
including 1923 positive samples and 1923 negative samples. Then randomly selected negative samples with 10
times the number of positive samples from all negative
samples and finally got 21 153 samples including 1923
positive samples and 19 230 negative samples. The samples are divided into 10 equal subsets, 90% are used for
training and 10% for testing at a time. We use the 10fold cross validation method to evaluate the proposed
model.


In the DTIs prediction model, the splicing high-latitude
node vector is dimensionally reduced by DAE, and the



_Heterogeneous GAT_ | 7


embedding vector dimension is 500. Adaptive moment
estimation algorithm (Adam) [40] is selected to minimize
the objective function. The learning rate is set to 0.001
and dropout percentage is 0.5. L2 regularization is added
to reduce overfitting, and the model number of epochs is
100.


**Performance evaluation**


We performed a cross validation of known experimental drug–protein scores to assess the performance of
DTIHNC. The performance is evaluated by the following
parameter indicators: ACC (overall accuracy), Recall, Precision and F1-score, which are widely used in computational biology and are expressed as


_TP_ + _TN_
_ACC_ = _TP_ + _FP_ + _FN_ + _TN_ (17)


_TP_
_Recall_ = _TP_ + _FN_ (18)


_TP_
_Precision_ = _TP_ + _FP_ (19)


_F_ 1 − _score_ = 2∗ _TP_ 2+∗ _FPTP_ + _FN_ (20)


where TP stands for true positives, TN for true negatives,
FP for false positives and FN for false negatives.

We adapt Area Under the Receiver Operating Characteristic Curve (AUROC) and Area Under the PrecisionRecall Curve (AUPR) as the evaluation metrics. The larger
the value of AUROC and AUPR, the better the performance of the method. We compare our proposed method
with the following three baseline algorithms: DTINET

[18], GADTI [29] and NeoDTI [26].


**Overall comparison**

We compare our method with several benchmark
methods, including DTINet, GADTI and NeoDTI. DTINet
predicted DTIs from a constructed heterogeneous network, which integrated diverse drug-related information.
GADTI integrated diverse drug-related and target-related
datasets into heterogeneous network, and predicted
DTIs by a graph autoencoder approach. NeoDTI integrated diverse information from heterogeneous network
data and automatically learned topology-preserving
representations of drugs and targets to facilitate DTI
prediction.

As shown in Figure 4, Figure 5 and Table 2, the experiment results show that our method achieves the highest
performance in all the tasks of predicting DTIs with
different ratios between positive samples (known DTI
pairs) and negative samples (unknown DTI pairs). Among
them, we observe that our method achieves significant
improvement in terms of AUPR (3.1% over GADTI with
the ratio = 1:10 between positive and negative samples
and 0.93% over NeoDTI with the ratio = 1:1 between
positive and negative samples), while GADTI achieves


8 | _Jiang_ et al.


Figure 4. The ROC curves and PR curves of different methods(# positive: # negative = 1:1 and # positive: # negative = 1:10).



the best performance among baselines with the ratio =
1:10 and NeoDTI achieves the best performance among
baselines with the ratio = 1:1. Also, we observe that
DTIHNC achieves improvement in terms of AUROC (0.4%
over GADTI with the ratio = 1:10 and 0.6% over GADTI
with the ratio = 1:1), while GADTI achieves the best
performance among baselines. In summary, the results
suggest that exploring multi-hop neighbor information
from a heterogeneous network and cross-modal similarities information can improve the accuracy for DTIs
prediction.


**Case study**

To further demonstrate the performance of our model
to discover potential DTIs, we conduct case studies on



five drugs, namely Ziprasidone, Amitriptyline, Asenapine, Quetiapine and Aripiprazole. The five drugs we have
chosen are for psychiatric diseases. For each of these
drugs, the drug candidate–protein interactions are prioritized according to their interaction scores, and the
top-10 candidate proteins are collected, and as listed in
Table 3.


We consult the following databases for drug interaction information. DrugCentral is a similar database
that collates information on pharmaceutical active
ingredients and pharmacological effects approved by the
FDA and other regulatory agencies. KEGG is a practical
database resource for understanding advanced functions
and biological systems, especially large molecular data
sets. CTD is a robust, publicly available database. It


_Heterogeneous GAT_ | 9


Figure 5. Comparison between DTINet, GADTI, NeoDTI and our method in terms of AUROC and AUPR based on 10-fold cross validation(# positive: #
negative = 1:1 and # positive: # negative = 1:10).


**Table 2.** Comparison of DTIHNC with DTInet, GADTI and NeoDTI models


**ACC** **Precision** **Recall** **F1-Score**


DTInet 1:1 0.527 1 0.057 0.108

1:10 0.913 1 0.052 0.099

GADTI 1:1 0.725 0.988 0.456 0.624

1:10 0.943 0.986 0.383 0.552

NeoDTI 1:1 0.881 0.893 0.865 0.879

1:10 0.962 0.919 0.648 0.760

DTIHNC 1:1 0.920 0.900 0.930 0.910

1:10 0.970 0.964 0.865 0.807



provides manually curated information about chemical–
gene/ protein interactions, chemical–disease and gene–
disease relationships. Among the 50 candidate proteins,
41 candidates are included by DrugCentral, four candidates are recorded by KEGG and five candidates are
recorded by CTD, respectively. It indicates that candidate
targets indeed interact with corresponding drugs. In
summary, the case studies reveal that our method can
accurately detect potential drug–protein interactions.



Conclusion


In this work, we propose DTIHNC, an integration
of multi-modal similarities and neighbor topology
information learned from multi-entity heterogeneous
network(e.g. drugs, proteins, diseases, side effects) for
the DTI prediction. Specifically, we first propose a new
feature encoding scheme associating drugs, proteins,
diseases and side effects, which employs DAE to extract


10 | _Jiang_ et al.


**Table 3.** The top 10 candidate targets for five drugs


**Drug name** **Rank** **Target** **Evidence** **Rank** **Target** **Evidence**


Ziprasidone 1 ADRA1A DrugCentral 6 CHRM5 DrugCentral
2 HTR1B DrugCentral 7 HTR1A CTD
3 DRD5 DrugCentral 8 CHRM3 DrugCentral
4 ADRA2C DrugCentral 9 ADRA2B DrugCentral
5 DRD2 CTD 10 CHRM1 DrugCentral
Amitriptyline 1 ADRA1A DrugCentral 6 ORPD1 DrugCentral
2 CHRM3 DrugCentral 7 CHRM2 DrugCentral
3 CHAM5 DrugCentral 8 HTR2A DrugCentral
4 ADRA1D DrugCentral 9 SLC6A4 KEGG
5 CHRM1 DrugCentral 10 CHRM4 DrugCentral
Asenapine 1 ADRA1A DrugCentral 6 DRD2 KEGG
2 ADRB1 DrugCentral 7 HTR2C KEGG
3 ADRA2C DrugCentral 8 DRD4 DrugCentral
4 HTR1B DrugCentral 9 ADRA2A DrugCentral
5 HRH2 CTD 10 DRD1 DrugCentral
Quetiapine 1 ADRA1A DrugCentral 6 CHRM3 DrugCentral
2 CHRM5 DrugCentral 7 CHRM1 DrugCentral
3 DRD5 DrugCentral 8 DRD1 DrugCentral
4 HTR1D DrugCentral 9 HTR2C DrugCentral
5 HTR1B DrugCentral 10 HTR1A DrugCentral
Aripiprazole 1 DRD5 DrugCentral 6 CHRM1 DrugCentral
2 ADRA2C DrugCentral 7 ADRA2B DrugCentral
3 CHRM5 DrugCentral 8 HTR1A KEGG
4 CHRM3 DrugCentral 9 ADRA2A DrugCentral

5 DRD2 CTD 10 HTR2C CTD



informative features. Then, we construct a heterogeneous network by combining with multiple biological
networks. Besides, a heterogeneous graph attention
operations is exploited to update the embedding of a
node based on information in its 1-hop neighbors, and
RWR aware graph attention is utilized to integrate multihop neighbor information through a larger neighborhood
region. Third, we calculate multi-modal drug similarities
and protein similarities, and a hierarchical attention
mechanism method was proposed to capture informative features from similarities vectors. Finally, a multiplelayer CNN is designed to deeply integrated similarity
information of drugs and proteins with the neighbor
topology information. Experimental results show that
our method outperforms baseline methods on a realworld DTI prediction task.


Funding


This work is supported by the Fundamental Research
Funds for the Central Universities 2412018QD022, NSFC
(under Grant No.61976050, 61972384) and Jilin Provincial
Science and Technology Department under Grant No.
20190302109GX.


**Key Points**


  - A new feature encoding scheme associating drugs, proteins, diseases and side effects is proposed, which
employs denoising autoencoder to extract informative
features.




  - In heterogeneous network, a heterogeneous graph attention operations is exploited to update the embedding
of a node based on information in its 1-hop neighbors,
and for multi-hop neighbor information, we propose
random walk with restart aware graph attention to integrate more information through a larger neighborhood
region.

  - We calculate cross-modal drug similarities and protein
similarities from which an attention mechanism method

was proposed to obtain attribute embedding.

  - A multiple-layer CNN is designed to deeply integrated
similarity information of drugs and proteins with the
embedding features obtained from heterogeneous network.


References


1. Huang Y, Zhu L, Tan H, _et al._ Predicting Drug-Target on Heterogeneous Network with Co-rank. _The 8th International Conference on_
_Computer Engineering and Networks (CENet)_ 2018.
2. Yao L, Evans JA, Rzhetsky A. Novel opportunities for computational biology and sociology in drug discovery: Corrected paper.

_Trends Biotechnol_ 2010.

3. Ezzat A, Min W. Xiaoli Li and Chee Keong Kwoh.
Computational prediction of drug-target interactions using
chemogenomic approaches: an empirical survey. _Brief Bioinform_

2018.

4. Bagherian M, Sabeti E, Wang K, _et al._ Machine learning
approaches and databases for prediction of drug-target interaction: a survey paper. _Brief Bioinform_ 2021.


5. Chen J, Zhang L, Cheng K, _et al._ Exploring Multi-level Mutual
Information for Drug-target Interaction Prediction. _2020 IEEE_
_International Conference on Bioinformatics and Biomedicine (BIBM)_

2020.

6. Mei J, Kwoh CK, Yang P, _et al._ Drug-target interaction prediction
by learning from local information and neighbors. _Bioinformatics_

2013.

7. Shi H, Liu S, Chen J, _et al._ Predicting drug-target interactions
using Lasso with random forest based on evolutionary informa
tion and chemical structure. _Genomics_ 2019.

8. Peng J, Wang Y, Guan J, _et al._ An end-to-end heterogeneous
graph representation learning-based framework for drug-target
interaction prediction. _Brief Bioinform_ 2021.
9. Lee I, Nam H. Identification of drug-target interaction by a

random walk with restart method on an interactome network.

_BMC Bioinformatics_ 2018.
10. Mst K, Md H, Kurata H. PreAIP: computational prediction of antiinflammatory peptides by integrating multiple complementary
features. _Front Genet_ 2019;129.

11. Mehedi HM, Schaduangrat N, Basith S, _et al._ Balachandran
HLPpred-Fuse: improved and robust prediction of hemolytic
peptide and its activity by fusing multiple feature representation. _Bioinformatics_ 2020;3350–6.
12. Mehedi HM, Ashad AM, Shoombuatong W, _et al._ NeuroPredFRL: an interpretable prediction model for identifying neuropeptide using feature representation learning. _Brief Bioinform_

2021.

13. Mehedi HM, Basith S, Khatun MS, _et al._ Meta-i6mA: an inter
species predictor for identifying DNA N 6-methyladenine sites
of plant genomes by exploiting informative features in an integrative machine-learning framework. _Brief Bioinform_ 2021.
14. Gang F, Ding Y, Seal A, _et al._ Predicting drug target interactions
using meta-path-based semantic network analysis. _BMC Bioinfor-_

_matics_ 2016.

15. Bleakley K, Yamanishi Y. Supervised prediction of drug-target
interactions using bipartite local models. _Bioinformatics_ 2009.
16. Ding Y, Tang J, Guo F. Identification of Drug-Target Interactions
via Multi-view Graph Regularized Link Propagation Model. _Neu-_
_rocomputing_ 2021.
17. Ding Y, Tang J, Guo F. Identification of Drug-Target Interactions via Dual Laplacian Regularized Least Squares with
Multiple Kernel Fusion. _Knowledge-Based Systems, pages_ 2020;

**106254** .

18. Luo Y, Zhou X, Zhou J, _et al._ A network integration approach
for drug-target interaction prediction and computational drug
repositioning from heterogeneous information. _Nat Commun_

2017.

19. Yan X, Zhang S, He C. Prediction of drug-target interaction
by integrating diverse heterogeneous information source with
multiple kernel learning and clustering methods. _Comput Biol_

_Chem_ 2019.

20. Zhang ZC, Zhang XF, Wu M, _et al._ A Graph Regularized Generalized Matrix Factorization Model for Predicting Links in Biomedical Bipartite Networks. _Bioinformatics, pages_ 2020; **11** .
21. Chen X, Liu M, Yan G. Drug-target interaction prediction
by random walk on the heterogeneous network. _Mol Biosyst_

2012.



_Heterogeneous GAT_ | 11


22. Zheng X, Ding H, Mamitsuka H, _et al._ Collaborative matrix factorization with multiple similarities for predicting drug-target
interactions. _Proceedings of the 19th ACM SIGKDD international_
_conference on Knowledge discovery and data mining_ 2013;1025–33.
23. Ezzat A, Zhao P, Min W, _et al._ Drug-Target Interaction Prediction
with Graph Regularized Matrix Factorization. _IEEE/ACM Trans_
_Comput Biol Bioinform_ 2018.
24. Liu Y, Min W, Miao C, _et al._ Neighborhood Regularized Logistic
Matrix Factorization for Drug-Target Interaction Prediction. _PLoS_
_Comput Biol_ 2016.
25. Andrac CA, Ricardo BC, Costa IG. A multiple kernel learning algorithm for drug-target interaction prediction. _BMC Bioinformatics_

2016;46.

26. Wan F, Hong L, Xiao A, _et al._ NeoDTI: neural integration of neighbor information from a heterogeneous network for discovering
new drug-target interactions. _Bioinformatics_ 2021.
27. Zheng X, He S, Song X, _et al._ DTI-RCNN: New Efficient Hybrid
Neural Network Model to Predict Drug-Target Interactions. _27th_
_International Conference on Artificial Neural Networks_ 2018.
28. Manoochehri HE, Pillai A, Nourani M. Graph Convolutional Networks for Predicting Drug-Protein Interactions. _IEEE International_
_Conference on Bioinformatics and Biomedicine(BIBM)_ 2019.
29. Liu Z, Chen Q, Lan W, _et al._ GADTI: Graph Autoencoder Approach
for DTI Prediction From Heterogeneous Network. _Front Genet_

2021.

30. Jin X, Liu MM, Wang L, _et al._ Multi-Resolutional Collaborative
Heterogeneous Graph Convolutional Auto-Encoder for DrugTarget Interaction Prediction. _IEEE International Conference on_
_Bioinformatics and Biomedicine(BIBM)_ 2020.
31. Luo Y, Zhao X, Zhou J, _et al._ A network integration approach
for drug-target interaction prediction and computational drug
repositioning from heterogeneous information. _Nat Commun_

2017.

32. Knox C, Law V, Jewison T, _et al._ DrugBank 3.0: a comprehensive
resource for ‘Omics’ research on drugs. _Nucleic Acids Res_ 2010.
33. Prasad TSK, Goel R, Kandasamy K, _et al._ Human Protein Reference Database - 2009 update. _Nucleic Acids Res_ 2009.
34. Kuhn M, Campillos M, Letunic I, _et al._ A side effect resource to
capture phenotypic effects of drugs. _Mol Syst Biol_ 2014.
35. Davis AP, Murphy CG, Johnson RJ, _et al._ The Comparative Toxicogenomics Database: update 2013. _Nucleic Acids Res_ 2012.
36. Yousefi-Azar M, Varadharajan V, Hamey L, _et al._ Autoencoderbased feature learning for cyber security applications. _2017_
_International Joint Conference on Neural Networks (IJCNN)_ 2017.
37. Jiang L, Wang P, Cheng K, _et al._ EduHawkes: A Neural Hawkes
Process Approach for Online Study Behavior Modeling. _Proceed-_
_ings of the 2021 SIAM International Conference on Data Mining(SDM)_

2021.

38. Tong H, Faloutsos C, Pan J-Y. Random walk with restart: fast
solutions and applications. _Knowledge and Information Systems_

2008.

39. Xuan P, Chen B. Tiangang Zhang and Yan Yang. Prediction of
drug-target interactions based on network representation learning and ensemble learning. _IEEE/ACM Trans Comput Biol Bioinform_

2020.

40. Kingma DP, Ba J. Adam: A Method for Stochastic Optimization.

_ICLR_ 2015.


