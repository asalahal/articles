This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 1

## **Predicting Drug-Target Interactions via Dual-Stream Graph Neural** **Network**


Yuhui Li [1], Wei Liang [1,2,3], Li Peng [2,3], Dafang Zhang [1], Cheng Yang [2,3], Kuan-Ching Li [2,3]

1 College of Computer Science and Electronic Engineering, Hunan University, Changsha, 410082, China
2 School of Computer Science and Engineering, Hunan University of Science and Technology, Xiangtan, 411201, China
3 Hunan Key Laboratory for Service Computing and Novel Software Technology, Xiangtan, 411201, China


**Drug target interaction prediction is a crucial stage in drug discovery. However, brute-force search over a compound database**
**is financially infeasible. We have witnessed the increasing measured drug-target interactions records in recent years, and the rich**
**drug/protein-related information allows the usage of graph machine learning. Despite the advances in deep learning-enabled drug-**
**target interaction, there are still open challenges: (1) rich and complex relationship between drugs and proteins can be explored; (2)**
**the intermediate node is not calibrated in the heterogeneous graph. To tackle with above issues, this paper proposed a framework**
**named DSG-DTI. Specifically, DSG-DTI has the heterogeneous graph autoencoder and heterogeneous attention network-based Matrix**
**Completion. Our framework ensures that the known types of nodes (e.g., drug, target, side effects, diseases) are precisely embedded**
**into high-dimensional space with our pretraining skills. Also, the attention-based heterogeneous graph-based matrix completion**
**achieves highly competitive results via effective long-range dependencies extraction. We verify our model on two public benchmarks.**
**The result of two publicly available benchmark application programs show that the proposed scheme effectively predicts drug-target**
**interactions and can generalize to newly registered drugs and targets with slight performance degradation, outperforming the best**
**accuracy compared with other baselines.**


_**Index Terms**_ **—deep learning, drug-target interactions, graph neural network, matrix completion**



I. I NTRODUCTION
# T HE prediction of Drug-Target Interactions (DTI) hasattracted broad interest as one of the most challenging

and critical tasks in the biochemistry domain. Finding drugtarget interactions is the foundation of drug development
that enables the downstream tasks such as drug discovery,
drug repositioning [1], drug resistance, and side-effect prediction. Different definitions have been introduced for these
downstream tasks. For example, drug repositioning refers to
finding suitable diseases for drugs that failed approval for new
therapeutic indications. In contrast, drug re-purposing suggests
that already approved drugs can also work for another disease

[2]. The core idea behind it is to maximize the potential of
existing drugs for both known and unknown diseases.
Finding interactions between drug and target is not a trivial
process, and significant challenges of novel drug development
are three folds: (1) the rarity of existing interactions between
drugs, targets and other entities such as side effects and
diseases, (2) Laborious and costly biochemical experiments,
and (3) the Not-always-deterministic nature of the DTIs
experiments. At an early stage, costly and time-consuming
high-throughput screening experiments were conducted to
examine the affinity of drug-target pairs. Unfortunately, these
experiments were unrealistic for searching for potential drugs
since millions of similar drugs and candidates were used.
Besides, many tasks like molecular properties, protein folds,
and compound interactions are arduous to have solved under
traditional approaches.
All these challenges hinder the advancement in medical
treatment. On average, it costs 2.6 billion US dollars and


Manuscript received April 19, 2005; revised August 26, 2015.



Fig. 1. Example of Drug-Target Interaction Heterogeneous Graph.


takes up to 17 years to complete the entire process of drug
development. Despite the high investment in drug research
and development, the process may fail. During the recent
COVID-19 pandemic, research institutes across the globe
investigated nearly 70 FDA-approved drugs to see whether
they could be used to treat patients. Notably, the computational
prediction methods significantly reduce the time and cost of
drug development and innovation.


_A. Prior Work and Limitations_


With the advance in Machine Learning (ML) and its successful applications in academia and industry, such as risk
management [3] and data verification [4], several computational approaches beyond traditional methods have been proposed to predict drug-target interaction. One feasible solution
is molecular docking, which predicts a complex stable 3D
structure through scoring functions. Although informative, it
needs prior expert knowledge of its spatial structure. Nevertheless, it is not always available and sometimes hard to
retrieve, especially for proteins. Other non-parameter learning



Disease


Drug


Target


Side Effect


(a) Node Types



(b) Heterogeneous Graph



(d) Neighbors



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 2



techniques, such as random forest (RF), have been used as
an alternative to design scoring functions artificially. Later,
researchers found that RF-based scoring functions failed in
virtual screening and docking tests. They oversimplified the
protein-ligand complex description and led to massive information loss, resulting in low accuracy in those tasks.
For parameter learning approaches, it is assumed that similar
drugs have similar interactions with targets and develop matrix
factorization-based methods [5] [6]. However, these methods
only model the implicit relationship for the drug-target pairs
without considering the rich chemical features, so those suffering from the poor performance have also encountered scalable
issues. Besides, [7] proposed the construction of a gradient
boosting machine by utilizing drug-target pair similarity.
Due to the steadily increasing labelled data and demanding
computation capacity, deep learning has become a state-ofthe-art solution in many other research fields [8]. Influenced
by the recent advancements in deep learning, several works in
the literature applied neural networks to drug-target interaction
tasks. For example, [9] proposed a 1D-convolution-based
neural network, named DeepDTA, targeting input sequences
for affinity regression. Based on DeepDTA, [10] enrich the
features and achieve better accuracy. As of now, deep learning
models perform the best in drug-target prediction benchmarks.
Notably, the drug-target affinity prediction tasks can be directly applied to drug-target interaction prediction by setting
a threshold.

Modelling drug and target sequence as 1D structure loses
information, as molecules and proteins have rich structural and
spatial information, and sequences cannot contain in SMILES
and sequences. Such kinds of representation may degrade the
prediction accuracy since they can not enable the model to
learn atoms and bonds’ functionality and spatial interactions.
Due to such, inspired by graph convolution, [11], [12], [13],

[14], [15], [16], [17], [18] adapted various kinds of message
passing network learning molecular representation for downstream tasks such as drug discovery, drug-drug interactions,
molecule property, and chemical reaction. However, graph
representation might not be easy to retrieve due to difficulty
incorporating coordinate information and the limitation in
graph pooling, even for small molecules.
In recent years, network-based methods have attracted
enormous research attention, with rapid advances in graph
machine learning. Compared to docking-based and ligandbased approaches, the advantages of network-based methods
are twofold. First, network-based methods generally perform
well without utilizing three-dimensional structural information; secondly, the data collection is much more friendly.
Docking-based methods are challenging to perform well
when the targets are never-seen, while ligand-based methods
require an extensive database with rich binding data. Various computational models based on network data have been
developed to predict DTIs. DTINet [1] extracted the lowdimensional representation from heterogeneous data sources
and applied inductive matrix completion to predict the probability of the drug-target interaction. MSCMF [19] applied matrix factorization on multiple similarity matrices and computed
the weighted aggregation to predict the interactions. BLM-NII




[20] integrated neighbour-based interaction-profile inferring
into a bipartite local model. These methods capture shallow
dependency in a heterogeneous graph, leading to sub-optimal
performance. GADTI [21] and HampDTI [22] utilize graph
auto-encoder and automatic meta-path learning mechanism
to learn representation for drug-target interactions prediction.
However, the graph convolution network they used may suffer
from an over-smooth problem [23], indicating that they may
fail to capture long-range dependencies in the heterogeneous
graph.


_B. Our Contribution_


In this article, we propose **DSG-DTI**, a novel neural network architecture capable of capturing heterogeneous relationships from biology networks. Evaluation analysis shows that
this approach performs well among the baseline methods on
two well-known drug-target interaction benchmark programs.
Our contributions are summarized as follows:


_•_ To propose a novel drug-target interaction prediction
model named DSG-DTI, which has two major components: a heterogeneous graph encoder for meta-pathbased relation learning and a homogeneous graph convolution network for node-level embedding refinement.
Compared with existing methods, the significant advantage is that we can fully utilize the power of heterogeneous graphs to calibrate and refine the embeddings.

_•_ To conduct experiments using two public benchmark
datasets to evaluate the performance of proposed methods, demonstrating their effectiveness and great potential.

The remainder of this article is organized as follows. Section
II categorizes various methods and reviews representative
works for each category. Section III introduces the problem
definition and presents the proposed methods. Section IV evaluates the proposed model with different experimentations, and
the results are presented and analyzed, and finally, concluding
remarks and future directions are given in Section V.


II. R ELATED W ORK


This section reviews recent works on DTIs, approached
under the following category: similarity-based, deep-learningbased, feature-based, matrix factorization-based, networkbased, and hybrid-based. Each of them is separately presented
and discussed.


_A. Similarity-based DTI_


Similarity-based methods focus on setting up similarity
(distance) functions to perform DTI prediction by first collecting similarity scores for known drug-drug, target-target,
and drug-target pairs [24]. Then, the similarity of a new
drug/target for the known pairs based on the collected data is
obtained. Intuitively, the similarity function’s quality largely
influences the prediction accuracy, and thus several ways
to define a similarity function are developed. In previous
works, Euclidean distance [25], biological information [26],
and pharmacological similarity [27] are employed to measure
the distance between drugs and targets. Based on the similarity



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 3



measures, Nearest-Neighbor (NN) based methods [28], [29],

[30], logistic regression [26], and clustering algorithm [31],

[32] can then be applied to facilitate the prediction task of
DTI. However, the similarity-based approaches heavily rely
on the interaction information between drugs and targets and
unfortunately, it fails to deal when a considerable amount of
unlabeled data is given.


_B. Deep Learning-based DTI_


Motivated by remarkable achievement in computer vision,
recommender systems [33], [34], [35], service management

[36], abnormal detection [37], [38], and several others, Deep
Learning (DL) quickly spread out to other research fields, including bioinformatics, medicine and pharmacology. In recent
works, various types of DL-based models, such as autoencoder

[39], multi-layer perceptron [40], [41], and convolutional
neural networks [9], [40], have been established for new DTI
prediction. The significant advantage of DL-based models is
the strong ability to learn high-quality representation from
raw data, and the non-linear transformation enables the model
to capture highly complicated hidden patterns from data.
Nevertheless, the performance of the DL-based methods may
be hindered when sufficient information is not available.


_C. Feature-based DTI_


A broad range of methods can be categorized into featurebased DTI methods. Feature-based methods map the pharmacological/genomic feature of drugs/targets into a feature
space and represent a pair of drugs and targets in terms of
a feature vector with a certain length [42], [43]. Later, various
feature-based machine learning methods were developed for
DTI prediction, such as SVM [44], [45], random forest [46],

[47], and kernel regression-based algorithm [48].


_D. Matrix Factorization-based DTI_


Matrix Factorization (MF) based DTI methods decompose
the interaction matrix into two low-rank matrices. The missing
entries of the matrix (e.g., no interaction records) will be
obtained by performing the matrix completion technique with
the two matrices. Derived from the idea of collaborative

filtering, [49] proposed a probabilistic matrix factorization
(PMF) model for DTI prediction. As an improved version,
by incorporating a target bias into the model, a Bayesian
Ranking MF-based approach is adopted for DTI prediction

[50]. Inspired by the excellent performance of the MF-based
methods, as they eliminate the dependency on the similarity of
the targets and drugs, several variants are found in the recent
works [51], [52], [53]. However, the ever-growing DTI data
may exceed the capacity of matrix representations.


_E. Network-based DTI_


Network-based DTI methods refer to those that seek drugtarget similarity through DTI topological structure [54]. Some
existing works [55], [56], [57] transform DTI data to a heterogeneous network graph, where random walk-based methods



can then be applied to achieve the DTI prediction. For instance, [1] developed a computational pipeline that learns the
topological properties of nodes in the heterogeneous network
to predict the DTI.


_F. Hybrid-based DTI_


Hybrid-based methods refer to any combination of the
categories mentioned earlier. [58] designed a framework for
utilizing both similarity matrix from kernel matrix and feature
transformation for DTI prediction. [59] combined PMF with a
denoising autoencoder and succeeded in facilitating the coldstart problem. Deriving the idea from recommender systems,

[60] employed the Deep Matrix Factorization approach to
reveal the non-linearity relations among DTI, and integrating
different machine learning methods can lead to better predicting performance. However, the high complexity of the hybrid
models remains of concern.


_G. Binding Affinity Prediction_


Predicting drug-target binding affinity has been of relevant
studies for quite a long time [42] [61], as it is a critical
stage in the entire drug development process. Earlier methods
artificially design docking [62], [63], [64] and score functions

[65], [66], [67], [68], [69], through extensive expert knowledge are required. Later, statistical machine learning methods
were developed. These data-driven methods extract features
from large-scale data to develop classification and regression
ability. For example, [46] used Random Forest (RF) to learn
scoring functions automatically from a given dataset. In [45],
a SVM-based model trained by associating sets of individual
energy terms retrieved from molecular docking with the known
binding affinity is introduced. These approaches are dependent
on hand-craft feature engineering, which also requires expert
knowledge. However, low generality is still predominant.


_H. Graph Neural Network for DTI_


In 2016, Kipf et al. [70] first proposed graph convolution
neural networks as the start of a new era of graph representation learning. To exemplify, GraphSAGE [71] proposed a
novel sampling and aggregation strategy to learn an inductive
and efficient model on a large graph, GAT was introduced
by the attention mechanism that gives different weights to
neighbourhood nodes when aggregating, and [72] analyzed
and discussed the power of GNNs through the WeisfeilerLehman test and designed a graph isomorphism network (GIN)
to maximize the discriminative ability.
Influenced by the significant advantages of graph learning
in modelling graph data, recent research has focused on applying them in computation chemistry tasks such as molecular
properties prediction, chemical reaction, and drug discovery.
MPNN reformulates graph convolution into a more general
framework, named MPNN, and utilizes it for predicting molecular properties. SchNet [73] proposes and achieves remarkable improvement in equilibrium molecules and molecular
dynamics trajectories benchmarks. The work [11] develops a
general spatial convolution operation for learning atomic-level



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 4



chemical interactions from the molecular graph and success in
the protein-ligand prediction task. [74] model molecules as a
graph and verified the effectiveness and superiority of graph
learning representation over SMILES.
Recent studies applied transformers in modelling graph
structure data and achieved similar performance, demonstrating that transformers can be an alternative solution beyond
GNNs. Transformer shows its potential on a unified model
for different tasks as it can learn cross-modal and multimodal

representation [75] [76]. In drug discovery, [77] leverages
transformer in learning the protein sequence and drug se
quence.


_I. Utilities_

Showed that machine learning and deep learning-enabled
methods are widely adopted in computational biology and
chemistry, many utilities are proposed to facilitate the entire
process. For data preprocessing, RDKit, DeepChem [78], and
OpenBabel introduce many functions to cope with molecules
and proteins in different formats. Several libraries provided
reproduced models and datasets for simplifying model development, such as DeepChem, DGL Lifescience, PytorchGeometric, and Paddle Helix.


III. P ROPOSED M ETHOD DSG-DTI

_A. DTI Workflow_

The drug-target interactions prediction for a new drug/target
consists of four steps, as per the illustration given in Figure 2.
First, we add the new drug/target into the database, then some
basic silico experiments to get its profile are conducted, such
as its similarity to known drugs, and the disease it is related
to. Next, we build a heterogeneous network to include the
new drug/target, so DTI-prediction methods are performed to
predict the interaction probability. Finally, some downstream
tasks could be enabled according to the prediction results.


_B. Heterogeneous Graph_

A heterogeneous graph is a complex network that contains
multiple types of entities and various kinds of edges.
_Definition 1:_ **Heterogeneous Graph.** A heterogeneous
graph consists of a set of nodes _V_ and a set of edges _E_, denoted
as _G_ = ( _V, E_ ). The nodes are in _t_ different types, denoted
as _{V_ [(1)] _, V_ [(2)] _, ..., V_ [(] _[t]_ [)] _}_, while the edges also have different
types, named relation, denoted as _{R_ [(1)] _, R_ [(2)] _, ..., R_ [(] _[l]_ [)] _}_ . Thus,
_R_
an edge is defined as _V_ _→_ _V_ .
**Example.** As shown in Figure 1, we construct a heterogeneous graph to model drug-target interactions with rich context
relationships, consisting of several types of nodes (drug, target,
side effect, disease) and relations (the dash lines in different
colours). In a heterogeneous graph, two objects can be linked
by different paths, also known as meta-paths.
_Definition 2:_ **Meta-path.** A meta-path Φ is a path in the
_R_ 1 _R_ 2 _R_ _k_
form of _V_ 1 _→_ _V_ 2 _→_ _..._ _→_ _V_ _k_ +1, consisted of multiple types
of relations between node _V_ 1 to node _V_ _k_ +1 .
**Example.** As shown in Figure 1, a drug can be connected to
a side effect via multiple meta-paths (e.g., Drug-Disease-DrugSide Effect). Given two types of nodes, a set of meta-paths



can connect to them yet reveal rich semantic information in a
heterogeneous graph.
_Definition 3:_ **Neighbours.** In a heterogeneous graph, the
neighbours are slightly different in the homogeneous graph,
given that neighbours of a node _V_ 1 are defined as a set of nodes
connecting to _V_ 1 via meth-path. In this article, the neighbours
of a node always contain themselves as we add a self-loop for
the convenience of message passing.
**Example.** Taking Figure 1 as an example, the node type of
drug’s neighbours can be drug, disease, target, and side effect.


_C. Problem Definition_


We formulate the Drug-Target Interaction (DTI) problem as
a matrix completion task to determine whether the drug-target
pair will interact. A unique id represents drugs and targets in
the proposed framework, where the drug is _D_, the target is _T_,
and the interaction _I_ .

_Problem_ _1:_ **DTI** **Prediction.** Given _m_ drugs embeddings _{E_ _d_ 1 _, E_ _d_ 2 _, ..., E_ _d_ _m_ _}_ and _n_ targets embeddings
_{E_ _t_ 1 _, E_ _t_ 2 _, ..., E_ _t_ _n_ _}_, the problem can be cast to learn a matrix
completion within a matrix of interactions _I ∈R_ _[m][×][n]_, assumed that the interest pair as ( _d_ _i_, _t_ _j_ ), the predicted interaction
probability _I_ _ij_ can be calculated as:


_Y_ ˆ _ij_ = **E** _d_ _i_ **E** _[⊤]_ _t_ _j_ _[,]_ (1)


where _Y_ [ˆ] _ij_ is the predicted interaction probability. Typically,
we set a threshold (e.g., 0.5) to binarize the probability to 0
and 1 for the convenience of binary classification.


_D. Learning from Heterogeneous Network_


As illustrated in Figure 1, there are multiple types of nodes
and complex relationships between entities. Our goal is to
predict whether there exist links between drugs and targets.
Thus, it is more challenging than predicting the link in the
drug-target bipartite graph since the graph convolution network
can not be directly applied to handle heterogeneous graphs.
The rich semantics may significantly impact prediction.
Heterogeneous graph convolution is proposed to learn on a
heterogeneous graph. This concept is to apply a conventional
graph convolution network on a local graph that only has
two types of node, which is equivalent to the application
of a graph convolution on each meta-path induced subgraph.
After node embeddings propagation, a node can aggregate
messages received from different meta-paths and compute a
new representation. The long-range dependencies via different
meta-paths can be extracted by stacking more heterogeneous
graph convolution layers. However, the information received
from different meta-paths does not contain the same volume
of helpful information. Therefore, an attention mechanism is
utilized to distinguish the importance and filter out the noise.
We adopt Heterogeneous Attention Network [79] as our graph
convolution layer. The HAN has a two-level attention mechanism named node-level attention and semantic-level atten
tion. Graph Attention Network (GAT) implements a weighted
scheme to aggregate node embeddings of direct neighbours
to achieve node-level attention. Semantic attention aggregate



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 5















**(a) New Drug/Target** **(b) Relationship Discovery** **(c) Heterogeneous Graph Learning** **(d) Downstream Tasks**


Fig. 2. Workflow of drug-target interaction prediction.







**Heterogeneous Stream**





























Fig. 3. The framework of Heterogeneous Graph-based Matrix Completion


embeddings from different meta-paths Φ. The semantic-level
attention is achieved by:



Meta-Path: Drug-Disease-Drug



Homo Graph: Drug-Drug



_H_ [(] _[l]_ [)] =



_|_ Φ _|_
� _α_ Φ _i_ _H_ Φ [(] _[l]_ _i_ [)] _[,]_ (2)


_i_



**Complete Hetero Graph** **Meta-Path Induced Hetero Graph** **Meta-Path Reachable Graph**



where _H_ [(] _[l]_ [)] denotes the _l_ -layer node representation, and the
weight fraction _α_ calculated by:


Φ _i_ [))]
_α_ Φ _i_ = _[exp]_ _|_ Φ [(] _|_ [attn][(] _[H]_ [(] _[l]_ [)] _,_ (3)
~~�~~ _k_ attn( _H_ Φ [(] _[l]_ _k_ [)] [)]


where attn( _·_ ) is a shallow neural network that compute the
attention score. We have implemented it, as:


attn( _x_ ) = **W** 2 (Tanh( **W** 1 **x** )) _,_ (4)


where **W** **1** and **W** **2** are trainable weights, Tanh( _x_ ) =
sinh( _x_ ) _/_ cosh( _x_ ) = ( _e_ _[x]_ _−_ _e_ _[−][x]_ ) _/_ ( _e_ _[x]_ + _e_ _[−][x]_ ). The bias term
is omitted for simplicity.
For drug-target interaction prediction, as the goal is to
extract high-quality drug and target representations for better
prediction, we design a branch structure, namely, the drug
HAN and target HAN, to compute the drug-centric and targetcentric embeddings, respectively.



Fig. 4. The preprocessing workflow of heterogeneous graph attention network.


_E. Learning from Homogeneous Graph_


The heterogeneous graph attention network pre-processes
the heterogeneous graph to a homogeneous graph, as illustrated in **??** . The heterogeneous graph attention network
first extracts the subgraph from the heterogeneous network
and then compresses the meta-path induced subgraph into
a homogeneous graph. These designs have two significant
problems. First, the homogeneous graph indicates that the
meta-path’s start and end node types should be the same. Second, the induced homogeneous graph ignores the intermediate
nodes. To overcome such limitations, we propose to leverage a
heterogeneous graph generated from a heterogeneous network
to eliminate the concept of meta-path and let the information
be highly liquid in the network. In a homogeneous graph, the
information is not limited to passing only inside drugs and
targets but can pass over all other kinds of nodes. The adjacent



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 6



with known drug-protein interactions as positive samples and
randomly select unknown pairs as negative samples.
With the guide of positive and negative samples, we can optimize our model weights by minimizing the loss via stochastic
gradient descent to learn meaningful node embeddings for
drugs and targets. The loss function is formulated as:









Loss = [1]

2



� ( _y_ _ui_ _−_ _y_ ˆ _ui_ ) [2] + [1] 2

( _u,i_ ) _∈Y_ [+]



�



2



ˆ

� ( _y_ _ui_ _−_ _y_ _ui_ ) [2] _,_ (9)

( _u,i_ ) _∈Y_ _[−]_








|Drug-Drug|Drug-<br>Side Effect|Drug-<br>Disease|Drug-<br>Target|
|---|---|---|---|
|**Side Effect-**<br>**Drug**|**Identity**|||
|**Disease-**<br>**Drug**||**Identity**|**Disease-**<br>**Target**|
|**Target-**<br>**Drug**||**Target-**<br>**Disease**|**Target-**<br>**Target**|



Fig. 5. Transform a heterogeneous graph into homogeneous graph.


matrix of the heterogeneous graph induced homogeneous
graph is presented in Figure 5.
With the adjacent matrix, we apply a graph convolution
network. The multi-hop information could be captured by
stacking multiple layers theoretically. However, in practice,
GCN may suffer from the over-smooth problem, which hinders
the performance of deep graph convolution networks. To solve
this problem, we introduce a random walk with a restart
(RWR) trick to alleviate such an issue. With RWR, the graph
convolution is formulated as:


_X_ [(] _[k]_ [+1)] = (1 _−_ _α_ ) _AX_ [(] _[k]_ [)] + _αX_ [(0)] _,_ (5)


where _k_ denotes the _k_ -layers output, _α_ denotes the restart
probability, and _A_ is the adjacent matrix which corresponds
to Figure 5. _X_ [(0)] is the initial embeddings.


_F. Matrix Completion-based Decoder_


The two graph neural networks learn node representations in
the heterogeneous network, the decoder reconstructs the drugtarget interaction matrix as the predicted scores. The decoder is
to learn a score mapping: R _[d]_ _×_ R _[d]_ _→_ R. The scoring function
is presented as:


**Y** = **E** _d_ **E** _[⊤]_ _t_ _[,]_ (6)


where the **E** denotes the final node representation that aggregated from the final output of two graph neural networks:


**E** _d_ = **E** [(] _d_ _[het]_ [)] + **E** [(] _d_ _[homo]_ [)] (7)

**E** _t_ = **E** [(] _t_ _[het]_ [)] + **E** [(] _t_ _[homo]_ [)] _._ (8)


Considering the computation complexity and the performance, we use an inner-product-based score function instead
of a neural network-based scoring function.


_G. Training_


After applying the previously-mentioned modules, we successfully obtained the drug and protein embeddings. These two
embeddings are the keys to different downstream tasks, so we
can train the proposed model in supervised learning paradigms



where _Y_ [+] and _Y_ _[−]_ denotes the positive sample index set and
the negative set, respectively.
As the number of positive records in a interaction matrix
is quite small while we still need to split them into training,
validation, and testing set, we apply K-Fold cross validation
strategy. The training algorithm in pseudo-code are given in
1:


**Algorithm 1** Matrix Completion Model Training Algorithm
**Input:** nodes embeddings _N_, epochs _Epochs_,
heterogeneous graph _G_, model _f_
**Output:** trained model _f_


1: initialize optimizer;
2: create k-fold datasets;

3: **for** each fold in k-fold dataset **do**

4: extract positive and negative samples;

5: **for** each _i ∈_ [1 _, Epochs_ ] **do**
6: _h_ _d_, _h_ _t_ = model.forward( _G_, _N_ );
7: compute DTI matrix via Equation 8;
8: compute the loss by Equation 9;
9: gradient descent optimization;
10: **end for**


11: validate and record model performance;
12: **end for**

13: **return** _f_ ;


IV. P ERFORMANCE A NALYSIS


This section compares existing baseline methods in
network-based drug-target interactions fields. We aim to show
the proposed model performance and analyze the model’s
sensitivity to hyper-parameters.


_A. Runtime Configuration_


The experiments are conducted on a server equipped with
one Intel Xeon(R) Processor, 32GB memory, with one accelerator card NVIDIA GPU K80 attached. The proposed model is
implemented using the open-source DL library PyTorch 1.7.0
[(https://pytorch.org/) and DGL 0.7.0 (https://www.dgl.ai/) with](https://pytorch.org/)
Python 3.8. Experimental results for performance evaluation
are obtained on a server equipped with one Intel 12-core
Xeon processor, 81GB memory, with one NVIDIA GPU K80
accelerator card. The hyper-parameters applied are presented
in Table I.

The experiments are conducted using the hyper-parameters
in Table I, and we will further explain how these values are
chosen in the hyper-parameter sensitivity analysis.



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 7



TABLE I

H YPER -P ARAMETER C ONFIGURATION IN E XPERIMENT


Hyper Parameter Name Setting
num layers 3
dimension 256

num heads 8
learning rate 1e-3
weight decay 1e-3
dropout rate 0.4
epochs 1000
cross validation 10


_B. Dataset_


We assess the proposed scheme under a drug-target-related
dataset, named Drug-Target Interaction Heterogeneous Network (DTI-HN) [80] [1], and includes four types of nodes.
The statistics of the heterogeneous network dataset are shown
in Table II.


TABLE II

S TATISTICS OF DTI-HN DATASET


**Nodes Types** # **Nodes** **Edge Types** # **Edges**
**Drug** 708 **drug-drug interactions** 10036
– – **drug-drug similarity** 489261
– – **drug-target interactions** 1923
**Target** 1512 **target-target interactions** 7363
– – **target-target similarity** 2286144
**Disease** 5603 **disease-drug associations** 199214
– – **disease-target associations** 1596745
**Side effect** 4192 **side effect-drug associations** 80164
**Total** **12015** **Total** **4670850**


TABLE III

M ETA     - PATH AND THEIR SEMANTIC


**Meta-path** **semantic**
_ρ_ _[D]_ 1 [:] _[ Dr]_ ~~_[inter]_~~ _[Dr]_ drug-drug interactions
_ρ_ _[D]_ 2 [:] _[ Dr]_ _[sim]_ _[Dr]_ drug-drug structure similarity
_ρ_ _[D]_ 3 [:] _[ Dr]_ _[assoc]_ _[Di]_ _[assoc]_ _[Dr]_ drug-drug with common disease
_ρ_ _[D]_ 4 [:] _[ Dr]_ _[assoc]_ _[Si]_ _[assoc]_ _[Dr]_ drug-drug with common side effect
_ρ_ _[D]_ 5 [:] _[ Dr]_ _[assoc]_ _[Ta]_ _[assoc]_ _[Dr]_ drug-drug with common target protein
_ρ_ _[T]_ 1 [:] _[ Ta]_ _[inter]_ _[Ta]_ target-target interactions
_ρ_ _[T]_ 2 [:] _[ Ta]_ _[sim]_ _[Ta]_ target-target structure similarity
_ρ_ _[T]_ 3 [:] _[ Ta]_ _[assoc]_ _[Di]_ _[assoc]_ _[Ta]_ target-target with common disease
_ρ_ _[T]_ 4 [:] _[ Ta]_ _[assoc]_ _[Dr]_ _[assoc]_ _[Ta]_ target-target with common drug


_C. Baselines_


To fairly evaluate the proposed model’s performance in two
benchmarks, we select the following baselines for comparison:


_•_ **DTI-CNN** [81] learns low-dimensional vector representations of features from heterogeneous networks and adopts
CNNs as a classification model. DTI-CNN contains three
components: heterogeneous network-based feature extractor, denoising-autoencoder-based feature selector, and
CNN-based interaction predictor. Unlike the proposed
method, the DPP network is not constructed in this work,
and the drug and target were directly concatenated as the
model’s input.

_•_ **DTINet** [1] focuses on learning a low-dimensional vector
representation of features, which accurately explains the



topological properties of individual nodes in the heterogeneous network and then makes a prediction based
on these representations via a vector space projection
scheme.


_•_ **NeoDTI** [80] develops a new nonlinear end-to-end learning model that integrates diverse information from heterogeneous network data and automatically learns topologypreserving representations of drugs and targets to facilitate DTI prediction.

_•_ **GADTI** [21] uses a custom graph convolution network
and a random walk with restart (RWR) as an encoder
to obtain low-dimensional feature representations of the
DTI heterogeneous network, so then applies matrix factorization as a decoder to obtain potential DTIs.

_•_ **HampDTI** [22] automatically learns the important metapaths between drugs and targets from the HN and generates meta-path graphs. The features learned from drug
molecule graphs and target protein sequences for each
meta-path graph serve as the node attributes. A nodetype specific graph convolutional network (NSGCN) efficiently considers node type information (drugs or targets)
and is designed to learn embeddings of drugs and targets.
Finally, the embeddings from multiple meta-path graphs
are combined to predict novel DTIs.


_D. Metrics_


The AUROC and AUPR scores are used to evaluate the

models in comparison. AUROC and AUPR are two commonly
used criteria in machine learning with an imbalance number
of labels. The higher scores in both metrics indicate better
performance in predicting potential drug-target interactions.
We briefly introduce these two performance metrics as follows:


_•_ **AUROC** . Area Under Curve (AUC) is defined as the area
under the ROC curve. The horizontal axis is the false positive rate (FPR), and the vertical axis is the true positive
rate (TPR). FP and TP are computed via FPR = FP /
(TN+FP) and TPR = TP / (TP+FN). AUC ranges from
0.5 to 1.0. The higher the AUC value indicates, the better
classification accuracy. Since the DTI score is neither zero
nor one, which can be cast to binary classification, we
measure AUC as a performance indicator.

_•_ **AUPR** . The Area Under Precision-Recall (AUPR) score
refers to the area under the Precision-Recall curve, in
which Precision = TP / (TP+FN), Recall=TP / (TP+FP).
AUPR considers both Precision and recall, and it is a
more accurate metric to describe the real performance
for the extremely unbalanced dataset. As some zero
entries are considered negative samples, it is necessary
to evaluate by this metric.


_E. Experiment One_


The known drug-target interactions are considered positive
samples in the experiments, and we randomly sample unknown
associations as negative samples. The number of negative
samples is set to one and ten in different experiments as a
hyper-parameter. If negative samples are ten times as many as
positive samples, the negative sample ratio is ten. The k-fold



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 8



cross-validation is adopted to split all drug-target samples into
training and testing sets. Further, we split 20% as the validation
set from the training set. In this way, the corresponding
sampled matrix of the training set is highly sparse.
We first compare the proposed method with other baselines
as presented in Table IV. Results show that the AUC and
AP are 0.9421 and 0.6337, respectively, showing that the proposed model is superior to the representative DTI prediction
methods.


TABLE IV

P ERFORMANCE C OMPARISON ON DTI-HN DATASET

(P OSITIVE :N EGATIVE =1:10)

|Methods|AUROC|AUPR|
|---|---|---|
|NeoDTI|0.9218|0.7420|
|DTI-CNN|-|-|
|HampDTI|0.7412|0.7016|
|GADTI|0.9208|0.7388|
|DTI-Net|0.9208|0.8183|
|Ours|0.9568|0.8629|



We further conduct another experiment that takes all unknown entries as negative samples, which results are shown in
Table V. We observe that the proposed model is significantly
better than other baseline methods and improves 2-3% on AUC
and AP.


TABLE V

P ERFORMANCE C OMPARISON ON DTI-HN DATASET

(P OSITIVE :N EGATIVE =1:1)

|Methods|AUROC|AUPR|
|---|---|---|
|NeoDTI|-|-|
|DTI-CNN|0.9385|0.9461|
|HampDTI|-|-|
|GADTI|-|-|
|DTI-Net|0.9111|0.9290|
|Ours|0.9501|0.9464|



_F. Experiment Two_


Deep learning models have tens of hyper-parameters to
tune. We only select a few of them as they significantly
impact the prediction performance and set others as default.
We investigate the following hyper-parameters: (1) the depth
of the graph neural network, (2) the number of attention heads,
and (3) the hidden size. We conduct a heuristic search instead
of the expensive grid search operation. Specifically, we first
select the best depth of GNN, then fixed the depth to select
the best number of attention heads. We do it sequentially
until all hyper-parameters are investigated. In this experiment,
we randomly select the negative samples as many as positive
samples.
**Depth of Graph Neural Networks** The depth of neural
networks can significantly influence the model performance.
Before the ResNet, the depth of the neural network is no more
than about 20 layers because of the well-known degradation
problem. Similarly, the graph neural network can not be too
deep. Otherwise, they will encounter the over-smooth problem,
which can not be alleviated by known tricks (e.g., residual
connection). We test the best depth on the DTI-HN dataset,



0 . 9 6 0


0 . 9 5 5


0 . 9 5 0


0 . 9 4 5


D e p t h s


Fig. 6. The performance under different GNN depths.


and the other hyper-parameters setting is the same as reported
in Table I. We set the same depth for both as we have
two-stream neural graph networks–the heterogeneous graph
attention network and the homogeneous graph propagation
network.

According to Figure 6, the best depth of our model is two.
Although stacking more layers indicates longer dependencies
could be extracted, the noise of high-order neighbours and the
over-squash problem may hinder the performance improvement. Besides, the computation complexity is high for deep
graph neural networks, and one should consider the balance
between model performance and the computation cost.
**Number of Attention Heads** With the best depth of
graph neural network selected, we assess how the attention
heads in heterogeneous attention networks influence the model
performance. We present the effect of the multi-head attention
mechanism, which empirically brings higher quality representations. From Figure 7, results show that both AUROC
and AUPR increase with more attention heads, and due to
the balance between training/inference speed and prediction
accuracy, we stop exploring more attention heads.
**Dimensions** Lastly, we investigate how the embedding
dimension and hidden units impact the model performance.
To reduce the number of hyper-parameters, we set the input
size, hidden size, and output size to the same. Precisely, the
input embeddings size and the hidden units in the proposed
model and the final drug/protein embeddings dimension are set
to the same. In this experiment, we set depth=2 and #attention
heads=8. We set the positive and negative samples to 1 and
conducted a 10-fold cross-validation on the DTI-HN dataset.

As presented in Figure 8, the results show that the performance
increase with the dimension enlarged. However, we could
discover that the improvement gradually becomes tiny with
dimension exponentially increased. Therefore, we stop at 256
and pick it as the best option.


V. C ONCLUSIONS AND F UTURE W ORK

The drug-target binding interaction prediction task is one
of the most critical stages in drug discovery and repurposing.



0 . 9 6 5





© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 9


enhance the proposed model under investigation.



0 . 9 5





0 . 9 0


0 . 8 5


0 . 8 0


N u m H e a d s


Fig. 7. The performance under different number of attention heads.


1 . 0 0





0 . 9 5


0 . 9 0


0 . 8 5


0 . 8 0

1 6 3 2 6 4 1 2 8 2 5 6


H i d d e n S i z e


Fig. 8. The performance under different dimensions.


Inspired by deep learning and Big Data, we propose a novel
model named HGMC. Experimental results demonstrate that
the proposed model outperforms other baselines. The superior
performance is mainly due to Graph Neural Ordinary Differential Equations (GNODE) combined with a heterogeneous
graph attention network. First, the heterogeneous graph attention network automatically learns latent node representations
from heterogeneous entity-relationship graphs with dynamic
weight, leveraging node-level and semantic-level information.
Secondly, we wrap a heterogeneous graph convolution layer
into the GNODE framework to alleviate the over-smooth

problem and extract long-range meta-path information. The
GNODE is compatible with the current optimization framework, and thus, the proposed model can be trained end-to-end.
We will focus on effective graph learning as future work
since the protein-ligand complex is a challenging topic. Besides, predicting complex may have better generality and
possibly bring new knowledge. Additionally, the proposed
framework is also compatible with many other representation
layers, and we will explore such a potential to extend and



A CKNOWLEDGMENT

This work was partially supported by the National Key
Research and Development Program of China under Grant
2021YFA1000600, the National Natural Science Foundation
of China under Grants 62072170 and 61976087, the Science
and Technology Project of Department of Communications
of Hunan Provincial under Grant 202101, the Key Research
and Development Program of Hunan Province under Grant
2022GK2015, and the Hunan Provincial Natural Science

Foundation of China under Grant 2021JJ30141.


R EFERENCES


[1] Y. Luo, X. Zhao, J. Zhou, J. Yang, Y. Zhang, W. Kuang, J. Peng, L. Chen,
and J. Zeng, “A network integration approach for drug-target interaction
prediction and computational drug repositioning from heterogeneous
information,” _Nature communications_, vol. 8, no. 1, pp. 1–13, 2017.

[2] X. Zhou, W. Liang, W. Li, K. Yan, S. Shimizu, I. Kevin, and K. Wang,
“Hierarchical adversarial attacks against graph neural network based iot
network intrusion detection system,” _IEEE Internet of Things Journal_,
2021.

[3] Q. Zhang, C. Zhou, Y.-C. Tian, N. Xiong, Y. Qin, and B. Hu, “A
fuzzy probability bayesian network approach for dynamic cybersecurity
risk assessment in industrial control systems,” _IEEE Transactions on_
_Industrial Informatics_, vol. 14, no. 6, pp. 2497–2506, 2017.

[4] S. Huang, A. Liu, S. Zhang, T. Wang, and N. N. Xiong, “Bd-vte: A novel
baseline data based verifiable trust evaluation scheme for smart network
systems,” _IEEE transactions on network science and engineering_, vol. 8,
no. 3, pp. 2087–2105, 2020.

[5] X. Zheng, H. Ding, H. Mamitsuka, and S. Zhu, “Collaborative matrix factorization with multiple similarities for predicting drug-target
interactions,” in _Proceedings of the 19th ACM SIGKDD international_
_conference on Knowledge discovery and data mining_, 2013, pp. 1025–
1033.

[6] M. C. Cobanoglu, C. Liu, F. Hu, Z. N. Oltvai, and I. Bahar, “Predicting
drug–target interactions using probabilistic matrix factorization,” _Journal_
_of chemical information and modeling_, vol. 53, no. 12, pp. 3399–3409,
2013.

[7] T. He, M. Heidemeyer, F. Ban, A. Cherkasov, and M. Ester, “Simboost:
a read-across approach for predicting drug–target binding affinities using
gradient boosting machines,” _Journal of cheminformatics_, vol. 9, no. 1,
pp. 1–14, 2017.

[8] X. Zhou, Y. Li, and W. Liang, “Cnn-rnn based intelligent recommendation for online medical pre-diagnosis support,” _IEEE/ACM Transactions_
_on Computational Biology and Bioinformatics_, vol. 18, no. 3, pp. 912–
921, 2020.

[9] H. Ozt¨urk, A. [¨] Ozg¨ur, and E. Ozkirimli, “Deepdta: deep drug–target [¨]
binding affinity prediction,” _Bioinformatics_, vol. 34, no. 17, pp. i821–
i829, 2018.

[10] H. Ozt¨urk, E. Ozkirimli, and A. [¨] Ozg¨ur, “Widedta: prediction of drug- [¨]
target binding affinity,” _arXiv preprint arXiv:1902.04166_, 2019.

[11] J. Gomes, B. Ramsundar, E. N. Feinberg, and V. S. Pande, “Atomic
convolutional networks for predicting protein-ligand binding affinity,”
_arXiv preprint arXiv:1703.10603_, 2017.

[12] Ł. Maziarka, T. Danel, S. Mucha, K. Rataj, J. Tabor, and S. Jastrzebski, “Molecule attention transformer,” _arXiv preprint arXiv:2002.08264_,
2020.

[13] J. Lim, S. Ryu, K. Park, Y. J. Choe, J. Ham, and W. Y. Kim,
“Predicting drug–target interaction using a novel graph neural network
with 3d structure-embedded graph representation,” _Journal of Chemical_
_Information and Modeling_, vol. 59, no. 9, pp. 3981–3988, 2019.

[14] M. Sun, S. Zhao, C. Gilvary, O. Elemento, J. Zhou, and F. Wang,
“Graph convolutional networks for computational drug development and
discovery,” _Briefings in bioinformatics_, vol. 21, no. 3, pp. 919–935, 2020.

[15] C. Zang and F. Wang, “Moflow: an invertible flow model for generating
molecular graphs,” in _Proceedings of the 26th ACM SIGKDD Interna-_
_tional Conference on Knowledge Discovery & Data Mining_, 2020, pp.
617–626.

[16] J. Klicpera, J. Groß, and S. G¨unnemann, “Directional message passing
for molecular graphs,” in _International Conference on Learning Repre-_
_sentations_, 2019.



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 10




[17] K. Yang, K. Swanson, W. Jin, C. Coley, P. Eiden, H. Gao, A. GuzmanPerez, T. Hopper, B. Kelley, M. Mathea _et al._, “Analyzing learned
molecular representations for property prediction,” _Journal of chemical_
_information and modeling_, vol. 59, no. 8, pp. 3370–3388, 2019.

[18] K. Do, T. Tran, and S. Venkatesh, “Graph transformation policy network
for chemical reaction prediction,” in _Proceedings of the 25th ACM_
_SIGKDD International Conference on Knowledge Discovery & Data_
_Mining_, 2019, pp. 750–760.

[19] X. Zheng, H. Ding, H. Mamitsuka, and S. Zhu, “Collaborative matrix factorization with multiple similarities for predicting drug-target
interactions,” in _Proceedings of the 19th ACM SIGKDD international_
_conference on Knowledge discovery and data mining_, 2013, pp. 1025–
1033.

[20] J.-P. Mei, C.-K. Kwoh, P. Yang, X.-L. Li, and J. Zheng, “Drug–target
interaction prediction by learning from local information and neighbors,”
_Bioinformatics_, vol. 29, no. 2, pp. 238–245, 2013.

[21] Z. Liu, Q. Chen, W. Lan, H. Pan, X. Hao, and S. Pan, “Gadti: Graph
autoencoder approach for dti prediction from heterogeneous network,”
_Frontiers in Genetics_, vol. 12, 2021.

[22] H. Wang, F. Huang, and W. Zhang, “Hampdti: a heterogeneous graph
automatic meta-path learning method for drug-target interaction prediction,” _arXiv preprint arXiv:2112.08567_, 2021.

[23] M. Poli, S. Massaroli, J. Park, A. Yamashita, H. Asama, and J. Park,
“Graph neural ordinary differential equations,” 11 2019.

[24] X. Zhou, W. Liang, I. Kevin, K. Wang, and S. Shimizu, “Multi-modality
behavioral influence analysis for personalized recommendations in
health social media environment,” _IEEE Transactions on Computational_
_Social Systems_, vol. 6, no. 5, pp. 888–897, 2019.

[25] Z. He, J. Zhang, X.-H. Shi, L.-L. Hu, X. Kong, Y.-D. Cai, and K.-C.
Chou, “Predicting drug-target interaction networks based on functional
groups and biological features,” _PloS one_, vol. 5, no. 3, p. e9603, 2010.

[26] L. Perlman, A. Gottlieb, N. Atias, E. Ruppin, and R. Sharan, “Combining
drug and gene similarity measures for drug-target elucidation,” _Journal_
_of computational biology_, vol. 18, no. 2, pp. 133–145, 2011.

[27] M. Takarabe, M. Kotera, Y. Nishimura, S. Goto, and Y. Yamanishi,
“Drug target prediction using adverse event report systems: a pharmacogenomic approach,” _Bioinformatics_, vol. 28, no. 18, pp. i611–i618,
2012.

[28] K. Buza and L. Peˇska, “Drug–target interaction prediction with bipartite
local models and hubness-aware regression,” _Neurocomputing_, vol. 260,
pp. 284–293, 2017.

[29] K. Buza, “Drug-target interaction prediction with hubness-aware machine learning,” in _2016 IEEE 11th International Symposium on Applied_
_Computational Intelligence and Informatics (SACI)_ . IEEE, 2016, pp.
437–440.

[30] K. Buza, A. Nanopoulos, and G. Nagy, “Nearest neighbor regression
in the presence of bad hubs,” _Knowledge-Based Systems_, vol. 86, pp.
250–260, 2015.

[31] X. Zhang, L. Li, M. K. Ng, and S. Zhang, “Drug–target interaction prediction by integrating multiview network data,” _Computational biology_
_and chemistry_, vol. 69, pp. 185–193, 2017.

[32] J.-Y. Shi, S.-M. Yiu, Y. Li, H. C. Leung, and F. Y. Chin, “Predicting
drug–target interaction for new drugs using enhanced similarity measures and super-target clustering,” _Methods_, vol. 83, pp. 98–104, 2015.

[33] J. Yin, W. Lo, S. Deng, Y. Li, Z. Wu, and N. Xiong, “Colbar: A collaborative location-based regularization framework for qos prediction,”
_Information Sciences_, vol. 265, pp. 68–84, 2014.

[34] X. Chen, W. Liang, J. Xu, C. Wang, K.-C. Li, and M. Qiu, “An efficient
service recommendation algorithm for cyber-physical-social systems,”
_IEEE Transactions on Network Science and Engineering_, pp. 1–1, 2021.

[35] M. Fu, H. Qu, Z. Yi, L. Lu, and Y. Liu, “A novel deep learningbased collaborative filtering model for recommendation system,” _IEEE_
_Transactions on Cybernetics_, vol. 49, no. 3, pp. 1084–1096, 2019.

[36] W. Liang, Y. Li, J. Xu, Z. Qin, and K.-C. Li, “QoS Prediction and
Adversarial Attack Protection for Distributed Services Under DLaaS,”
_IEEE Transactions on Computers_, vol. PP, pp. 1–14, 2021.

[37] W. Liang, S. Xie, D. Zhang, X. Li, and K.-c. Li, “A mutual
security authentication method for rfid-puf circuit based on deep
[learning,” vol. 22, no. 2, oct 2021. [Online]. Available: https:](https://doi.org/10.1145/3426968)
[//doi.org/10.1145/3426968](https://doi.org/10.1145/3426968)

[38] W. Liang, L. Xiao, K. Zhang, M. Tang, D. He, and K.-C. Li, “Data fusion
approach for collaborative anomaly intrusion detection in blockchainbased systems,” _IEEE Internet of Things Journal_, pp. 1–1, 2021.

[39] L. Wang, Z.-H. You, X. Chen, S.-X. Xia, F. Liu, X. Yan, Y. Zhou, and
K.-J. Song, “A computational-based method for predicting drug–target
interactions by using stacked autoencoder deep neural network,” _Journal_
_of Computational Biology_, vol. 25, no. 3, pp. 361–373, 2018.




[40] I. Lee, J. Keum, and H. Nam, “Deepconv-dti: Prediction of drug-target
interactions via deep learning with convolution on protein sequences,”
_PLoS computational biology_, vol. 15, no. 6, p. e1007129, 2019.

[41] J. You, R. D. McLeod, and P. Hu, “Predicting drug-target interaction
network using deep learning model,” _Computational Biology and Chem-_
_istry_, vol. 80, pp. 90–101, 2019.

[42] L. Jacob and J.-P. Vert, “Protein-ligand interaction prediction: an improved chemogenomics approach,” _Bioinformatics_, vol. 24, no. 19, pp.
2149–2156, 2008.

[43] N. Nagamine and Y. Sakakibara, “Statistical prediction of protein–
chemical interactions based on chemical structure and mass spectrometry
data,” _Bioinformatics_, vol. 23, no. 15, pp. 2004–2012, 2007.

[44] D.-S. Cao, L.-X. Zhang, G.-S. Tan, Z. Xiang, W.-B. Zeng, Q.-S. Xu, and
A. F. Chen, “Computational prediction of drug target interactions using
chemical, biological, and network features,” _Molecular informatics_,
vol. 33, no. 10, pp. 669–681, 2014.

[45] S. L. Kinnings, N. Liu, P. J. Tonge, R. M. Jackson, L. Xie, and P. E.
Bourne, “A machine learning-based method to improve docking scoring
functions and its application to drug repurposing,” _Journal of chemical_
_information and modeling_, vol. 51, no. 2, pp. 408–419, 2011.

[46] P. J. Ballester and J. B. Mitchell, “A machine learning approach to
predicting protein–ligand binding affinity with applications to molecular
docking,” _Bioinformatics_, vol. 26, no. 9, pp. 1169–1175, 2010.

[47] H. Shi, S. Liu, J. Chen, X. Li, Q. Ma, and B. Yu, “Predicting drugtarget interactions using lasso with random forest based on evolutionary
information and chemical structure,” _Genomics_, vol. 111, no. 6, pp.
1839–1852, 2019.

[48] Y. Yamanishi, M. Kotera, M. Kanehisa, and S. Goto, “Drug-target
interaction prediction from chemical, genomic and pharmacological data
in an integrated framework,” _Bioinformatics_, vol. 26, no. 12, pp. i246–
i254, 2010.

[49] M. C. Cobanoglu, C. Liu, F. Hu, Z. N. Oltvai, and I. Bahar, “Predicting
drug–target interactions using probabilistic matrix factorization,” _Journal_
_of chemical information and modeling_, vol. 53, no. 12, pp. 3399–3409,
2013.

[50] L. Peska, K. Buza, and J. Koller, “Drug-target interaction prediction:
a bayesian ranking approach,” _Computer methods and programs in_
_biomedicine_, vol. 152, pp. 15–21, 2017.

[51] L. Li and M. Cai, “Drug target prediction by multi-view low rank
embedding,” _IEEE/ACM Transactions on Computational Biology and_
_Bioinformatics_, vol. 16, no. 5, pp. 1712–1721, 2017.

[52] Y. Liu, M. Wu, C. Miao, P. Zhao, and X.-L. Li, “Neighborhood regularized logistic matrix factorization for drug-target interaction prediction,”
_PLoS computational biology_, vol. 12, no. 2, p. e1004760, 2016.

[53] M. Wang, C. Tang, and J. Chen, “Drug-target interaction prediction via
dual laplacian graph regularized matrix completion,” _BioMed Research_
_International_, vol. 2018, 2018.

[54] F. Cheng, C. Liu, J. Jiang, W. Lu, W. Li, G. Liu, W. Zhou, J. Huang, and
Y. Tang, “Prediction of drug-target interactions and drug repositioning
via network-based inference,” _PLoS computational biology_, vol. 8, no. 5,
p. e1002503, 2012.

[55] X. Chen, M.-X. Liu, and G.-Y. Yan, “Drug–target interaction prediction
by random walk on the heterogeneous network,” _Molecular BioSystems_,
vol. 8, no. 7, pp. 1970–1978, 2012.

[56] Y. Huang, L. Zhu, H. Tan, F. Tian, and F. Zheng, “Predicting drug-target
on heterogeneous network with co-rank,” in _International Conference on_
_Computer Engineering and Networks_ . Springer, 2018, pp. 571–581.

[57] A. Seal, Y.-Y. Ahn, and D. J. Wild, “Optimizing drug–target interaction
prediction based on random walk on heterogeneous networks,” _Journal_
_of cheminformatics_, vol. 7, no. 1, pp. 1–12, 2015.

[58] Q. Kuang, Y. Li, Y. Wu, R. Li, Y. Dong, Y. Li, Q. Xiong, Z. Huang, and
M. Li, “A kernel matrix dimension reduction method for predicting drugtarget interaction,” _Chemometrics and Intelligent Laboratory Systems_,
vol. 162, pp. 104–110, 2017.

[59] N. Yasuo, Y. Nakashima, and M. Sekijima, “Code-dti: Collaborative
deep learning-based drug-target interaction prediction,” in _2018 IEEE_
_International Conference on Bioinformatics and Biomedicine (BIBM)_ .
IEEE, 2018, pp. 792–797.

[60] H. E. Manoochehri and M. Nourani, “Predicting drug-target interaction
using deep matrix factorization,” in _2018 IEEE Biomedical Circuits and_
_Systems Conference (BioCAS)_ . IEEE, 2018, pp. 1–4.

[61] S. F. Sousa, P. A. Fernandes, and M. J. Ramos, “Protein–ligand docking:
current status and future challenges,” _Proteins: Structure, Function, and_
_Bioinformatics_, vol. 65, no. 1, pp. 15–26, 2006.

[62] A. N. Jain, “Surflex: fully automatic flexible molecular docking using a
molecular similarity-based search engine,” _Journal of medicinal chem-_
_istry_, vol. 46, no. 4, pp. 499–511, 2003.



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


This article has been accepted for publication in IEEE/ACM Transactions on Computational Biology and Bioinformatics. This is the author's version which has not been fully edited and


content may change prior to final publication. Citation information: DOI 10.1109/TCBB.2022.3204188


JOURNAL OF L [A] TEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2015 11




[63] N. Moitessier, P. Englebienne, D. Lee, J. Lawandi, Corbeil, and CR,
“Towards the development of universal, fast and highly accurate docking/scoring methods: a long way to go,” _British journal of pharmacol-_
_ogy_, vol. 153, no. S1, pp. S7–S26, 2008.

[64] W. J. Allen, T. E. Balius, S. Mukherjee, S. R. Brozell, D. T. Moustakas,
P. T. Lang, D. A. Case, I. D. Kuntz, and R. C. Rizzo, “Dock 6:
Impact of new features and current docking performance,” _Journal of_
_computational chemistry_, vol. 36, no. 15, pp. 1132–1156, 2015.

[65] H. Gohlke, M. Hendlich, and G. Klebe, “Knowledge-based scoring
function to predict protein-ligand interactions,” _Journal of molecular_
_biology_, vol. 295, no. 2, pp. 337–356, 2000.

[66] A. Krammer, P. D. Kirchhoff, X. Jiang, C. Venkatachalam, and M. Waldman, “Ligscore: a novel scoring function for predicting binding affinities,” _Journal of Molecular Graphics and Modelling_, vol. 23, no. 5, pp.
395–407, 2005.

[67] S. Yin, L. Biedermannova, J. Vondrasek, and N. V. Dokholyan,
“Medusascore: an accurate force field-based scoring function for virtual
drug screening,” _Journal of chemical information and modeling_, vol. 48,
no. 8, pp. 1656–1662, 2008.

[68] J. Dittrich, D. Schmidt, C. Pfleger, and H. Gohlke, “Converging a
knowledge-based scoring function: Drugscore2018,” _Journal of chemical_
_information and modeling_, vol. 59, no. 1, pp. 509–521, 2018.

[69] O. Trott and A. J. Olson, “Autodock vina: improving the speed and
accuracy of docking with a new scoring function, efficient optimization,
and multithreading,” _Journal of computational chemistry_, vol. 31, no. 2,
pp. 455–461, 2010.

[70] T. N. Kipf and M. Welling, “Semi-supervised classification with graph
convolutional networks,” _arXiv preprint arXiv:1609.02907_, 2016.

[71] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation
learning on large graphs,” in _Proceedings of the 31st International_
_Conference on Neural Information Processing Systems_, 2017, pp. 1025–
1035.

[72] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph
neural networks?” in _International Conference on Learning Represen-_
_tations_, 2018.

[73] K. Sch¨utt, P.-J. Kindermans, H. Sauceda, S. Chmiela, A. Tkatchenko,
and K.-R. M¨uller, “Schnet: a continuous-filter convolutional neural
network for modeling quantum interactions,” in _Proceedings of the 31st_
_International Conference on Neural Information Processing Systems_,
2017, pp. 992–1002.

[74] T. Nguyen, H. Le, T. P. Quinn, T. Nguyen, T. D. Le, and S. Venkatesh,
“Graphdta: Predicting drug–target binding affinity with graph neural
networks,” _Bioinformatics_, vol. 37, no. 8, pp. 1140–1147, 2021.

[75] H. Tan and M. Bansal, “Lxmert: Learning cross-modality encoder
representations from transformers,” in _Proceedings of the 2019 Con-_
_ference on Empirical Methods in Natural Language Processing and_
_the 9th International Joint Conference on Natural Language Processing_
_(EMNLP-IJCNLP)_, 2019, pp. 5100–5111.

[76] Y. Khare, V. Bagal, M. Mathew, A. Devi, U. D. Priyakumar, and
C. Jawahar, “Mmbert: Multimodal bert pretraining for improved medical
vqa,” in _2021 IEEE 18th International Symposium on Biomedical_
_Imaging (ISBI)_ . IEEE, 2021, pp. 1033–1036.

[77] K. Huang, C. Xiao, L. M. Glass, and J. Sun, “Moltrans: Molecular
interaction transformer for drug–target interaction prediction,” _Bioinfor-_
_matics_, vol. 37, no. 6, pp. 830–836, 2021.

[78] B. Ramsundar, P. Eastman, P. Walters, V. Pande, K. Leswing, and
Z. Wu, _Deep Learning for the Life Sciences_ . O’Reilly Media, 2019,
[https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/](https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837)
[dp/1492039837.](https://www.amazon.com/Deep-Learning-Life-Sciences-Microscopy/dp/1492039837)

[79] X. Wang, H. Ji, C. Shi, B. Wang, P. Cui, P. Yu, and Y. Ye, “Heterogeneous graph attention network,” in _The World Wide Web Conference_,
2019.

[80] F. Wan, L. Hong, A. Xiao, T. Jiang, and J. Zeng, “Neodti: neural
integration of neighbor information from a heterogeneous network for
discovering new drug–target interactions,” _Bioinformatics_, vol. 35, no. 1,
pp. 104–111, 2019.

[81] J. Peng, J. Li, and X. Shang, “A learning-based method for drug-target
interaction prediction based on feature representation learning and deep
neural network,” _BMC bioinformatics_, vol. 21, no. 13, pp. 1–13, 2020.



**Yuhui Li** is a graduate student at Hunan University.
He has published several high-quality peer-reviewed
journal and conference papers, including IEEE TC,
IEEE ICME and IEEE TITS. His research interests
include graph neural networks, network measurement, and bio-informatics.


**Wei Liang** received a Ph.D. degree in computer
science and technology from Hunan University,
China, in 2013. He was a Postdoctoral Scholar with
Lehigh University, USA, from 2014 to 2016. He is
currently an Associate Professor with the College
of Computer Science and Electronic Engineering,
Hunan University. He has authored or co-authored
more than 110 journal/conference papers such as
IEEE Transactions on Industrial Informatics, IEEE
Transactions on Emerging Topics in Computing,
IEEE Transactions on Computational Biology and
Bioinformatics, and IEEE Internet of Things Journal. His research interests
include Blockchain security, network security protection, embedded system
and hardware IP protection, fog computing, and security management in
Wireless Sensor Networks (WSN).


**Li Peng** received her PhD degree in computer science and technology from Hunan University, China,
in 2018. She is currently an Associate Professor with
the College of Computer Science and Engineering,
Hunan University of Science and Technology. Her
current research interests include data mining, bioinformatics, and machine learning.


**Dafang Zhang** received a Ph.D. degree in applied mathematics from Hunan University, China,
in 1997. He is currently a Professor at the College
of Computer Science and Electronic Engineering,
Hunan University, China. He was a Visiting Fellow
with Regina University, Canada, during 2002–2003,
and a Senior Visiting Fellow with Michigan State
University, USA, in 2013. He has authored or coauthored more than 230 journal/conference papers
and is a principal investigator (PI) for more than 30
large-scale scientific projects. His research interests
include dependable systems/networks, network security, network measurement, hardware security, and IP protection.


**Cheng Yang** is a postgraduate student at Hunan
University of Science and Technology. His research
interests include bioinformatics and machine learning.


**Kuan-Ching Li** (Senior Member, IEEE) is a professor in the Department of Computer Science and
Information Engineering (CSIE) at Providence University, where he also serves as the Director of
the High-Performance Computing and Networking
Center. He received the Licenciatura in Mathematics,
and MS and Ph.D. degrees in electrical engineering
from the University of Sao Paulo (USP), Brazil, in
1994, 1996, and 2001, respectively. Besides publishing articles in renowned journals and conferences,
he is co-author or co-editor of more than 40 books
published by leading publishers. He is a Fellow of IET and a senior member
of the IEEE. Professor Li’s research interests include parallel and distributed
computing, Big Data, and emerging technologies.



© 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.��See https://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Tsinghua University. Downloaded on November 13,2022 at 03:15:48 UTC from IEEE Xplore. Restrictions apply.


