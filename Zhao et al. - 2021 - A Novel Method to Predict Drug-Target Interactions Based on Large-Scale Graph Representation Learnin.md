# **_cancers_**

_Article_
## **A Novel Method to Predict Drug-Target Interactions Based on** **Large-Scale Graph Representation Learning**


**Bo-Wei Zhao** **[1,2,3]** **, Zhu-Hong You** **[1,2,3,]** ***, Lun Hu** **[1,2,3]** **, Zhen-Hao Guo** **[1,2,3]** **, Lei Wang** **[1,2,3]** **, Zhan-Heng Chen** **[4]**

**and Leon Wong** **[1,2,3]**


1 The Xinjiang Technical Institute of Physics & Chemistry, Chinese Academy of Sciences,
Urumqi 830011, China; zhaobowei19@mails.ucas.ac.cn (B.-W.Z.); hulun@ms.xjb.ac.cn (L.H.);
guozhenhao17@mails.ucas.ac.cn (Z.-H.G.); leiwang@ms.xjb.ac.cn (L.W.);
huangliguang18@mails.ucas.ac.cn (L.W.)
2 University of Chinese Academy of Sciences, Beijing 100049, China
3 Xinjiang Laboratory of Minority Speech and Language Information Processing, Urumqi 830011, China
4 College of Computer Science and Software Engineering, Shenzhen University, Shenzhen 518060, China;
chenzhanheng17@mails.ucas.ac.cn
***** Correspondence: zhuhongyou@nwpu.edu.cn; Tel.: +86-991-367-2967



���������
**�������**


**Citation:** Zhao, B.-W.; You, Z.-H.; Hu,


L.; Guo, Z.-H.; Wang, L.; Chen, Z.-H.;


Wong, L. A Novel Method to Predict


Drug-Target Interactions Based on


Large-Scale Graph Representation


Learning. _Cancers_ **2021**, _13_, 2111.


[https://doi.org/10.3390/cancers](https://doi.org/10.3390/cancers13092111)


[13092111](https://doi.org/10.3390/cancers13092111)


Academic Editor: Christos K. Kontos


Received: 27 March 2021


Accepted: 22 April 2021


Published: 27 April 2021


**Publisher’s Note:** MDPI stays neutral


with regard to jurisdictional claims in


published maps and institutional affil

iations.


**Copyright:** © 2021 by the authors.


Licensee MDPI, Basel, Switzerland.


This article is an open access article


distributed under the terms and


conditions of the Creative Commons


[Attribution (CC BY) license (https://](https://creativecommons.org/licenses/by/4.0/)


[creativecommons.org/licenses/by/](https://creativecommons.org/licenses/by/4.0/)


4.0/).



**Simple Summary:** The traditional process of drug development is lengthy, time-consuming, and
costly, whereas very few drugs ever make it to the clinic. The use of computational methods to detect
drug side effects greatly reduces the deficiencies in drug clinical trials. Prediction of drug-target interactions is a key step in drug discovery and repositioning. In this article, we proposed a novel method
for the prediction of drug-target interactions based on large-scale graph representation learning. This
method can be helpful to researchers in clinical trials and drug research and development.


**Abstract:** Identification of drug-target interactions (DTIs) is a significant step in the drug discovery or
repositioning process. Compared with the time-consuming and labor-intensive in vivo experimental
methods, the computational models can provide high-quality DTI candidates in an instant. In
this study, we propose a novel method called LGDTI to predict DTIs based on large-scale graph
representation learning. LGDTI can capture the local and global structural information of the
graph. Specifically, the first-order neighbor information of nodes can be aggregated by the graph
convolutional network (GCN); on the other hand, the high-order neighbor information of nodes
can be learned by the graph embedding method called DeepWalk. Finally, the two kinds of feature
are fed into the random forest classifier to train and predict potential DTIs. The results show that
our method obtained area under the receiver operating characteristic curve (AUROC) of 0.9455 and
area under the precision-recall curve (AUPR) of 0.9491 under 5-fold cross-validation. Moreover, we
compare the presented method with some existing state-of-the-art methods. These results imply that
LGDTI can efficiently and robustly capture undiscovered DTIs. Moreover, the proposed model is
expected to bring new inspiration and provide novel perspectives to relevant researchers.


**Keywords:** drug discovery; drug-target interactions; large-scale graph representation learning;
computational method


**1. Introduction**


Drug repositioning is the process of exploring the new effects of existing drugs except
for the original indications for medical treatment. It is a direction with great opportunities
and challenges. In addition, it has the advantages of low-cost, short-time and low-risk [ 1, 2 ].
The drug-target interactions (DTIs) play an important role in drug discovery and drug
repositioning. Accurate prediction of DTIs can improve the accuracy of drug clinical
trials, thus greatly reducing the risks of experiments. For a long time, the accumulation



_Cancers_ **2021**, _13_ [, 2111. https://doi.org/10.3390/cancers13092111](https://doi.org/10.3390/cancers13092111) [https://www.mdpi.com/journal/cancers](https://www.mdpi.com/journal/cancers)


_Cancers_ **2021**, _13_, 2111 2 of 12


of a large number of biological experimental data and related literature makes the biological database richer and richer, which provides a favorable condition for the use of
computational methods.
Traditional computing methods are mainly divided into two categories: ligand-based
methods and structure-based methods. However, structure-based approaches are limited
when the 3D structures of the target protein are absent, and ligand-based approaches have
low accuracy when there are only a few binding ligands for the target protein [ 3 – 7 ]. In recent
years, the widespread recognition of data-driven methods has made machine learning
algorithms widely used in biomolecular correlation prediction [ 8 – 11 ]. There are mainly four
related methods of in-silico methods: machine learning-based methods, network-based
methods, matrix factor-based methods, and deep learning-based methods [ 12 – 14 ]. For
example, Ding et al. [ 15 ] used substructure fingerprints, physical and chemical properties
of organisms, and DTIs as feature extraction methods and input features, and further
used SVM for classification. Chen et al. [ 16 ] employed gradient boosting decision tree
(GBDT) to predict drug-target interactions based on three properties, including IDs of
the drug and target, the descriptor of drug and target, DTIs. Luo et al. [ 17 ] constructed
a heterogeneous network to predict the potential DTIs by integrating the information of
multiple drugs. Chen et al. [ 18 ] and _Ji_ et al. [ 19 ] proposed a multi-molecular network model
based on network embedding to predict novel DTIs. Liu et al. [ 20 ] proposed a model called
NRLMF, which calculates the score of DTIs through logical matrix decomposition, where
the properties of the drug and target are expressed in terms of their specificity. Zheng
et al. [ 21 ] proposed to map the drug and target into a low-rank matrix and to establish the
weighted similarity matrix, and solve the problem by using the small square algorithm.
Wen et al. [ 22 ] used unsupervised learning to extract representations from the original
input descriptors to predict DTIs.
Recently, the extensive application of non-Euclidean structured data in graph neural
networks has led to various graph-based algorithms [ 23 – 30 ], such as graph convolution
networks (GCN), graph attention networks (GAT), graph autoencoders (GAE), graph
generative networks, graph spatial-temporal networks, etc. Based on the analysis of
biological data, it is found that the biological data network has a good preference for
the graph neural network. Gao et al. [ 31 ] used long short-term memory (LSTM) and
graph convolutional networks (GCN) to represent protein and drug structures, to predict
DTIs. Previous work has shown the preferable performance of graph neural network for
DTIs [ 27, 32 ], however, a single understanding of the data relationship between DTIs cannot
mine out the hidden information of the graph data well. Therefore, it is necessary to explore
the depth information of the drug and target protein through the graph neural network.
In the actual graph, the relationship between two nodes is complex, and the features
of each node are usually composed of a variety of attributes. It is necessary to clearly
understand the relationship between nodes. Therefore, the extraction of node features
should be multi-angle and multi-dimensional. To solve these challenges, we propose a
novel method to predict DTIs based on large-scale graph representation learning (LGDTI).
Unlike previous graph-based neural network-based approaches, LGDTI aims to gain an
in-depth understanding of known drugs and targets association networks through different
graph-based representation learning methods. To extract hidden graph features of drugs
and targets in a complex biological network, two types of graph representation learning
were used to excavate them.


**2. Materials and Methods**

_2.1. Datasets_


In this article, the multi-graph data were collected from DrugBank5.0 [ 33 ]. DrugBank5.0 is an open, free, comprehensive database, including drug molecular structures,
mechanisms, and drug-target interactions that are constantly being updated. We downloaded 11,396 known DTIs from Drugbank5.0, including 984 drugs and 635 proteins;


_Cancers_ **2021**, _13_, 2111 3 of 12


11,396 known DTIs are conducted as the benchmark dataset, and in training as the positive sample.


_2.2. Drug Attribute Representation_


The molecular structure of the drug was extracted from the DrugBank database. The
molecular structure is complex and difficult to use directly. To facilitate the calculation of
drug molecular structure, it was necessary to vectorize its molecular structure [ 34 ]. The
molecular fingerprint [ 35 ] is an abstract representation of a molecule, which encodes a
molecule as a series of bit vectors, in which each bit on the molecular fingerprint corresponds to a molecular fragment, as shown in Figure 1. For the drug data, RDKit [ 36 ] was
selected to calculate the Morgan fingerprint of the drug molecule.


**Figure 1.** A schematic diagram of the drug molecular structure is constructed as bit vectors. A is the
structure of a drug molecule, and B, C, and D are all substructures of the drug molecule, corresponding
to the converted bit (represented by the small black box), respectively.


_2.3. Protein Attribute Representation_


Protein sequence information was extracted from the STRING database [ 37 ]. Proteins
are important biological macromolecules. All proteins are polymers formed by the linkage
of 20 different amino acids, including (Ala, Val, Leu, Ile, Met, Phe, Trp, Pro), (Gly, Ser, Thr,
Cys, Asn, Gln, Tyr), (Arg, Lys, His), and (Asp, Glu). Subsequently, the k-mer method is
used [ 38 ], and k is set to 3, which translates each protein sequence into a 64-dimensional
(4 * 4 * 4) feature vector by calculating the occurrence frequency of each sub-sequence in
the entire protein sequence.


_2.4. Graph Convolutional Network for Drug-Target Interactions (DTIs)_


A graph convolutional network (GCN) [ 39 ] is a semi-supervised approach that turns
topological associations into topological diagrams. In the algorithm, the input of GCN is
the structure of the graph and the characteristics of each node, and the output includes the
results at the node level, the results at the graph level, and the pooling information at the
node level. Consequently, it is widely used in non-Euclidean spaces.
Let us assume that we have a bipartite graph G = with _V_ = [ _v_ 1, _· · ·_, _v_ _n_, _· · ·_, _v_ _m_ + _n_ ]
representing _n_ drugs and _m_ proteins, _E_ = � _e_ _ij_ � representing the relationship of drug _i_ and
protein _j_ . If _e_ _ij_ = 1, _v_ _i_ and _v_ _j_ has a connection. Furthermore, in the graph the attributes of


_Cancers_ **2021**, _13_, 2111 4 of 12



_X_ _d_
all nodes _X_ =
_X_
� _p_



_T_
, the attributes of the drug _X_ _d_ = [ _x_ 1 _[d]_ [,] _[ · · ·]_ [,] _[ x]_ _n_ _[d]_ []] and the attributes of
�



_T_
the protein _X_ _p_ = [ _x_ 1 _[p]_ [,] _[ · · ·]_ [,] _[ x]_ _m_ _[p]_ []] .
In this work, we define the function _f_ ( _X_, _A_ ) using the spatial method of GCN, where
_X_ is the feature set of each node, and _A_ is the adjacency matrix. Therefore, the network
communication rules of GCN are as follows:



�
_f_ ( _X_, _A_ ) _[l]_ [+] [1] = _σ_ _D_ _[−]_ [1] 2
�




[1]

2 _X_ _[l]_ _W_ _[l]_ [�], (1)




[1] �

2 _A_ � _D_ _[−]_ 2 [1]



in which, _A_ [�] = _A_ + _I_ _n_ + _m_ is the adjacency matrix added to the self-loop, _D_ [�] is represented
as the degree matrix of _A_ [�] . _W_ is the weight of the randomly initialized the network. _σ_
represents the activation function of each layer of the neural network, here _σ_ is _ReLU_ ( _·_ ) .
Although GCN has a natural preference for graph data, for DTIs data, we finally
determined _l_ = 1 and _W_ is 64 * 64 after analysis and experiment. Then, in the initial
training, we found that the algorithm had the problem of over-smoothing. To solve this
challenge, we adjusted the defect of the original algorithm for this data. Specifically, after
each convolution, we added node features for training, the formula is as follows:



�
_f_ ( _X_, _A_ ) _[l]_ [+] [1] = _T_ _σ_ _D_ _[−]_ 2 [1]
� �




[1]

2 _X_ _[l]_ _W_ _[l]_ [�], _X_, (2)
�




[1] �

2 _A_ � _D_ _[−]_ 2 [1]



we adopted this adjusted graph convolution definition in this work.


_2.5. Graph Embedding—DeepWalk for DTIs_


DeepWalk [ 40 ] is a method to learn the potential representation of nodes in a graph
and is a widely used algorithm in graph embedding. The main idea of the algorithm is
divided into two parts. The first part is to sample the graph based on the random walk
and map the node adjacency structure into sequence structure. The second part is to
train the Skip-gram model by using the sequences obtained from sampling so that the
expression of learning can capture the connectivity between nodes. Let us assume that we
have a bipartite graph G =( _V_, _E_ ) . _V_ is the set of nodes in the graph, and _E_ is the edge of
nodes. Each calculation starts from a given starting point, and then carries out a random
walk through the sampled neighbor nodes, repeating the operation until the length of the
sampled sequence is equal to the given maximum length, as shown in Algorithm 1.


_S_ _i_ = ( _v_ _i_ _|_ ( _v_ 1, _v_ 2, _v_ 3, _· · ·_, _v_ _i_ _−_ 1 )), (3)


where, _S_ _i_ is the random walk collection sequence, and _v_ _i_ is the random node.
Therefore, in the second part of the algorithm, _S_ is computed by the Skip-gram
model. Specifically, a two-layer neural network model is established. The input is the node
sequence matrix of _S_ _[n]_ _[∗]_ _[m]_, and the weights in the neural network model are set as _W_ 1 _[m]_ _[∗]_ _[h]_
and _W_ 2 _[h]_ _[∗]_ _[m]_ respectively. Secondly, through backpropagation, the weight parameters are
updated to obtain the representation of the target node, as shown in Algorithm 2.


_S_ = [ _S_ 1 _[m]_ [,] _[ S]_ 2 _[m]_ [,] _[ · · ·]_ [,] _[ S]_ _n_ _[m]_ []] _[T]_ [,] (4)


_Cancers_ **2021**, _13_, 2111 5 of 12


**Algorithm 1 DeepWalk** ( _G_, _w_, _d_, _γ_,)


**Input:** graph _G_ ( _V_, _E_ )
windows size _w_

representation size _d_
epoch _γ_
step length _t_
**Output:** matrix of nodes representation _ψ_ _∈_ R _[|]_ _[V]_ _[|×]_ _[d]_

1: Initialization: _ψ_
2: Build a binary Tree _T_ from _V_
3: **for** _i_ = 0 to _µ_ **do**
4: _V_ _[′]_ = **Shuffle** ( _V_ )
5: **for each** _v_ _i_ _∈_ _V_ _[′]_ **do**
6: _M_ _v_ _i_ = _**RandomWalk**_ ( _G_, _v_ _i_, _t_ )
7: _**SkipGram**_ ( _ψ_, _M_ _v_ _i_, _w_ )
8: **end for**

9: **end for**


**Algorithm 2 SkipGram** ( _ψ_, _M_ _v_ _i_, _w_ )


1: **for each** _v_ _j_ _∈_ _S_ _v_ _b_ do
2: **for each** _u_ _k_ _∈_ _S_ _v_ _i_ [ _j_ _−_ _w_ : _j_ + _w_ ] **do**



3: _J_ ( _ψ_ ) = _−_ _logPr_ ( _u_ _k_ ��� _ψ_ � _v_ _j_ � )

4: _ψ_ = _ψ_ _−_ _α_ _×_ _∂ψ_ _[∂][J]_

5: **end for**

6: **end for**



_2.6. Construction of the Large-Scale Graph Representation Learning Network_

Given a graph _G_ ( _V_, _E_ ) containing vertices _V_ and edges _E_, where _e_ _ij_ is regard as a
connection of _v_ _i_ and _v_ _j_ . a graph is considered as an adjacency matrix or an incidence
matrix [41]. For an adjacency matrix _A_, _A_ _∈_ _R_ _[N]_ _[×]_ _[N]_, is defined as:

_A_ _ij_ = � 1 if0 else� _v_ _i_, v _j_ � _⊆_ _E_, (5)


Here, we used an undirected cycled graph, so _a_ _ii_ = 1. For an incidence matrix _B_,
_B_ _∈_ _R_ _[N]_ _[×]_ _[M]_, is defined as:

_B_ _ij_ = � 1 if v0 else _i_ and v _j_ are connected, (6)


The function of graph representation learning is to map data from complex graph
space to multi-dimensional space. Its form is as follows:


_f_ : _V_ _→_ _X_ _∈ℜ_ _[d]_, (7)


where _d_ _≪|_ _V_ _|_, _V_ = [ _v_ 1, _v_ 2, _v_ 3, _· · ·_, _v_ _n_ + _m_ ] is the original set of spatial variables and
_X_ = [ _x_ 1, _x_ 2, _x_ 3, _· · ·_, _x_ _d_ ] is the projected vector (or the embedded vector) that contains the
structural information.

The first-order information is generally used to describe the local similarity between
pairs of vertices in a graph [ 42 ]. Specifically, if there is an edge between two vertices, the
two vertices should be close to each other in the embedded space. If there is no edge
connection between two vertices, the first-order proximity between them is 0. Such work
usually uses the KL-divergence [43] to calculate the distance by minimizing:

### O 1 = − ∑ ( i, j ) ∈ E w ij logp i � v i, v j �, (8)


_Cancers_ **2021**, _13_, 2111 6 of 12


in which _p_ 1 � _**v**_ _**i**_, _**v**_ _**j**_ � = 1 + exp ( 1 _−_ _v_ _**i**_ _[T]_ _[·]_ _[v]_ _**[j]**_ [)] [,] _**[ v]**_ _**[i]**_ [ and] _**[ v]**_ _**[j]**_ [ are the low-dimensional vector representation]

of the node _v_ _i_ and _v_ _j_ . _W_ _**ij**_ is the edge weight between node _i_ and _j_ . Although the methods
based on the first-order neighbor of nodes are successful in graph embedding, they often
fail to combine node substructure and node attributes for optimization. To address this
challenge, the advantages of graph convolutional networks in vertex local feature extraction
are utilized in Equation (1) to remedy this defect. An example of this algorithm is shown in
Figure 2C.


**Figure 2.** An example of large-scale graph representation learning. ( **A** ) The schematic diagram of
the relationship between drugs and targets. ( **B** ) An example of the graph embedding in drug-target
interactions (DTIs). ( **C** ) An example of the graph convolutional network.


The high-order information is learning the relationship between vertex _v_ _i_ and the other
vertices separately [ 44, 45 ]. Although there is no direct connection between the two vertices
in the high-order information, learning that their representation vectors are close means
that they should have similar or identical neighbors in the actual relational graph. For
example, Figure 2B shows that drug _d_ 1 has a second-order relationship with the target _t_ 2,
drug _d_ 2 and drug _d_ 1 have a shared target _t_ 1, and target _t_ 3 is a high-order potential candidate
for drug _d_ 1 . Then, we abstract high-order information (or global structure information) for
each node by the graph embedding method: DeepWalk.
Consequently, we constructed a large-scale graph representation learning network to
learn the features of each node, as shown in Figure 2. In which Figure 2A is the drug-target
interactions sub-network.


_2.7. The Large-Scale Graph Representation Learning DTI (LGDTI) Model Framework_


In this study, the proposed LGDTI model contains not only first-order but also highorder graph information. In the first-order graph information, the graph convolutional
network is used to capture the first-order neighbor information of the nodes in the graph;
in the high-order graph information, the graph embedding algorithm DeepWalk is used
to capture the high-order neighbor information of the nodes in the graph. Through these
two different methods, the local and global information of each node in the graph is
captured by LGDTI. The first-order neighbor information contains the attributes of nodes,
which are internal to the node; the high-order neighbor information contains the whole
network information of the node, which is called the behavior information. In the end, the
two kinds of representation features of nodes obtained from LGDTI are predicted by the
random forest classifier. The framework of large-scale graph representation learning as
shown in Figure 3. In short, we have three main contributions: (i) we propose to employ
specific GCN to learn first-order neighbors’ information (or local structural information) of
nodes. (ii) This article proposes to utilize a graph embedding algorithm to learn high-order
neighbors’ information (or global structural information) of nodes. (iii) In conclusion,
LGDTI can view the DTIs network from multiple perspectives, including three features in
the whole feature extraction process: node attributes, node first-order information, and
node high-order information.


_Cancers_ **2021**, _13_, 2111 7 of 12


**Figure 3.** The flowchart of the proposed large-scale graph representation learning DTI (LGDTI). ( **a** ) A bipartite graph of
DTIs. The solid black line is described as known DTIs, and the dashed red line is described as latent DTIs. ( **b** ) Part A
constructed an adjacency graph containing a self-loop, in which green nodes are drugs and purple nodes are targets, and
the information of _first-order_ neighbors of each node is aggregated through graph convolutional network. Part B represented
_high-order_ information of each node in a bipartite graph by DeepWalk. ( **c** ) The two kinds of representation features are
integrated. ( **d** ) Random forest classifier is trained and used for predicting new DTIs.


**3. Results and Discussion**

_3.1. Performance Evaluation of LGDTI Using 5-Fold Cross-Validation_


To accurately evaluate the stability and robustness of LGDTI, 5-fold cross-validation
was adopted. In detail, the original data set was randomly divided into 5 subsets, among
which 4 subsets were selected for each training, and the remaining subsets were used
as the test set and repeated 5 times. Additionally, we used five evaluation indicators,
including Acc. (Accuracy), MCC. (Matthews’s Correlation Coefficient), Sen. (Sensitivity),
Spec. (Specificity), and Perc. (Precision). Moreover, for binary classification, the receiver
operating characteristic (ROC) curve can reflect the capability of the model, while the AUC
is the area under the ROC curve. The closer the ROC curve is to the upper left corner,
the better the performance of the model. Similarly, the value of AUC is also high. The
precision-recall (PR) curve contains precision and recall, with recall as the horizontal axis
and precision as the vertical axis. On very skewed data sets, the PR curve can give us a
comprehensive understanding of the performance of the model. The details of LGDTI
under 5-fold cross-validation are shown in Table 1 and Figure 4. The results of each fold
AUC, AUPR, and various evaluation criteria show that the proposed method has a better
predictive ability. Studying it carefully, the results of each training are close to each other,
which shows that the model has preferable stability and robustness.


**Table 1.** Five-fold cross-validation results by random forest classifier.


**Fold** **Acc. (%)** **MCC (%)** **Sen. (%)** **Spec. (%)** **Prec. (%)** **AUC (%)**


0 88.36 77.11 83.25 93.46 92.72 93.93

1 88.60 77.54 83.90 93.29 92.59 94.43

2 88.22 76.89 82.85 93.60 92.83 94.66

3 88.40 77.22 83.16 93.64 92.90 94.51

4 89.61 79.52 85.18 94.04 93.46 95.23
Average 88.64 _±_ 0.56 77.66 _±_ 1.07 83.67 _±_ 0.93 93.61 _±_ 0.28 92.90 _±_ 0.33 94.55 _±_ 0.47


i



i


**Figure 4.** The receiver operating characteristic (ROC) and precision-recall (PR) curves under 5-fold

cross-validation.


_3.2. Comparison LGDTI with the Different Machine Learning Algorithms_


Different machine learning algorithms have different representations of features. By
comparing different classification algorithms, including logistic regression (LR), K-nearest
neighbor (KNN), gradient boosting decision tree (GBDT), and random forest classifier (RF),
we can intuitively see the feature advantages of LGDTI. To make the comparison fairer and
more objective, all classification algorithms choose the default parameters. The detailed
evaluation results of 5-fold cross-validation are shown in Table 2 and Figure 5.


~~**Table 2.**~~ ~~Comparison of different machine learning classifer.~~ i


**Classifier** **Acc. (%)** **MCC (%)** **Sen. (%)** **Spec. (%)** **Prec. (%)** **AUC (%)**


LR 72.54 _±_ 1.23 45.23 _±_ 2.47 76.57 _±_ 1.29 68.51 _±_ 1.49 70.86 _±_ 1.21 78.26 _±_ 0.78

KNN 71.07 _±_ 1.15 46.90 _±_ 1.82 92.99 _±_ 0.68 49.15 _±_ 2.69 64.67 _±_ 1.09 82.63 _±_ 0.46

~~GBDT~~ ~~84.98~~ ~~_±_~~ ~~0.23~~ ~~70.23~~ ~~_±_~~ ~~0.41~~ ~~80.54~~ ~~_±_~~ ~~0.65~~ ~~89.41~~ ~~_±_~~ ~~0.26~~ ~~88.38~~ ~~_±_~~ ~~0.19~~ ~~91.62~~ ~~_±_~~ ~~0.38~~

RF 88.64 _±_ 0.56 77.66 _±_ 1.07 83.67 _±_ 0.93 93.61 _±_ 0.28 92.90 _±_ 0.33 94.55 _±_ 0.47

i



**Figure 5.** Comparison of the ROC and PR curves performed based on different machine learning



i


_Cancers_ **2021**, _13_, 2111 9 of 12


_3.3. Comparison of the Different Feature with Attribute, GF and LGDTI_


In summary, LGDTI constructs a graph and combines the first-order and high-order
information of the nodes in the graph to denote the characteristics of each node. The
first-order graph information aggregates the direct neighbor information of nodes. In
graph theory, two nodes have similarities if the structure is similar to the subgraph. The
high-order graph information provides a preferable representation of each node’s indirect
neighbor information. Therefore, we conducted experiments on the different features of
nodes, in which random forest classifier was used, as shown in Table 3 and Figure 6. In
Table 3, Attribute has exemplified the feature of drug molecular structure and protein
sequence; only first-order graph information is represented as GF; LGDTI includes the
first-order and high-order graph information. When only node self-attributes are the
worst, while self-attributes of nodes can be enhanced through GCN. Therefore, only the
combination of first-order graph information and high-order graph information can better
explore the potential features of nodes.


**Table 3.** Comparison of different feature using random forest classifier.


**Feature** **Acc. (%)** **MCC (%)** **Sen. (%)** **Spec. (%)** **Prec. (%)** **AUC (%)**


Attribute 83.86 _±_ 0.32 67.78 _±_ 0.65 81.62 _±_ 0.69 86.09 _±_ 0.56 85.44 _±_ 0.47 90.89 _±_ 0.38

GF 84.28 _±_ 0.46 68.76 _±_ 0.90 80.67 _±_ 0.89 87.90 _±_ 0.62 86.96 _±_ 0.56 91.41 _±_ 0.36

LGDTI 88.64 _±_ 0.56 77.66 _±_ 1.07 83.67 _±_ 0.93 93.61 _±_ 0.28 92.90 _±_ 0.33 94.55 _±_ 0.47


**Figure 6.** Comparison of the ROC and PR curves performed by random forest classifier based on

different features.


_3.4. Compared with Existing State-of-the-Art Prediction Methods_


To evaluate the advantage of the proposed method, it is compared with other advanced
methods. Although the method proposed by Chen et al. [ 18 ] and Ji et al. [ 19 ], considers
the network information of nodes, it fully expressed the local information of nodes in the
network. Then, LGDTI is relatively sufficient for information extraction of nodes, and its
high AUROC, AUPR, and ACC are stronger than other methods, as shown in Table 4.


**Table 4.** Compared with existing state-of-the-art prediction methods.


**Methods** **Datasets** **AUROC** **AUPR** **ACC**


Ji et al. methods (Only Attribute) DrugBank 0.8777 0.8828 0.8073
Chen et al. methods (Only Attribute) DrugBank 0.8779 N/A 0.8127
LGDTI (Only Attribute) DrugBank 0.9089 0.9109 0.8386
LGDTI (GF) DrugBank 0.9141 0.9177 0.8428
Chen et al. methods (Only Behavior) DrugBank 0.9206 N/A 0.8545
Ji et al. methods (Only Behavior) DrugBank 0.9218 0.9286 0.8575
~~Ji et al. methods (Attribute+Behavior)~~ ~~DrugBank~~ ~~0.9233~~ ~~0.9301~~ ~~0.8583~~
LGDTI DrugBank 0.9455 0.9491 0.8864


Compared with other methods, node attributes (LGDTI (Only Attribute)), node firstorder information (LGDTI (GF)), and the LDGTI model are all better. Among them, in the


_Cancers_ **2021**, _13_, 2111 10 of 12


case of only node attributes, the AUROC, AUPR, and ACC of our model are at least 0.031,
0.0281, and 0.0259 higher respectively. Meanwhile, LGDTI (GF) still has some advantages.
Definitively, the AUROC, AUPR, and ACC of the LGDTI model are at least 0.0222, 0.019,
and 0.0281 higher than that of Ji et al. methods (Attribute+Behavior), respectively. The
first-order neighborhood information aggregation makes node attribute characteristics
are enhanced. Furthermore, the integration of first-order information and high-order
information of the node will make our method have better prediction ability.


_3.5. Case Studies_


To test the practical ability of our model, the drugs clozapine and risperidone were exploited to predict potential targets, respectively. Clozapine can be used to treat many types
of schizophrenia, and it can directly inhibit the brain stem reticulum up-activation system
and has a powerful sedative and hypnotic effect. Risperidone is a psychiatric drug used to
treat schizophrenia. In particular, it has an improved effect on the positive and negative
symptoms and their accompanying emotional symptoms. It may also reduce the emotional
symptoms associated with schizophrenia. In this case study, all known associations in the
benchmark dataset were trained by our method, and we sorted the predicted scores of the
remaining candidate targets and selected the top 5 targets, as shown in Table 5. The experiment showed that there were 3 targets of the drugs clozapine and risperidone predicted by
LGDTI, which could be proved in the SuperTarget database [ 46 ]. The remaining unproven
targets may be candidates, hopefully, to be explored by medical researchers.


**Table 5.** New association prediction results for the top 5 targets with clozapine and risperidone.



|Drug Name|Target Name|Confirmed|
|---|---|---|
|Clozapine|Alpha-1D adrenergic receptor<br>Cytochrome P450 3A5<br>UDP-glucuronosyltransferase 1A1<br>Solute carrier family 22 member 3<br>Sodium-dependent serotonin transporter|SuperTarget<br>SuperTarget<br>Unconﬁrmed<br>Unconﬁrmed<br>SuperTarget|
|Risperidone|Alpha-1D adrenergic receptor<br>Solute carrier family 22 member 8<br>Cytochrome P450 2C19<br>Sodium-dependent serotonin transporter<br>Potassium voltage-gated channel subfamily H member 2|SuperTarget<br>Unconﬁrmed<br>Unconﬁrmed<br>SuperTarget<br>SuperTarget|


**4. Conclusions**





Although the accurate and efficient computational model could greatly accelerate the
process of identification of DTIs, there is still a huge gap between academia and industry. In
this study, we developed a novel method called LGDTI for predicting DTIs. Specifically, the
nodes in LGDTI can be represented by 2 kinds of feature including first-order information
learned by GCN and high-order information learned by DeepWalk from the graph. in
which molecular fingerprint technology was used to extract the attribute of drugs, and
the k-mer method was used to extract the attribute of targets. Then, the Random Forest
classifier was applied to carry out the relationship prediction task. The presented method
obtained the AUC of 0.9455 and the AUPR of 0.9491 under 5-fold cross-validation which

is more competitive than several state-of-the-art methods. Moreover, our method can
learn three kinds of information about the node, including the node’s attributes, local
structure, and global structure. Specifically, LGDTI can integrate attribute information
with structural information for learning. The experimental results show that LGDTI
has a prominent predictive ability for DTIs. Nevertheless, due to the limitation of the
benchmark dataset, the performance of LGDTI cannot be shown collectively in multiple
data. Moreover, LGDTI may be greatly improved if two kinds of node information can be
better integrated. Consequently, we hope that the proposed model could be utilized to
guide drug development and other biological wet experiments.


_Cancers_ **2021**, _13_, 2111 11 of 12


**Author Contributions:** B.-W.Z., Z.-H.Y. and L.H. considered the algorithm, arranged the dataset,
and performed the analyses. Z.-H.G., L.W. (Lei Wang), Z.-H.C. and L.W. (Leon Wong) wrote the
manuscript. All authors have read and agreed to the published version of the manuscript.


**Funding:** This work was supported by the grant of the National Key R&D Program of China
(2018YFA0902600), and the grants of the National Science Foundation of China, Nos. 61722212,
61861146002 & 61732012.


**Institutional Review Board Statement:** Not applicable for studies not involving humans or animals.


**Informed Consent Statement:** Not applicable.


**Data Availability Statement:** The data presented in this study are available on request from the
corresponding author.


**Conflicts of Interest:** The authors declare no conflict of interest.


**References**


1. Dickson, M.; Gagnon, J.P. The cost of new drug discovery and development. _Discov. Med._ **2009**, _4_, 172–179.
2. DiMasi, J.A.; Hansen, R.W.; Grabowski, H.G. The price of innovation: New estimates of drug development costs. _J. Health Econ._
**2003**, _22_ [, 151–185. [CrossRef]](http://doi.org/10.1016/S0167-6296(02)00126-1)
3. Li, J.; Zheng, S.; Chen, B.; Butte, A.J.; Swamidass, S.J.; Lu, Z. A survey of current trends in computational drug repositioning.
_Brief. Bioinform._ **2016**, _17_ [, 2–12. [CrossRef]](http://doi.org/10.1093/bib/bbv020)
4. Napolitano, F.; Zhao, Y.; Moreira, V.M.; Tagliaferri, R.; Kere, J.; D’Amato, M.; Greco, D. Drug repositioning: A machine-learning
approach through data integration. _J. Cheminform._ **2013**, _5_ [, 30. [CrossRef]](http://doi.org/10.1186/1758-2946-5-30)
5. Wu, C.; Gudivada, R.C.; Aronow, B.J.; Jegga, A.G. Computational drug repositioning through heterogeneous network clustering.
_BMC Syst. Biol._ **2013**, _7_ [, S6. [CrossRef]](http://doi.org/10.1186/1752-0509-7-S5-S6)
6. Kinnings, S.L.; Liu, N.; Buchmeier, N.; Tonge, P.J.; Xie, L.; Bourne, P.E. Drug discovery using chemical systems biology:
Repositioning the safe medicine Comtan to treat multi-drug and extensively drug resistant tuberculosis. _PLoS Comput. Biol._ **2009**,
_5_ [, e1000423. [CrossRef] [PubMed]](http://doi.org/10.1371/journal.pcbi.1000423)
7. Liu, Z.; Fang, H.; Reagan, K.; Xu, X.; Mendrick, D.L.; Slikker, W., Jr.; Tong, W. In silico drug repositioning–what we need to know.
_Drug Discov. Today_ **2013**, _18_ [, 110–115. [CrossRef]](http://doi.org/10.1016/j.drudis.2012.08.005)
8. Bagherian, M.; Sabeti, E.; Wang, K.; Sartor, M.A.; Nikolovska-Coleska, Z.; Najarian, K. Machine learning approaches and databases
for prediction of drug-target interaction: A survey paper. _Brief. Bioinform._ **2021**, _22_ [, 247–269. [CrossRef] [PubMed]](http://doi.org/10.1093/bib/bbz157)
9. Agamah, F.E.; Mazandu, G.K.; Hassan, R.; Bope, C.D.; Thomford, N.E.; Ghansah, A.; Chimusa, E.R. Computational/in silico
methods in drug target and lead prediction. _Brief. Bioinform._ **2020**, _21_ [, 1663–1675. [CrossRef] [PubMed]](http://doi.org/10.1093/bib/bbz103)
10. Manoochehri, H.E.; Nourani, M. Drug-target interaction prediction using semi-bipartite graph model and deep learning. _BMC_
_Bioinform._ **2020**, _21_ [, 1–16. [CrossRef] [PubMed]](http://doi.org/10.1186/s12859-020-3518-6)
11. D’Souza, S.; Prema, K.; Balaji, S. Machine learning models for drug-target interactions: Current knowledge and future directions.
_Drug Discov. Today_ **2020**, _25_ [, 748–756. [CrossRef]](http://doi.org/10.1016/j.drudis.2020.03.003)
12. Xue, H.; Li, J.; Xie, H.; Wang, Y. Review of drug repositioning approaches and resources. _Int. J. Biol. Sci._ **2018**, _14_ [, 1232. [CrossRef]](http://doi.org/10.7150/ijbs.24612)
13. Luo, H.; Li, M.; Yang, M.; Wu, F.-X.; Li, Y.; Wang, J. Biomedical data and computational models for drug repositioning: A
comprehensive review. _Brief. Bioinform._ **2021**, _22_ [, 1604–1619. [CrossRef]](http://doi.org/10.1093/bib/bbz176)
14. Yella, J.K.; Yaddanapudi, S.; Wang, Y.; Jegga, A.G. Changing trends in computational drug repositioning. _Pharmaceuticals_ **2018**, _11_,
[57. [CrossRef] [PubMed]](http://doi.org/10.3390/ph11020057)
15. Ding, Y.; Tang, J.; Guo, F. Identification of drug-target interactions via multiple information integration. _Inf. Sci._ **2017**, _418_, 546–560.

[[CrossRef]](http://doi.org/10.1016/j.ins.2017.08.045)
16. Chen, J.; Wang, J.; Wang, X.; Du, Y.; Chang, H. Predicting Drug Target Interactions Based on GBDT. In Proceedings of the
International Conference on Machine Learning and Data Mining in Pattern Recognition, New York, NY, USA, 15–19 July 2018;
pp. 202–212.
17. Luo, Y.; Zhao, X.; Zhou, J.; Yang, J.; Zhang, Y.; Kuang, W.; Peng, J.; Chen, L.; Zeng, J. A network integration approach for
drug-target interaction prediction and computational drug repositioning from heterogeneous information. _Nat. Commun._ **2017**, _8_,
[1–13. [CrossRef] [PubMed]](http://doi.org/10.1038/s41467-017-00680-8)
18. Chen, Z.-H.; You, Z.-H.; Guo, Z.-H.; Yi, H.-C.; Luo, G.-X.; Wang, Y.-B. Prediction of Drug-Target Interactions from Multi-Molecular
Network Based on Deep Walk Embedding Model. _Front. Bioeng. Biotechnol._ **2020**, _8_ [, 338. [CrossRef] [PubMed]](http://doi.org/10.3389/fbioe.2020.00338)
19. Ji, B.-Y.; You, Z.-H.; Jiang, H.-J.; Guo, Z.-H.; Zheng, K. Prediction of drug-target interactions from multi-molecular network based
on LINE network representation method. _J. Transl. Med._ **2020**, _18_ [, 1–11. [CrossRef]](http://doi.org/10.1186/s12967-020-02490-x)
20. Liu, Y.; Wu, M.; Miao, C.; Zhao, P.; Li, X.-L. Neighborhood regularized logistic matrix factorization for drug-target interaction
prediction. _PLoS Comput. Biol._ **2016**, _12_ [, e1004760. [CrossRef] [PubMed]](http://doi.org/10.1371/journal.pcbi.1004760)


_Cancers_ **2021**, _13_, 2111 12 of 12


21. Zheng, X.; Ding, H.; Mamitsuka, H.; Zhu, S. Collaborative Matrix Factorization with Multiple Similarities for Predicting DrugTarget Interactions. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data
Mining, Chicago, IL, USA, 11–14 August 2013; pp. 1025–1033.
22. Wen, M.; Zhang, Z.; Niu, S.; Sha, H.; Yang, R.; Yun, Y.; Lu, H. Deep-learning-based drug-target interaction prediction. _J. Proteome_
_Res._ **2017**, _16_ [, 1401–1409. [CrossRef] [PubMed]](http://doi.org/10.1021/acs.jproteome.6b00618)
23. Sun, M.; Zhao, S.; Gilvary, C.; Elemento, O.; Zhou, J.; Wang, F. Graph convolutional networks for computational drug development
and discovery. _Brief. Bioinform._ **2020**, _21_ [, 919–935. [CrossRef]](http://doi.org/10.1093/bib/bbz042)
24. Zhou, J.; Cui, G.; Zhang, Z.; Yang, C.; Liu, Z.; Wang, L.; Li, C.; Sun, M. Graph neural networks: A review of methods and
applications. _arXiv_ **2018**, arXiv:1812.08434.
25. Wu, Z.; Pan, S.; Chen, F.; Long, G.; Zhang, C.; Philip, S.Y. A comprehensive survey on graph neural networks. _IEEE Trans. Neural_
_Netw. Learn. Syst._ **2020**, _32_ [, 4–24. [CrossRef]](http://doi.org/10.1109/TNNLS.2020.2978386)
26. Guo, Z.-H.; You, Z.-H.; Yi, H.-C. Integrative construction and analysis of molecular association network in human cells by fusing
node attribute and behavior information. _Mol. Ther. Nucleic Acids_ **2020**, _19_ [, 498–506. [CrossRef]](http://doi.org/10.1016/j.omtn.2019.10.046)
27. Zhao, T.; Hu, Y.; Valsdottir, L.R.; Zang, T.; Peng, J. Identifying drug-target interactions based on graph convolutional network and
deep neural network. _Brief. Bioinform._ **2020**, _22_ [, 2141–2150. [CrossRef]](http://doi.org/10.1093/bib/bbaa044)
28. Lim, J.; Ryu, S.; Park, K.; Choe, Y.J.; Ham, J.; Kim, W.Y. Predicting drug-target interaction using a novel graph neural network
with 3D structure-embedded graph representation. _J. Chem. Inf. Modeling_ **2019**, _59_ [, 3981–3988. [CrossRef] [PubMed]](http://doi.org/10.1021/acs.jcim.9b00387)
29. Jiang, M.; Li, Z.; Zhang, S.; Wang, S.; Wang, X.; Yuan, Q.; Wei, Z. Drug-target affinity prediction using graph neural network and
contact maps. _RSC Adv._ **2020**, _10_ [, 20701–20712. [CrossRef]](http://doi.org/10.1039/D0RA02297G)
30. Yue, X.; Wang, Z.; Huang, J.; Parthasarathy, S.; Moosavinasab, S.; Huang, Y.; Lin, S.M.; Zhang, W.; Zhang, P.; Sun, H. Graph
embedding on biomedical networks: Methods, applications and evaluations. _Bioinformatics_ **2020**, _36_ [, 1241–1251. [CrossRef]](http://doi.org/10.1093/bioinformatics/btz718)

[[PubMed]](http://www.ncbi.nlm.nih.gov/pubmed/31584634)
31. Gao, K.Y.; Fokoue, A.; Luo, H.; Iyengar, A.; Dey, S.; Zhang, P. Interpretable Drug Target Prediction Using Deep Neural
Representation. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, Stockholm, Sweden, 13–19
July 2018; pp. 3371–3377.
32. Torng, W.; Altman, R.B. Graph convolutional neural networks for predicting drug-target interactions. _J. Chem. Inf. Modeling_ **2019**,
_59_ [, 4131–4149. [CrossRef]](http://doi.org/10.1021/acs.jcim.9b00628)
33. Wishart, D.S.; Feunang, Y.D.; Guo, A.C.; Lo, E.J.; Marcu, A.; Grant, J.R.; Sajed, T.; Johnson, D.; Li, C.; Sayeeda, Z. DrugBank 5.0: A
major update to the DrugBank database for 2018. _Nucleic Acids Res._ **2017**, _46_ [, D1074–D1082. [CrossRef]](http://doi.org/10.1093/nar/gkx1037)
34. Li, Y.; Liu, X.z.; You, Z.H.; Li, L.P.; Guo, J.X.; Wang, Z. A computational approach for predicting drug-target interactions from
protein sequence and drug substructure fingerprint information. _Int. J. Intell. Syst._ **2021**, _36_ [, 593–609. [CrossRef]](http://doi.org/10.1002/int.22332)
35. Rogers, D.; Hahn, M. Extended-connectivity fingerprints. _J. Chem. Inf. Modeling_ **2010**, _50_ [, 742–754. [CrossRef] [PubMed]](http://doi.org/10.1021/ci100050t)
36. Landrum, G. Rdkit documentation. _Release_ **2013**, _1_, 1–79.
37. Szklarczyk, D.; Morris, J.H.; Cook, H.; Kuhn, M.; Wyder, S.; Simonovic, M.; Santos, A.; Doncheva, N.T.; Roth, A.; Bork, P. The
STRING database in 2017: Quality-controlled protein–protein association networks, made broadly accessible. _Nucleic Acids Res._
**2016**, _45_ [, D362–D368. [CrossRef] [PubMed]](http://doi.org/10.1093/nar/gkw937)
38. Rizk, G.; Lavenier, D.; Chikhi, R. DSK: K-mer counting with very low memory usage. _Bioinformatics_ **2013**, _29_ [, 652–653. [CrossRef]](http://doi.org/10.1093/bioinformatics/btt020)

[[PubMed]](http://www.ncbi.nlm.nih.gov/pubmed/23325618)
39. Kipf, T.N.; Welling, M. Semi-supervised classification with graph convolutional networks. _arXiv_ **2016**, arXiv:1609.02907.
40. Perozzi, B.; Al-Rfou, R.; Skiena, S. Deepwalk: Online Learning of Social Representations. In Proceedings of the 20th ACM
SIGKDD International Conference on Knowledge Discovery and Data Mining, Association for Computing Machinery, New York,
NY, USA, 24–27 August 2014; pp. 701–710.
41. Ding, C.H.; He, X.; Zha, H.; Gu, M.; Simon, H.D. A Min-Max Cut Algorithm for Graph Partitioning and Data Clustering.
In Proceedings of the 2001 IEEE International Conference on Data Mining, IEEE, California, CA, USA, 29 November 2001;
pp. 107–114.
42. Cavallari, S.; Zheng, V.W.; Cai, H.; Chang, K.C.-C.; Cambria, E. Learning community embedding with community detection
and node embedding on graphs. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management,
Association for Computing Machinery, Singapore, 6–10 November 2017; pp. 377–386.
43. Goldberger, J.; Gordon, S.; Greenspan, H. An Efficient Image Similarity Measure Based on Approximations of KL-Divergence
between Two Gaussian Mixtures. In Proceedings of the Ninth IEEE International Conference on Computer Vision, Nice, France,
[13–16 October 2003; Volume 1, pp. 487–493. [CrossRef]](http://doi.org/10.1109/ICCV.2003.1238387)
44. Zhou, C.; Liu, Y.; Liu, X.; Liu, Z.; Gao, J. Scalable graph embedding for asymmetric proximity. _Proc. AAAI Conf. Artif. Intell._ **2017**,
_31_, 2942–2948.
45. Yang, J.-H.; Chen, C.-M.; Wang, C.-J.; Tsai, M.-F. HOP-rec: High-order proximity for implicit recommendation. In Proceedings of
the 12th ACM Conference on Recommender Systems, Vancouver British Columbia, Canada, 2 October 2018; pp. 140–144.
46. Günther, S.; Kuhn, M.; Dunkel, M.; Campillos, M.; Senger, C.; Petsalaki, E.; Ahmed, J.; Urdiales, E.G.; Gewiess, A.; Jensen, L.J.
SuperTarget and Matador: resources for exploring drug-target relationships. _Nucleic Acids Res._ **2007**, _36_ [, D919–D922. [CrossRef]](http://doi.org/10.1093/nar/gkm862)


