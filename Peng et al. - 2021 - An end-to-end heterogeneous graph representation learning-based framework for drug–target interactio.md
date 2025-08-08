_Briefings in Bioinformatics,_ 22(5), 2021, 1–9


**[https://doi.org/10.1093/bib/bbaa430](https://doi.org/10.1093/bib/bbaa430)**
Problem Solving Protocol

# **An end-to-end heterogeneous graph representation** **learning-based framework for drug–target interaction** **prediction**

## Jiajie Peng, Yuxian Wang, Jiaojiao Guan, Jingyi Li, Ruijiang Han, Jianye Hao, Zhongyu Wei and Xuequn Shang


Corresponding authors: Jiajie Peng, School of Computer Science, Northwestern Polytechnical University, Xi’an 710072, China. Tel: 029-88431519;
Fax: 029-88431519; E-mail: jiajiepeng@nwpu.edu.cn; Zhongyu Wei, School of Data Science, Fudan University, Shanghai 200433, China.
E-mail: zywei@fudan.edu.cn; Xuequn Shang, School of Computer Science, Northwestern Polytechnical University, Xi’an 710072, China.
E-mail: shang@nwpu.edu.cn


Abstract


Accurately identifying potential drug–target interactions (DTIs) is a key step in drug discovery. Although many related
experimental studies have been carried out for identifying DTIs in the past few decades, the biological experiment-based
DTI identification is still timeconsuming and expensive. Therefore, it is of great significance to develop effective
computational methods for identifying DTIs. In this paper, we develop a novel ‘end-to-end’ learning-based framework based
on heterogeneous ‘graph’ convolutional networks for ‘DTI’ prediction called end-to-end graph (EEG)-DTI. Given a
heterogeneous network containing multiple types of biological entities (i.e. drug, protein, disease, side-effect), EEG-DTI
learns the low-dimensional feature representation of drugs and targets using a graph convolutional networks-based model
and predicts DTIs based on the learned features. During the training process, EEG-DTI learns the feature representation of
nodes in an end-to-end mode. The evaluation test shows that EEG-DTI performs better than existing state-of-art methods.
[The data and source code are available at: https://github.com/MedicineBiology-AI/EEG-DTI.](https://github.com/MedicineBiology-AI/EEG-DTI)


**Key words:** drug–target interaction prediction; heterogeneous network; end-to-end learning; graph convolutional networks.


**Jiajie Peng** is an associate professor in the School of Computer Science at Northwestern Polytechnical University, Xi’an, China. His expertise is
computational biology and machine learning.
**Yuxian Wang** is a master student in the School of Computer Science at Northwestern Polytechnical University, Xi’an, China. His expertise is computational
biology and machine learning.
**Jiaojiao Guan** is a master student in the School of Computer Science at Northwestern Polytechnical University, Xi’an, China. Her expertise is computational
biology and machine learning.
**Jingyi Li** is a master student in the School of Computer Science at Northwestern Polytechnical University, Xi’an, China. Her expertise is computational
biology and machine learning.
**Ruijiang Han** is a master student in the School of Computer Science at Northwestern Polytechnical University, Xi’an, China. His expertise is computational
biology and medical imaging.
**Jianye Hao** is an associate professor in the School of Software at Tianjian University, Tianjin, China. His expertise is artificial intelligence and machine
learning.
**Zhongyu Wei** is an associate professor in the School of Data Science at Fudan University, Shanghai, China. His expertise is natural language processing
and machine learning.
**Xuequn Shang** is a professor in the School of Computer Science at Northwestern Polytechnical University, Xi’an, China. Her expertise is data mining and
computational biology.
**Submitted:** 21 October 2020; **Received (in revised form):** 1 December 2020


© The Author(s) 2021. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


1


2 _Peng_ et al _._


Introduction


Drug–target interaction (DTI) identification is of great significance for drug repositioning [1, 2] and drug discovery [3]. Particularly, to find safe and effective drugs in the drug discovery
process, thousands of chemical compounds have been tested in
the past few decades. Two biological experimental methods have
been used to identify DTIs: protein microarrays [4] and affinity
chromatography [5].
However, experiment-based drug development is an expensive and time-consuming process. To accelerate drug discovery,
it is crucial to develop effective computational methods to identify DTIs [6, 7, 8, 9, 10]. The existing computation-based method
of DTI identification can be divided into three categories: text
mining-based methods, biological feature-based methods and
network-based methods.

The first category is text mining-based methods. In detail,
the text mining-based methods extract information from the
literatures and use the descriptions of drugs and targets as
features to identify DTIs [11]. A semantic similarity-based model
using random forest (RF) and support vector machine (SVM)
method is proposed [12] to identify DTIs. The model constructs a
semantic network across the chemical and biological space and
extracts the features based on the semantic network. However,
text mining-based methods are affected by the differences in
semantic expression and conflicts among different literatures
limit the performance.
The second category is biological feature-based methods,
which are also known as feature engineering-based method. The
main idea of these methods is to extract the biological features of
drugs and targets. Based on extracted features, machine learning
models are used to identify DTIs. A method based on SVM called
bipartite local models (BLM) transforms the DTI identification
problem into a binary classification problem [13]. It considers
both DTIs and the drug–drug, target–target similarity based on
chemical and genomic data. Based on the BLM, a computing
framework called BLMNII [14] is proposed. BLMNII combines
neighbor-based interaction profile inferring (NII) method with
BLM. In summary, these methods identify DTIs based on similarity networks. Meng _et al_ . used a model named Predicting Drug
Targets with Protein Sequence (PDTPS) [15] to predict the DTIs via
integrating the protein sequences and drug chemical structures.
In detail, given a protein sequence, the model uses positionspecific iterated Basic Local Alignment Search Tool (BLAST) [16]
to calculate a position-specific scoring matrix (PSSM) [17]. Then,
the model uses a bi-gram probabilities (BIGP) model [18] to
extract features based on the PSSM. Finally, given a feature
of a protein sequence extracted by BIGP, PDTPS uses principal
component analysis (PCA) to reduce the feature dimensions.
For drugs, the model obtains the drug feature representation
based on the drug chemical structures. Then, the features of
drug and protein are concatenated as the feature of drug–protein
pair. Finally, PDTPS applies the relevance vector machine [19] to
predict the DTIs. Similar to PDTPS, Wang _et al_ . proposed a stacked
autoencoder-based model [20] to learn the feature based on the
PSSM. Then, the model uses a RF to predict the DTIs. However,
these methods do not consider the interactions of drug–drug or
protein–protein.
The third category is network-based methods. Network can
describe the complex interaction between different types of
biological entities (i e. drug, protein). Several network-based
methods have been developed for DTI identification [2, 21, 22,
23, 24, 25]. In detail, Zheng _et al_ . proposed a model called collaborative matrix factorization (CMF) for DTI identification [21]. CMF



uses collaborative matrix factorization to learn low-dimensional

feature representations of drugs and targets. These low-rank
features are estimated by an alternating least squares algorithm.
It predicts the interactions between drugs and targets based
on their low-rank representations. Xia _et al_ . proposed a method
called NetLapRLS [22]. NetLapRLS is a semi-supervised learning
algorithm using Laplacian regularized least squares. NetLapRLS
considers the interactions between drug and target as well as
drug–drug similarity and protein–protein similarity. Particularly,
Luo _et al_ . proposed a method called DTINet, which predicts DTIs
from a heterogeneous network [2]. DTINet integrates diverse
drug-related information and protein-related information. First,
it computes the similarity matrix of drugs and proteins through
multiple networks. Then, the random walk with restart and
singular value decomposition (SVD) decomposition are applied
on the similarity matrices sequentially to obtain the feature
representations of drugs and proteins. DTINet identifies the
DTIs based on these low-dimensional representations. Recently,
Zhao _et al_ . proposed a network-based method combining graph
convolutional neural network and deep neural network for DTI
identification [26]. In detail, this method builds a drug–protein
pair (DPP) network through the drug–drug interaction network,
protein–protein interaction network and drug–protein interaction network. A node in the DPP network represents a drug–
target pair, and the edge represents the link strength between
these pairs. Then, the DTI identification problem is converted
into a node classification problem. The categories of drug and
the amino acid information of the target are used as the feature
of the DPP. In the feature extraction stage, the model applies
the graph convolution operation to capture the feature of DPP
node.

The relations between drug and target are complicated. Several types of information, such as drug–disease associations,
drug–drug interactions, drug–side-effect associations and disease–protein associations, should be considered for DTI prediction. The existing methods usually extract the features based
on each type of interactions separately. Then, features based
on each type of interactions are concatenated together. Those
methods do not model the heterogeneous information in a single heterogeneous network. Therefore, existing methods cannot
well consider the associations among multiple types of biological entities, such as drug, disease, protein, side-effect, etc.
Furthermore, the existing methods are mainly divided into two
independent steps that are feature extraction step and DTI prediction step, which are not end-to-end models. The parameters
involved in the feature extraction step cannot be optimized by
the final DTI prediction task.
Recently, the heterogeneous graph representation learningbased method has achieved great success in many tasks, e.g.
item recommendation [27, 28] and polypharmacy side-effects
prediction [29]. Compared with homogeneous networks, heterogeneous network-based method can model multiple types
of entities and complex interactions between different types
of entities in a single heterogeneous network. Graph convolutional networks (GCN) is a powerful deep representation learning
method for network data, which has shown superior performance on network analysis and aroused considerable research
interest. The introduction of GCN can be found in the Supplementary Document.
Inspired by the success of heterogeneous network-based and
GCN-based model, we propose a novel end-to-end heterogeneous graph representation learning-based framework, named
EEG-DTI, to identify the interactions between drug and target.


_End-to-end heterogeneous graph representation learning-based framework_ 3



To optimize all parameters based on the final DTI prediction
task, EEG-DTI is designed as an end-to-end fashion. Here are four
major contributions:


 - To better describe the relations between drugs and targets, we construct a heterogeneous network with multiple
entities (i.e. drug, protein, disease, side-effect) and multiple
types of edges.

 - We propose a heterogeneous GCN-based method to learn
the drug and target feature representation based on a heterogeneous network.

 - We propose an end-to-end framework to predict DTIS,
which can optimize parameters in the model based on the
final DTI prediction task.

 - The evaluation results show that EEG-DTI outperforms
some state-of-the-art approaches for DTI prediction.


Methodology


We propose a novel end-to-end heterogeneous graph representation learning-based method called EEG-DTI to identify DTIs.
The workflow of EEG-DTI is shown in Figure 1. Our work contains
three parts. First, a heterogeneous network is constructed by
combining eight types of biological networks (Figure 1A). Second,
we propose a novel heterogeneous graph convolutional neural
network to obtain the low-dimensional representations of drugs
and targets based on the constructed network (Figure 1B). Third,
we use the inner product method to calculate the interaction
score between drugs and targets based on the low-dimensional
representations and optimize the model by cross-entropy
(Figure 1C).


**Construct a heterogeneous network**


In this section, we introduce how to construct a heterogeneous
network. In detail, we combine eight types of networks to construct a heterogeneous network, including drug–drug interaction
network, drug–protein interaction network, drug–disease association network, drug–side-effect association network, drug–drug
similarity network, protein–protein interaction network, protein–disease association network and protein–protein similarity
network. Two types of edges are included in the constructed
heterogeneous network. One type of edges is the original interactions included in the combined networks. The other type
of edges, named ‘similarity edge’, is added based on similarities between these biological entities based on each combined
network.

In addition to the original associations in the networks, we
add extra drug–drug similarity and protein–protein similarity
information to the heterogeneous network. The similarities
between drug and drug, protein and protein are calculated
by Jaccard similarity coefficient. Jaccard similarity is used to
measure the similarity of two sets. We take the drug–disease
association network as an example to describe how to calculate
a drug–drug similarity network. We use the following formula to
calculate the similarity value between drug _i_ and drug _j_ :



network. Therefore, we can obtain three similarity scores for a
given pair of drugs (drug _i_ and drug _j_ ). If one of the scores is larger
than a given threshold and there is no original associations
between the drug _i_ and drug _j_, we add a similarity edge between
drug _i_ and drug _j_ (see Supplementary Document for detail).
For targets (proteins), we also calculate their similarities
based on two networks, such as protein–disease association network and protein–protein interaction network. Similar to adding
similarity edges between drugs, we add the similarity edges
between proteins (see Supplementary Document for detail).


**Heterogeneous network-based feature extraction**
**framework**


The key step of identifying DTI is feature extraction. Recently,
GCN is widely used for aggregating node features in the network.
The essence of performing graph convolution on the networks is
to achieve feature aggregation between related nodes.
In this section, we introduce how to generate feature
representation of each node through heterogeneous graph
convolutional neural networks in each layer. In the constructed
heterogeneous network, there are multiple types of edges. To
extract the features of drugs and targets in the heterogeneous
network, we proposed a heterogeneous graph convolutional
networks (HGCN) to perform message passing based on different
types of edges in the heterogeneous network.
A heterogeneous network is a network with multiple types
of nodes and edges. Given a heterogeneous network, it can
be represented as _G_ = ( _V_, _E_, _R_ ), where _v_ _i_ ∈ _V_ represents the
node in the heterogeneous network, ( _v_ _i_, _r_, _v_ _j_ ) ∈ _E_ represents
the edge in the heterogeneous network and _r_ ∈ _R_ represents
the edge type in the heterogeneous network. Specifically, in
the heterogeneous network, there are four types of nodes ( _v_ _i_ )
(i.e. drug, protein, disease, side-effect). Therefore, there are eight
types of edges included in _R_, such as drug–drug interaction,
drug–protein interaction, drug–disease association, drug–sideeffect association, protein–protein interaction, protein–disease
association, drug–drug similarity and protein–protein similarity.
In a heterogeneous network, different types of edges should
be deferentially considered during the information 30], in each
layer, we model the type information of edges as follows:



_h_ ( _il_ +1) = _φ_



⎛



_r_

⎝ [�]



⎞

(2)
⎠



_c_ _ijr_ _[W]_ _r_ ( _l_ ) _[h]_ ( _jl_ ) [+] _[ c]_ _r_ _[i]_ _[h]_ ( _il_ )



_r_



�

_j_ ∈ _N_ _[i]_ _r_



_A_ _ij_ = | [|] _D_ _[ D]_ _[i]_ _i_ [ ∩] ∪ _[D]_ _D_ _[j]_ _j_ [ |] | (1)



where _c_ _ijr_ [=][ 1] _[/]_ ~~�~~ | _N_ _[i]_ _r_ [||] _[N]_ _jr_ [|][, which is symmetric sqrt normalization]

constant; _c_ _[i]_ _r_ [=][ 1] _[/]_ [|] _[N]_ _[i]_ _r_ [|][, in which] _[ N]_ _[i]_ _r_ [denotes the set of neighbors]
of _v_ _i_ with edge type _r_ ; _W_ _r_ ( _l_ ) [represents the trainable parameters]
at the _l_ th layer with edge type _r_ ; _h_ ( _il_ ) ∈ R _[d]_ represents the feature

representation of the _v_ _i_ at the _l_ th layer; _h_ ( _jl_ ) [represents the feature]

representation of neighbors of _v_ _i_ at the _l_ th layer; _h_ _i_ ( _l_ +1) represents

the feature representation of the _v_ _i_ at the _l_ +1th layer; _h_ _i_ ( _l_ +1)
represents the feature representation after the feature aggregation operation; _φ_ is the rectified linear unit activation function.
Specifically, when _l_ =0, we encode each node with one-hot vector
as the original feature. In summary, in this step, the model can
get the feature representation of each layer of all nodes in the
heterogeneous network.
We have introduce how to generate feature representation for
one node in each GCN layer. In this section, we will introduce the
feature extraction framework.

For one node in the network, we know that if the number
of graph convolution neural network layer is only one layer,



where _D_ _i_ represents the set of diseases of drug _i_, _A_ _ij_ represents
the similarity value between drug _i_ and drug _j_ and _A_ _ij_ ∈ [0, 1].
For each pair of drugs, we calculate their similarities based
on three networks, such as drug–disease association network,
drug–drug interaction network and drug–side-effect association


4 _Peng_ et al _._


**Figure 1.** The workflow of EEG-DTI. We identify the interaction between _dr_ 7 and _pr_ 3 as an example to describe the process of our model. EEG-DTI contains three
main steps. (A) In the first step, we construct a heterogeneous network. There are four kinds of entities and eight kinds of edge types in the heterogeneous network.

Among them, dr represents drug, pr represents protein, se represents side-effect and di represents disease. The eight kinds of edge types are drug–drug interaction,
drug–side-effect association, drug–disease association, drug–protein interaction, protein–protein interaction, protein–disease association, drug–drug similarity and
protein–protein similarity. (B) To generate low-dimensional embedding for drugs and proteins, the features of different types of nodes will be considered in the process
of generating embedding, and finally, neighbor’s information of multitype is aggregated. In this framework, we implement a three-layers GCN. For instance, in the 1st

layer, dr 7 aggregates dr 1, dr 3 and its own feature in the homogeneous nodes. And it aggregates the feature of pr 2, pr 5, di 1 and se 1 in heterogeneous nodes. For pr 3,
in the homogeneous nodes, it aggregates pr 1, pr 2 and its own feature, In heterogeneous nodes, it aggregates the feature of dr 1, di 2 . (C) Finally, we concatenate the
three layers of embedding. Then, we use the inner product to get the score between dr 7 and pr 3 and use the cross-entropy to optimize the model by an end-to-end

fashion.



the node feature representation that the model get will only
aggregate its 1st-order information of neighbor. Therefore,
stack _N_ layers of graph convolutional layers can make the
feature representation effectively convolve information from
its information of _N_ -order neighbors [31]. In our model, we
implement a three-layers graph convolution neural network. In
each layer, the feature representation of each node is generated
by aggregating the features of its neighbors connecting by
different types of edges. The details of the model are shown in
Figure 1.
It is worth noting that stacking more layers into a GCN model
has probability to lead to the common vanishing gradient problem [32, 33]. In other words, it may cause over-smoothing when
using back-propagating to train the parameters of GCN-based
model. The features of vertices within a connected component
may converge to the same value because of the over-smoothing
problem [34]. Besides, in representation learning, given a feature learned by a multilayer neural network-based model, there
might be loss of feature information. Based on this mechanism,
in the fields of computer vision and natural language processing,
many methods have been proposed to solve this problem. For
instance, in order to prevent the loss of information in the recurrent neural network [35], a long short-term memory network [36,
37] model is proposed. To prevent the loss of information in the



convolutional neural network [38], a residual net (ResNet) model
is proposed [39].
Inspired by He _et al_ . [27] and Wang _et al_ . [40], He _et al_ . proposed
a model named LightGCN; it considers the representations of
different GCN layers. It means that the model considers the
information loss in different layers. Wang _et al_ . developed a
model named convolution spatial graph embedding network (CSGEN) to predict the molecule property. In their framework, the
model uses concatenate operation to prevent information loss
like the ResNet [39]. In order to prevent information loss and
overcome over-smoothing, we propose a simple but effective
method. In detail, we concatenate the representation of each
node in different layers. Given a drug or target _v_ _i_ in the heterogeneous network, the feature representation _v_ _i_ can be described
as follows:


_h_ _i_ = _h_ (1) _i_ ⊕ _h_ (2) _i_ ⊕ _h_ (3) _i_ (3)


where _h_ (1) _i_ ∈ R _[d]_, _h_ (2) _i_ ∈ R _[d]_ and _h_ (3) _i_ ∈ R _[d]_ represent the feature
representation of _v_ _i_ obtained in the 1st layer, the 2nd layer and
the 3rd layer of graph convolutional neural network, respectively.
⊕ represents the vector concatenate operation. After aforementioned operation, we obtain the feature representation _h_ _i_ ∈ R [3] _[d]_ of
each drug or target.


_End-to-end heterogeneous graph representation learning-based framework_ 5



**DTI prediction**


In this section, we introduce how to predict the DTI and optimize
the model by an end-to-end fashion.
After obtaining the representation of drugs and proteins, we
use the inner product method [41] to predict DTIs. In detail,
given two nodes _v_ _i_ and _v_ _j_, _h_ _i_ and _h_ _j_ represent their feature
representations. The probability of existing interaction between
_v_ _i_ and _v_ _j_ can be calculated as:


_p_ _[ij]_ = _σ_ ( _h_ _[T]_ _i_ _[h]_ _[j]_ [)] (4)


where _σ_ ( _x_ ) = 1 _/_ (1 + _e_ [−] _[x]_ ) is the sigmoid function, _p_ _[ij]_ represents the
interaction score between _v_ _i_ and _v_ _j_ .
We use the cross-entropy loss to train the model. In the drug–
target identification problem, the number of negative samples
are much larger than the number of positive samples. Therefore,
we use negative sampling [29, 42, 43] to optimize the model. The
loss function is



_L_ = � − log _p_ _ijr_ [−] [E] _n_ ∼ _P_ _r_ ( _j_ ) [log] �1 − _p_ _[in]_ _r_ � (5)

_r_



binary network. The detail of how to construct binary networks
can be found in the Supplementary Document. The edge information of binary networks can be found in the Supplementary
Document. Then, we construct a heterogeneous network by
combining drug–drug binary network, protein–protein binary
network and drug–protein interaction network. An example of
a heterogeneous network based on Yamanishi _et al_ . dataset is
shown in the Supplementary Document.
In summary, the difference between Luo _et al_ . dataset and
Yamanishi _et al_ . dataset is that the types of nodes and edges
contained in the heterogeneous network. In the Luo _et al_ . dataset,
it contains four kinds of nodes (i.e. drug, protein, disease, sideeffect). However, in the Yamanishi _et al_ . dataset, there are only
two kinds of nodes (i.e. drug, protein). Thus, the different types of
nodes leads to the different types of edges in the heterogeneous
network.


**Experimental settings**


_**Data generation**_


Following the same method used in [2], we generate the evaluation dataset. In detail, the known DTIs are considered as
the positive samples. We randomly select the same number
of unknown DTIs as negative samples. We use 10-fold crossvalidation method for evaluation. For positive samples, we randomly hold out 10% of the whole labeled DTIs as the testing set
and utilize the remaining 90% as the training set. For negative
samples, the method to generate the training set and testing set
is same as the method to generate positive samples.


_**Performance evaluation**_


DTI identification can be treated as a link prediction task

[29]. Thus, we adapt Area Under the Receiver Operating
Characteristic Curve (AUROC) and Area Under the PrecisionRecall Curve (AUPR) as the model evaluation criteria. In the
evaluation test, we compare four existing algorithms, which
are BLMNII [14], NetLapRLS [22], CMF [21] and DTINet [2]. The
introduction of BLMNII, NetLapRLS, CMF and DTINet can be
found in the Supplementary Document. We do not compare
GCN-DTI [26] because it requires extra features of drug and
target, such as categories of drug, amino acid information of
target. Besides, the parameters settings of EEG-DTI can be found
in the Supplementary Document.


**Performance evaluation on Luo dataset**


We apply EEG-DTI on Luo _et al_ . dataset to validate its performance. The error bar chart of AUROC and AUPR is shown in

Figure 2. The detailed AUROC and AUPR performance of Luo _et_
_al_ . datasets on the DTI identification task is shown in Table 1.

We compare EEG-DTI with four methods (i.e. BLMNII, CMF,
NetLapRLS, DTINet). The experiment results show that EEG-DTI
achieves the highest performance among all methods according
to AUROC and AUPR. Compared with other methods, the AUROC
is 1.68% of EEG-DTI higher than other methods (from 0.9391 to
0.9559), and AUPR is 1.41% higher than other methods (from
0.9504 to 0.9645). The improvement of the results is due to
the fact that EEG-DTI can capture the information of neighbors
in the heterogeneous network and learn the feature representation of nodes by an end-to-end fashion. In summary, EEGDTI performs better than some state-of-the-art approaches in
identifying DTIs.



where _r_ is drug–protein edge type or protein–drug edge type.
_ij_
_p_ _r_ [represents the probabilities of positive samples that are cal-]
culated by the inner product with edge type _r_, _p_ _[in]_ _r_ represents
the probabilities of negative samples that are calculated by the
inner product by negative sampling randomly, which follows
the sampling distribution _P_ _r_ with edge type _r_ . We hope that the
model assigns the probabilities for the observed edges as high
as possible and the probabilities for the random edges as low as
possible by using the cross-entropy loss.


Results


**Data preparation**


To evaluate the performance of the end-to-end heterogeneous
graph representation learning-based framework for DTI prediction, we test our model on two datasets, namely, Luo _et al_ .
dataset [2] and Yamanishi _et al_ . dataset [44]. These two datasets
are widely used for evaluating DTI identification algorithms in
previous studies.
Luo _et al_ . dataset contains six drug/protein-related networks:
drug–drug interaction network [DrugBank (Version 3.0)] [45],
protein–protein interaction network [HPRD database (Release
9)] [46], drug–protein interaction network [DrugBank (Version
3.0)] [45], drug–disease association network (Comparative
Toxicogenomics Database) [47], protein–disease association
network (Comparative Toxicogenomics Database) [47] and
drug–side-effect association network [sider database (version
2)] [48]. The details of Luo _et al_ . dataset can be found in
the Supplementary Document. Besides, the details of how to
construct the heterogeneous network based on Luo _et al_ . dataset
can be found in the Supplementary Document.
Yamanishi _et al_ . dataset contains four subdatasets: nuclear

receptor (NR), G-protein-coupled receptors (GPCR), ion channels (IC) and enzyme. Each subdataset contains three networks: drug–drug structure similarity network, protein–protein
sequence similarity network and the drug–protein interaction
network. The detailed information of Yamanishi _et al_ . dataset

can be found in the Supplementary Document. For every subdataset of Yamanishi _et al_ . dataset, we construct two binary networks including drug–drug binary network and protein–protein


6 _Peng_ et al _._


**Table 1.** The detailed AUROC and AUPR performance of Luo _etal_ . dataset on DTI identification task _Notes_ : The five methods are BLMNII, CMF,
NetLapRLS, DTINet and EEG-DTI, respectively. Bolded numbers are the best performance.


BLMNII CMF NetLapRLS DTINet EEG-DTI


AUROC 0.6595 0.9222 0.9391 0.9308 **0.9559**

AUPR 0.6382 0.9413 0.9476 0.9504 **0.9645**


Notes: The five methods are BLMNII, CMF, NetLapRLS, DTINet and EEG-DTI, respectively. Bolded numbers are the best performance.


**Table 2.** The detailed AUROC and AUPR performance of the Yamanishi _etal_ . dataset on DTI identification task _Notes_ : The five methods are
BLMNII, CMF, NetLapRLS, DTINet and EEG-DTI, respectively. Bolded numbers are the best performance.


Dataset BLMNII CMF NetLapRLS DTINet EEG-DTI


AUROC GPCR 0.9107 0.8883 0.8849 0.8833 **0.9606**

Enzyme 0.9681 0.9066 0.948 0.9380 **0.9880**

NR **0.9111** 0.7963 0.8148 0.8284 0.8988

IC 0.9712 0.9531 0.9649 0.9139 **0.9861**

AUPR GPCR 0.9206 0.9135 0.9038 0.8789 **0.9522**

Enzyme 0.9678 0.9359 0.9637 0.9479 **0.9858**

NR **0.9461** 0.7621 0.8639 0.7308 0.8936

IC 0.9715 0.9649 0.9731 0.9089 **0.9846**


Notes: The five methods are BLMNII, CMF, NetLapRLS, DTINet and EEG-DTI,respectively. Bolded numbers are the best performance." as table legend in Table 2.


**Table 3.** The AUROC and AUPR performance when removing nodes
or its combinations on _Luoetal_ . dataset of EEG-DTI model


Settings AUROC AUPR


Whole network **0.9559** **0.9645**

Without side-effects 0.9522 0.9614

Without diseases 0.9234 0.9455

Without diseases and side-effects 0.9178 0.9394


Bolded numbers are the best performance.



**Figure 2.** EEG-DTI improves DTI prediction performance on Luo _etal_ . dataset.
We compared the DTI prediction performance of EEG-DTI to other state-of-theart approaches, BLMNNI, CMF, NetLapRLS and DTINet. In the figure, the left
one and right one represent the AUROC and AUPR results of different methods,

respectively.


**Performance evaluation on Yamanishi** _etal_ . **dataset**


For further evaluation, we also implement EEG-DTI algorithm on
Yamanishi _et al_ . dataset to evaluate its performance. Comparing
EEG-DTI algorithm with four other approaches, we can observe
consistent improvement in all four subdatasets of Yamanishi _et_
_al_ . dataset (see Figure 3 and Table 2).
The experiment results show that EEG-DTI achieves the highest performance among all methods according to AUROC, AUPR
on GPCR, Enzyme and IC. Compared with other methods, the
AUROC of EEG-DTI increases by 4.99%, 1.99%, 1.49% on GPCR,
Enzyme and IC, respectively. In AUPR performance, it increases
by 3.16%, 1.8%, 1.15% on GPCR, Enzyme and IC, respectively. In
the NR dataset, the AUROC and AUPR of EEG-DTI rank 2nd. It is
important to note that in the design of the end-to-end learning
model, the premise is that there is a large amount of training
data. However, in NR dataset, there are very few drug nodes and
protein nodes in the heterogeneous network, so the results are
not as good as other subdatasets. The detailed statistics of the
number of nodes and edges can be found in the Supplementary
Document. In summary, this experiment shows that EEG-DTI
can achieve significant improvement in identifying the DTIs
compared with some state-of-the-art methods.



**The effects of different types of nodes and their**
**combinations in the heterogeneous network**


The EEG-DTI DTI prediction approach proposed in this paper
mainly contains two parts: the heterogeneous network construction and GCN-based DTI prediction. In order to evaluate
the effects of different types of nodes and its combinations
in the heterogeneous network, we implement three ablation
experiments on Luo _et al_ . dataset. The experimental results are
shown in Table 3. Furthermore, the essence of removing nodes
is to remove the related edges in the heterogeneous network.
In order to evaluate the effect of edges in the heterogeneous
network, we make some experiments to evaluate the effect of
edges and their combinations. The results are shown in the
Supplementary Document).
It is shown in Table 3 that EEG-DTI algorithm shows substantial superiority if all networks are used. If the side-effects
are removed, the AUROC and AUPR decrease, but it just little
drops. However, if the diseases are removed, the AUROC and
AUPR decrease greatly. The reason is that side-effects are only
related to drugs in heterogeneous networks, but diseases are
related to drugs and proteins. We can infer that when performing the task of DTI identification, adding disease nodes
and related edges can improve the performance of identifying
DTIs. Finally, if the diseases and side-effects are removed at
the same time, compared with removing diseases or removing
side-effects, respectively, the AUROC and AUPR decrease. It is
proved that combining diseases and side-effects combinations
can improve the performance of DTI identification.


_End-to-end heterogeneous graph representation learning-based framework_ 7


**Figure 3.** EEG-DTI improves DTI prediction performance on Yamanishi _etal_ . dataset. We compared the DTI prediction performance of EEG-DTI to other state-of-the-art

approaches, BLMNNI, CMF, NetLapRLS and DTINet. In the figure, the 1st row and 2nd row represent the AUROC and AUPR results of different methods, respectively.
From left to right are GPCR, Enzyme, NR and IC, repectively.



**Table 4.** The AUROC and AUPR of different number of GCN layers


Settings AUROC AUPR


One layer 0.9390 0.9542
Two layers 0.9470 0.9610
Three layers **0.9559** **0.9645**
Four layers 0.9543 0.9639
Five layers 0.9542 0.9641


Bolded numbers are the best performance.


**The effects of GCN layers**


We mentioned that stacking _N_ layers of GCN layer, it can convolve information from its _N_ -order neighbors [31]. In order to
evaluate the effects of different numbers of GCN layers to EEGDTI, we run our model with different numbers of GCN layers
on Luo _et al_ . dataset. The experimental results are shown in
Table 4. Besides, in order to evaluate the effectiveness of the
concatenate operation, we apply the five ablation experiments
to demonstrate it. The detailed results of the effectiveness of the

concatenate operation can be found in the Supplementary Document. The results demonstrate that the concatenate operation
improve the performance of the EEG-DTI model.
It is shown in Table 4 that EEG-DTI algorithm shows substantial superiority if the number of layers equals to three. In detail,
if the number of layers is less than or equal to three, the AUROC
and AUPR increase with the increase of the number of GCN

layers. Comparing with three-layers GCN, there is no significant
increase if the number of layers is four or five. The reason might
be that GCN has explored deep enough neighbors. Therefore, the
performance does not increase significantly with the increase of
the number of GCN layers.


Conclusion


In this paper, we propose a novel end-to-end HGCN algorithm,
termed as EEG-DTI, and apply it on DTI prediction. We construct
a heterogeneous network by combining with multiple biological
networks and learn the low-dimensional feature representation



based on heterogeneous network. Then, we optimize the model
by end-to-end learning. To demonstrate the performance of EEGDTI, we compare our method with four state-of-the-art measures. The evaluation on two datasets, named Luo _et al_ . and
Yamanishi _et al_ ., demonstrates that EEG-DTI performs better
than other existing state-of-the-art approaches. Furthermore,
we test the contribution of different types of nodes in the heterogeneous network and the effects of different numbers of GCN
layers.
Besides, although EEG-DTI is mainly designed for predicting
DTIs, it is an extendible method and can also be used to predict other biological links, such as microRNA–small molecule
association [49, 50, 51], microRNA–disease association [52] and
disease–disease association [53]. Furthermore, we will develop a
new version of EEG-DTI for handling the weighted and directed
networks.


Key Points


   - To better describe the relations between drugs and targets, we construct a heterogeneous network with multiple entities (i.e. drug, protein, disease, side-effect)
and multiple types of edges.

   - We propose a HGCN-based method to learn the drug
and target feature representation based on a heterogeneous network.

   - We propose an end-to-end framework to predict DTIs,
which can optimize parameters in the model based on
the final DTI prediction task.

   - The evaluation results show that EEG-DTI outperforms some state-of-the-art approaches for DTI prediction.


Supplementary material


[Supplementary](https://academic.oup.com/bib/article-lookup/doi/10.1093/bib/bbaa430#supplementary-data) data are available online at _Briefings_ _in_
_Bioinformatics_ .


8 _Peng_ et al _._


Acknowledgments


This work was funded by National Natural Science Foundation of China (No. 62072376, U1811262, 61772426), the International Postdoctoral Fellowship Program (No. 20180029),
China Postdoctoral Science Foundation (No. 2017M610651),
Top International University Visiting Program for Outstanding Young scholars of Northwestern Polytechnical
University.


Conflict of Interest


The authors declare no conflicts of interest.


Authors’ Contributions Statement


J.P. and Y.W. conceived the experiments, Y.W. conducted the
experiments and all authors analyzed the results. J.P., Y.W.
and J.G. wrote the manuscript, and all authors reviewed the
manuscript.


References


1. Cheng F, Liu C, Jiang J, _et al._ Prediction of drug-target interactions and drug repositioning via network-based inference.
_PLoS Comput Biol_ 2012; **8** (5):e1002503.
2. Luo Y, Zhao X, Zhou J, _et al._ A network integration approach
for drug-target interaction prediction and computational
drug repositioning from heterogeneous information. _Nat_
_Commun_ 2017; **8** (1): 1–13.
3. Huang Y, Zhu L, Tan H, _et al._ Predicting drug-target on heterogeneous network with co-rank. In: _International Conference_
_on Computer Engineering and Networks_ . Springer, 2018, 571–81.
4. Lee H, Lee JW. Target identification for biologically active
small molecules using chemical biology approaches. _Arch_
_Pharm Res_ 2016; **39** (9): 1193–201.
5. Schirle M, Jenkins JL. Identifying compound efficacy targets
in phenotypic drug discovery. _Drug Discov Today_ 2016; **21** (1):

82–9.

6. Chen X, Yan CC, Zhang X, _et al._ Drug–target interaction prediction: databases, web servers and computational models.
_Brief Bioinform_ 2016; **17** (4): 696–712.
7. Ezzat A, Min W, Li X-L, _et al._ Computational prediction of
drug–target interactions using chemogenomic approaches:
an empirical survey. _Brief Bioinform_ 2019; **20** (4): 1337–57.
8. Chen R, Liu X, Jin S, _et al._ Machine learning for drug-target
interaction prediction. _Molecules_ 2018; **23** (9): 2208.
9. Bagherian M, Sabeti E, Wang K, _et al._ Machine learning
approaches and databases for prediction of drug–target
interaction: a survey paper. _Brief Bioinform_ 2020.
10. Dai Y-F, Zhao X-M. A survey on the computational
approaches to identify drug targets in the postgenomic era.
_Biomed Res Int_ 2015; **2015** .
11. Fleuren WWM, Alkema W. Application of text mining in the
biomedical domain. _Methods_ 2015; **74** :97–106.
12. Gang Fu, Ying Ding, Abhik Seal, _et al._ Predicting drug target interactions using meta-path-based semantic network
analysis. _BMC bioinformatics_, **17** (1):160, 2016.
13. Bleakley K, Yamanishi Y. Supervised prediction of drug–
target interactions using bipartite local models. _Bioinformat-_
_ics_ 2009; **25** (18): 2397–403.
14. Mei J-P, Kwoh C-K, Yang P, _et al._ Drug–target interaction
prediction by learning from local information and neighbors.
_Bioinformatics_ 2013; **29** (2): 238–45.



15. Meng F-R, You Z-H, Chen X, _et al._ Prediction of drug–
target interaction networks from the integration of protein
sequences and drug chemical structures. _Molecules_ 2017;
**22** (7): 1119.
16. Altschul SF, Koonin EV. Iterated profile searches with psiblast-a tool for discovery in protein databases. _Trends Biochem_
_Sci_ 1998; **23** (11): 444–7.
17. Gribskov M, McLachlan AD, Eisenberg D. Profile analysis:
detection of distantly related proteins. _Proc Natl Acad Sci_
1987; **84** (13): 4355–8.
18. Sharma A, Lyons J, Dehzangi A, _et al._ A feature extraction
technique using bi-gram probabilities of position specific
scoring matrix for protein fold recognition. _J Theor Biol_ 2013;

**320** :41–6.

19. Tipping ME. Sparse bayesian learning and the relevance vector machine. _Journal of machine learning research_ 2001; **1** (Jun):

211–44.

20. Wang L, You Z-H, Chen X, _et al._ A computational-based
method for predicting drug–target interactions by using
stacked autoencoder deep neural network. _J Comput Biol_
2018; **25** (3): 361–73.
21. Zheng X, Ding H, Mamitsuka H, _et al._ Collaborative matrix
factorization with multiple similarities for predicting drugtarget interactions. In: _Proceedings of the 19th ACM SIGKDD_
_international conference on Knowledge discovery and data mining_,
2013, 1025–33.
22. Xia Z, Wu L-Y, Zhou X, _et al._ Semi-supervised drugprotein interaction prediction from heterogeneous biological spaces. In: _BMC systems biology_, Vol. **4** . Springer, 2010,

S6.

23. Chen X, Liu M-X, Yan G-Y. Drug–target interaction prediction
by random walk on the heterogeneous network. _Mol Biosyst_
2012; **8** (7): 1970–8.
24. Wang W, Yang S, Zhang X, _et al._ Drug repositioning
by integrating target information through a
heterogeneous network model. _Bioinformatics_ 2014; **30** (20):

2923–30.

25. Yan X-Y, Zhang S-W, He C-R. Prediction of drugtarget interaction by integrating diverse heterogeneous
information source with multiple kernel learning
and clustering methods. _Comput_ _Biol_ _Chem_ 2019; **78** :

460–7.

26. Zhao T, Yang H, Valsdottir LR, _et al._ Identifying drug–target
interactions based on graph convolutional network and
deep neural network. _Brief Bioinform_ 2020.
27. He X, Deng K, Wang X, _et al._ Lightgcn: simplifying and powering graph convolution network for recommendation. _arXiv_
_preprint arXiv:200202126_ 2020.
28. Wu S, Zhang Y, Gao C, _et al._ Garg: anonymous recommendation of point-of-interest in mobile networks by graph
convolution network. _Data Science and Engineering_ 2020; **5** (4):

433–47.

29. Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy
side effects with graph convolutional networks. _Bioinformat-_
_ics_ 2018; **34** (13): i457–66.
30. Schlichtkrull M, Kipf TN, Bloem P, _et al._ Modeling relational
data with graph convolutional networks. In: _European Seman-_
_tic Web Conference_ . Springer, 2018, 593–607.
31. Kipf TN, Welling M. Semi-supervised classification
with graph convolutional networks. _arXiv preprint arXiv:_

_160902907_ 2016.

32. Li G, Muller M, Thabet A, _et al._ Deepgcns: Can gcns go as deep
as cnns? In: _Proceedings of the IEEE International Conference on_
_Computer Vision_, 2019, 9267–76.


_End-to-end heterogeneous graph representation learning-based framework_ 9



33. Rong Y, Huang W, Xu T, _et al._ Dropedge: Towards deep
graph convolutional networks on node classification. In:
_International Conference on Learning Representations_, 2019.
34. Li Q, Han Z, Wu X-M. Deeper insights into graph convolutional networks for semi-supervised learning. _arXiv preprint_
_arXiv:180107606_ 2018.

35. Pearlmutter BA. Learning state space trajectories in recurrent neural networks. _Neural Comput_ 1989; **1** (2): 263–9.
36. Hochreiter S, Schmidhuber J. Long short-term memory. _Neu-_
_ral Comput_ 1997; **9** (8): 1735–80.
37. Gers FA, Schmidhuber J, Cummins F. _Learning to forget: Con-_
_tinual prediction with lstm_, 1999.
38. Krizhevsky A, Sutskever I, Hinton GE. Imagenet classification with deep convolutional neural networks. In: _Advances_
_in neural information processing systems_, 2012, 1097–105.
39. He K, Zhang X, Ren S, _et al._ Deep residual learning for image
recognition. In: _Proceedings of the IEEE conference on computer_
_vision and pattern recognition_, 2016, 770–8.
40. Wang X, Li Z, Jiang M, _et al._ Molecule property prediction
based on spatial graph embedding. _J Chem Inf Model_ 2019;
**59** (9): 3817–28.
41. Long Q, Jin Y, Song G, _et al._ Graph structural-topic neural
network. In: _Proceedings of the 26th ACM SIGKDD International_
_Conference on Knowledge Discovery & Data Mining_, 2020, 1065–

73.

42. Mikolov T, Sutskever I, Chen K, _et al._ Distributed representations of words and phrases and their compositionality.
In: _Advances in neural nformation processing systems_, 2013,

3111–9.

43. Trouillon T, Welbl J, Riedel S, _et al._ Complex embeddings
for simple link prediction. _International Conference on Machine_
_Learning (ICML)_ 2016.



44. Yamanishi Y, Araki M, Gutteridge A, _et al._ Prediction of drug–
target interaction networks from the integration of chemical
and genomic spaces. _Bioinformatics_ 2008; **24** (13): i232–40.
45. Knox C, Law V, Jewison T, _et al._ Drugbank 3.0: a comprehensive resource for ‘omics’ research on drugs. _Nucleic Acids Res_
2010; **39** (suppl_1): D1035–41.
46. Prasad TSK, Goel R, Kandasamy K, _et al._ Human protein
reference database-2009 update. _Nucleic Acids Res_ 2009;
**37** (suppl_1): D767–72.
47. Davis AP, Murphy CG, Johnson R, _et al._ The comparative toxicogenomics database: update 2013. _Nucleic Acids Res_ 2013;
**41** (D1): D1104–14.
48. Kuhn M, Campillos M, Letunic I, _et al._ A side effect resource
to capture phenotypic effects of drugs. _Mol Syst Biol_ 2010;
**6** (1): 343.
49. Wang C-C, Chen X. A unified framework for the prediction of
small molecule–microrna association based on cross-layer
dependency inference on multilayered networks. _J Chem Inf_
_Model_ 2019; **59** (12): 5281–93.
50. Zhao Y, Chen X, Yin J, _et al._ Snmfsmma: using symmetric
nonnegative matrix factorization and kronecker regularized
least squares to predict potential small molecule-microrna
association. _RNA Biol_ 2020; **17** (2): 281–91.
51. Chen X, Guan N-N, Sun Y-Z, _et al._ Microrna-small molecule
association identification: from experimental results to
computational models. _Brief Bioinform_ 2020; **21** (1): 47–61.
52. Peng J, Hui W, Li Q, _et al._ A learning-based framework
for mirna-disease association identification using neural
networks. _Bioinformatics_ 2019; **35** (21): 4364–71.
53. Peng J, Guan J, Hui W, _et al._ A novel subnetwork representation learning method for uncovering disease-disease
relationships. _Methods_ 2020.


