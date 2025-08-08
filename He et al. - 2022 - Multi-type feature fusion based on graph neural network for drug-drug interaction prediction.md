## He et al. BMC Bioinformatics     (2022) 23:224  BMC Bioinformatics

https://doi.org/10.1186/s12859-022-04763-2


### **RESEARCH**


### **Open Access**


# Multi‑type feature fusion based on graph neural network for drug‑drug interaction prediction

Changxiang He [1], Yuru Liu [1], Hao Li [2], Hui Zhang [3], Yaping Mao [4], Xiaofei Qin [2], Lele Liu [1*] and Xuedian Zhang [2]



*Correspondence:
ahhylau@outlook.com


1 College of Science,
University of Shanghai
for Science and Technology,
Shanghai 200093, China
2 School of Optical‑Electrical
and Computer Engineering,
University of Shanghai
for Science and Technology,
Shanghai 200093, China
3 Institute of Interdisciplinary
Integrative Medicine
Research, Shanghai University
of Traditional Chinese Medicine,
Shanghai 201203, China
4 School of Mathematics

and Statistis, Qinghai Normal
University, Xining 810008, China



**Abstract**

**Background:** Drug-Drug interactions (DDIs) are a challenging problem in drug
research. Drug combination therapy is an effective solution to treat diseases, but it can
also cause serious side effects. Therefore, DDIs prediction is critical in pharmacology.
Recently, researchers have been using deep learning techniques to predict DDIs. However, these methods only consider single information of the drug and have shortcomings in robustness and scalability.

**Results:** In this paper, we propose a multi-type feature fusion based on graph neural
network model (MFFGNN) for DDI prediction, which can effectively fuse the topological information in molecular graphs, the interaction information between drugs and
the local chemical context in SMILES sequences. In MFFGNN, to fully learn the topological information of drugs, we propose a novel feature extraction module to capture
the global features for the molecular graph and the local features for each atom of the
molecular graph. In addition, in the multi-type feature fusion module, we use the gating mechanism in each graph convolution layer to solve the over-smoothing problem
during information delivery. We perform extensive experiments on multiple real datasets. The results show that MFFGNN outperforms some state-of-the-art models for DDI
prediction. Moreover, the cross-dataset experiment results further show that MFFGNN
has good generalization performance.
**Conclusions:** Our proposed model can efficiently integrate the information from
SMILES sequences, molecular graphs and drug-drug interaction networks. We find that
a multi-type feature fusion model can accurately predict DDIs. It may contribute to
discovering novel DDIs.


**Keywords:** Multi-type feature fusion, Graph neural network, Gating mechanism, Link
prediction


**Introduction**

Drug-Drug interactions (DDIs) refer to the presence of one drug changing the pharmacological activity of another, which may produce some side effects and even injury
or death. At the same time, multiple drug combinations to treat diseases are inevita
ble. So, it is crucial to predict potential DDI. Traditional methods of DDI prediction


© The Author(s) 2022. **Open Access** This article is licensed under a Creative Commons Attribution 4.0 International License, which permits
use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original
author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third
party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or
[exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://​](http://creativecommons.org/licenses/by/4.0/)
[creat​iveco​mmons.​org/​licen​ses/​by/4.​0/. The Creative Commons Public Domain Dedication waiver (http://​creat​iveco​mmons.​org/​publi​](http://creativecommons.org/licenses/by/4.0/)
[cdoma​in/​zero/1.​0/) applies to the data made available in this article, unless otherwise stated in a credit line to the data.](http://creativecommons.org/publicdomain/zero/1.0/)


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 2 of 18


depend on in vivo and in vitro experiments. However, due to its limited environment,

too small scale, cumbersome and expensive process, the ability to predicting DDI is
greatly limited. Therefore, an efficient computational method is needed to predict
DDI.

In the past several years, people have proposed methods based on machine learn
ing [1–4] to solve this problem. Qiu et al. [5] summarized some methods based on

machine learning. Deng et al. [6] used chemical structure to learn the representa
tion of DDIs in representation module, and then predicted some rare events with few
examples in comparing module. Deng et al. [7] predicted DDI using different drug
features and constructed deep neural networks (DNN). Zhang et al. [8] predicted DDI

using manifold regularization.

Recently, graph-based representation learning has been applied to Drug-Drug inter
action. Drugs are compounds, each of which can be represented by a molecular graph
with the atom as the node and the chemical bond as the edge, or a Simplified Molecular Input Line Entry System (SMILES) sequence. In Drug-Drug interaction networks,

by treating the drug as the node and the interaction as the edge, DDI prediction can
be regarded as link prediction tasks. Graph neural network (GNN) has made some

progress in DDI prediction [9–13]. Feng et al. [14] predicted DDI using Graph Convolutional Network (GCN) and DNN. In addition, there are also many methods about

multi-type DDI prediction [15–17]. Nyamabo et al. [18] proposed to predict DDIs by
the interactions between drug substructures. Then, Nyamabo et al. [19] used gating
devices to learn the chemical substructures of drugs. Chen et al. [20] used the bi-level

cross strategy to fuse the structural information and knowledge graph information of

drugs.
Although the models mentioned have achieved significant results, there are still some
limitations: (i) The models mentioned are generally limited to only considering the structure, sequence or interaction information of the drugs, without considering the synergistic effects between them. (ii) For molecular graphs, only applying GNN can extract the
local features for the atoms of the molecular graph, but it is difficult to propagate the
information in the graph remotely to capture the global features for the molecular graph.
(iii) In drug-drug interaction networks, node features obtained by stacking multi-layer

GNNs will be smoothed and blurred, which loses the diversity of node features.

To address above issues, this paper proposes an end-to-end learning framework for
DDI prediction, namely MFFGNN. In MFFGNN, we first utilize deep neural networks
to capture the intra-drug features from SMILES sequences and molecular graphs. For

SMILES sequences, MFFGNN applies the bi-directional gate recurrent unit neural net
work [21] to extract local chemical context information from the sequences. For molec
ular graphs, MFFGNN not only utilizes graph interaction networks [22] but also graph

warp unit [23] to extract both the global features for the molecular graph and the local

features for each atom of the molecular graph. In addition, MFFGNN takes the intra
drug features as the initial features of the nodes in the DDI network and uses GCN

encoder to fuse the intra-drug features and external DDI features to update the drug

representation. Finally, we predict the missing interactions in the DDI graph through
Multi-layer Perceptron (MLP).

Overall, the main contributions of this paper are summarized as follows:


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 3 of 18


        - We propose a novel model MFFGNN for DDI prediction, which fuses the topologi
cal information in molecular graphs, the interaction information between drugs and

the local chemical context in SMILES sequences.

         - To better learn the topological structure of drugs, we propose a molecular graph feature extraction module (MGFEM) to extract the global features for the molecular

graph and the local features for each atom of the molecular graph.

        - We conduct extensive experiments on three real datasets with different scales to
demonstrate the superiority of our model.


**Related works**


**Drug‑drug prediction**

Drug-Drug prediction has always been a worthy research direction in pharmacology.

Most of previous work depended on in vivo and in vitro experiments. However, they do

not scale well due to the limitations of the laboratory environment [24]. Subsequently,

machine learning has been proposed to solve this problem. Similarity-based methods
calculated specific similarity measures [25–29], e.g., drug structure, targets, side effects,
genomic properties, therapeutic, etc., while combined with machine learning models for

drug prediction. Ryu et al. [30] predicted the type of drug-drug interactions using DNN

based on the similarity of the chemical structure of drugs. Graph-based methods pre
dicted drug-drug interactions by learning the molecular graph [31] or interaction graph

[32]. Shang et al. [33] modeled drugs as nodes and DDI as links, so tasks as link predic
tion problems.


**Graph neural network**

Recently, as a neural network method on graph domain, the study of graph neural network (GNN) has received great attention. With the development of GNN, many vari
ants based on GNN came out one after another [34–36]. Rahimi et al. [37] proposed to

control the transmission of neighbourhood information through gating operation. With

the increasing popularity of GNN, researchers are using GNN models for DDIs [38]. For

example, Duvenaud et al. [39] used GNN to perform molecular modeling by extracting molecular circular fingerprints. Lin et al. [40] used knowledge graph neural network
(KGNN) to mine their associated relations in knowledge graph to solve the DDI predic
tion problem. Bai et al. [41] proposed to learn drug feature representation by a Bi-level
Graph Neural Network (BI-GNN) to solve biological link prediction tasks. MIRACLE

[42] is most relevant to our work.


**Methods**


**Preliminaries**
We define the drug set as D={d 1, . . ., d n } and its corresponding SMILES sequence set as
Q = {q 1, q 2, . . ., q n }, where _n_
represents the number of drugs. We define the molecular
graph as G = (V, E), where V and E represent the sets of atoms and chemical bonds,

respectively, and interaction graph as G = (G, L), where L represents the links between
drugs. We use d h to define the dimension of the representation of the atom and chemical
bond and d
g to define the dimension of the representation of the drug.


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 4 of 18


**Fig. 1** Overview of MFFGNN, where [�] is sum. The MFFGNN uses SMILES sequences and molecular graphs
as inputs to the model, and then extracts the intra-drug features through the MGFEM and SSFEM modules,
respectively. Then, MFFGNN fuses the intra-drug features and external DDI features through MFFM module to
obtain the updated drug features. Finally, the final predicted value is obtained by DDI predictor









































































Super



























**Input:** **Graph interaction network**
**Molecular Graph** **Initial features** **and graph warp unit**



**Output:**
**updated atoms and**
**super node features**



**Fig. 2** Overview of MGFEM. The MGFEM module applies graph interaction network and graph wrap unit
to extract local information and global information of the molecular graph. When extracting the local
information, the module updates the edge feature before updating the node feature. When extracting the
global information, the module utilizes a supernode to promote the global propagation of information


_Problem description_ The DDI prediction problem is regarded as the link prediction
task on the graph. The interaction graph N can be represented by an adjacency matrix
**A** ∈ R [n][×][n] with each element a ij ∈{0, 1} . Given two drug nodes, the DDI prediction
problem is defined to predict whether there is an interaction between them.


**Overview of MFFGNN**
The framework of MFFGNN is shown in Fig. 1, which is divided into the following four
modules. In Molecular Graph Feature Extraction Module (MGFEM), we use the graph

interaction network with graph wrap unit to extract the topological structure features of

the drug from a given molecular graph. In SMILES Sequence Feature Extraction Module
(SSFEM), we employ the bi-directional gate recurrent unit to extract local chemical context from a given SMILES sequence. In Multi-type Feature Fusion Module (MFFM), we

apply GCN encoder to fuse the intra-drug features and external DDI features to update

the drug representation. Finally, we predict the missing interactions in the DDI graph

through MLP.


**Molecular graph feature extraction module**
The Molecular Graph Feature Extraction Module (MGFEM) is shown in Fig. 2. Molecular graphs are an important expression for drugs. We use RDKit [43] tool to construct the
molecular graph G based on SMILES sequence. First, we obtain the initial features **v** i [(][in][)]
of each atom according to atom symbol, formal charge, whether the atom is aromatic,
its hybridization, chirality, etc. Similarly, we obtain the initial features **e** ij [(][in][)] [ of each bond ]
according on the type of bond, whether the bond is in a ring, whether it is conjugated, etc.


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 5 of 18


Then, the initial atom and chemical bond features are transformed to R [d] [h] through a layer
neural network, and the calculation process is as follows:


**v** i [(][0][)] = ReLU( **W** v [(][0][)] **[v]** i [(][in][)] ) (1)



**e** [(][0][)] = ReLU( **W** e [(][0][)] **[e]** [(][in][)] ) (2)



ij [(][0][)] = ReLU( **W** e [(][0][)] **[e]** ij [(][in][)]



)
ij



where ReLU is the activation function, **W** v [(][0][)] [ and ] **[W]** e [(][0][)] [ are the learnable weight matrices. ]

To fully extract atom and chemical bond features, we apply graph interaction networks

[22]. In graph interaction network, firstly, the features of edge e ij are updated according
to the features of its connected nodes and itself, and the process is as follows:



**e** [(][l][+][1][)] = ReLU[( **e** [(][l][)] [||] **[v]** i [(][l][)] [||] **[v]** [(][l][)] [)] **[W]** e [(][l][)] + **b** [(] e [l][)] []] (3)




[(][l][+][1][)] = ReLU[( **e** [(][l][)]

ij ij



ij [(][l][)] [||] **[v]** i [(][l][)]



i [(][l][)] [||] **[v]** [(][l][)]



j [(][l][)] [)] **[W]** e [(][l][)]



e [(][l][)] + **b** [(] e [l][)]



e []]



where || is concatenation operation, **W** e [(][l][)] [ and ] **[b]** [(] e [l][)] [ are the learnable weight matrix and ]
the bias of the edge update, respectively. Then, the node features are updated according
to the features of its connected edges and itself, and the calculation process is as follows:



˜
**v** i [(][l][+][1][)] = ReLU[( **v** i [(][l][)] [||] � e [(] ij [l][+][1][)] ) **W** v [(][l][)] [+] **[ b]** [(] v [l][)] []] (4)



i [(][l][+][1][)] = ReLU[( **v** i [(][l][)]



e [(][l][+][1][)]




[(] ij [l][+][1][)] ) **W** v [(][l][)] [+] **[ b]** [(] v [l][)] []]



i [||]



�



j∈N (i)



where _N_ ( _i_ ) represents the neighbor of node _i_ .
The above processes can only spread the features of atoms and chemical bonds locally,
but cannot spread information globally. Therefore, we propose to extract the global features
of the molecular graph by applying graph warp unit (GWU) [23]. The properties of the
whole drug often influence drug-drug interaction prediction. The GWU consists of three
parts: supernode, transmitter and warp gate.

Supernode: We add a supernode to the graph, which can connect every atom in the
molecular graph. Then, the sum of all atom features is taken as the initial feature of the
supernode, **g** [(][0][)] ∈ R [d] [h], that is:



�



**v** [(][0][)] .



**g** [(][0][)] = � **v** i [(][0][)] . (5)



i .



i∈V


Then, the features of the supernode are updated by a single-layer neural network:


˜
**g** [(][l][)] = tanh( **W** g [(][l][)] **[g]** [(][l][−][1][)] [)] (6)


where **W** g [(][l][)] [ are the learnable weight matrix.]
Transmitter: The transmitter part gathers information from the atoms and the supernode. Before propagating the atom features to the supernode, we need to transform the
form of the information. Different atom features have different degrees of importance relative to the global features. Therefore, the transmitter part applies the multi-head attention
mechanism to aggregate different atom features. The calculation process is as follows:


**v** v [(][l] → [)] s [=][ tanh][(] **[W]** v [(][l] → [)] s [[||] [K] k=1 � α v [(][k],i [,][l][)] **[v]** i [(][l][−][1][)] ]) (7)

i∈V


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 6 of 18



a [(][k][,][l][)] **o** [(] v [k] i [,][l][)]



α [(][k] i [,][l][)] = softmax( **W** a [(][k][,][l][)] **o** [(][k] i [,][l][)] [)] (8)



v [(][k],i [,][l][)] = softmax( **W** a [(][k][,][l][)]



v,i [,] [)]



**o** v [(][k],i [,][l][)] = tanh( **W** a [(][k] 1 [,][l][)] **[v]** i [(][l][−][1][)] ) ⊙ tanh( **W** a [(][k] 2 [,][l][)] **[g]** [(][l][−][1][)] [)] (9)


where **v** v [(][l] → [)] s [ represents the information propagated from each atom to the supernode ]
at the l [th] layer, α v [(][k],i [,][l][)] [ represents the significance score of node ] _[i]_ [ at the ] [k] [th] [ head and the ]
l [th] layer, ⊙ represents the product of the elements and k = 1, 2, . . ., K, _K_ represents the
number of heads. The information propagated from the supernode to each atom is calculated by the following formula:



�



**W** s [(] → [l][)] v **[g]** [(][l][−][1][)] [�]



**g** s [(] → [l][)] v [=][ tanh] **W** s [(] → [l][)] v **[g]** [(][l][−][1][)] (10)



where **g** s [(] → [l][)] v [ represents the information propagated from the supernode to each atom at ]

the l [th] layer.
Warp Gate: The warp gate combines the transmitted information and sets the gating
coefficients to control the fusion of information. For each atom, gated interpolation is
used to fuse the information from the supernode **g** s [(] → [l][)] v [ with the updated atom features ]
**v** i [(][l][)] [:]



�



**v** s [(] → [l][)] i [=] **1** − α s [(] → [l][)] i ⊙˜ **v** i [(][l][)] + α s [(] → [l][)] i [⊙] **[g]** s [(] → [l][)] v (11)



**1** − α s [(] → [l][)] i



�



⊙˜ **v** i [(][l][)] + α s [(] → [l][)] i [⊙] **[g]** s [(] → [l][)] v



�



α s [(] → [l][)] i [=][ σ] **W** b [(][l] 1 [)] **[v]** [˜] i [(][l][)] + **W** b [(][l] 2 [)] **[g]** s [(] → [l][)] v (12)



**W** b [(][l] 1 [)] **[v]** [˜] i [(][l][)] + **W** b [(][l] 2 [)] **[g]** s [(] → [l][)] v



�



where α s [(] → [l][)] i [ represents the gating coefficient during the transmission from supernode ]

to each atom and **v** s [(] → [l][)] i [ represents the information transmitted to each atom. For super-]

node, gated interpolation is used to fuse information from atoms **v** v [(][l] → [)] s [ with updated ]
supernode features **g** ˜ [(][l][)] :



�



**g** i [(] → [l][)] s [=] **1** − α s [(] → [l][)] i ⊙˜ **g** [(][l][)] + α s [(] → [l][)] i [⊙] **[v]** v [(][l] → [)] s (13)



**1** − α s [(] → [l][)] i



�



⊙˜ **g** [(][l][)] + α s [(] → [l][)] i [⊙] **[v]** v [(][l] → [)] s



�



α i [(] → [l][)] s [=][ σ] **W** s [(] 1 [l][)] **[g]** [˜] [(][l][)] [ +] **[ W]** s [(] 2 [l][)] **[v]** v [(][l] → [)] s (14)



**W** s [(] 1 [l][)] **[g]** [˜] [(][l][)] [ +] **[ W]** s [(] 2 [l][)] **[v]** v [(][l] → [)] s



�



where α i [(] → [l][)] s [ represents the gating coefficient during the transmission from atom to ]
supernode and **g** i [(] → [l][)] s [ represents the information transmitted to supernode. Finally, the ]
updated features of each atom and supernode are calculated through the gated recurrent

units (GRU) [44]:



�



�



**v** i [(][l][)] = GRU v **v** i [(][l][−][1][)], **v** s [(] → [l][)] i (15)



**v** i [(][l][−][1][)], **v** s [(] → [l][)] i



�



.
�



**g** [(][l][−][1][)], **g** i [(] → [l][)] s



**g** [(][l][)] = GRU g **g** [(][l][−][1][)], **g** i [(] → [l][)] s . (16)



By applying this module to the whole dataset, we obtain the feature matrix **G** ∈ R [n][×][d] [g]

based on the molecular graph.


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 7 of 18









**READOUT**



**NCC1(CC(O)=O)CCC**

**CC1**













**Input:** **Output:**
**SMILES sequence** **Smi2Vec** **Embedding** **BiGRU** **Drug features**

**Fig. 3** Overview of SSFEM. The SSFEM module applies Smi2Vec and BiGRU to extract features from SMILES
sequences. Then, the whole drug features are obtained through the readout layer



**Smi2Vec**



**Embedding**



**BiGRU**



**SMILES sequence feature extraction module**

Drugs are commonly represented by the SMILES sequences, which are composed

of molecular symbols. SMILES sequences also contain rich features compared with
molecular graphs. The molecular graphs of the drug provide how the atoms are connected, while the SMILES sequences provide the functional information of the atoms

and long-term dependency representations. To capture the local chemical context in
SMILES sequences, we first utilized the embedding method to construct an atomic
embedding matrix, and then input it into the Bi-directional Gate Recurrent Unit
(BiGRU) neural network to obtain the entire drug representation. SMILES Sequence
Feature Extraction Module (SSFEM) is shown in Fig. 3.

Nowadays, most methods encode SMILES sequence by label or one-hot encoding.

However, one-hot encoding and label ignore the context information of the atom.
Therefore, to explore the function of the atom in the context, we propose to encode
SMILES sequences by an advanced embedding method, _Smi2Vec_ [45]. Specifically, for
SMILES sequences q 1, we divide them into a series of atomic symbols by space. Then,
we map each atom to an embedding vector according to the pre-trained embed
ding dictionary. Finally, we aggregate the embedding vectors of atoms to obtain an
embedding matrix **X** ∈ R [m][×][d] [h], in which _m_ is the number of atoms and each row is the

embedding of an atom.

We apply a layer of BiGRU [21] on the embedding matrix **X** . BiGRU trains the input
data with two GRUs in opposite directions, as shown in Fig. 3. The current hidden
state of BiGRU can be described as follows: [−→] **s** t =GRU( **x** t, [−−→] **s** t−1 ) and [←−] **s** t =GRU( **x** t, [←−−] **s** t−1 )
, where GRU(·) represents a non-linear transformation of the input vector. Therefore,
the hidden state **s** t at time _t_ can be expressed by the weighted sum of [−→] **s** t and [←−] **s** t, which

is expressed as follows:


**s** t = **W** t [−→] **s** t + **V** t [←−] **s** t + **b** t (17)



where **W** t and **V** t represent the weights, and **b** t represents the bias. Then, we use a fully
connected layer as the readout layer to obtain the drug representation. By applying this
module to the whole dataset, we obtain the sequence-based feature matrix **S** ∈ R [n][×][d] [g] .
Note that we should input a fix-sized matrix into the BiGRU layer. However, the
length of the SMILES sequence varies. We use the approximately average length of
the sequences in the dataset as the fixed length and apply zero-padding and cutting
operations.


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 8 of 18


**Fig. 4** Overview of MFFM, where **G** is gating and **G** [˜] is 1-gating. The MFFM takes the intra-drug features as
the initial node features in DDI network, and then update the node representation by multi-layer graph
convolution neural network with gating


**Multi‑type feature fusion module**

We combine the feature matrices **G** and **S** obtained above to obtain the intra-drug features,
namely **H** = **G** [�] **S** . In order to fuse the intra-drug features with the external DDI features,
we design a GCN encoder with the gating mechanism. Specifically, we take the intra-drug
features as the initial node features in the interaction graphs, and then update the node representation by multi-layer GCN. The Multi-type Feature Fusion Module (MFFM) is shown
in Fig. 4.
For drug d i, the output of r [th] layer is as follows:




[r][−] **W** [r]

j



**z** i [r] [=][ ReLU][(] � **A** ij **z** j [r][−] **W** u [r] [)] (18)



i [=][ ReLU][(]



�



**A** ˜ **z** [r][−][1]
ij



u [)]



j∈ N (i)



where **W** u [r] [ is learnable weight parameter. ] **[A]** [˜] [ij] [ is the component of the normalized adja-]


ˆ

cency matrix **A** [˜] . **A** [˜] = **K** [ˆ] [−] 2 [1] ( **A** + **I** n ) **K** ˆ [−] 2 [1] where **K** **ii** = [�] [(] **[A]** [ +] **[ I]** [n] [)] [ . We can add multiple ]




[1]

2 ( **A** + **I** n ) **K** ˆ [−] 2 [1]



ˆ

cency matrix **A** [˜] . **A** [˜] = **K** [ˆ] [−] 2 ( **A** + **I** n ) **K** ˆ [−] 2 where **K** **ii** = [�] j [(] **[A]** [ +] **[ I]** [n] [)] ij [ . We can add multiple ]

GCN layers to expand the neighborhood of label propagation, but it may also cause the
increase of noisy information. Meanwhile, the neighborhoods of different orders contain
different information. Therefore, we utilize the gating mechanism [37] to control how
much neighborhood information is passed to the node. The process is as follows:


T ( **z** i [r][−][1] ) = σ( **W** [r][−][1] **z** i [r][−][1] + **b** [r][−][1] ) (19)


**z** i [r] [=] **[ z]** i [r] [⊙] [T] [(] **[z]** i [r][−][1] ) + **z** i [r][−][1] ⊙ (1 − T ( **z** i [r][−][1] )) (20)


where T ( **c** [r][−][1] ) represents the gating weight of the (r − 1) [th] layer, ( **W** [r][−][1], **b** [r][−][1] ) are weight
matrix and bias variable of the (r − 1) [th] layer. After multi-layer GCN, we finally obtain
the feature matrix **Z** ∈ R [n][×][d] [g] for drugs in DDI Network.

In addition, inspired by MIRACLE, the module uses the graph contrastive learning

approach to balance the information inside and outside of the drug. For the drug d i, we
take itself and its first-order neighboring nodes as positive samples _P_ and the nodes not in
first-order neighbors as negative samples _N_ . We design a learning objective, which made
external features of drug d i consistent with internal features of positive samples and distinct
from internal features of negative samples, defined as follows:


L c = − log σ(f D ( **h** i, **z** i )) − log σ(1−f D ( **h** [˜] i, **z** i )) (21)


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 9 of 18


**Table 1** Detailed information about the datasets


**Dataset** **Drugs** **DDI links** **Information**


ZhangDDI [46] 548 48,548 Similarity


ChCh-Miner [47] 1514 48,514 –

DeepDDI [30] 1861 192,284 Polypharmacy side-effect


where f D (·) : R [d] [g] ×R [d] [g] �−→ R is the discriminator function, which scores agreement

between the two vectors of the input. Here we set it to the point product operation.


**DDI prediction**

Firstly, we obtain an interaction link representation by multiplying two drug representation. Then, we input it into the MLP to get the prediction score:



�



z i ⊙ z j ��



yˆ ij = σ �MLP�z i ⊙ z j �� (22)



MLP



�



where MLP consists of two fully connected layers.

Our learning objective is to minimize the distance between the predictions and the
true labels. The specific formula is as follows:



�



ˆ

� y ij log(y ij ) + (1 − y ij ) log(1 −ˆy ij )

l ij ∈ L



L r = − � y ij log(y ij ) + (1 − y ij ) log(1 −ˆy ij )
(23)



where y ij is the real label for drug pair (d i, d j ) . Then, we unify the DDI prediction task
and the contrastive learning task into a learning framework. Formally, the learning

objective of our model is:


L = L r + αL c (24)


where α is a hyper-parameter used to control the magnitude of contrastive task.


**Results**

In this section, we design various experiments to demonstrate the superiority of the

model MFFGNN.


**Experimental setup**
**Datasets.** To verify the validity of our model on datasets with different scales, we evaluate the proposed model in small, medium, and large datasets. In the small-scale dataset,
the number of drugs is relatively small, but fingerprints of all drugs are available. In the
medium-scale dataset, although the number of drugs is relatively large, there is only the

same number of labeled DDI links as in small-scale dataset. In the large-scale dataset,
most of drugs lack many fingerprints. Detailed information about the datasets can be
seen in Table 1.

Note that we removed the SMILES sequences that cannot construct the graph in the

dataset.

_Baselines_ To demonstrate the superiority of our model, we compare MFFGNN with

the following state-of-the-art models:


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 10 of 18


            - _SSP-MLP_ [30]: This approach used the names and structural information of drugdrug or drug-food pairs as inputs and applied Structural Similarity Profile (SSP) and
MLP for classification. We name this model as SSP-MLP.

            - _Multi-Feature Ensemble_ [46]: This approach combined multiple types of data and
proposed a collective framework. We name this model as Ens.

            - _GCN_ [48]: This approach applied GCN to perform semi-supervised node classification. We use GCN to extract structural information of drugs for DDI prediction.

            - _GAT​_ [35]: This approach used GAT to perform node classification task. We apply
GAT to extract drug features in interaction graph for DDI prediction.

            - _SEAL-C/AI_ [49]: This approach performs semi-supervised graph classification tasks
from a hierarchical graph perspective. We apply this model to obtain drug features

for DDI prediction.

            - _NFP-GCN_ [39]: This approach designs a GCN for learning molecular fingerprints.
We name this model as NFP-GCN.

            - _MIRACLE_ [42]: This approach simultaneously learned the inter-view molecular
structure information and intra-view interaction information of drugs for DDI pre
diction.

            - _MFs_ [50]: This approach only used molecular fingerprints as input to the DDI network to predict DDIs, we name this model as MFs.

        - We also consider several multi-type DDI prediction methods and apply them to
binary classification tasks, i.e. DPDDI [14], SSI-DDI [18], DDIMDL [7], MUFFIN

[20].


_Implementation details_ For the division of the datasets, the splitting method is the same

as MIRACLE [42]. We divide 80% of each dataset into the training set, 20% into the test
set, and 20% of the training set are randomly sampled as the validation set. The dataset only contains positive drug pairs. For negative training samples, we select the same

number of negative drug pairs [51].

We utilize Adam [52] optimizer to train the model and Xavier [53] initialization to ini
tialize the model. We utilize the exponential decay method to set the learning rate, where
the initial learning rate is 0.0001 and the multiplication factor is 0.96. The model applies
a dropout [54] layer to the output of each intermediate layer, where the dropout rate is

0.3. We set the dimension of the atom-level and drug-level representations as 256. We

set K = 2
in the multi-head attention mechanism. To evaluate the effectiveness of the
model MFFGNN, we consider three metrics, including Area Under the Receiver Operating Characteristic curve (AUROC), Area Under the Precision-recall Curve (AUPRC)

and F1.


**Comparison results**

To verify the validity of the proposed MFFGNN, we compare MFFGNN with stateof-the-art models for DDI prediction on three datasets with different scales. Over ten
repeated experiments, we give the mean and standard deviation. The best results are
highlighted in bold.

_Comparison on the ZhangDDI dataset_ We compare the MFFGNN model with stateof-the-art models on the ZhangDDI dataset, and the results are shown in Table 2. The


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 11 of 18


**Table 2** Comparison results on ZhangDDI dataset


**Method** **AUROC** **AUPRC** **F1**


SSP-MLP 92.51 ± 0.15 88.51 ± 0.66 80.69 ± 0.81


Ens 95.20 ± 0.14 92.51 ± 0.15 85.41 ± 0.16


GCN 91.91 ± 0.62 88.73 ± 0.84 81.61 ± 0.39


GAT​ 91.49 ± 0.29 90.69 ± 0.10 80.93 ± 0.25


SEAL-C/AI 92.93 ± 0.19 92.82 ± 0.17 84.74 ± 0.17


NFP-GCN 93.22 ± 0.09 93.07 ± 0.46 85.29 ± 0.17


MIRACLE 98.95 ± 0.15 98.17 ± 0.06 93.20 ± 0.27


MFFGNN **99.06 ± 0.08** **98.83 ± 0.16** **97.97 ± 0.25**


**Table 3** Comparison results on ChCh-Miner dataset


**Method** **AUROC** **AUPRC** **F1**


GCN 82.84 ± 0.61 84.27 ± 0.66 70.54 ± 0.87


GAT​ 85.84 ± 0.23 88.14 ± 0.25 76.51 ± 0.38


SEAL-C/AI 90.93 ± 0.19 89.38 ± 0.39 84.74 ± 0.48


NFP-GCN 92.12 ± 0.09 93.07 ± 0.69 85.41 ± 0.18


MIRACLE 96.15 ± 0.29 95.57 ± 0.19 92.26 ± 0.09


MFFGNN **97.02 ± 0.25** **98.45 ± 0.06** **96.94 ± 0.39**


results of these baselines are obtained from Table 2 in Ref. [42]. As can be seen, the

methods considering multiple features, such as Ens, SEAL-C/AI, NFP-GCN and MIRA
CLE, perform better than the methods considering only one feature. However, the MFF
GNN has the best performance. MFFGNN considers not only the topological structure

information in molecular graphs and the interaction information between drugs, but
also the local chemical context in SMILES sequences. This indicates that multi-type feature fusion can improve the performance of the model.
_Comparison on the ChCh-Miner dataset_ Because the ChCh-Miner dataset lacks fingerprints and side-effect information, we only compare the MFFGNN with the graphbased models, and the results are shown in Table 3. The results of these baselines are
obtained from Table 3 in Ref. [42]. As shown in Table 3, MFFGNN outperforms all
baselines in all metrics, indicating that MFFGNN still maintain its effectiveness on the
dataset with few labeled data. In addition, we obtain labeled training data with different amounts by adjusting the proportion of the training set on the ChCh-Miner dataset.
This can analyze the robustness of the MFFGNN. We compare MFFGNN with other
methods, and the results are shown in Fig. 5a. The results show that MFFGNN has high
performance even in a small amount of labeled data. The reason could be that (i) our
model fuses topological structure, local chemical context and DDI relationships; (ii) our

model extracts both the global features for the molecular graph and the local features for
the atoms of the molecular graph; (iii) our model sets a gating mechanism for each graph

convolution layer to prevent over-smoothing when stacking multi-layer GCN.

_Comparison on the DeepDDI dataset_ To verify the scalability of MFFGNN, we perform

comparative experiments on the DeepDDI dataset, and the results are shown in Table 4.

Because there may be missing information in the large-scale dataset, we only choose the


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 12 of 18



(a) Comparison results on ChCh-Miner dataset with different
training proportion


**Fig. 5** Experimental results on ChCh-Miner dataset



(b) Ablation experimental results on ChCh-Miner datasets



SSP-MLP model. And the NFP-GCN model has worse performance and space limitation. We also ignore the experimental results. We use 881 dimensional molecular fingerprints as the initial node features in the DDI graph for DDIs prediction. Meanwhile,

we degrade multi-type DDI prediction methods and obtain binary prediction results on

DeepDDI dataset.
As shown in Table 4, MFFGNN has high AUROC, AUPRC and F1. The MFs model is
relatively poor in all metrics, which only contains one drug feature. Single feature can not
comprehensively represent drug information, which will ultimately affect the prediction
results. However, MFFGNN integrates the features from drug sequences and molecular

graphs to input into DDI graph, so that a more comprehensive drug information can be

learned. Although the SSI-DDI and MIRACLE models have higher AUROC metric than

MFFGNN, MFFGNN has the highest AUPRC and F1 values. In general, the AUPRC

metric is more important than the AUROC metric, because it penalizes false positive
DDIs better. F1 focuses on the proportion that can correctly predict DDIs. The imbalance of the data in the DeepDDI dataset may have a negative impact on the AUROC
metrics of our model. However, this does not affect the performance of MFFGNN.
_Cross-dataset evaluations_ To further evaluate that MFFGNN has good generalization

performance, we perform cross-dataset evaluations. One dataset serves as the training

set, while the other two serve as test sets. Because of the poor performance of other

methods, we compare MFFGNN to three methods, including GAT, SEAL-C/AI and
MIRACLE, and the results are shown in Fig. 6. As shown in figures, MFFGNN outperforms the other methods in AUROC, AUPRC and F1. From the above results, it can be

shown that our model can predict drug-drug interaction with steady accuracy, independent of the scale of the datasets. Through this experiment, we can also verify that
MFFGNN has good generalization performance.


**Ablation study**

In order to verify the validity of each type of feature of drugs, we carry out DDI predictions using each type of feature or combination of feature on ChCh-Miner datasets. The
experimental results are shown in Table 5. The best results are highlighted in bold.


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 13 of 18


**Table 4** Comparison results on DeepDDI dataset


**Method** **AUROC** **AUPRC** **F1**


SSP-MLP 92.28 ± 0.18 90.27 ± 0.28 79.71 ± 0.16


GCN 85.53 ± 0.17 83.27 ± 0.31 72.18 ± 0.22


GAT​ 84.84 ± 0.23 81.14 ± 0.25 73.51 ± 0.38


SEAL-C/AI 92.83 ± 0.19 90.44 ± 0.39 80.70 ± 0.48


MFs 91.54 ± 0.04 89.82 ± 0.24 83.05 ± 0.5


DPDDI 92.79 ± 0.38 91.15 ± 0.52 85.54 ± 0.40


SSI-DDI **96.14 ± 0.06** 94.63 ± 0.47 92.27 ± 0.14


DDIMDL 94.85 ± 0.71 93.48 ± 0.07 82.31 ± 0.44


MUFFIN 95.26 ± 0.12 94.47 ± 0.28 91.22 ± 0.48


MIRACLE 95.51 ± 0.27 92.34 ± 0.17 83.60 ± 0.33


MFFGNN 95.39 ± 0.25 **96.81 ± 0.16** **92.54 ± 0.61**


The best results are highlighted in bold



(a) Training set: ZhangDDI Test set:
ChCh-Miner



(b) Training set: ChCh-Miner Test
set: ZhangDDI



(c) Training set: DeepDDI Test set:
ZhangDDI



(d) Training set: ZhangDDI Test set: (e) Training set: ChCh-Miner Test set: (f) Training set: DeepDDI Test set:
DeepDDI DeepDDI ChCh-Miner


**Fig. 6** Cross-dataset experimental results



(d) Training set: ZhangDDI Test set:
DeepDDI



(e) Training set: ChCh-Miner Test set:
DeepDDI



**Table 5** The performance of different types of features on ChCh-Miner dataset


**Method** **AUROC** **AUPRC** **F1**


S 90.17 ± 0.04 90.27 ± 0.18 89.14 ± 0.08


M 92.87 ± 0.74 92.55 ± 0.40 90.93 ± 0.56


I 93.23 ± 0.01 92.74 ± 0.15 90.28 ± 0.31


S+I 96.01 ± 0.83 96.89 ± 0.76 94.99 ± 0.23


S+M 95.49 ± 0.72 95.33 ± 0.54 95.02 ± 0.16


M+I 96.25 ± 0.05 97.23 ± 0.02 94.87 ± 0.05


S+M+I **97.02 ± 0.25** **98.45 ± 0.06** **96.94 ± 0.39**


The best results are highlighted in bold


_S_ SMILES sequence, _M_ molecular graph, _I_ interaction


As shown in Table 5, deleting any one of these three types of the features will damage performance. The performance is best when the three types of features are considered simultaneously. In addition, among single feature, considering only the interaction


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 14 of 18


**Table 6** Ablation experimental results on ChCh-Miner dataset


**Method** **AUROC** **AUPRC** **F1**


–GWU​ 95.89 ± 0.15 97.26 ± 0.18 94.97 ± 0.67


–Gating 96.28 ± 0.23 97.78 ± 0.31 95.28 ± 0.20


–Contrastive 96.07 ± 0.28 97.85 ± 0.15 94.38 ± 0.06


MFFGNN **97.02 ± 0.25** **98.45 ± 0.06** **96.94 ± 0.39**


The best results are highlighted in bold


information between drugs or the topological information of the molecular graph, the

model has the great performance. Among pairwise feature combinations, consider
ing the interaction information between drugs and the topological information of the

molecular graph simultaneously performs best, and pairwise feature combinations can
significantly improve performance than single feature. This suggests that multi-feature
integration can better represent drugs and improve prediction results.

Our model considers the global features for the molecular graph and the local features
for the atoms of the molecular graph. In order to study its effectiveness, we design a
variant, namely -GWU. -GWU ignores the global information in molecular graphs. As

shown in Table 6, deleting the global features will damage performance. To study the
validity of contrastive learning, we design a variant, called -Contrastive. This variant
removes the contrastive learning from the framework. As shown in Table 6, -Contrastive
is inferior to MFFGNN in all metrics. The results show that contrastive learning is beneficial to assist drug feature learning.
MFFGNN contains a GCN encoder with the gating mechanism to fully utilize the
neighborhood information of different order. In order to study its effectiveness, we conduct a comparative experiment based on whether there is gating or not, and the results
are shown in Table 6. The performance of the model without gating is lower than that of
the model with gating. It can be proved that GCN encoder with gating is beneficial to
predict DDI. From Fig. 5b, we can intuitively see the effectiveness of each component of
the proposed MFFGNN.


**Parameter analysis**

In this section, we analyze several key parameters in the model by performing experi
ments on the ZhangDDI dataset, including α in the objective function of our model, the
dimensionality of drug representation d g, sequence length L s, learning rate l r, the number of GCN layers L m and _k_ of the k-head attention in the MGFEM module. We study the
influence of different key parameters settings on MFFGNN by fixing other parameters.
In order to study the optimal setting of α in the objective function of our model, we
vary α from 0.1 to 1.0 and fix other parameters, the results are shown in Fig. 7a. We
observe that the three metrics are optimal when α is set to 0.9. On the whole, the non
zero nature of α proves the importance of contrastive learning in the model.
When training the BiGRU, we need to input a fix-sized matrix. However, the length
of SMILES sequences varies. Therefore, we fix the length of the input sequence at some
value and apply zero-padding and cutting operations. To study the optimal setting of
sequence length, we vary L s from 50 to 250 and fix other parameters, the results are


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 15 of 18


(a) Parameter study of _α_ (b) Parameter study of _L_ _s_ (c) Parameter study of _d_ _g_


(d) Parameter study of _l_ _r_ (e) Parameter study of _k_ (f) Parameter study of _L_ _m_

**Fig. 7** Parameter study on ZhangDDI dataset


shown in Fig. 7b. Because most of the SMILES sequences in the dataset are less than

150 and greater than 100, the model performance is optimal when L s = 150 . When

L s = 150, most of the sequences do not need to be cut, and little information is lost.

But, when L s = 100, most of the sequences will lose information, and the performance

is low. When the sequence length is greater than 150, even if zero-paddings are applied,

the performance degradation could be trivial, because it contains enough sequence

information.
In order to study the optimal setting of d g, we change it from 2 to 1024 and fix other
parameters, and the results are shown in Fig. 7c. When d g is set to 256, the three metrics
are optimal, and the model achieves the best performance. Specifically, with the increase
of the dimensionality of drug representation, MFFGNN can extract more useful infor
mation. However, a too high dimensionality may increase noise and lead to performance

degradation. Similarly, in order to study the optimal setting of l r, we change l r with
{0.01, 0.001, 0.0001, 0.00001} and fix other parameters, the results are shown in Fig. 7d.
When l r = 0.0001, the model performance is best.

In order to study the optimal setting of L m and _k_ of the k-head attention in the MGFEM
module, we change it from 1 to 4 and fix other parameters, the results are shown in
Fig. 7e, f. For _k_ of k-head attention, when k = 2, the model performance is the best. As
seen from the figure, as the L m increases, the MFFGNN performance improves. When
L m = 3, the three metrics are optimal and the model achieves the best performance.
However, too many layers may cause overfitting and lead to performance degradation.


**Discussions**

Drug-Drug prediction has always been a worthy research direction in pharmacology.

Most of the existing methods for predicting drug-drug interactions only consider sin
gle drug feature. However, single drug feature cannot comprehensively represent drug
information, which will ultimately affect the prediction results. Our proposed model
takes into account not only the topological structure information in molecular graphs

and the interaction information between drugs, but also the local chemical context in

SMILES sequences. Multiple drug features will represent the drug information more


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 16 of 18


comprehensively. We perform DDI predictions using each type of feature or combina
tion of features, and the experimenta results are shown in Table 5. When the three types

of features are considered simultaneously, the model has the best performance.

When extracting information from the molecular graph, we extract the local feature of the atoms and the global feature of the whole molecular graph. This facilitates
the remote propagation of the information in graph. We demonstrate the importance

of the global features of the molecular graphs in the ablation experiments, and the

results are given in Table 6. In addition, To verify evaluate that MFFGNN has good

generalization performance, we perform cross-dataset evaluations, and the results
are given in Fig. 6. As shown in figures, our model can predict drug-drug interaction
with stable accuracy, regardless of the scale of the dataset. However, our model also

has some limitations, for example, it does not extend to multi-type DDI prediction

tasks. In future work, we will further generalize the model to predict multi-type DDIs

events.


**Conclusions**

In this paper, we propose a novel end-to-end learning framework for DDI prediction,
namely MFFGNN, which can efficiently fuse the information from drug molecular
graphs, SMILES sequences and DDI graphs. The MFFGNN model utilizes the molecular graph feature extraction module to extract global and local features in molecu
lar graphs. Moreover, in the multi-type feature fusion module, we set up the gating

mechanism to control how much neighborhood information is passed to the node.
We perform extensive experiments on multiple real datasets. The results show that
the MFFGNN model consistently outperforms other state-of-the-art models.


**Abbreviations**

MFFGNN Multi-type Feature Fusion based on Graph Neural Network
DDIs Drug-Drug interactions
SMILES Simplified Molecular Input Line Entry System
GNN Graph neural network
GCN Graph convolution network
MLP Multi-layer perceptro
SSFEM SMILES sequence feature extraction module
MGFEM Molecular graph feature extraction module
MFFM Multi-type feature fusion module
GWU​ Graph warp unit
BiGRU​ Bi-directional gate recurrent unit


**Acknowledgments**
Not applicable.


**Author contributions**

CH, YL, HL and XQ conceived the experiments, CH and YL conducted the experiments, HL, HZ, YM, LL and XZ analysed
the results. All authors read and approved the final manuscript.


**Funding**
This work was supported by the Artificial Intelligence Program of Shanghai (2019-RGZN-01077), and the National Nature
Science Foundation of China (12001370).


**Availability of data and materials**
[The datasets generated and/or analysed during the current study are available in the Github repository, https://​github.​](https://github.com/kaola111/mff)
[com/​kaola​111/​mff](https://github.com/kaola111/mff)


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 17 of 18


**Declarations**


**Ethics approval and consent to participate**
Not applicable.


**Consent for publication**
Not applicable.


**Competing interests**
The authors declare that they have no competing interests.


Received: 20 December 2021  Accepted: 26 April 2022


**References**

1. Zhang W, Jing K, Huang F, Chen Y, Li B, Li J, Gong J. Sf **l** n: a sparse feature learning ensemble method with linear
neighborhood regularization for predicting drug-drug interactions. Inf Sci. 2019;497:189–201.
2. Yan C, Duan G, Zhang Y, Wu FX, Pan Y, Wang J. Predicting drug-drug interactions based on integrated similarity and
semi-supervised learning. IEEE/ACM Trans Comput Biol Bioinform. 2020;2:1147.
3. Zhang Y, Qiu Y, Cui Y, Liu S, Zhang W. Predicting drug-drug interactions using multi-modal deep auto-encoders
based network embedding and positive-unlabeled learning. Methods. 2020;179:37–46.
4. Zhu J, Liu Y, Zhang Y, Li D. Attribute supervised probabilistic dependent matrix tri-factorization model for the prediction of adverse drug-drug interaction. IEEE J Biomed Health Inf. 2020;25(7):2820–32.
5. Qiu Y, Zhang Y, Deng Y, Liu S, Zhang W. A comprehensive review of computational methods for drug-drug interaction detection. IEEE/ACM Trans Comput Biol Bioinform. 2021;3:7487.
6. Deng Y, Qiu Y, Xu X, Liu S, Zhang Z, Zhu S, Zhang W. Meta-ddie: predicting drug-drug interaction events with fewshot learning. Brief Bioinform. 2022;23(1):514.
7. Deng Y, Xu X, Qiu Y, Xia J, Zhang W, Liu S. A multimodal deep learning framework for predicting drug-drug interaction events. Bioinformatics. 2020;36(15):4316–22.
8. Zhang W, Chen Y, Li D, Yue X. Manifold regularized matrix factorization for drug-drug interaction prediction. J
Biomed Inform. 2018;88:90–7.
9. Huang K, Xiao C, Hoang T, Glass L, Sun J. Caster: Predicting drug interactions with chemical substructure representation. In: Proceedings of the AAAI Conference on Artificial Intelligence. 2020;34:702–9.
10. Li P, Wang J, Qiao Y, Chen H, Yu Y, Yao X, Gao P, Xie G, Song S. An effective self-supervised framework for learning
expressive molecular global representations to drug discovery. Brief Bioinform. 2021;22(6):109.
11. Wang F, Lei X, Liao B, Wu F-X. Predicting drug-drug interactions by graph convolutional network with multi-kernel.
Brief Bioinform. 2022;23(1):511.
12. Feeney A et al. Relation matters in sampling: A scalable multi-relational graph neural network for drug-drug interac[tion prediction. arXiv preprint arXiv:​2105.​13975 2021.](http://arxiv.org/abs/2105.13975)
13. Purkayastha S, Mondal I, Sarkar S, Goyal P, Pillai JK. Drug-drug interactions prediction based on drug embedding
and graph auto-encoder. In: 2019 IEEE 19th International Conference on Bioinformatics and Bioengineering (BIBE),
2019;547–552 . IEEE
14. Feng Y-H, Zhang S-W, Shi J-Y. Dpddi: a deep predictor for drug-drug interactions. BMC Bioinform. 2020;21(1):1–15.
15. Dai Y, Guo C, Guo W, Eickhoff C. Drug-drug interaction prediction with wasserstein adversarial autoencoder-based
knowledge graph embeddings. Brief Bioinform. 2021;22(4):256.
16. Lyu T, Gao J, Tian L, Li Z, Zhang P, Zhang J. Mdnn: a multimodal deep neural network for predicting drug-drug interaction events. Science. 2019;5:1147.
17. Yu Y, Huang K, Zhang C, Glass LM, Sun J, Xiao C. Sumgnn: multi-typed drug interaction prediction via efficient
knowledge graph summarization. Bioinformatics. 2021;37(18):2988–95.
18. Nyamabo AK, Yu H, Shi J-Y. Ssi-ddi: substructure-substructure interactions for drug-drug interaction prediction. Brief
Bioinform. 2021;22(6):133.
19. Nyamabo AK, Yu H, Liu Z, Shi J-Y. Drug-drug interaction prediction with learnable size-adaptive molecular substructures. Brief Bioinform. 2022;23(1):441.
20. Chen Y, Ma T, Yang X, Wang J, Song B, Zeng X. Muffin: multi-scale feature fusion for drug-drug interaction prediction.
Bioinformatics. 2021;7:1148.
[21. Bahdanau D et al. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:​1409.​](http://arxiv.org/abs/1409.0473)

[0473 2014.](http://arxiv.org/abs/1409.0473)

22. Battaglia PW, Pascanu R, Lai M, Rezende D, Kavukcuoglu K. Interaction networks for learning about objects, relations
and physics. Science. 2016;2:7740.
23. Ishiguro K, Maeda Si, Koyama M. Graph warp module: an auxiliary module for boosting the power of graph neural
[networks in molecular graph analysis. arXiv preprint arXiv:​1902.​01020 2019.](http://arxiv.org/abs/1902.01020)
24. Duke JD, et al. Literature based drug interaction prediction with clinical assessment using electronic medical
records: novel myopathy associated drug interactions 2012.
25. Takeda T, Hao M, Cheng T, Bryant SH, Wang Y. Predicting drug-drug interactions through drug structural similarities
and interaction networks incorporating pharmacokinetics and pharmacodynamics knowledge. J Cheminform.
2017;9(1):1–9.
26. Vilar S, Uriarte E, Santana L, Lorberbaum T, Hripcsak G, Friedman C, Tatonetti NP. Similarity-based modeling in largescale prediction of drug-drug interactions. Nat Protoc. 2014;9(9):2147–63.


He _et al. BMC Bioinformatics     (2022) 23:224_ Page 18 of 18


27. Fokoue A, Sadoghi M, Hassanzadeh O, Zhang P. Predicting drug-drug interactions through large-scale similaritybased link prediction. In: European Semantic Web Conference, 2016;774–789 . Springer
28. Ma T, Xiao C, Zhou J, Wang F. Drug similarity integration through attentive multi-view graph auto-encoders. arXiv
[preprint arXiv:​1804.​10850 2018.](http://arxiv.org/abs/1804.10850)
29. Kastrin A, Ferk P, Leskošek B. Predicting potential drug-drug interactions on topological and semantic similarity
features using statistical learning. PLoS ONE. 2018;13(5):0196865.
30. Ryu JY, et al. Deep learning improves prediction of drug-drug and drug-food interactions. Proc Natl Acad Sci.
2018;115(18):4304–11.
31. Xu N et al. Mr-gnn: Multi-resolution and dual graph neural network for predicting structured entity interactions.
[arXiv preprint arXiv:​1905.​09558 2019.](http://arxiv.org/abs/1905.09558)
32. Ma T et al. Genn: predicting correlated drug-drug interactions with graph energy neural networks. arXiv preprint

[arXiv:​1910.​02107 2019.](http://arxiv.org/abs/1910.02107)
33. Shang J, Xiao C, Ma T, Li H, Sun J. Gamenet: Graph augmented memory networks for recommending medication
combination 2018.

34. Hamilton, W.L., Ying, R., Leskovec, J.: Inductive representation learning on large graphs. In: Proceedings of the 31st
International Conference on Neural Information Processing Systems, 2017;1025–1035
[35. Veličković P et al. Graph attention networks. arXiv preprint arXiv:​1710.​10903 2017.](http://arxiv.org/abs/1710.10903)
36. Schlichtkrull M, Kipf TN, Bloem P, Van Den Berg R, Titov I, Welling M. Modeling relational data with graph convolutional networks. In: European Semantic Web Conference, 2018;593–607 . Springer
[37. Rahimi A et al. Semi-supervised user geolocation via graph convolutional networks. arXiv preprint arXiv:​1804.​08049](http://arxiv.org/abs/1804.08049)
2018.
38. Zitnik M, et al. Modeling polypharmacy side effects with graph convolutional networks. Bioinformatics.
2018;34(13):457–66.
[39. Duvenaud D et al. Convolutional networks on graphs for learning molecular fingerprints. arXiv preprint arXiv:​1509.​](http://arxiv.org/abs/1509.09292)

[09292 2015.](http://arxiv.org/abs/1509.09292)

40. Lin, X., et al.: Kgnn: Knowledge graph neural network for drug-drug interaction prediction. In: IJCAI,
2020;380:2739–2745.
[41. Bai Y et al. Bi-level graph neural networks for drug-drug interaction prediction. arXiv preprint arXiv:​2006.​14002 2020.](http://arxiv.org/abs/2006.14002)
42. Wang Y et al. Multi-view graph contrastive representation learning for drug-drug interaction prediction. In: Proceedings of the Web Conference 2021, 2021;2921–2933.
43. Landrum G. RDKit: a software suite for cheminformatics, computational chemistry, and predictive modeling. London: Academic Press; 2013.
44. Chung J, Gulcehre C, Cho K, Bengio Y. Empirical evaluation of gated recurrent neural networks on sequence mod[eling. arXiv preprint arXiv:​1412.​3555 2014.](http://arxiv.org/abs/1412.3555)
45. Quan Z et al. A system for learning atoms based on long short-term memory recurrent neural networks. In: 2018
IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2018;728–733. IEEE
46. Zhang W, et al. Predicting potential drug-drug interactions by integrating chemical, biological, phenotypic and
network data. BMC Bioinformatics. 2017;18(1):1–12.
[47. Marinka Zitnik SM, Rok Sosič, Leskovec J. BioSNAP Datasets: Stanford Biomedical Network Dataset Collection. http://​](http://snap.stanford.edu/biodata)

[snap.​stanf​ord.​edu/​bioda​ta 2018](http://snap.stanford.edu/biodata)
[48. Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:​1609.​](http://arxiv.org/abs/1609.02907)

[02907 2016](http://arxiv.org/abs/1609.02907)
49. Li J et al. Semi-supervised graph classification: A hierarchical graph perspective. In: The World Wide Web Conference,
2019;972–982
50. Kim S, Chen J, Cheng T, Gindulyte A, He J, He S, Li Q, Shoemaker BA, Thiessen PA, Yu B, et al. Pubchem 2019 update:
improved access to chemical data. Nucleic Acids Res. 2019;47(D1):1102–9.
51. Chen X, Liu X, Wu J. Drug-drug interaction prediction with graph representation learning. In: 2019 IEEE International
Conference on Bioinformatics and Biomedicine (BIBM), 2019;354–361. IEEE
[52. Kingma DP, Ba J. Adam: A method for stochastic optimization. arXiv preprint arXiv:​1412.​6980 2014](http://arxiv.org/abs/1412.6980)
53. Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks. In: Proceedings of
the Thirteenth International Conference on Artificial Intelligence and Statistics, 2010;249–256. JMLR Workshop and
Conference Proceedings
54. Srivastava N, et al. Dropout: a simple way to prevent neural networks from overfitting. J Mach Learn Res.
2014;15(1):1929–58.


