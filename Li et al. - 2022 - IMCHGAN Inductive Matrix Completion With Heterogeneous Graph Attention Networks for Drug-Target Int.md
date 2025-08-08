IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 2, MARCH/APRIL 2022 655

# IMCHGAN: Inductive Matrix Completion With Heterogeneous Graph Attention Networks for Drug-Target Interactions Prediction


Jin Li, Jingru Wang, Hao Lv, Zhuoxuan Zhang, and Zaixia Wang


Abstract—Identification of targets among known drugs plays an important role in drug repurposing and discovery. Computational
approaches for prediction of drug–target interactions (DTIs) are highly desired in comparison to traditional biological experiments as its
fast and low price. Moreover, recent advances of systems biology approaches have generated large-scale heterogeneous, biological
information networks data, which offer opportunities for machine learning-based identification of DTIs. We present a novel Inductive
Matrix Completion with Heterogeneous Graph Attention Network approach (IMCHGAN) for predicting DTIs. IMCHGAN first adopts a
two-level neural attention mechanism approach to learn drug and target latent feature representations from the DTI heterogeneous
network respectively. Then, the learned latent features are fed into the Inductive Matrix Completion (IMC) prediction score model which
computes the best projection from drug space onto target space and output DTI score via the inner product of projected drug and target
feature representations. IMCHGAN is an end-to-end neural network learning framework where the parameters of both the prediction
score model and the feature representation learning model are simultaneously optimized via backpropagation under supervising of the
observed known drug-target interactions data. We compare IMCHGAN with other state-of-the-art baselines on two real DTI
experimental datasets. The results show that our method is superior to existing methods in term of AUC and AUPR. Moreover,
[IMCHGAN also shows it has strong predictive power for novel (unknown) DTIs. All datasets and code can be obtained from https://](https://github.com/ljatynu/IMCHGAN/)
[github.com/ljatynu/IMCHGAN/](https://github.com/ljatynu/IMCHGAN/)


Index Terms—Drug-target interactions, graph attention network, heterogeneous network, end-to-end learning


Ç


_



1 I NTRODUCTION
# I DENTIFYING down the search scope of drug candidates for downstream drug-target interactions will greatly narrow

drug discovery experimental validation and thus significantly reduce the high cost and the long period of developing
a new drug [1], [2]. As identification of the drug-target interactions using biological experiments is time-consuming and
expensive, computational methods are highly desired to be
used to determine the potential interactions between drugs
and targets (hereafter abbreviate DTIs) [3].

The ligand-similarity based, docking simulation, networkbased methods, and machine learning based approaches are
currently the four main categories of computational methods
for predicting DTIs [3]. Ligand-similarity based methods,
such as [4], predict interactions by comparing a new ligand to
known proteins ligands. However, ligand-based methods
perform poorly when the number of known ligands is insufficient. As for docking simulation methods [5], the threedimensional (3D) structures of proteins are required for
simulation hence becoming inapplicable when there are
numerous proteins with unavailable 3D structures. Networkbased methods, such as [6], [7], [8] use the network and graph


� The authors are with the School of Software, Yunnan University, Kunming
650091, China. E-mail: [lijin@ynu.edu.cn,](mailto:lijin@ynu.edu.cn) [15073239620@163.com,](mailto:15073239620@163.com)
[{1139798570, 1323099669, 623771147}@qq.com.](mailto:1139798570@qq.com)


Manuscript received 31 Aug. 2020; revised 18 May 2021; accepted 8 June
2021. Date of publication 11 June 2021; date of current version 1 Apr. 2022.
(Corresponding author: Jin Li.)
Digital Object Identifier no. 10.1109/TCBB.2021.3088614


_



theory to infer the potential drug-target interaction. However,
the prediction qualities of network-based approaches are
strongly limited by available linked information. Thus, these
methods perform not very well on association predictions for
new drugs or targets with rare linked information. Additionally, some useful information, such as drugs and targets feature information, cannot be fully utilized to improve
prediction accuracy for these methods.

In the past few years, machine learning based approaches [9], [10], [11], [12] have gained the most attention for
their high-quality prediction results for DTIs. Most of these
methods are DTI interactive data-driven methods which
generally yield the latent feature from the known DTI interactive data, and then adopt various machine learning techniques to offer prediction of DTI. In addition to known DTI
data, there are other types of biological objects frequently
involved in DTI prediction, such as protein, disease, gene,
and side effect. DTI interactive data and other heterogeneous data are usually available with the form of biomedical information networks where drugs, targets and other
biomedical objects connected with each other. The link relationship and topological structure contained in these biomedical information networks provide rich interactive
semantics between different biological objects, which can be
used by network-based machine learning algorithms to further improve the accuracy of DTI prediction.

For example, DTINet [11] automatically learn lowdimensional feature representations of drugs and targets
from heterogeneous network (HN) data, and then applies


_



1545-5963 © 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.

See ht_tps://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


656 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 2, MARCH/APRIL 2022



inductive matrix completion [13] to predict DTIs based
on the learnt features. With the same heterogeneous data
provided by DTINet, NeoDTI [14], that integrates diverse
information from heterogeneous network data and automatically learns topology-preserving representations of
drugs and targets to make DTI prediction. Recently,
AOPEDF [15] learns a low-dimensional vector representation of features that preserve arbitrary-order proximity
from a highly integrated, drug–target–disease heterogeneous biological network, and adopts deep forest as a classifier to make a DTI prediction. GCN-DTI [16] first built a
DPP (drug–protein pairs) network based on multiple drugs
and proteins in which DPPs are the nodes and the associations between DPPs are the edges of the network. GCN-DTI
then uses a graph convolutional network to learn the features for each DPP. With the learned feature representation
as an input, it uses a deep neural network to identify DTIs.

Despite the effectiveness of the above-mentioned methods for DTIs predictions, there are still some limitations to
current research results. First, the feature representations of
drugs and targets are critical issues for machine learning
based-prediction approaches determining the quality of the
results of DTI prediction. However, some existing methods,
such as DTINet [11] and AOPEDF [15], use unsupervised manner to learn low-dimensional feature representations of
drugs and targets from the heterogeneous network (HN). In
these methods, feature learning was separated from the prediction task which may not yield the optimal solution, as the
features learned from the unsupervised learning procedure
may not be the most suitable representations of drugs or targets for the final DTI prediction task. Second, there are other
existing methods, such as NeoDTI [14] and GCN-DTI [16],
adopted graph neural network-based representation learning to obtain latent features of drugs and targets, and established end-to-end pipelines which can simultaneously
optimize the feature extraction process and the DTI prediction model. However, these categories of methods either
ignore the meta-path information of HN or fail to provide good
interpretability for DTI heterogeneous network analysis.

To overcome the mentioned limitations of current
approaches for DTI prediction, in this study, we proposed a
novel Inductive Matrix Completion with Heterogeneous
Graph Attention Network approach (IMCHGAN) to
address the problem of DTI prediction. The basic idea of
our proposed method is as follows. First, IMCHGAN
adopts a two-level representation learning approach to
learn drug and target latent feature representations from
the heterogeneous network respectively. On the bottom
level, Graph Attention Network (GAT) is leveraged to learn
drugs (or targets) latent feature representations for a specific
meta-path. On the top level, an attention-based learning
approach is further employed to integrate different metapath latent representations into the final latent feature. As
reported in previous studies [13], [17], [18], due to the full
use of the auxiliary information of entities, Inductive Matrix
Completion (IMC) has achieved good performance in biological entity association prediction. In addition, a crucial
advantage of IMC is that it is inductive. It can be applied to
entities not seen at training time, unlike traditional matrixcompletion approaches and network-based inference methods that are transductive. Therefore, in our study, IMC was



adopted as a scoring model for drug-target interaction. Specifically, the learned latent features are fed into the Inductive Matrix Completion (IMC) prediction score model
which computes the best projection from drug space onto
target space, such that the projected feature vectors of drugs
are geometrically as close to the corresponding feature vectors of their known interacting targets as possible. It is
worth mentioning that in IMCHGAN the parameters of
GAT layers, attention-based integration layer, and IMCbased prediction model are collectively optimized via backpropagation in an end-to-end learning way under supervising of the observed known drug-target interactions.
Extensive experiments are carried on two real DTI data to
evaluate the performance of IMCHGAN. The experimental
results show that IMCHGAN can outperform several existing baseline prediction methods. Moreover, we also show
the two-level attention-based representation learning
approach allows IMCHGAN to focus on task-relevant parts
of network and meta-paths which also bring good interpretability for DTI heterogeneous network analysis.


2 M ATERIALS AND M ETHODS


In this section, we first introduce the heterogeneous information network of drug-target interactions (DTI-HN) which
includes different types of biological information entities
and different types of relationships between them. Some
concepts about DTI-HN are also presented. Second, a twolevel feature representation learning method is proposed to
learn feature representation from DTI-HN. Finally, a novel
end-to-end learning framework IMCHGAN is presented to
solve the problem of DTIs prediction.


2.1 The Drug-Target Biological Heterogeneous
Network

A biological heterogeneous network DTI-HN can be constructed based on several public databases for DTIs prediction which includes the following four types of biological
entities and eight types of relationships (interactions, associations, or similarities). Specifically, four types of entities are
protein, disease, gene, and side effects. Each entity is represented as a node in the heterogeneous network. Eight types
of relationships are drug-drug interactions, drug-drug
structure similarity, drug-target interactions, drug-disease
associations, drug-side effect associations, target-target
interactions, target-target sequence similarity, target-disease
associations. Each relationship is represented as an edge in
the heterogeneous network. Fig. 1 illustrates an example of
the DTI heterogeneous network.

The concept of meta-path and meta-path-based neighbors are introduced which are closely related to the methods of feature representation learning from DTI-HN.

In the DTI-HN, two nodes can be connected via different
semantic paths, which are called meta-paths [19]. A metapath, denoted by r, is a path which describes a composite
relation between two nodes. We use the form of r ¼
T 1 [sem1] T 2 [sem2] . . . [sem][l] T l (sem is the abbreviation of semantic
relationship) to denote a meta path, where T i denotes a
type of nodes (e.g., disease, drug, or target). Different metapaths capture different semantic relationship between various types of nodes. For instance, as shown in Fig. 1, the



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


LI ET AL.: IMCHGAN: INDUCTIVE MATRIX COMPLETION WITH HETEROGENEOUS GRAPH ATTENTION NETWORKS FOR DRUG-TARGET... 657



Fig. 1. An Illustration of a DTIs heterogeneous information network.


meta-path Dr [inter] Dr represents a direct interaction relationship between two drugs, such as dr 2 [inter] dr 4 . The meta-path
Dr [assoc] Di [assoc] Dr presents a relationship that two drugs are
connected to a common disease, such as dr 3 [assoc] di 1 [assoc] dr 4 .
It’s obvious that link semantics underneath these meta
paths are quite different. Thus, meta-path is a powerful
approach to describe semantic relationships among specific
types of nodes in the DTI-HN.

We then introduce meta-path-based neighbors which
contains a set of neighbors connected by a specific semantic
meta-path in the DTI-HN. Given a node i and a meta-path r
in the DTI-HN, the meta-path-based neighbors of node i is
denoted by N [r] ðiÞ and defined as the set of nodes which connect with node i via meta-path r and i itself.

For example, given the target sequence similarity metapath r 1 ¼ Ta [se][q][sim] Ta, N [r] [1] ðta 3 Þ ¼ taf 3 ; ta 2 ; ta 4 g. For the
meta-path r 2 ¼ Ta [inter] Dr [inter] Ta, N [r] [2] ðta 3 Þ ¼ taf 3 ; ta 1 g since
ta 3 and ta 1 both connect to dr 4 . So, for the same node different meta-path-based neighbors can be obtained through different meta-path. In addition, we note that ta 3 is not directly
connected with ta 1 in the DTI-HN, however, ta 1 is a metapath-based neighbor of ta 3 via the meta-path r 2 . Therefore,
it should be emphasized that meta-path-based neighbors
provide higher-order connections between nodes from different semantic perspectives.


2.2 Learning Feature Representations With
Attention Mechanism

In recent years, certain studies [20], [21] have attempted to
automatically learn network topology-preserving node-level
latent feature representations (embedding) from networks.
Particularly, graph neural networks, such as the graph convolutional network (GCN), the graph attention network
(GAT), and its variants [17], [22], [23], [24], [25] have significantly improved many networks related prediction tasks,
such as predicting the biological activities of small molecules
and recommendation. Particularly GAT has successfully
achieved or surpassed the performance of other state-of-theart methods such as GCN by leveraging self-attention mechanism to aggregate feature information of neighboring nodes
to generate feature representations of nodes.

As illustrated by the Fig. 1, a DTI-HN with multi-types of
biological entities and various relationships between these
entities contains comprehensive information and rich



semantics. Thus, a DTI-HN can be considered as an important data source for generating drug and target feature representations. In this paper, we propose a novel two-level
approach to learn drug and target embeddings from the
DTI-HN. On the bottom level, GAT is leveraged to learn all
drugs embeddings X [r] [i] given a specific meta-path r i . Different meta-path-based embeddings can be obtained for different meta-paths. Since different meta-paths represent
different connection semantic, higher quality embeddings
can be obtained by integrating different meta-path-based
embeddings. Therefore, on the top level, an attention-based
approach is further employed to integrate different metapath embeddings fX [r] [1] ; X [r] [2] ; . . . ; X [r] [s] g into final output
embeddings X. The final embeddings Y for targets are
obtained in the same way. Details of the learning approach
are discussed in the follows.


2.2.1 Obtaining Node Embeddings via Graph Attention
Layers

Like other graph neural network models, such as GCN,
GAT employs weighty neighborhood aggregation to generate an embedding for a node. We will start by describing a
single graph attention layer which leverages self-attention

[26] to learn the aggregation weight for different neighbors.
The flowchart of GAT for learning drug (target) embeddings is illustrated by Fig. 2.

Specifically, given a meta-path r, N [r] ðiÞ is the meta-pathbased neighbors for a drug (or target) node i in the DTI-HN.
j 2 N [r] ðiÞ is a meta-path neighbor of i. The l-layer embedding of i under the meta-path r is denoted by x [r] i [2 R] [d] [l] [ and]

d l is the vector dimension of embedding at l-layer. Let w [r] i;j

be the influence weight of j to i which means how important
node j will be for node i. w [r] i;j [depends on the embeddings of]

both i and j which is




[r] i;j [¼][ s][ð][g] [>] r




[r] j [�Þ][;] (1)



w [r]




[>] r [�½][W] [r] [x] i [r]




[r] i [k][W] [r] [x] [r]



where s denotes the activation function (such as LeakyRelu), k denotes the vector concatenate operation, g [>] [2 R] [2][d] [l][þ][1]



elu), k denotes the vector concatenate operation, g [>] r [2 R] [2][d] [l][þ][1]

is the influence weight vector with the length of 2d lþ1 for
meta-path r and W [r] 2 R [d] [l][þ][1] [�][d] [l] is a shared linear transformation weight matrix. After obtaining the influence weight
for all meta-path-based neighbors, we normalize them to
get the attention coefficient a [r] [by a Softmax function]




[r] i;j [by a Softmax function]



a [r]




[r] i;j [¼][ softmax] [j][2][N] [r] [ð][i][Þ] [w] [r]



� [w] [r] i;j �



exp w� [r] i;j �



(2)



¼



k2N [r] ðiÞ [exp][ w] [r]



~~P~~



:

[ w] ~~�~~ [r] i;k ~~�~~



Then, the meta-path-based embedding of node i at next
(l+1)-layer can be weighty aggregated by the neighbor’s
embeddings at l-layer with the corresponding attention
coefficients as follows:



1: (3)

A



x [r] i [¼][ s]



0 X �a [r] i;j [�] [W] [r] [x] [r] j �



@



X



a [r]




[r] i;j [�] [W] [r] [x] [r] j



j2N [r] ðiÞ



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


658 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 2, MARCH/APRIL 2022


Fig. 2. The flowchart of GAT for learning drug (target) embeddings.



Like the idea of the feature extraction with multiple convolutional kernels in Convolutional Neural Network, multihead attention is employed to make the learning process of
self-attention more robust[24]. Specifically, K attention
mechanisms are independently used to implement the feature transformation described by the Eq. (3), and then the
transformed features are concatenated, resulting in the following output feature representation which is a vector with
the length of Kd lþ1



importance of different meta-paths and integrating them for
the prediction task. Specifically, the meta-path-level attention score b [r] [i] for the meta-path r i is calculated by



s [r] [i] ¼ [1]


n



X

j2V d



q [>] s Wx� [r] j [i] [þ] **[b]** �; (6)



where W is the weight matrix, b is the bias vector, q is the
semantic level attention vector, and V d is the set of drugs
(The same equation is suitable for targets). Note that for the
meaningful comparation, all above parameters are shared
for all meta-paths and semantic-specific embedding.

The final meta-path-level attention values are obtained
by normalizing the above attentive scores with the Softmax
function given in Eq. (6), which can be interpreted as the
contributions of different meta-path embeddings to the
aggregated embedding

a [r] [i] ¼ ~~P~~ ~~s~~ j ex ¼1 p [exp] sð [r] [ s] ð [i] Þ [r] [j] Þ [:] (7)


With the learned attention values as coefficients, we can
integrate these meta-path-specific embeddings to obtain the
final embedding X as follows:



x [r] i [¼ k] k¼1�K [s]



0 X �a [r] i;j [�] [W] [r] [x] [r] j �



@



1: (4)

A



X



a [r]




[r] i;j [�] [W] [r] [x] [r] j



j2N [r][ð][i][Þ]



As we can see from the Eqs. (1), (2), and (3), w [r]



As we can see from the Eqs. (1), (2), and (3), w [r] i;j [depends on]

the x [r] i [,][ x] [r] j [and][ g] [>] r [which both are learnable target-task rele-]

vant-parameters. This allows the model to dynamically assign
larger aggregation weights to neighboring nodes with higher
relevance to the DTIs prediction task. In turn, it also enables
the embeddings of the nodes to be aggregated according to
the dynamic weight. These characteristics make our approach
have strong capability of representation learning.

All drug embeddings at l-layer under the meta-path r is
denoted by X [r] ðlÞ 2 R [n][�][Kd] [l] . We can obtain X [r] ðl þ 1Þ 2
R [n][�][Kd] [l][þ][1] via a graph attention transformation denoted by as
follows:


X [r] ðl þ 1Þ ¼ GATð X [r] ðlÞÞ: (5)


The whole L-layers GAT architecture is stacked with several
graph attention layers.




[r] j [and][ g] [>] r




[r] i [,][ x] [r]



X ¼



s
X

i¼1



a [r] [i] X [r] [i] : (8)



2.2.2 Attention-Based Integration of Embeddings
Given a meta-path r i, an embedding matrix X [r] [i] for all nodes
in DTI-HN can be obtained by the graph attention network
transformations. Thus, we can obtain s groups of node
embeddings fX [r] [1] ; X [r] [2] ; . . . ; X [r] [s] g for a meta-path set
fr 1 ; r 2 ; . . . ; r s g. Different meta-path semantics bring different meta-path embeddings. To obtain a more comprehensive node embedding, we need to integrate multiple metapath embeddings. To address this problem, a novel metapath-level attention is proposed to automatically learn the



Totally, given a group of meta-paths fr 1 ; r 2 ; . . . ; r s g for
drugs in DTI-HN, for each meta-path r i, starting from the
randomly initialized embedding X [r] [i] ð0Þ, our learning
approach transforms the embeddings in a layer by layer
manner and finally outputs the attention-based aggregated
embeddings X. The learned embeddings X will be used as
the input for the downstream IMC-based rating model to
make final association predictions. Note that the same learning process is used to generate targets embedding.



2.3 The End-to-End Learning Framework of
IMCHGAN

In this paper, the DTIs prediction is formulated as a unified
end-to-end neural network learning framework IMCHGAN



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


LI ET AL.: IMCHGAN: INDUCTIVE MATRIX COMPLETION WITH HETEROGENEOUS GRAPH ATTENTION NETWORKS FOR DRUG-TARGET... 659


Fig. 3. The framework of IMCHGAN.



as shown in Fig. 3. The total framework of IMCHGAN
includes three following modules. (1) The preprocessing
module uses different sources of information to construct
the drug-target interactions heterogenous information network DTI-HN. (2) In the learning module, graph attention
network (GAT, described in Section 2.2.1) is first leveraged
to learn drug and target embeddings under a specific metapath over DTI-HN. Then, the learned embeddings of different meta-paths are integrated by meta-path level attention
mechanism (ATT, described in Section 2.2.2) and are input
the inductive matrix completion model (IMC). The learning
module learns the parameters of GAT layers, attentionbased integration layer and IMC via backpropagation in an
end-to-end learning way under supervising of the observed
known drug-target associations. (3) The prediction module
makes DTIs prediction through an inductive matrix completion based on the well-trained model. In the follows, we
introduce the details of the IMCHGAN.

An adjacent matrix T 2 f0; 1g [m][�][n] with 0-1 entries was
constructed to represent the partially observed drug-target
associations, where Tði; jÞ ¼ 1 if a drug i 2 Dr is known to
be associated with a target j 2 Ta. For instance, in Fig. 1,
T tað 3 ; dr 3 Þ ¼ 1. Tði; jÞ ¼ 0 if the association between a drug
i and a target j is unknown or unobserved. For instance,
T tað 3 ; dr 5 Þ ¼ 0 in Fig. 1. Thus, DTIs prediction can be formulated as a matrix completion which is the task of filling in
the missing entries of the partially observed matrix T. More
specifically, without loss of generality, V and V [�] were used
to denote the set of the known and unobserved or unknown
drug-target entries from T respectively. The observation V
consisted only of the known associations, i.e., if 8ði; jÞ 2
V; Tði; jÞ ¼ 1. V [�] is the set of unknown or unobserved entries
if 8ði; jÞ 2 V [�] ; Tði; jÞ ¼ 0. A sample of observed entries V
from a true underlying matrix Q was considered. The objective was to estimate missing entries under some additional
assumptions on the structure of the interaction matrix T.
The most common assumption is that Q is low-rank, i.e.,
Q ¼ FG [>], where F 2 R [m][�][k] and G 2 R [n][�][k] are of rank k �
m; n. With these notations, the basic DTIs prediction can be
formulated as the following matrix completion problems:



where P V ð�Þ is the projection of the matrix onto the set V.

One limitation of the matrix completion for DTIs prediction formulated by Eq. (9) is that it cannot directly leverage
side information, such as the feature representation of drug
and target learned by attention-based model, to predict.
Inductive matrix completion (IMC) [13] was proposed to
circumvent this limitation. When DTIs prediction is solved
using IMC, an association predicted rating is modeled as an
inner product of the features of a drug and a target projected onto a latent space. IMC assumes that the association
matrix is generated by applying feature vectors associated
with its row as well as column entities to a projection matrix
Z. The goal was to recover Z using observations from T.
Furthermore, to learn parameters effectively from a small
number of observed ratings, the latent space is constrained
to be low-dimensional, which implies that the parameter
matrix is constrained to be low-rank.

Specifically, let X 2 R [m][�][f] [d] be the feature matrix learned
from attention-based model (i.e., Eq. (9)). Similarly, Y 2
R [n][�][f] [t] is the feature matrix for targets. The IMC tries to
recover a feature projection matrix Z 2 R [f] [d] [�][f] [t] using the
observed entries from the known drug-target association
matrix T and the feature matrix of X and Y. To learn the
parameters effectively from a small number of observed ratings, the latent space was constrained to be low-dimensional, which implies that the feature projection matrix Z is
constrained to be low-rank, i.e., Z ¼ Z 1 Z [T] 2 [, where][ Z] [1] [ 2][ R] [f] [d] [�][k]

and Z 2 2 R [f] [t] [�][k] are of rank k � f d ; f t . Additionally, to avoid
to yield degenerate results, a bias item a 2 ð0; 1Þ that appropriately weighs observed and unobserved entries is introduced in the inductive matrix completion formulation. Let
C d ¼ �q d ; W d ; b d ; W [r] d [;][ g] [r] d � and C t ¼ �q t ; W t ; b t ; W [r] t [;][ g] [r] t � be

the parameters involved in graph attention layers and attention-based integration model for drug and target, respectively. With these notions, the IMCHGAN is formulated as
the following optimization problem:



min

F;G



1 �� � ��� 2
2 [P] [V] [ T][ �] [FG] [T]



F [þ][�] [k][F][k] [2]




[k][F][k] [2] F [þ k][G][k] [2] F

� �




[2] F [þ k][G][k] [2]



; (9)



�q d ; W d ; b d ; W [r] d [;][ g] [r] d �



�q t ; W t ; b t ; W [r] t [;][ g] [r] t �




[r] d [;][ g] [r] d



and C t ¼ �q t ; W t ; b t ; W [r] t




[r] t [;][ g] [r] t



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


660 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 2, MARCH/APRIL 2022



TABLE 1
Statistics of DTI-HN


Nodes Types # Edge Types #


Drug 708 drug-drug interactions 10036
Target 1512 drug-drug structure similarity 489261
Disease 5603 drug-target interactions 1923
Side effect 4192 drug-disease associations 199214
– – drug-side effect associations 80164
– – target-target interactions 7363
– – target-target sequence similarity 2286144
– – target-disease associations 1596745
Total 12015 Total 4670850



T� � XZ 1 Z [>] 2 [Y] [>] �



k [2] F



min
Z 1 ;Z 2 ;C d ;C t



ð1 � aÞ



kP V T� � XZ 1 Z [>] 2
2



þ [a]



2
�� [P] [V][�] [ T] � [ �] [XZ] [1] [Z] 2 [>] [Y] [>] ��� F



F



�� � 2

2 [P] [V][�] [ T][ �] [XZ] [1] [Z] [>]



(10)



TABLE 2
Statistics of GSD


Dataset Drug # Target # Interactions #


Es 445 664 2926

ICs 210 204 1476

GPCRs 223 95 635

NRs 54 26 90


(https://pytorch.org). Graph convolutional network
encoders are implemented based on the open source deep
leaning on graph library (https://www.dgl.ai/). All experiments are carried on Windows 10 operation system with a
Dell Precision T5820 workstation computer of an intel W2145 8 cores, 3.7GHz CPU and 64G memory.


3.2 Optimization of Parameters in IMCHGAN
The following parameters will affect the performance of
IMCHGAN: (1) the biased item a in the loss function of
inductive matrix completion formulation defined by
Eq. (10), (2) the layer number of graph attention networks l,
(3) the dimension of drug or target embedding d, and (4) the
head number of multi-head attention mechanism K defined
in Eq. (4). We carry on experiments on DTI-HN in terms of
AUC and AUPR to evaluate the effect of these parameters
on the performance of IMCHGAN.

First, the biased item a in the Eq. (10) is introduced to
appropriately weigh observed and unobserved entries. The
loss function is optimized only using positive samples if
a = 0 and only using unobserved samples if a = 1. Fig. 4a
shows the effects of different a on the prediction performance of IMCHGAN. The performances when a = 0.6 which
appropriately weighs observed and unobserved entries are
superior to the performance when a is set to be other values.

Second, we analyze the effect of the number of
graph attention layers in IMCHGAN to the prediction


Fig. 4. Effect of different parameters on performance of IMCHGAN.



kZ 1 k [2] F [þ][ Z] k [2] k [2] F

� �



þ � 1 kZ 1 k [2] F

�

þ � 2 kC d k [2]




[2] F [þ][ Z] k [2] k [2]



kC d k [2] F [þ][ C] k [t] k [2] F :

� �




[2] F [þ][ C] k [t] k [2] F



:



A gradient descent with adaptive moment estimation

[27] is adopted to optimize the parameters of IMCHGAN.
First, for each training iteration, some batches of miRNAdisease pairs from the set of positive association entries V
and the set of unobserved entries V [�] were sampled. Second,
in the process of forward propagation, the embeddings for
the sampled drugs and targets are learned with two-level
attention mechanism according to Eqs. (5) and (8), respectively. Then, the drug-target association scores are predicted
using IMC. The all parameters of IMC and two-level attention mechanism are learned via back propagations.


3 R ESULTS


3.1 Experimental Data and Settings
Two drug-target related datasets are used in the experiments. First, the drug-target interaction heterogeneous
network dataset (DTI-HN) adopted from the previous
study [11], [12], [14] which includes four types of entities:
protein, disease, gene and side effect and eight types of
relationships: drug-drug interactions, drug-drug structure
similarity, drug-target interactions, drug-disease associations, drug-side effect associations, target-target interactions, target-target sequence similarity, target-disease
associations. The details of constructing this dataset can be
found in [11]. The statistics about DTI-HN are listed in

Table 1.

Second, several research works used golden standard
dataset (GSD) proposed in Yamanishi [9] as their evaluation dataset. GSD is composed of the four datasets corresponding to the classes of target proteins: (i) enzymes (Es),
(ii) ion channels (ICs), (iii) G-protein-coupled receptors
(GPCRs) and (iv) nuclear receptors (NRs). We also used
these datasets in our experiments to compare with the
method that used GSD. The statistics about GSD are in the

Table 2.



The experimental code is implemented based on the
open source machine learning framework Pytorch



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


LI ET AL.: IMCHGAN: INDUCTIVE MATRIX COMPLETION WITH HETEROGENEOUS GRAPH ATTENTION NETWORKS FOR DRUG-TARGET... 661


TABLE 3
Meta-Paths for Drug-Drug and Target-Target


Meta-path semantic



r [D] 1



r [D] 1 [: Dr] [inter] [Dr] drug-drug interactions

r [D] 2 [: Dr] [sim] [Dr]



r [D] 2 [: Dr] [sim] [Dr] drug-drug structure similarity

r [D] [: Dr] [assoc] [Di] [assoc] [Dr] drug-drug with common disease



r 3 [: Dr] [assoc] [Di] [assoc] [Dr] drug-drug with common disease

r [D] 4 [: Dr] [assoc] [Si] [assoc] [Dr] drug-drug with common side effect



r 4 [: Dr] [assoc] [Si] [assoc] [Dr] drug-drug with common side effect

r [D] [: Dr] [assoc] [Ta] [assoc] [ Dr][ drug-drug with common target protein]



r 5 [: Dr] [assoc] [Ta] [assoc] [ Dr][ drug-drug with common target protein]

r [T] 1 [: Ta] [inter] [Ta] target-target interactions



r [T] 1 [: Ta] [inter] [Ta] target-target interactions

r [T] 2 [: Ta] [sim] [Ta]



r [T] 2 [: Ta] [sim] [Ta] target-target structure similarity

r [T] [: Ta] [assoc] [Di] [assoc] [Ta] target-target with common disease



r 3 [: Ta] [assoc] [Di] [assoc] [Ta] target-target with common disease

r [T] 4 [: Ta] [assoc] [Dr] [assoc] [Ta][ target-target with common drug]



4 [: Ta] [assoc] [Dr] [assoc] [Ta][ target-target with common drug]



TABLE 4
Compare Results of Different Groups of Meta-Paths


Drug group Target group AUC AUPR



�r [D] 1 �



r [T] 1



r [D] 1



r [T] 1



r 1 � �r 1 � 0.8082 0.8285

r [D] 1 [;][ r] [D] 2 � �r [T] 1 [;][ r] [T] 2 � 0.8944 0.8787

r [D] 1 [;][ r] [D] 2 [;][ r] [D] 3 � �r [T] 1 [;][ r] [T] 2 [;][ r] [T] 3 � 0.9205 0.9054

r [D] 1 [;][ r] [D] 2 [;][ r] [D] 3 [;][ r] [D] 4 � �r [T] 1 [;][ r] [T] 2 [;][ r] [T] 3 � 0.9414 0.9158

r [D] 1 [;][ r] [D] 2 [;][ r] [D] 4 [;][ r] [D] 5 � �r [T] 1 [;][ r] [T] 2 [;][ r] [T] 4 � 0.9386 0.8935

r [D] 1 [;][ r] [D] 2 [;][ r] [D] [;][ r] [D] 4 [;][ r] [D] � �r [T] 1 [;][ r] [T] 2 [;][ r] [T] [;][ r] [T] 4 � 0.9544 0.9203



�r [D] 1 [;][ r] [D] 2 �



�r [T] 1 �




[D] 1 [;][ r] [D] 2

[D] 1 [;][ r] [D] 2

[D] 1 [;][ r] [D] 2

[D] 1 [;][ r] [D] 2

[D] 1 [;][ r] [D] 2




[D] 3 [;][ r] [D] 4

[D] 4 [;][ r] [D] 5

[D] 3 [;][ r] [D] 4



�r [D] 1 [;][ r] [D] 2 [;][ r] [D] 3 �



�r [T] 1 [;][ r] [T] 2 �




[T] 1 [;][ r] [T] 2




[D] 2 [;][ r] [D] 3

[D] 2 [;][ r] [D] 3

[D] 2 [;][ r] [D] 4

[D] 2 [;][ r] [D] 3




[D] 4 [;][ r] [D] 5




[T] 3 [;][ r] [T] 4



�r [D] 1 [;][ r] [D] 2 [;][ r] [D] 3 [;][ r] [D] 4 �

�r [D] 1 [;][ r] [D] 2 [;][ r] [D] 4 [;][ r] [D] 5 �



r [T] 1



r [T] 1

r [T] 1

r [T] 1



�r [D] 1 [;][ r] [D] 2 [;][ r] [D] 3 [;][ r] [D] 4 [;][ r] [D] 5 �



�r [T] 1 [;][ r] [T] 2 [;][ r] [T] 3 [;][ r] [T] 4 �



�r [T] 1 [;][ r] [T] 2 [;][ r] [T] 3 �

�r [T] 1 [;][ r] [T] 2 [;][ r] [T] 3 �

�r [T] 1 [;][ r] [T] 2 [;][ r] [T] 4 �




[T] 1 [;][ r] [T] 2




[T] 1 [;][ r] [T] 2

[T] 1 [;][ r] [T] 2

[T] 1 [;][ r] [T] 2




[T] 2 [;][ r] [T] 3




[T] 2 [;][ r] [T] 3

[T] 2 [;][ r] [T] 3

[T] 2 [;][ r] [T] 4



0.9544 0.9203



performance. The comparative result is shown in Fig. 4b.
From Fig. 4b, GAT encoders with l = 2 provides the best performance. Additionally, we also note that with graph attention layers increasing (l � 2), the performance of GAT
encoders slightly decreases as higher graph attention layers
oversmoothed the encoded embeddings.

The Fig. 4c shows the effect of the dimension of drug or
target embedding on the performance of IMCHGAN. When
l = 128 achieves the best performance both AUC and AUPR.

Finally, multi-head attention mechanism defined in
Eq. (4) is employed to provide more robust representation
learning capability to IMCHGAN. K = 8 give us the best
performance.

Based on the above evaluation experimental results, in
the following experiments, we use a = 0.6, l = 2, d = 128 and
K = 8 as experimental settings. The learning rate of Adaptive Moment Estimation as 0.001 and the regulation coefficients � 1 ; � 2 as 0.001.


3.3 The Attention-Based Meta-Path Integration
Enhances the Performance of IMCHGAN

In this experiment, we show that multiple meta-path
embeddings are integrated to obtain a more comprehensive
node embedding which effectively enhances the performance of DTIs prediction. The group of meta-paths for
learning drug embeddings and target embeddings are
shown in Table 3.

To evaluate the effect of meta-path on the performance of
IMCHGAN-DTI, we used different groups of meta-paths to
generate the embeddings and compare their prediction performance. The compare results are shown in Table 4 which
shows the prediction performance of IMCHGAN is
improved as more meta-paths-based embeddings being
integrated into final embeddings.



Fig. 5. Esmolol’s top-5 meta-path neighbors with aggregation coefficients under different meta-paths.


Fig. 6. PSMB1’s top-5 meta-path neighbors with aggregation coefficients
under different meta-paths.


3.4 The Attention-Based Interpretability for
IMCHGAN

In this study, the feature representation learning is based on
a two-level attention models which brings the interpretability to IMCHGAN. Specifically, the bottom-level attention
aggregates meta-path neighbors’ embeddings to generate
node embedding where the aggregation coefficients are
learned via backpropagation and measure the prediction
task-related importance of neighbors embedding to the
node embedding under a specific meta-path. Fig. 5 shows
the drug Esmolol’s top-5 meta-path neighbors with aggregation coefficients under the meta-path Dr [inter] Dr, and
Dr [assoc] Di [assoc] Dr, respectively. Similarly, Fig. 6 shows the
target PSMB1’s top-5 meta-path neighbors with aggregation
coefficients under the meta-paths Ta [inter] Ta and
Ta [assoc] Dr [assoc] Ta, respectively.

The upper-level attention further aggregates different
meta-path-based embeddings into the final node embedding where the aggregation coefficients are learned via
backpropagation and measure the importance of different
meta-paths for the DTIs prediction. Fig. 7 shows the results
that the Heat map of aggregation coefficients for 5 drugs
under 5 different meta-paths. From Fig. 7, IMCHGAN gives
Dr [inter] Dr the largest aggregation coefficient, which means
that IMCHGAN considers the Dr [inter] Dr as the most related
meta-path in identifying the drug Esmolol’s associated targets. Interestingly, it seems that for the drug Cevimeline,
the meta-path Dr [assoc] Ta [assoc] Dr plays the most important
role in identifying the associated targets. Totally, both
Dr [inter] Dr and Dr [assoc] Ta [assoc] Dr are important for identifying


Fig. 7. Heat map of aggregation coefficients for different drugs under different meta-paths.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


662 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 2, MARCH/APRIL 2022


Fig. 8. Heat map of aggregation coefficients for different targets under
different meta-paths.



the potential drug-target associations. Similar Heat map is
also given in Fig. 8 for the 5 targets under 4 different metapaths.


3.5 Comparisons With Existing Work
In the experiments, K-Fold-Cross-Validation (K-FCV) are
used to evaluate the predictive performance of IMCHGAN.
In K-FCV, the known drug-target associations are considered positive samples and randomly divided into K subsets.
A part of them was considered the testing set and the rest
was considered the training set. When the ranking score of
the drug-target pair (i; j) was higher than a specific threshold, the model was considered to successfully predict the
(i; j) drug-target pair. The area under precision recall
(AUPR) curve and the area under receiver operating characteristic (AUROC) curve were used to evaluate prediction
performance of all prediction methods.

First, on the DTI-HN dataset, we compared IMCHGAN
with other 6 representative DTIs prediction baselines
including NetLapRLS [28], MSCMF [29], HNM [30],
LPMIHN [8], and recently proposed DTINet [11], NeoDTI

[14] (all of these baselines used DTI-HN as their evaluated
datasets). We carried 10-fold cross-validation (10-FCV)
experiments on all positive pairs (the known interacting
drug-target pairs) and a set of randomly sampled negative
pairs (the unknown interacting drug-target pairs). The number of the negative samples was 10 times as many as that of
positive samples. In the experiment, the following settings
were used for IMCHGAN: the used meta-paths for drug



�r [D] 1 [;][ r] [D] 2 [;][ r] [D] 3 [;][ r] [D] 4 [;][ r] [D] 5 �



r [D] 1




[D] 1 [;][ r] [D] 2

[T] 1 [;][ r] [T] 2




[D] 2 [;][ r] [D] 3

[T] 2 [;][ r] [T] 3




[D] 3 [;][ r] [D] 4

[T] 3 [;][ r] [T] 4




[D] 4 [;][ r] [D] 5



r 1 [;][ r] 2 [;][ r] 3 [;][ r] 4 [;][ r] 5 � and the meta-paths for target

r [T] 1 [;][ r] [T] 2 [;][ r] [T] [;][ r] [T] 4 � in Table 3, the biased item a=0.6, the head



Fig. 10. The 10-fold cross-validation comparison of prediction performance with other baselines on DTI-HN dataset where all unknown
drug–target interacting pairs were considered as negative samples.


layer l=2. The result shows that the mean and the standard
deviation AUC and AUPR of IMCHGAN were
0.9571�0.0065 and 0.9036�0.0057, respectively which are
obviously superior to the results of other compared methods. The details of compared results are shown in Fig. 9.

Next, we further evaluate the performance of IMCHGAN
by 10-FCV using the same experimental settings as NeoDTI
where all the unknown drug-target pairs are adopted as
negative samples. Fig. 10 shows the experimental result. We
observed an AUPR improvement (2.8 percent) over the second-best method NeoDTI. IMCHGAN also outperformed
other baseline methods in terms of AUROC.

Finally, we compare our proposed IMCHGAN with
NormMulInf [31] on GSD. As the same way in [31], we first
constructed the DTI heterogeneous information network
which includes drug-drug chemical structure similarity, the
sequence similarity of target proteins and the known DTI
interaction data. It is worth noting that since there is no
other side information for GSD, in this experiment,
IMCHGAN only use graph attention network to obtain
drugs and targets embeddings, the meta-path based methods are not employed to generate latent embeddings. Then,
5-FCV are adopted to evaluate the performance of prediction methods. Table 5 summarize the performance of two
methods in terms of AUC and AUPR. The highest performances are presented in boldface. As shown in Table 5,
IMCHGAN outperformed NormMulInf in terms of AUPR
on Es, ICs, GPCRs datasets, and NormMulInf slightly outperformed IMCHGAN on NRs dataset. However, the AUCs


TABLE 5
Comparison Results on Golden Standard Dataset (GSD)


Dataset Method AUC AUPR


ES NormMulInf 0.958 0.932

IMCHGAN 0.926 0.940


ICS NormMulInf 0.939 0.913

IMCHGAN 0.904 0.920


GPCRs NormMulInf 0.948 0.879

IMCHGAN 0.936 0.947


NRs NormMulInf 0.941 0.857

IMCHGAN 0.893 0.855



�r [T] 1 [;][ r] [T] 2 [;][ r] [T] 3 [;][ r] [T] 4 �



�r 1 [;][ r] 2 [;][ r] 3 [;][ r] 4 � in Table 3, the biased item a=0.6, the head

number of multi-head attention K=8, the graph attention



Fig. 9. The 10-fold cross-validation comparison of prediction performance with other baselines on DTI-HN dataset where the positive and
negative samples was set to 1 : 10.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


LI ET AL.: IMCHGAN: INDUCTIVE MATRIX COMPLETION WITH HETEROGENEOUS GRAPH ATTENTION NETWORKS FOR DRUG-TARGET... 663


TABLE 6
The Prediction and Validation of Novel (Unknown) DTIs


Dataset Drug name Drug ID Target name Target ID Valid Evidence


Dipivefrin DB00449 ADRA1D P25100 YES KEGG
Biperiden DB00810 CHRM2 P08172 YES KEGG
DTI-HN Glipizide DB01067 CHUK O15111 Unknown –
Methyldopa DB00968 ADRA2C P18825 YES KEGG
Valproic Acid DB00313 GRIA1 Q13002 YES CTD


Methoxsalen D00139 CYP1A1 hsa:1543 YES DB
Metyrapone D00410 CYP1A1 hsa:1543 YES CTD
ES Nifedipine D00437 CYP2C9 hsa:1559 YES CTD
Salicylic acid D00097 PTGS2 hsa:5743 YES DB
Imatinib mesylate D01441 MAPK1 hsa:5594 YES CTD


Benzocaine D00552 SCN5A hsa:6331 YES KEGG
Metoclopramide D00726 CHRNA5 hsa:1138 Unknown –
ICs Diazoxide D00294 ABCC9 hsa:10060 YES CTD
Nicotine D03365 CHRNA4 hsa:1137 YES DB
Zonisamide D00538 SCN5A hsa:6331 YES DB


Enprostil D01891 CHRM5 hsa:1133 Unknown –
Metoprolol D02358 ADRB2 hsa:154 YES DB
GPCRs Epinephrine D00095 ADRA1D hsa:146 YES KEGG
Niacin D00049 HCAR3 hsa:8843 YES DB
Risperidone D00426 DRD2 hsa:1813 YES KEGG


Isotretinoin D00348 RXRA has:6256 YES CTD
Isotretinoin D00348 RARB hsa:5915 YES KEGG
NRs Dydrogesterone D01217 NR3C2 hsa:4306 Unknown –
Mifepriston D00585 ESR1 has:2099 YES CTD
Dienestrol D00898 ESR2 hsa:2100 YES KEGG



of IMCHGAN are lower than that of NormMulInf. The reason may be that comparing with DTI-HN, the GSD have too
little DTI interaction data and only use simple drug similar
network and target sequence similar network to build heterogeneous network, while IMCHGAN is more suitable for
drug target prediction of biological heterogeneous network
with large data scale.


3.6 Prediction and Validation of Novel (Unknown)
DTIs

We also evaluate the capability of IMCHGAN for predicting novel DTIs (i.e., those that are not known DTIs in our
evaluated datasets) on DTI-HN, golden standard datasets,
separately. To verify these novel predictions, we considered several reference databases to find supporting evidences, such as DrugBank [32], KEGG [33], CTD [34] that
contain information obtained from curated/experimental/
published results on DTIs. The results of prediction and
validation can be shown in Table 6. In summary, these
novel DTIs predicted by IMCHGAN with validation databases supports further demonstrated its strong predictive

power.


4 D ISCUSSION AND C ONCLUSION


In this study, we developed a novel model called IMCHGAN
for drug-target interaction prediction. IMCHGAN was compared in 10-FCV on DTI-HN dataset with the existing computational prediction baselines, including NetLapRLS [28],
MSCMF [29], HNM [30], LPMIHN [8], and recently proposed
DTINet [11], NeoDTI [14], and 5-FCV on golden standard



dataset (GSD) with NormMulInf [31]. The experimental
results show that IMCHGAN has high accuracy in both 10FCV and 5-FCV on corresponding datasets. Moreover,
IMCHGAN also shows it has strong predictive power for
novel (unknown) DTIs.

The superior performance of IMCHGAN mainly attributes to the following reasons. First, we adopted the graph
attention mechanism automatically learn latent feature representations from biological information networks. Comparing with the other recent graph neural network-based
methods NeoDTI [14] and GCN-DTI [16] which mainly
employed the topological structure-related static weight
feature aggregation to learning representation, graph attention mechanism instead employed dynamic weight
(through learnable attention weight) feature aggregation to
obtain feature representation. The dynamic weigh aggregation provides more flexible and powerful representation
learning capability. Second, DTI biological information network is a typical heterogeneous information network.
IMCHGAN fully consider the heterogeneity of DTI network
and employed a two-level attention mechanism to obtain
more comprehensive drug and target feature representation. Finally, IMCHGAN is an end-to-end neural network
learning framework where the parameters of both the prediction score model and the feature representation learning
model are simultaneously optimized via backpropagation
under supervising of the observed known drug-target associations. This leads IMCHGAN to obtain prediction taskrelated drug and target feature representation.

However, IMCHGAN has certain limitations, which
require further investigations. First, the structural



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


664 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 2, MARCH/APRIL 2022



information regarding drug and target similarity networks
significantly affect the learned feature representations,
which further affect the final prediction results. Methods of
gathering different valuable biological information to effectively construct drug and target similarity networks are
worth investigating in the future. Second, in IMCHGAN,
feature representations of drug and target are learned with
a two-level attention mechanism from a DTI heterogeneous
information network. Other feature representation learning
approaches, such as heterogeneous graph transformer [35],
warrant further investigation. In addition, we are aware
that the method proposed in this paper has a wide range of
applicability. Therefore, in addition to solving problem DTI,
it can also be used to solve other biological entity association
prediction problems, such as association between small
molecule-microRNA association [36], [37], [38].


A CKNOWLEDGMENTS


This work was supported by the Fundamental Research
Project of Yunnan Province under Grant 202001BB050052.


R EFERENCES


[1] G. R. Langley et al., “Towards a 21st-century roadmap for biomedical research and drug discovery: Consensus report and recommendations,” Drug Discov. Today, vol. 22, no. 2, pp. 327–339, 2017.

[2] X. Chen et al., “Drug-target interaction prediction: Databases, web
servers and computational models,” Brief. Bioinf., vol. 17, no. 4,
pp. 696–712, 2016.

[3] R. Chen, X. Liu, S. Jin, J. Lin, and J. Liu, “Machine learning for
drug-target interaction prediction,” Molecules, vol. 23, no. 9, 2018,
Art. no. 2208.

[4] M. J. Keiser, B. L. Roth, B. N. Armbruster, P. Ernsberger, J. J. Irwin,
and B. K. Shoichet, “Relating protein pharmacology by LIGAND
chemistry,” Nat. Biotechnol., vol. 25, no. 2, pp. 197–206, 2007.

[5] G. Pujadas et al., “Protein-ligand docking: A review of recent
advances and future perspectives,” Curr. Pharmaceutical Anal., vol.
4, no. 1, pp. 1–19, 2008.

[6] X. Chen, M. X. Liu, and G. Y. Yan, “Drug–target interaction prediction by random walk on the heterogeneous network,” Mol. BioSyst., vol. 8, no. 7, pp. 1970–1978, 2012.

[7] F. Cheng et al., “Prediction of drug-target interactions and drug
repositioning via network-based inference,” PLoS Comput. Biol.,
vol. 8, no. 5, 2012, Art. no. e1002503.

[8] X. Y. Yan, S. W. Zhang, and S. Y. Zhang, “Prediction of drug–target interaction by label propagation with mutual interaction information derived from heterogeneous network,” Mol. BioSyst., vol.
12, no. 2, pp. 520–531, 2016.

[9] Y. Yamanishi, M. Araki, A. Gutteridge, W. Honda, and M. Kanehisa, “Prediction of drug-target interaction networks from the
integration of chemical and genomic spaces,” Bioinformatics, vol.
24, no. 13, pp. 232–240, 2008.

[10] J. P. Mei, C. K. Kwoh, P. Yang, X. L. Li, and J. Zheng, “Drug–target

interaction prediction by learning from local information and
neighbors,” Bioinformatics, vol. 29, no. 2, pp. 238–245, 2013.

[11] Y. Luo et al., “A network integration approach for drug-target

interaction prediction and computational drug repositioning from
heterogeneous information,” Nat. Commun., vol. 8, no. 1, 2017,
Art. no. 573.

[12] R. S. Olayan, H. Ashoor, and V. B. Bajic, “DDR: Efficient computa
tional method to predict drug–target interactions using graph
mining and machine learning approaches,” Bioinformatics, vol. 34,
no. 7, pp. 1164–1173, 2018.

[13] N. Natarajan and I. S. Dhillon, “Inductive matrix completion for

predicting gene-disease associations,” Bioinformatics, vol. 30, no.
12, pp. 60–68, 2014.

[14] F. Wan, L. Hong, A. Xiao, T. Jiang, and J. Zeng, “NEODTI: Neural

integration of neighbor information from a heterogeneous network for discovering new drug–target interactions,” Bioinformatics, vol. 35, no. 1, pp. 104–111, 2019.




[15] X. Zeng et al., “Network-based prediction of drug–target interac
tions using an arbitrary-order proximity embedded deep forest,”
Bioinformatics, vol. 36, no. 9, pp. 2805–2812, 2020.

[16] T. Zhao, Y. Hu, L. R. Valsdottir, T. Zang, and J. Peng, “Identifying

drug–target interactions based on graph convolutional network
and deep neural network,” Brief. Bioinf., vol. 22, no. 2, pp. 2141–
2150, 2021.

[17] J. Li, S. Zhang, T. Liu, C. Ning, Z. Zhang, and W. Zhou, “Neural

inductive matrix completion with graph convolutional networks
for mirna-disease association prediction,” Bioinformatics, vol. 36,
no. 8, pp. 2538–2546, 2020.

[18] X. Chen, L. Wang, J. Qu, N.-N. Guan, and J.-Q. Li, “Predicting

MIRNA-disease association based on inductive matrix completion,” Bioinformatics, vol. 34, no. 24, pp. 4256–4265, 2018.

[19] Y. Sun, J. Han, X. Yan, P. S. Yu, and T. Wu, “Pathsim: Meta path
based top-k similarity search in heterogeneous information
networks,” Proc. VLDB Endowment, vol. 4, no. 11, pp. 992–1003,
2011.

[20] C. Su, J. Tong, Y. Zhu, P. Cui, and F. Wang, “Network embedding

in biomedical data science,” Brief. Bioinf., vol. 21, no. 1, pp. 182–
197, 2020.

[21] D. Zhang, J. Yin, X. Zhu, and C. Zhang, “Network representation

learning: A survey,” IEEE Trans. Big Data, vol. 6, no. 1, pp. 3–28,
Mar. 2020.

[22] M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional

neural networks on graphs with fast localized spectral filtering,”
in Proc. Adv. Neural Inf. Process. Syst., 2016, pp. 3844–3852.

[23] T. N. Kipf and M. Welling, “Semi-supervised classification with

graph convolutional networks,” in Proc. 5th Int. Conf. Learn. Repre
[24] P. Velisentations�ckovi, 2017, pp. 1–14.�c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y.�

Bengio, “Graph attention networks,” in Proc. Int. Conf. Learn. Representations, 2018, pp. 1–12.

[25] X. Wang, H. Ji, B. Wang, P. Cui, P. Yu, and Y. Ye, “Heterogeneous

graph attention network,” in Proc. Web Conf, pp. 2022–2032, 2019.

[26] A. Vaswani et al. “Attention is all you need,” in Proc. 31st Int. Conf.

Neural Inf. Process. Syst., 2017, pp. 5998–6008.

[27] D. Kingma and J. Ba, “Adam: A method for stochastic opti
mization,” in Proc. 3rd Int. Conf. Learn. Representations, pp. 1–13,
2015.

[28] Z. Xia, L. Y. Wu, X. Zhou, and S. T. C. Wong, “Semi-supervised

drug-protein interaction prediction from heterogeneous biological
spaces,” BMC Syst. Biol., vol. 4, no. 2, 2010, Art. no. S6.

[29] X. Zheng, H. Ding, H. Mamitsuka, and S. Zhu, “Collaborative

matrix factorization with multiple similarities for predicting
drug–target interactions,” in Proc. 19th ACM SIGKDD Int. Conf.
Knowl. Discov. Data Mining, 2013, pp. 1025–1033.

[30] W. Wang, S. Yang, X. Zhang, and J. Li, “Drug repositioning by

integrating target information through a heterogeneous network
model,” Bioinformatics, vol. 30, no. 20, pp. 2923–2930, 2014.

[31] L. Peng, B. Liao, W. Zhu, Z. Li, and K. Li, “Predicting drug-target

interactions with multi information fusion,” IEEE J. Biomed. Health
Inf., vol. 21, no. 2, pp. 561–572, 2017.

[32] C. Knox et al., “Drugbank 3.0: A comprehensive resource for

‘omics’ research on drugs,” Nucleic Acids Res., vol. 39, no. Database issue, pp. 1035–1041, 2011.

[33] M. Kanehisa, M. Furumichi, M. Tanabe, Y. Sato, and K. Morish
ima, “Kegg: New perspectives on genomes, pathways, diseases
and drugs,” Nucleic Acids Res., vol. 45, no. 1, pp. 353–361, 2017.

[34] A. P. Davis et al., “The comparative toxicogenomics database:

Update 2013,” Nucleic Acids Res., vol. 41, no. 1, pp. 1104–1114,
2013.

[35] Z. Hu, Y. Dong, K. Wang, and Y. Sun, “Heterogeneous graph

transformer,” in Proc. Web Conf., 2020, pp. 2704–2710.

[36] X. Chen, N.-N. Guan, Y.-Z. Sun, J.-Q. Li, and J. Qu, “Microrna
small molecule association identification: from experimental
results to computational models,” Brief. Bioinf., vol. 21, no. 1, pp.
47–61, 2020.

[37] Y. Zhao, X. Chen, J. Yin, and J. Qu, “SNMFSMMA: Using symmet
ric nonnegative matrix factorization and kronecker regularized
least squares to predict potential small molecule-microrna
association,” RNA Biol., vol. 17, no. 2, pp. 281–291, 2020.

[38] C. Wang and X. Chen, “A unified framework for the prediction of

small molecule-microrna association based on cross-layer dependency inference on multilayered networks,” J. Chem. Inf. Model.,
vol. 59, no. 12, pp. 5281–5293, 2019.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


LI ET AL.: IMCHGAN: INDUCTIVE MATRIX COMPLETION WITH HETEROGENEOUS GRAPH ATTENTION NETWORKS FOR DRUG-TARGET... 665



Jin Li received the BSc degree in computer science, the MSc degree in computational mathematics, and the PhD degree in telecommunication and
information system from Yunnan University in
1998, 2004, and 2012, respectively. He is currently
a professor of machine learning with the National
Pilot School of Software, Yunnan University, Kunming, China. He has authored or coauthored more
than 30 papers in the peer-reviewed international
journals, including the Briefings in Bioinformatics,
Bioinformatics, Artificial Intelligence in Medicine,
IEEE/ACM Transactions on Computational Biology and Bioinformatics,
Future Generation Computation System, IEEE Transactions on Cybernetics, Knowledge-based System, Applied Intelligence. His research interests
include machine learning and bioinformatics.


Jingru Wang received the bachelor’s degree
from the School of Information Science and Engineering, Hunan Institute of Science and Technology, China, in 2018. She is currently working
toward the master’s degree with the School of
Software, Yunnan University, Kunming, Yunnan,
China. Her main research interests include bioinformatics and machine learning.


Hao Lv received the bachelor’s degree from the
School of Computer Science and Technology,
Chongqing University of Posts and Telecommunications, China, in 2017. He is currently working
toward the master’s degree with the School of
Software, Yunnan University, Kunming, Yunnan,
China. His main research interests include bioinformatics and machine learning.



Zhuoxuan Zhang received the bachelor’s
degree with the School of Software from Yunnan
University, China, in 2020. He is currently working
toward the master’s degree with the School of
Software, Yunnan University, Kunming, Yunnan,
China. His main research interests include bioinformatics and machine learning.


Zaixia Wang received the bachelor’s degree
from the School of Software from Yunnan University, China, in 2018. She is currently working
toward the master’s degree with the School of
Software, Yunnan University, Kunming, Yunnan,
China. Her main research interests include bioinformatics and machine learning.


" For more information on this or any other computing topic,
please visit our Digital Library at www.computer.org/csdl.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:59 UTC from IEEE Xplore. Restrictions apply.


