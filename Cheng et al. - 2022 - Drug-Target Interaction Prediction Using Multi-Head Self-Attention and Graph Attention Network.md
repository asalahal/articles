2208 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 4, JULY/AUGUST 2022

# Drug-Target Interaction Prediction Using Multi-Head Self-Attention and Graph Attention Network


Zhongjian Cheng, Cheng Yan, Fang-Xiang Wu, and Jianxin Wang


Abstract—Identifying drug-target interactions (DTIs) is an important step in the process of new drug discovery and drug repositioning.
Accurate predictions for DTIs can improve the efficiency in the drug discovery and development. Although rapid advances in deep
learning technologies have generated various computational methods, it is still appealing to further investigate how to design efficient
networks for predicting DTIs. In this study, we propose an end-to-end deep learning method (called MHSADTI) to predict DTIs based
on the graph attention network and multi-head self-attention mechanism. First, the characteristics of drugs and proteins are extracted
by the graph attention network and multi-head self-attention mechanism, respectively. Then, the attention scores are used to consider
which amino acid subsequence in a protein is more important for the drug to predict its interactions. Finally, we predict DTIs by a fully
connected layer after obtaining the feature vectors of drugs and proteins. MHSADTI takes advantage of self-attention mechanism for
obtaining long-dependent contextual relationship in amino acid sequences and predicting DTI interpretability. More effective molecular
characteristics are also obtained by the attention mechanism in graph attention networks. Multiple cross validation experiments are
adopted to assess the performance of our MHSADTI. The experiments on four datasets, human, C.elegans, DUD-E and DrugBank
show our method outperforms the state-of-the-art methods in terms of AUC, Precision, Recall, AUPR and F1-score. In addition,
the case studies further demonstrate that our method can provide effective visualizations to interpret the prediction results from
biological insights.


Index Terms—Drug-target interactions, multi-head self-attention, graph attention network


Ç


_



1 I NTRODUCTION
# T HE (DTIs) is a critical task for new drug discovery and drug accurate identification of drug-target interactions

repositioning. Because biological experiments cause excessive time and expensive lab cost [1], predicting potential
DTIs through efficient computational methods, which can
reduce the scope of conducting biological experiments, is
demanded urgently. In recent years, although many effective prediction methods have been proposed by researchers,
how to design more accurate and efficient methods for predicting DTIs is still a major challenge.

In order to accurately predict potential DTIs, many
machine learning-based methods have been proposed

[2], [3], [4], [5], [6], [7], [8], [9]. These methods can use


� Zhongjian Cheng and Jianxin Wang are with Hunan Provincial Key Lab
on Bioinformatics, School of Computer Science and Engineering, Central
[South University, Changsha 410083, China. E-mail: zj_cheng@csu.edu.](mailto:zj_cheng@csu.edu.cn)
[cn, jxwang@mail.csu.edu.cn.](mailto:zj_cheng@csu.edu.cn)
� Cheng Yan is with Hunan Provincial Key Lab on Bioinformatics, School of
Computer Science and Engineering, Central South University, Changsha
410083, China, and also with the School of Computer and Information,
Qiannan Normal University for Nationalities, Duyun, Guizhou 558000,
[China. E-mail: yancheng01@mail.csu.edu.cn.](mailto:yancheng01@mail.csu.edu.cn)
� Fang-Xiang Wu is with the Division of Biomedical Engineering and
Department of Mechanical Engineering, University of Saskatchewan, Sas[katoon, SK S7N5A9, Canada. E-mail: faw341@mail.usask.ca.](mailto:faw341@mail.usask.ca)


Manuscript received 9 May 2020; revised 23 October 2020; accepted 25 April
2021. Date of publication 6 May 2021; date of current version 8 August 2022.
(Corresponding author: Cheng Yan.)
Digital Object Identifier no. 10.1109/TCBB.2021.3077905


_



more data from both drugs and targets in a unified
framework simultaneously from chemogenomic perspective [10]. For example, Bleakley and Yamanishi [2]
develop a bipartite local model which applies SVMs on
known interactions after employing the similarity measurements between protein sequences and chemical structures. Cheng et al. [3] employ the multi-target
quantitative structure-activity relationships (mt-QSAR)
and computational chemogenomics for predicting DTIs
based on feature selection techniques. Van Laarhoven
and Marchiori [4] propose a noval model which integrates a simple weighted nearest neighbor method into a
machine learning method previously proposed and get a
high accuracy prediction result. Wang and Zeng [5] propose a model based on restricted Boltzmann machines.
Compared with the method that only mixes multiple
types of interactions, their model achieves the better prediction performances for different types of DTIs. Yuan
et al. [6] propose a new method (called DrugE-Rank) to
improve the prediction performance and a variety of
well-known similarity-based methods are used as components of ensemble learning in DrugE-Rank. Based on
the heterogeneous map of known DTIs with multiple
similarities between drugs and multiple similarities
between targets, Olayan et al. [7] develop a novel method
(called DDR) which applies a non-linear similarity fusion
method to multiple similarities selected in a heuristic
process, then they extract the features from the heterogeneous graph and train a random forest to predict DTIs.


_



1545-5963 © 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.

See ht_tps://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


CHENG ET AL.: DRUG-TARGET INTERACTION PREDICTION USING MULTI-HEAD SELF-ATTENTION AND GRAPH ATTENTION NETWORK 2209



Lee and Nam [8] use a random walk with restart algorithm to utilize global network topology information for
solving the problem which only considers directly connected nodes on the concept of “guilt-by-association”.
Mohamed et al. [9] regard the problem as a link prediction in knowledge graphs and propose a knowledge
graph embedding model (called TriModel) to learn vector representations for downstream. Because the traditional machine learning-based methods are relatively
simple, they may not be able to learn the complicated
concepts in the datasets which contain a large amount of
data and the methods based on machine learning gradually perform poorly on large amounts of data.

Recently, deep learning-based methods have achieved
excellent results for lots of prediction problems. Many endto-end deep learning methods have also been used for predicting DTIs and achieved better performance than traditional machine learning methods [11], [12], [13], [14], [15],

[16]. Compared to traditional feature selection methods,
Wang et al. [11] use the stacked autoencoder to mine the
information from raw data adequately and the method generates highly representative features automatically. By representing proteins with 2D distance maps from the
monomer structure, Zheng et al. [16] follow the Visual Question Answering mode and propose a deep learning framework which extracts information of drugs by the Bi-LSTM
structure and characteristics of proteins by dynamic CNN
structure. Lin et al. [12] introduce a probabilistic model
named FNML for dealing with overwhelming negative
samples and it achieves a robust performance on highly
imbalanced datasets of DTIs. Tsubaki et al. [13] propose an
end-to-end representation learning model for DTIs which
extracts representative features of drugs and proteins by a
graph neural network and a convolution neural network
(CNN), respectively. Although the method achieves very
good performance, it is obvious that the amino acid sequences of proteins have association information from the context in real biological environments. There are many
potential interactions between different positions in the
sequences. However, the convolutional window of CNN is
fixed and then CNN structure can not model these characteristics of contextual information. Based on a two-way
attention mechanism, Gao et al. [14] propose an interpretable model which uses the long short-term memory (LSTM)
to extract contextual information for amino acid sequences
and graph convolutional networks (GCN) to obtain feature
vectors of drugs. The LSTM structure effectively solves the
problem that the contextual information of amino acid
sequences can not be obtained. However, as the length of
amino acid sequences are usually large, LSTM can not learn
long-dependent contextual association information. Zheng
et al. [15] propose a hybrid model that integrates the CNN
and LSTM for DTIs, similarly, it also does not solve these
problems.

In this study, we propose a model (called MHSADTI) to
predict DTIs with the multi-head self-attention mechanism

[17] and graph attention networks (GATs) [18]. When
extracting the characteristics of amino acid sequences of
proteins, the multi-head self-attention mechanism can solve
the problem that CNN can not obtain contextual association
information. At the same time, it can also gain the long


TABLE 1
Summary of the Four Datasets, Human, C.elegans, DUD-E and

DrugBank



Datasets Number of



Number
of Drugs



Number of
total sample



Number of



Proteins



Proteins of Drugs total sample positive

interactions

human 852 1,052 6,738 3,369
C.elegans 2,504 1,434 8,000 4,000
DUD-E 20,489 102 68,070 22,886
DrugBank 4,794 6,707 37,283 18,816



dependent information in the sequences which can not be
learned by RNN, LSTM and other structures when the
lengths of amino acid sequences are too long. When extracting the characteristics of drugs, we use the GAT instead of
the graph convolutional network. This is advantage that
GAT can specify implicitly different weights to different
nodes in a neighborhood which can avoid the possible
effects of noise connection on nodes in the graph. The GAT
model further enhances the validity of the representative
feature vectors of the graphs. Finally, we also use attention
scores to help us consider which amino acid subsequence in
a protein is more important for the drug to predict its interaction through drug-protein attention network modules

[13]. Evaluated on the four datasets, human, C.elegans,
DUD-E and DrugBank, the proposed model achieves the
better performances than state-of-the-art methods in terms
of AUC (area under the receiver operating characteristics
curve), Precision, Recall, AUPR (area under the precisionrecall curve) and F-score. The results illustrate the validity
of the proposed model. In order to help understand what
the network structure has been learned intuitively, we provide the visualized results for multi-head self-attention and
drug-protein attention.


2 D ATASETS


In the task of predicting DTIs, randomly selected negative
samples may include potentially positive samples, which
causes the model to have higher accuracy during the training phase but not perform well in the test datasets [19]. So
an effective sampling method for negative samples in datasets is critical to predict DTIs. However, there are many
existing datasets with only random negative sampling have
been applied to evaluate the performance of algorithms for
predicting DTIs [20], [21].

In this study, we focus on evaluating the model on two
different datasets (human and C.elegans). The two datasets
are created by an effective negative sampling method [22],
where human dataset contains 3,369 positive interactions
between 1,052 unique compounds and 852 unique proteins;
the C.elegans dataset contains 4,000 positive interactions
between 1,434 unique compounds and 2,504 unique proteins. Based on the assumption that similar compounds
interact with the proteins that are similar to known proteins,
more credible negative samples are obtained after applying
specific dissimilarity rules to the screening framework.
Table 1 shows the details of two datasets. The human dataset has been used in the DeepCPI [13], [16] and [22]. The C.
elegans has been used in the [13] for predicting DTIs. In the



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


2210 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 4, JULY/AUGUST 2022


based on multi-head self-attention mechanism (Section 3.2),
are concatenated as input to the classifier for predicting
DTIs. The input of the algorithm is the raw data of drug
molecules and proteins (SMILES and amino acid sequences). After the embedding layers, molecular graphs
obtained from SMILES preprocessed by RDKit are input to
GAT and protein sequences based on n-gram are input to
the encoder.



Fig. 1. An overview of the proposed prediction approach.


experiments, we get the two datasets created from [13]
whose ratio of positive and negative samples is 1:1. The two
[datasets can be obtained from the link address (https://](https://github.com/masashitsubaki/CPI_prediction)
[github.com/masashitsubaki/CPI_prediction).](https://github.com/masashitsubaki/CPI_prediction)

In order to further illustrate the accuracy of the proposed
model, we also conduct the experiment on the DUD-E
benchmark, a robust dataset for structure-based virtual
screening methods. It contains 102 diverse target proteins
(provided as PDB files), 22,886 active pairs of proteins and
compounds (provided as SMILES (simplified molecular
input line entry specification)). The DUD-E dataset has been
used in [16] and [23]. We follow the experimental setting in

[[23] and the dataset is available at (https://github.com/](https://github.com/prokia/drugVQA)
[prokia/drugVQA). Since the training, validation and test](https://github.com/prokia/drugVQA)
sets of DUD-E are well-divided and the competing method
(DeepCPI [13]) has also been evaluated with three-fold cross
validation in previous works, we also conduct three-fold
cross validation to fairly evaluate and compare the performances of methods on DUD-E.

Compared to smaller datasets (human and C.elegans), we
also test our proposed method on datasets with larger
amounts of data. We organize the data in the DrugBank
database and obtain a dataset containing a large amount of
data. The DrugBank dataset contains 18,816 positive interactions between 4,794 unique compounds and 6,707 unique
proteins. Finally, the prediction performances of our
method and competing methods are evaluated in terms of
AUC, Precision, Recall, AUPR and F1-score on the above
four datasets.


3 M ETHOD


In this study, we propose a novel approach, named
MHSADTI, to predict DTIs. We first give the brief description of the approach, and then focus on the different modules of the approach.

Fig. 1 shows an overview of the proposed prediction
approach. The feature vectors of drug molecules and proteins, which are low-dimensional real-valued representations obtained using a GAT (Section 3.1) and an encoder



3.1 Graph Attention Network for Drugs
With the advent of graph structure data in recent years,
many methods have focused on the representation and
learning of graphs, and thus great progress has been made
in graph-based studies [24], [25], [26], [27]. Compared to
grid-like structure data, graph structure data usually lies in
an irregular domain which makes it more difficult to accurately represent the characteristics of graphs. When modeling the characteristics of drugs, drug molecules represented
by SMILES can be transformed into the graphs of molecules.
In the graph, the atoms represent nodes and chemical bonds
between atoms are represented as existing edges. Finally,
we can effectively extract map information based on existing works [24], [25], [26], [27] which can help us complete
prediction tasks.

Due to the difficulties of representation learning in molecules which have few types of atoms and chemical bonds,
we use the r-radius subgraphs [28] to get the vertical integer
encoding sequence of each drug. These subgraphs contain
large amounts of information about the drug molecules for
fully learning parameters in networks. Using the same procedure as [13], we can get a set of vertex features after
embedding for each drug molecule, D ¼ �d 1 ; d 2 ; . . . ; d j Dj �,

d i 2 R [d], where d is the embedding dimension. Through the
transformation of a weight matrix, W 2 R [F] [�][d], we can obtain
more representation information in the high dimension
than the original vector space. In order to learn the structural information of the graph, a self-attention mechanism
is applied to the graph and the attention coefficient can be
obtained as follows.



e ij ¼ a Wd� i ; Wd j �



j 2 N i : (1)



The coefficient e ij indicates the importance of vertex j to
vertex i in the graph, where N i is the neighborhood of vertex i in the graph. Instead of allowing each vertex to attend
others, we keep the structural information between atomic
vertices by N i . Next, we normalize them using a softmax
function



a ij ¼ softmax e� ij �



¼ exp e ij



exp e� ij �

~~P~~ k2N [exp e] ð



(2)
k2N i [exp e] ð [ik] Þ [:]



Finally, combining Equations (1) and (2), the attention
coefficients are expressed as



� a� [T] �Wd i jjWd j ���



exp LeakyRelu� a� [T] �Wd i jjWd j
a ij ¼



k2N i [exp LeakyRelu] ~~�~~ ~~�~~ [ a] [T] [ Wd] ~~�~~ [i] [jj][Wd] [j]



~~P~~



; (3)
~~�~~ ~~�~~ [ a] [T] [ Wd] ~~�~~ [i] [jj][Wd] [j] ~~���~~



where a 2 R [2][�][F] is the weight vector, the softmax function
applies LeakyRelu [29] nonlinear transformation and jj indicates the concatenated operation. As a result, we can get the



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


CHENG ET AL.: DRUG-TARGET INTERACTION PREDICTION USING MULTI-HEAD SELF-ATTENTION AND GRAPH ATTENTION NETWORK 2211



final output features of each vertex after combining the
characteristic information of its neighbors.



3.2.1 Preprocessing and Position Encoding
As explained in [33], we use the n-gram to define the
“word” in amino acid sequences. Given an amino acid
sequence, by splitting the protein sequences into overlapping n-gram amino acids, we can get its vector representation P ¼ �p 1 ; p 2 ; . . . ; p j j P �, where p i 2 R [d] is the embedding

vector of the ith word and d is the embedding dimension. In
the input part of the data, the model containing no recurrence and no convolution can not make full use of the order
of the sequence. In order to solve this problem, a “position
encoding” is set to obtain relative position information for
each “word” in the sequence. The “position encoding“ can
provide context information for each position of the

sequence.



0
i [¼][ s]



X

j2N i



a ij Wd j



1A; (4)



1



d



0

@



where s is the Relu activation function, W is the dimensional transformation matrix, d j is the jth neighbor feature
vector. Each vertex integrates the neighbors‘ features
according to the attention weights with its neighbors, that is
to say, the vertex is represented as a weighted average of its
neighbor feature vectors. After the activation function, the
semantic feature vector of the vertex is obtained. In order to
make the learning process of the self-attention mechanism
more stable and effective, we introduce the idea of multiattention, averaging the feature vectors learned by multiple
attentions to obtain the final feature vectors of the hidden
layer.



PE ðpos;2jþ1Þ ¼ cos pos



� �10000 [2][j=d] �



�



� �10000 [2][j=d] �



�



PE ðpos;2jÞ ¼ sin pos



(7)


; (8)



where pos and j 2 ½0; [d] 2 ~~[�]~~ [represent the position and the]

dimension, respectively. In Equations (7) and (8), the function PE �ð Þ [can effectively learn information between relative]
positions because there is a linear relationship between
PE pos and PE posþk for the fixed offset k.



X

j2N i



1A; (5)



d



00 1

0

i [¼][ s] ~~@~~ H



H
X

h¼1



a [h]

ij [W] [ h] [d] [j]



where H is the number of attention. By adjusting the dimensions of the transformation matrix W each time, we can
ensure that the final transformed output feature dimension
is F . After the convolution of t layers, the final feature representation of each vertex in the graph can be obtained, D [ð Þ][t] ¼



�



sin að þ bÞ ¼ sin að Þ cos bð Þ þ cos að Þ sin bð Þ
cos að þ bÞ ¼ cos að Þ cos bð Þ � sin að Þ sin bð Þ



: (9)



nd [ð Þ] 1 [t] [; d] [ð Þ] 2 [ t] [;][ . . .][ ; d] [ð Þ] j [ t] Dj o



d [ð Þ][t]




[ð Þ] 1 [t] [; d] [ð Þ] 2 [ t]




[ð Þ] 2 [ t] [;][ . . .][ ; d] [ð Þ][ t] D



d 1 [; d] 2 [;][ . . .][ ; d] jDj . Averaging the vectors of vertices, we

can obtain the feature vector of the drug, y drug as follows.



So we can get the linear relationship between the pos
position and the posþk position.



(10)



y drug ¼ [1]

jDj



jDj
X

i¼1



d [ð Þ] i [t] [:] (6)



PE ð posþk;2jÞ [¼][ PE] ðpos;2jÞ [�] [PE] ðk;2jþ1Þ [þ]


PE pos;ð 2jþ1Þ [�] [PE] ðk;2jÞ



3.2 Multi-Head Self-Attention for Protein Sequences
The self-attention mechanism has been widely used in deep
learning tasks with contextual relationships including
machine translation, semantic understanding [30], [31].
Through the parameters which can be learned by the backward propagation in the network, the model is able to relate
different positions of the sequence to calculate the importance of the sequence representation. In particular, Transformer [17], the use of self-attention to calculate the
representation of its inputs and outputs instead of sequencealigned RNN or CNN, achieves a major breakthrough in the
machine translation. Since the beginning of this work, there
are many excellent and advanced research works evolved in
the field of natural language processing [32].

In this section, we use the fully connected self-attention
module to learn the characteristic information of the amino
acid sequences with the encoder structure of Transformer.
Along with the effectiveness and robustness of the multihead self-attention mechanism, this structure addresses
some critical problems that CNN can not obtain contextual
information in the sequences. That LSTM can not obtain
long-dependent information when the sequence is long has
also been overcome. In the experimental part, the visualization results learned by the attention mechanism are shown
to illustrate the effectiveness of MHSADTI.



3.2.2 Encoder Based on the Multi-Head Self-Attention

Mechanism

Fig. 2 shows the detailed process of the self-attention structure. The top and bottom of Fig. 2 are the output vectors
processed by the self-attention structure and input vectors
preprocessed in the Section 3.2.1, respectively. The part
enclosed by the red dotted line is the self-attention learning
process between each “word” vector in an amino acid
sequence and the “word” vectors in its contextual sequence.
As depicted in Fig. 2, each “word” vector can be represented as a query, key-value pair through three mapping
matrices whose parameters can be learned by backward
propagation. For the output of each position, the attention
mechanism maps a query and a set of key-value pairs to get



PE posð þk;2jþ1Þ [¼][ PE] ðpos;2jþ1Þ [�] [PE] ðk;2jþ1Þ [�]

(11)
PE pos;ð 2jÞ [�] [PE] ðk;2jÞ [:]


For each position in the sequence, through the function
PE ð Þ � [, we can model the contextual feature vectors of differ-]
ent positions. Finally, the sum of the “word” embedding
vector and the position encoding is used as the input to the
encoder for downstream tasks.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


2212 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 4, JULY/AUGUST 2022


The second part of the encoder is a fully connected forward network which contains two linear transformation
structures applied to each position in the sequence separately and identically.


MLP xð Þ ¼ W 2 relu Wð ð 1 x þ b 1 ÞÞ þ b 2 ; (16)


where W 1 2 R [d] [inner] [�][d], W 2 2 R [d][�][d] [inner] are transformation
parameters, relu is the activation function, x is the output
from the first part of encoder.


**f**


**f**


**f**



**f**


**f**


**f**



**f**


**f**


**f**



Fig. 2. The detailed process of the self-attention structure.


the weighted sum at different locations in the context. The
weight assigned to each value is calculated according to the
query and the corresponding key.

More specifically, after preprocessing in Section 3.2.1 and
mapping, we can get the representation of query, key, value
for each amino acid sequence, Q ¼ �q 1 [T] [; q] 2 [T] [;][ . . .][ ; q] L [T] �; K ¼


**f**


**f**


**f**



�q 1 [T] [; q] 2 [T] [;][ . . .][ ; q] L [T] �


**f**


**f**


**f**



1 [T] [; q] 2 [T]


**f**


**f**


**f**



2 [T] [;][ . . .][ ; q] L [T]


**f**


**f**


**f**



; K ¼


**f**


**f**


**f**



As shown in Fig. 1, similar to most network structures, in
addition to two parts of the encoder, the residual connection

[34] structure and the layer normalization structure [35] are
used as the input and output. The normalization operations
and jump connections can improve the generalization ability of the model. The layer normalization can make the network converging faster by keeping the input of each layer
with the same data distribution during the training. The
residual connection can get rich characteristic information
of hidden layers by retaining not only the information of the
last layer but also the output information of the multi-layer
network processing. Finally, we can obtain the characteristics of each amino acid sequence after learning the k-layer
encoder structure, where k is a hyper-parameter.


**f**


**f**


**f**



�k [T] 1 [; k] [T] 2 [;][ . . .][ ; k] [T] L �


**f**


**f**


**f**



�v [T] 1 [; v] [T] 2 [;][ . . .][ ; v] [T] L �


**f**


**f**


**f**



k [T]


**f**


**f**


**f**




[T] 1 [; k] [T] 2


**f**


**f**


**f**




[T] 2 [;][ . . .][ ; k] [T] L


**f**


**f**


**f**



, V ¼ �v [T]


**f**


**f**


**f**




[T] 1 [; v] [T] 2


**f**


**f**


**f**




[T] 2 [;][ . . .][ ; v] [T] L


**f**


**f**


**f**



, where L represents


**f**


**f**


**f**



the length of the sequence and q i [T] [2][ R] [d] [,][ k] [T] i [2][ R] [d] [,][ v] [T] i [2][ R] [d] [.]

Finally, we can get the output of the ith position after selfattention processing.


**f**


**f**


**f**



i [T] [2][ R] [d] [,][ k] [T] i


**f**


**f**


**f**



the length of the sequence and q i [T]


**f**


**f**


**f**




[T] i [2][ R] [d] [,][ v] [T] i


**f**


**f**


**f**



Attention qð i ; K; V Þ ¼ softmax [q] [i] [K] ~~f~~ **f** [T]

� ~~p~~ d �

**f**


**f**



Attention qð i ; K; V Þ ¼ softmax [q] [i] [K] ~~f~~ **f** [T]


**f**


**f**



**f** V; (12)


**f**


**f**



3.2.3 Interaction Site Between a Drug and a Protein

Similar to [13], a neural attention mechanism can be used
between the drug molecule and the amino acid sequence
which can help us understand where in the protein

**f**

sequence are more likely to interact with drug molecules

[36]. For the protein information the output by the multi
**f**

layer encoder, P [ð Þ][t] ¼ p [ð Þ] 1 [t] [; p] [ t][ð Þ] 2 [;][ . . .][ ; p] [ð Þ][ t] P, p [ð Þ] i [t] 2 R [d], we can


**f**



**f**


**f**

, p [ð Þ] i [t]


**f**



**f**


**f**

where 1


**f**



**f**


**f**

where 1�pd is the scaling factor which is defined to be effec
tive for dealing with the small gradients when d is too large
after the softmax function [17].

Compared to recurrent neural networks, the model can
guarantee the parallel computing of data while learning the
contextual information from the sequences and greatly
reduce the time complexity for processing large amounts of
data. We can calculate the output of the entire self-attention
layer simultaneously.


**f**



**f**


f **f**
�pd


**f**



**f**


**f**

layer encoder, P [ð Þ][t] ¼ np 1 [; p] [ t] 2 [;][ . . .][ ; p] j j [ t] P o, p i 2 R [d], we can

calculate the importance of the drug molecule for different
positions of the protein by dot-product-based scalar values.


**f**



**f**


**f**

np [ð Þ] 1 [t] [; p] [ t][ð Þ] 2 [;][ . . .][ ; p] [ð Þ] j j [ t] P o


**f**



**f**


**f**

[ð Þ] 1 [t] [; p] [ t][ð Þ] 2


**f**



**f**


**f**

[ t][ð Þ] 2 [;][ . . .][ ; p] [ð Þ][ t] P


**f**



**f**


**f**


h drug ¼ d� W attention y drug þ b attention �


**f**



**f**


**f**


W attention d W trans p [ð Þ] i [t] þ b trans þ b attention

� � � �


**f**



**f**


**f**


W trans p [ð Þ] i [t] þ b trans

� �


**f**



**f**


**f**


(17)


(18)


**f**



**f**


**f**


h i ¼ d W attention d W trans p [ð Þ] i [t]


**f**



**f**


**f**


Attention Q; K; Vð Þ ¼ softmax [Q][K] ~~f~~ **f** [T]



**f**


**f**


[Q][K] ~~f~~ **f** [T]

� ~~p~~ d �



**f**


**f**


**f** V: (13)



**f**


**f**


@ i ¼ tanh h� [T] drug [h] [i] �; (19)

**f**



**f**


**f**


**f**


In order to maintain its validity and stability, the model
also incorporates the learning results of the multi-attention
which is able to learn the various semantic information of
the features in different spaces using multiple self-attention
weight matrices. Finally, the results learned by the multiattention mechanism are concatenated as output.



**f**


**f**


**f**

where W trans 2 R [F] [�][d], W attention 2 R [d] [atten] [�][F], b attention, b trans are
transformation parameters, d represents the non-linear activation function. By calculating the sum of the weights of the
attention mechanisms, we get the characteristic representation of the protein.



**f**


**f**


**f**


y protein ¼



**f**


**f**


**f**


j jP
X

i¼1



**f**


**f**


**f**


@ i h i : (20)



**f**


**f**


**f**


i [Q] [; KW] [ K] i [; VW] [ V] i

� �



**f**


**f**


**f**


head i ¼ Attention QW i [Q]



**f**


**f**


**f**


[ K] i [; VW] [ V] i



**f**


**f**


**f**


i [Q] [; KW] [ K] i



**f**


**f**


**f**


3.3 Classifier and Training
Similar to previous works [13], [14], [37], [38], [39] which
predict drug-target interactions based on data-driven deep
learning methods, we can also concatenate representations



**f**


**f**


**f**


MultiHead Q; K; Vð Þ ¼ Concat headð 1 ; . . . :; head h ÞW [O] ;



**f**


**f**


**f**


(14)


(15)



**f**


**f**


**f**


The characteristic representation of proteins obtained by
the self-attention structure can acquire the main features of
the protein according to different small molecules and
enhance the robustness of the model.



**f**


**f**


**f**


where W [Q]



**f**


**f**


**f**


i [ K] 2 R [d][�][d] [k], W i [V]



**f**


**f**


**f**


where W i [Q] [2][ R] [d][�][d] [k] [,][ W] i [ K] 2 R [d][�][d] [k], W i [V] [2][ R] [d][�][d] [v] [ and][ W] [ O] [ 2]

R [h][�][d] [v] [�][d] are used for feature space transformation, Concatð Þ �
is the vector concatenated operation.



**f**


**f**


**f**


i [Q] [2][ R] [d][�][d] [k] [,][ W] i [ K]



**f**


**f**


**f**


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


CHENG ET AL.: DRUG-TARGET INTERACTION PREDICTION USING MULTI-HEAD SELF-ATTENTION AND GRAPH ATTENTION NETWORK 2213



of drugs and proteins extracted by GAT and encoder for
downstream drug-target interactions prediction. After
concatenating the feature vectors extracted from small molecules and proteins, we can transform them through the
fully connected layer and softmax function to get the possibility of interaction between them.



Z ¼ W output y� drug ; y protein �



þ b output (21)



p i ¼ [ex][p][ z] ð [i] Þ
~~P~~ k [z] [k]



k [z] [k]



~~i~~ ¼ 0f ; 1g; (22)



where Z 2 R [2], W output 2 R [2][�ð][2][�][F] [Þ] is the weight matrix.

Because the final prediction is a supervised binary labeling problem, a cross-entropy loss function is used. In order
to avoid overfitting, a regularization term is added to the
expected loss function.



‘ uð Þ¼ �



N
X

k¼1



log p i k þ [�]



2 k k [u] [2] 2



2 [;] (23)



TABLE 2
Default Parameter Settings of MHSADTI Learned

by the Grid Search


Parameters Value


Activation Function(FC) ReLU
Activation Function(Output) Sigmoid
Epoch 200
Optimizer Adam
The number of layers of GAT 4
The number of encoder 1
The output dimension F 10
The embedding dimension d 10
The Regularization � 1e-5
The learning rate 1e-3
The radius r 2
The n-gram 3
The number of multi-head self-attention 8
The decay interval 10
The learning rate decay ratio 0.5


the feature vectors of the weighted sum of protein sequences is obtained. Finally, it completes the training process
within performing multi-layer perceptron structure processing on the concatenated two-part feature vectors. This
method has achieved good performances in various binary
classification predictions for DTIs and so it has attracted the
attention of many researchers.

On the other hand, we also compare the newly proposed
prediction method called DeepConv-DTI [37]. It captures
local residue patterns of proteins by performing convolutional neural network on various lengths of amino acid subsequences. For the characteristic representation of drugs, it
uses the Morgan/Circular drug fingerprint and each drug
is represented as a 2048-dimensional binary vector, whose
index indicates the existence of specific substructures.
Finally, the model fully concatenates the feature vectors of
drugs and proteins extracted by neural network to get prediction results after a fully connected layer.


4.2 Result

4.2.1 Effectiveness

In order to learn a generalized model, we get the experimental results with the cross validation for ten times on
four datasets and take the average as the final result.

In this section, Table 3 shows the experimental results of
the compared methods and MHSADTI. Similarly, we also
evaluate the two MHSADTI-ablation models (MHSADTI
(only gat), MHSADTI (only encoder)) shown in Table 4. As
shown in Table 3, on the human dataset, MHSADTI exceeds
the baseline in all evaluation metrics. Compared to DeepCPI, AUC, Precision and AUPR all achieve great improvement, from 0.9692 to 0.9822, 0.9187 to 0.9472, 0.9399 to
0.9568, respectively. Compared to DeepConv-DTI, the AUC
and AUPR are 0.8 1.3 percent higher, respectively.

As shown in Table 3, on the C.elegans dataset, AUPR has
the most obvious change, which increases from 0.9571 to
0.9832, which indicates an improvement of over the competing method. AUC, F1-score are also better than DeepCPI,
from 0.9758 to 0.9838, 0.9394 to 0.9763, respectively. Compared to DeepConv-DTI, MHSADTI is also better than it in
all indicators.



where u is the set of all parameters in the network, N is the
total number of the training samples, i k is the kth label and
� is the coefficient of the regularization term.


4 E XPERIMENTS


In this study, we have proposed a deep learning method
(called MHSADTI) for predicting DTIs. In the experiment,
the AUC, Precision, Recall, AUPR and F1-score are used as
the main metrics for the performance.

We perform the grid search to learn the model’s best
hyper-parameters. In the experiments, we initialize the
weight parameters of the model using Xavier [40] which is a
method of initializing weights in neural network. We optimize the training loss using the Adam [41] optimizer. The
hyper-parameters of the neural networks are set as follows

[13], the regularization: 1e-7, 1e-6, and 1e-5; the radius r: 0(i.e.,
each atom and chemical bond), 1 and 2; the n-gram: 1, 2 and 3;
the learning rate: 1e-5, 1e-4 and 1e-3; the number of layers of
GAT: 2, 3 and 4; the number of layers of encoder: 1, 2; the output dimension F: 5, 10 and 20; the embedding dimension d:
10, 20 and 30. Table 2 summarizes the default parameter settings for this architecture learned by grid search.

We use Pytorch(GPU) along with Python 3.6.7 to perform
our experiments. All experiments are executed on a Linux
machine with processor Intel(R) Xeon(R) Gold 6230 CPU @
2.10 GHz, 256 GB RAM, and 8 GeForce RTX 2080Ti GPU.
We conduct the multiple cross validation to evaluate the
prediction ability of the proposed algorithm.



4.1 Baseline

In order to assess the performance of MHSADTI, we have
compared it with other two state-of-the-art methods: DeepCPI [13] and DeepConv-DTI [37], which are also used to
predict DTIs. For the basic method, DeepCPI constructs a 4layer graph convolutional network and a 4-layer convolutional network to extract characteristics of drug molecules
and amino acid sequences of proteins, respectively. When
obtaining feature vectors of proteins, the weighting effects
of drug molecules on protein fragments are considered, and



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


2214 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 4, JULY/AUGUST 2022



TABLE 3
The Results on all the Dataset: AUC, Precision, Recall, AUPR

and F1-Score of the Compared Methods



Datasets Methods AUC

(std)


human RWR 0.8375
(0.008)



AUPR

(std)


0.8165
(0.006)

0.8257
(0.005)

0.9437
(0.009)

0.9399
(0.008)

0.9568
(0.009)


0.8212
(0.004)

0.8322
(0.006)

0.9711
(0.004)

0.9571
(0.011)

0.9832
(0.004)


0.4264
(0.019)

0.4351
(0.016)

0.4462
(0.010)


0.7854
(0.002)

0.7598
(0.005)

0.8238
(0.008)

0.7421
(0.016)

0.8351
(0.007)



0.7836
(0.011)



Precision

(std)



Recall

(std)


0.7243
(0.017)

0.8668
(0.016)

0.9175
(0.007)

0.9210
(0.008)

0.9365
(0.011)


0.7128
(0.010)

0.7474
(0.007)

0.9423
(0.003)

0.9271
(0.009)

0.9451
(0.005)


0.6301
(0.013)

0.6425
(0.017)

0.7124
(0.009)


0.6511
(0.004)

0.6289
(0.013)

0.7385
(0.014)

0.5563
(0.011)

0.7918
(0.012)



F1-score

(std)



0.4462). The results show that GAT used in MHSADTI can
further enhance the robustness of the model and improve
the prediction ability.

There is no doubt that extracting effective contextual
information of proteins and drugs is very important for predicting DTIs. Based on the convolution, both DeepConv-DTI
and DeepCPI perform well for predicting DTIs. DeepConvDTI achieves (0.9738, 0.9782, 0.9204, 0.8531), (0.9437, 0.9711,
0.4264, 0.8238) and (0.9204, 0.9579, 0.3155, 0.7643) in terms of
AUC, AUPR and F1-score for four datasets, respectively.
DeepCPI achieves (0.9692, 0.9758, 0.9334, 0.7003), (0.9399,
0.9571, 0.4351, 0.7421) and (0.9096, 0.9394, 0.3401, 0.6552) in
terms of AUC, AUPR and F1-score for four datasets, respectively. However, the spatial structure of proteins is complicated and there are many interactions and mutual influences
between protein structures [42], [43]. The convolutional deep
learning-based method is limited by the size of the neuron’s
perception window. Therefore, it only learns local feature
information which is not sufficient. In our methods, the
encoder based on multi-head self-attention not only has the
ability to extract local feature information which is similar to
CNN structure, but also explores the long-dependent information that may exist between different positions in the
amino acid sequence. When extracting the characteristics of
drugs, the possible noise connections of drug molecules are
omitted by GAT based on attention mechanism. Simultaneously, the more effective and generalized characteristic
information of drugs is extracted with GAT. Therefore,
MHSADTI based on multi-head self-attention mechanism
and GAT further improves the prediction performances and
achieves AUC values of 0.9822, 0.9838, 0.9484, 0.8628 on four
datasets. Furthermore, it also achieves the improvement in
terms of AUPR(0.9568, 0.9832, 0.4462, 0.8351) and F1-score
(0.9346, 0.9763, 0.3451, 0.7836) compared with others.

The DTI prediction is a very popular research topic and
many non-deep learning related methods have been proposed in recent years. In this section, we compare two
recently published methods (called DrugE-Rank [6], RWR

[8]) for predicting DTIs on human, C.elegans and DrugBank
datasets. For DUD-E, it is mainly used to evaluate methods
based on molecular docking(independent test dataset).
Therefore, we do not conduct experiment in DUD-E dataset
about RWR and DrugE-Rank. DrugE-Rank performs an
ensemble learning which integrates the prediction by
diverse cutting-edge techniques and adopts different prediction methods which includes the k-nearest neighbor
(kNN), BLM [2], LapRLS [44], WNN-GIP [4], [45]. RWR [8]
generates the global network topology information by using
random walk with restart algorithm, then the global topology information is used to train the model to predict DTIs.
As shown in Table 3, MHSADTI achieves the better performances than these two methods. It also indicates that
data-driven end-to-end deep learning models are more
robust than non-deep learning methods because they can
learn effective semantic features based on massive data.



RWR 0.8375 0.7707 0.7243 0.8165 0.7466
(0.008) (0.008) (0.017) (0.006) (0.010)

DrugE- 0.8562 0.7181 0.8668 0.8257 0.7851
Rank (0.008) (0.015) (0.016) (0.005) (0.005)



DrugE- 0.8562 0.7181 0.8668 0.8257 0.7851
Rank (0.008) (0.015) (0.016) (0.005) (0.005)

DeepConv- 0.9738 0.9295 0.9175 0.9437 0.9204
DTI (0.001) (0.006) (0.007) (0.009) (0.013)



0.8562
(0.008)

0.9738
(0.001)



DeepConv- 0.9738 0.9295 0.9175 0.9437 0.9204
DTI (0.001) (0.006) (0.007) (0.009) (0.013)

DeepCPI 0.9692 0.9187 0.9210 0.9399 0.9096
(0.003) (0.010) (0.008) (0.008) (0.009)



DeepCPI 0.9692 0.9187 0.9210 0.9399 0.9096
(0.003) (0.010) (0.008) (0.008) (0.009)

MHSADTI 0.9822 0.9472 0.9365 0.9568 0.9346



MHSADTI 0.9822 0.9472 0.9365 0.9568 0.9346

(0.001) (0.007) (0.011) (0.009) (0.009)


C.elegans RWR 0.8493 0.7860 0.7128 0.8212 0.7475
(0.007) (0.008) (0.010) (0.004) (0.005)



(0.001)



RWR 0.8493 0.7860 0.7128 0.8212 0.7475
(0.007) (0.008) (0.010) (0.004) (0.005)

DrugE- 0.8221 0.7906 0.7474 0.8322 0.7684
Rank (0.011) (0.011) (0.007) (0.006) (0.013)



DrugE- 0.8221 0.7906 0.7474 0.8322 0.7684
Rank (0.011) (0.011) (0.007) (0.006) (0.013)

DeepConv- 0.9782 0.9435 0.9423 0.9711 0.9579
DTI (0.001) (0.004) (0.003) (0.004) (0.006)



0.8221
(0.011)

0.9782
(0.001)



DeepConv- 0.9782 0.9435 0.9423 0.9711 0.9579
DTI (0.001) (0.004) (0.003) (0.004) (0.006)

DeepCPI 0.9758 0.9393 0.9271 0.9571 0.9394
(0.003) (0.006) (0.009) (0.011) (0.011)



DeepCPI 0.9758 0.9393 0.9271 0.9571 0.9394
(0.003) (0.006) (0.009) (0.011) (0.011)

MHSADTI 0.9838 0.9465 0.9451 0.9832 0.9763



MHSADTI 0.9838 0.9465 0.9451 0.9832 0.9763

(0.001) (0.009) (0.005) (0.004) (0.009)


DUD-E DeepConv- 0.9204 0.2131 0.6301 0.4264 0.3155



DeepConv- 0.9204 0.2131 0.6301 0.4264 0.3155

DTI (0.006) (0.005) (0.013) (0.019) (0.025)

DeepCPI 0.9334 0.2684 0.6425 0.4351 0.3401
(0.005) (0.006) (0.017) (0.016) (0.024)



DTI



(0.001)


0.9204
(0.006)



DeepCPI 0.9334 0.2684 0.6425 0.4351 0.3401
(0.005) (0.006) (0.017) (0.016) (0.024)

MHSADTI 0.9484 0.2462 0.7124 0.4462 0.3451



MHSADTI 0.9484 0.2462 0.7124 0.4462 0.3451

(0.001) (0.005) (0.009) (0.010) (0.025)


DrugBank RWR 0.7595 0.7406 0.6511 0.7854 0.6929
(0.004) (0.003) (0.004) (0.002) (0.003)



(0.001)



RWR 0.7595 0.7406 0.6511 0.7854 0.6929
(0.004) (0.003) (0.004) (0.002) (0.003)

DrugE- 0.7591 0.7070 0.6289 0.7598 0.6656
Rank (0.004) (0.005) (0.013) (0.005) (0.009)



DrugE- 0.7591 0.7070 0.6289 0.7598 0.6656
Rank (0.004) (0.005) (0.013) (0.005) (0.009)

DeepConv- 0.8531 0.7891 0.7385 0.8238 0.7643
DTI (0.003) (0.009) (0.014) (0.008) (0.016)



0.7591
(0.004)

0.8531
(0.003)



DeepConv- 0.8531 0.7891 0.7385 0.8238 0.7643
DTI (0.003) (0.009) (0.014) (0.008) (0.016)

DeepCPI 0.7003 0.7006 0.5563 0.7421 0.6552
(0.006) (0.003) (0.011) (0.016) (0.029)



DeepCPI 0.7003 0.7006 0.5563 0.7421 0.6552
(0.006) (0.003) (0.011) (0.016) (0.029)

MHSADTI 0.8628 0.7706 0.7918 0.8351 0.7836



(0.012)



0.7707
(0.008)

0.7181
(0.015)

0.9295
(0.006)

0.9187
(0.010)

0.9472
(0.007)


0.7860
(0.008)

0.7906
(0.011)

0.9435
(0.004)

0.9393
(0.006)

0.9465
(0.009)


0.2131
(0.005)

0.2684
(0.006)

0.2462
(0.005)


0.7406
(0.003)

0.7070
(0.005)

0.7891
(0.009)

0.7006
(0.003)

0.7706
(0.009)



On the DUD-E dataset, MHSADTI achieves the greater
performances than DeepCPI and DeepConv-DTI in terms of
AUC. MHSADTI achieves a 1.6, 3.0 percent improvement
over the DeepCPI and DeepConv-DTI, respectively.

On the DrugBank dataset, MHSADTI obtains the better
results in terms of all indicators. Especially, compared to
DeepConv-DTI, MHSADTI achieves a 1.1, 1.3 percent
improvement in terms of AUC and AUPR, respectively.

As shown in Table 4, for the four datasets, MHSADTI
gets slightly better performances than MHSADTI (only
encoder) in terms of AUC. According to the AUPR,
MHSADTI achieves the great improvement compared with
MHSADTI (only encoder) and the improvement is more
obvious when the dataset is larger. This indicates that GAT
used in MHSADTI is important for improving the generalization ability of the model when the two parts of the structure are merged and processed for downstream tasks. As
shown in Table 4, compared with MHSADTI(only encoder),
MHSADTI has an average 0.4, 1.7 percent improvement in
indicators (AUC, AUPR) on all datasets. Especially on C.elegans and DUD-E, MHSADTI achieves the greater improvement in terms of AUPR (from 0.9625 to 0.9832, 0.4337 to



4.2.2 Theoretical Analysis of Multi-Head Self-Attention
From the experimental results in Section 4.2.1 (shown by
Table 3), we can clearly see that the model evaluation index
has been improved after using the encoder structure of the



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


CHENG ET AL.: DRUG-TARGET INTERACTION PREDICTION USING MULTI-HEAD SELF-ATTENTION AND GRAPH ATTENTION NETWORK 2215



TABLE 4
The Results on all the Dataset: AUC, Precision, Recall, AUPR

and F1-Score of the Two MHSADTI-Ablation Models and

MHSADTI



TABLE 5
The Complexity, Degree of Parallelization, Maximum Step Size

on Self-Attention, Multi-Head Self-Attention, Convolutional,

Recurrent and the Corresponding Methods



Datasets Methods AUC


(std)



Precision


(std)



AUPR


(std)


0.9402

(0.008)


0.9511
(0.006)



0.9360

(0.007)



Recall


(std)


0.9196

(0.008)


0.9419

(0.009)



F1-score


(std)



Network

Structure (each
layer)



Methods Complexity Degree of
Parallelization



Maximum


Step Size



human MHSADTI
(only gat)



0.9716

(0.003)


0.9809
(0.001)



MHSADTI 0.9716 0.9255 0.9196 0.9402 0.9120
(only gat) (0.003) (0.010) (0.008) (0.008) (0.010)

MHSADTI 0.9809 0.9359 0.9419 0.9511 0.9360
(only (0.001) (0.009) (0.009) (0.006) (0.007)
encoder)



0.9255

(0.010)


0.9359
(0.009)



self-attention – Oðn [2] � dÞ OðnÞ Oð1Þ
multi-head self- MHSADTI Oðn [2] � d � mÞ OðnÞ Oð1Þ
attention



Oðk � n � d [2] Þ OðnÞ Oðlog k nÞ



MHSADTI 0.9822

(0.001)



0.9568

(0.009)


0.9524

(0.008)


0.9625

(0.008)



convolutional DeepConvDTI, DeepCPI



0.9365

(0.011)


0.9293

(0.010)


0.9423

(0.007)



0.9346

(0.009)



0.9479

(0.008)



C.elegans MHSADTI
(only gat)



0.9767

(0.003)


0.9833

(0.002)



MHSADTI 0.9767 0.9387 0.9293 0.9524 0.9337
(only gat) (0.003) (0.009) (0.010) (0.008) (0.010)

MHSADTI 0.9833 0.9502 0.9423 0.9625 0.9479
(only (0.002) (0.006) (0.007) (0.008) (0.008)
encoder)



0.9472

(0.007)


0.9387

(0.009)


0.9502

(0.006)



MHSADTI 0.9838
(0.001)



0.9832

(0.004)


0.4326

(0.010)


0.4337

(0.004)



0.9763

(0.009)



0.3427

(0.021)



DUD-E MHSADTI
(only gat)



0.9330

(0.002)


0.9429

(0.006)



MHSADTI 0.9330 0.2110 0.6676 0.4326 0.3383
(only gat) (0.002) (0.017) (0.018) (0.010) (0.012)

MHSADTI 0.9429 0.2625 0.6614 0.4337 0.3427
(only (0.006) (0.007) (0.013) (0.004) (0.021)
encoder)



0.9465
(0.009)


0.2110

(0.017)


0.2625

(0.007)



0.9451

(0.005)


0.6676

(0.018)


0.6614

(0.013)



MHSADTI 0.9484

(0.001)



0.7124

(0.009)


0.5502
(0.015)


0.7832

(0.011)



0.4462

(0.010)


0.7383
(0.011)


0.8267

(0.006)



0.3451

(0.025)



0.7722

(0.025)



DrugBank MHSADTI
(only gat)



0.6964
(0.005)


0.8566

(0.013)



MHSADTI 0.6964 0.7087 0.5502 0.7383 0.6328
(only gat) (0.005) (0.005) (0.015) (0.011) (0.013)

MHSADTI 0.8566 0.7719 0.7832 0.8267 0.7722
(only (0.013) (0.014) (0.011) (0.006) (0.025)
encoder)



0.2462

(0.005)


0.7087
(0.005)


0.7719

(0.014)



MHSADTI 0.8628

(0.012)



0.7706

(0.009)



0.7918

(0.012)



0.8351

(0.007)



0.7836

(0.011)



multi-head self-attention mechanism. In this section, we
give a theoretical analysis. As depicted in Fig. 3, it shows
the information transfer process from the previous layer to
the next layer in the network. Each black dot represents a
vector of amino acid fragments. Different colored arrows
indicate different weights for the next layer to obtain information from different positions on the previous layer. In
extracting the amino acid sequence information of proteins,
the CNN structure is used in the baseline and a multi-head

self-attention mechanism is used in MHSADTI. First of all,
compared with LSTM, both CNN and MHSADTI have the
common advantage that they can perform parallel data
processing. From Fig. 3, when using the CNN structure,
limited by the size of the convolution window, each position
in the sequence can only fuse local features within the current position range, and these features do not contain the
contextual information of the sequence (the window size in
Fig. 3 is 3). In this study, the encoder structure based on
multi-head self-attention is used. The self-attention obtains
the weight and output of this layer through the similarity


Fig. 3. Information extraction diagram of CNN, self-attention,multi-head
self-attention.



recurrent – Oðn � d [2] Þ Oð1Þ OðnÞ


weights of the current position and different positions in
sequences, avoiding the lack of learning local information
of different positions in the sequence in CNN structure.
After adopting the multi-head method, MHSADTI can
obtain multiple layers of semantic information between different positions in the sequence, improving the robustness
and stability of the model. With the “Position Encoding”,
the encoder can effectively extract the long-dependent information from amino acid sequences which can not be
obtained by CNN or RNN.


4.2.3 Complexity Analysis of MHSADTI

It is undeniable that the introduction of the attention mechanism can make the model more complicated, especially selfattention. In this section, we analyze the complexity of
MHSADTI through two aspects: qualitative analysis [17]
and quantitative analysis.

(A) Qualitative Analysis. As shown in Table 5, we have
analyzed the computational complexity of the network
structure based on self-attention, multi-head self-attention,
convolution and recurrent. Different from MHSDATI, both
DeepCPI and DeepConv-DTI are based on convolution.
So MHSADTI takes more time than DeepCPI, DeepConvDTI because of its attention mechanism. For an amino
acid sequence of length n, it can be vectorized into an
embedding matrix A 2 R [n][�][d] as input to models. self-attention includes three operation steps (similarity calculation,
softmax and weighted average) [17]. Given A 2 R [n][�][d], we
can get similarity matrix Smi 2 R [n][�][n] whose elements represent the similarity between different positions. The complexity of similarity calculation is n; dð Þ � ðd; nÞ ¼ O nð [2] � dÞ.
The weighted average is represented as the product of the
similarity matrix and the embedding matrix and its complexity is ðn; nÞ � ðn; dÞ ¼ O nð [2] � dÞ. As a result, the total
complexity of self-attention is O nð [2] � dÞ and the complexity
of multi-head self-attention that contains m self-attention is
O nð [2] � d � mÞ. For a convolutional network with a local perception window size of k, each convolution operation is a
matrix multiplication of size ðn; dÞ and ðd; kÞ, its complexity
is n; dð Þ � ðd; kÞ ¼ O kð � n � d [2] Þ. Therefore, the computational
complexity of the RNN structure at each time step is

ðn; dÞ � ðd; 1Þ ¼ O nð � d [2] Þ.
(B) Quantitative Analysis. In order to have a clear understanding of the running time of the model, we have quantitatively counted the average running time of each model



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


2216 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 4, JULY/AUGUST 2022


TABLE 6
The Average Time Consumption (in seconds) on Extracting the Feature Vector of Drugs and Proteins, Running the Entire

Model for Each Epoch


Model Drug Protein All


human C.elegans DUD-E DrugBank human C.elegans DUD-E DrugBank human C.elegans DUD-E DrugBank


DeepConv-DTI 2.52(0.02) 3.06(0.06) 15.4(0.64) 25.0(0.71) 7.72(0.07) 9.40(0.20) 47.4(1.97) 59.0(2.65) 42.59(0.47) 52.07(1.59) 260.8(13.1) 410.5(13.8)
DeepCPI 5.37(0.09) 4.82(0.07) 17.1(2.31) 29.6(0.59) 6.58(0.11) 6.99(0.11) 24.1(2.80) 41.1(0.90) 258.6(0.65) 266.1(0.42) 735.7(11.6) 1241(4.65)
MHSADTI 10.7(1.09) 10.8(0.54) 51.3(0.98) 59.5(0.63) 11.9(0.79) 13.0(0.68) 59.6(1.04) 68.7(0.81) 111.3(6.97) 117.3(4.73) 567.7(7.63) 615.1(6.91)



during 200 epochs. The running time of each epoch represents the time consumption required for the model to complete the training of all data once. In the experiment, we
ensure that all controllable environmental factors (early
stop mechanism, parallelization degree and so on) are the
same. As shown in Table 6, each column corresponding to
“Drug” and “Protein” represents the time (in seconds) taken
for the forward propagation of the modules in the model to
extract the feature vectors of drugs and proteins. We can see
from Table 6 that MHSADTI takes more time due to the use
of GAT based on the attention mechanism in the “Drug”
column. Similarly, MHSADTI also takes more time due to
multi-head self-attention for extracting feature vectors of
proteins in “Protein” column. In summary, in the process of



forward propagation, the feature extraction of drugs based
on GAT and the feature extraction of proteins based on the
encoder take more time and it is consistent with the qualitative analysis of Table 5. The “All” column represents the
sum time of forward propagation and backward propagation. The backward propagation contains a complicated
process of continuous derivative function. Table 6 indicates
that MHSADTI can take less time than DeepCPI in an average epoch. It seems inconsistent with the theoretical analysis in Table 5, and the reason may be that the backward
propagation of the deep learning-based method exists
unpredictability. Considering the improvement of prediction performance, it should be acceptable for MHSADTI to
take more time than DeepConv-DTI.



Fig. 4. The complex of imatinib and Tyrosine-protein kinase SYK (PDB ID:1XBB) [(a), (b)], aspirin and Phospholipase A2 (PDB ID: 1TGM)[(c), (d)].
(a) and (c) are visualizations of DTI with attention weights of mulit-head self-attention, the regions in proteins with high weight values are highlighted
in red and they are predicted correctly. (b) and (d) are visualizations of DTI with attention weights of drug-protein attention, the regions in protein with
high weight values are highlighted in red and they are correctly predicted as binding sites.


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


CHENG ET AL.: DRUG-TARGET INTERACTION PREDICTION USING MULTI-HEAD SELF-ATTENTION AND GRAPH ATTENTION NETWORK 2217



4.2.4 Case Study for Interpretability

In this study, while giving effective improvement results,
the interpretability of the learning content of the two selfattention models is visualized. To demonstrate this, those
regions with high weight values calculated from the two
attention models are mapped onto a known 3D protein
structure. Fig. 4 shows the complex of imatinib and Tyrosine protein kinase SYK (PDB ID: 1XBB) and the complex of
aspirin and Phospholipase A2 (PDB ID: 1TGM).

(a) and (c) shown in Fig. 4 are the visualization results of
the high-weight parts learned by the multi-head self-attention
mechanism. The black fragment is a known binding site. As
explained by Equations (12), we use the representation vectors of the query and key-value pairs in the encoder to calculate the weight of the remaining positions of the amino acid
sequence to its feature vector. The yellow fragments are the
known binding sites and should have a high weight for the
representation of the black fragment. The red fragments are
the high-weight areas and the known binding positions that
we correctly predicted through the calculated weight value.
The green fragments are predicted locations that have a high
weight for black fragments but are not known binding sites.
Through this way of calculating the self-attention, it can be
found which fragments have a strong connection with each
other, and the strong or weak connections between such fragments can effectively help the discovery of drugs.

(b) and (d) shown in Fig. 4 are the visualization results of
the high-weight areas learned by the drug-protein attention
mechanism. In the 3D structure of proteins, the part that is
not a binding site while predicted to be a binding site is
highlighted in green. The parts marked as yellow are true
binding sites while not giving a high weight score. The red
part is the position with a high weight score predicted by
the algorithm and the real binding site for drug action. It
can be seen that the algorithm can effectively capture the
binding sites.


5 C ONCLUSION


Accurately predicting DTIs can greatly improve efficiency
and reduce cost during the drug development. Although
advanced studies have greatly improved the accuracy of
DTI predictions, numerous challenges remain in the prediction of DTIs.

In this study, we have proposed an end-to-end neural
network model (called MHSADTI) for predicting DTIs.
MHSADTI extracts characteristics of drug molecules and
proteins by GATs and encoder based on multi-head selfattention. We have assessed the performance of the proposed method by conducting multiple cross validation. The
computational experimental results show that MHSADTI
can obtain better performance than the baseline. More
importantly, the visualization of weight scores learned by
the multi-head self-attention in the method confirms the
effectiveness of this learning method.

Although MHSADTI has obtained valid experimental
results, it is clear that it still has shortcomings for being further
improved. First, in the data input to the model, we only use
the one-dimensional data representation in the structural
characteristic information of drugs and proteins (SMILES and
amino acid sequences). However, drugs and proteins actually



have very complicated spatial structures. Although onedimensional data representation is easy to obtain, it does lose
the high-level characteristic information of many drugs and
proteins in prediction. Second, in the input of the model, we
only use a biological representation data for drugs and proteins. It is believed that integrating more comprehensive biological data to the deep learning model and eliminating
various noises are very helpful for more accurate predictions
of DTIs. We would consider integrating more useful data to
deep learning models in the future.


A CKNOWLEDGMENTS


This work supported in part by the National Natural Science Foundation of China under Grants 62072473, 61772552,
and 61832019, and in part by the NSFC-Zhejiang Joint Fund
for the Integration of Industrialization and Informatization
under Grant U1909208, in part by the 111 Project under
Grant B18059, and in part by the Hunan Provinvial Science
and Technology Program under Grant 2018WK4001.


R EFERENCES


[1] A. Ezzat, M. Wu, X.-L. Li, and C.-K. Kwoh, “Computational prediction of drug-target interactions using chemogenomic
approaches: An empirical survey,” Brief. Bioinf., vol. 20, pp. 1337–
1357, 2018.

[2] K. Bleakley and Y. Yamanishi, “Supervised prediction of drug–
target interactions using bipartite local models,” Bioinformatics,
vol. 25, no. 18, pp. 2397–2403, 2009.

[3] F. Cheng, Y. Zhou, J. Li, W. Li, G. Liu, and Y. Tang, “Prediction of
chemical–protein interactions: Multitarget-qsar versus computational chemogenomic methods,” Mol. BioSyst., vol. 8, no. 9, pp.
2373–2384, 2012.

[4] T. Van Laarhoven and E. Marchiori, “Predicting drug-target interactions for new drug compounds using a weighted nearest neighbor profile,” PLoS one, vol. 8, no. 6, 2013, Art. no. e66952.

[5] Y. Wang and J. Zeng, “Predicting drug-target interactions using
restricted boltzmann machines,” Bioinformatics, vol. 29, no. 13,
pp. i126–i134, 2013.

[6] Q. Yuan, J. Gao, D. Wu, S. Zhang, H. Mamitsuka, and S. Zhu,
“Druge-rank: Improving drug–target interaction prediction of
new candidate drugs or targets by ensemble learning to rank,”
Bioinformatics, vol. 32, no. 12, pp. i18–i27, 2016.

[7] R. S. Olayan, H. Ashoor, and V. B. Bajic, “DDR: Efficient computational method to predict drug–target interactions using graph
mining and machine learning approaches,” Bioinformatics, vol. 34,
no. 7, pp. 1164–1173, 2017.

[8] I. Lee and H. Nam, “Identification of drug-target interaction by a
random walk with restart method on an interactome network,”

[9] BMC bioinf.S. K. Mohamed, V. Nov, vol. 19, no. 8, 2018, Art. no. 208.�a�cek, and A. Nounu, “Discovering protein
drug targets using knowledge graph embeddings,” Bioinformatics,
vol. 36, no. 2, pp. 603–610, 2020.

[10] M. Bredel and E. Jacoby, “Chemogenomics: An emerging strategy

for rapid target and drug discovery,” Nat. Rev. Genet., vol. 5, no. 4,
pp. 262–275, 2004.

[11] L. Wang et al., “A computational-based method for predicting

drug–target interactions by using stacked autoencoder deep neural network,” J. Comput. Biol., vol. 25, no. 3, pp. 361–373, 2018.

[12] C. Lin, S. Ni, Y. Liang, X. Zeng, and X. Liu, “Learning to predict

drug target interaction from missing not at random labels,” IEEE
Trans. Nanobiosci., vol. 18, no. 3, pp. 353–359, Jul. 2019.

[13] M. Tsubaki, K. Tomii, and J. Sese, “Compound–protein interaction

prediction with end-to-end learning of neural networks for graphs
and sequences,” Bioinformatics, vol. 35, no. 2, pp. 309–318, 2018.

[14] K. Y. Gao, A. Fokoue, H. Luo, A. Iyengar, S. Dey, and P. Zhang,

“Interpretable drug target prediction using deep neural representation,” in Proc. 27th Int. Joint Conf. Artif. Int., 2018, pp. 3371–3377.

[15] X. Zheng, S. He, X. Song, Z. Zhang, and X. Bo, “DTI-RCNN: New

efficient hybrid neural network model to predict drug–target interactions,” in Proc. Int. Conf. Artif. Neural Netw., 2018, pp. 104–114.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


2218 IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS, VOL. 19, NO. 4, JULY/AUGUST 2022




[16] S. Zheng, Y. Li, S. Chen, J. Xu, and Y. Yang, “Predicting drug pro
tein interaction using quasi-visual question answering system,”
Nat. Mach. Int., vol. 2, pp. 134–140, 2020.

[17] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural

[18] P. VeliInf. Process. Syst.�ckovi�c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and, 2017, pp. 5998–6008.

Y. Bengio, “Graph attention networks,” 2017, arXiv:1710.10903.

[19] H. Ding, I. Takigawa, H. Mamitsuka, and S. Zhu, “Similarity-based

machine learning methods for predicting drug–target interactions:
A brief review,” Brief. Bioinf., vol. 15, no. 5, pp. 734–747, 2013.

[20] M. Hamanaka et al., “CGBVS-DNN: Prediction of compound-pro
tein interactions based on deep learning,” Mol. Inf., vol. 36, no. 1–
2, 2017, Art. no. 1600045.

[21] F. Wan and J. Zeng, “Deep learning with feature embedding for

compound-protein interaction prediction,” BioRxiv, p. 086033,
Jan. 2016.

[22] H. Liu, J. Sun, J. Guan, J. Zheng, and S. Zhou, “Improving com
pound–protein interaction prediction by building up highly credible negative samples,” Bioinformatics, vol. 31, no. 12, pp. i221–i229,
2015.

[23] I. Wallach, M. Dzamba, and A. Heifets, “AtomNet: A deep convo
lutional neural network for bioactivity prediction in structurebased drug discovery,” 2015, arXiv:1510.02855.

[24] T. N. Kipf and M. Welling, “Semi-supervised classification with

[25] J. Klicpera, A. Bojchevski, and S. Ggraph convolutional networks,” 2016,unnemann, “Predict then prop-€ arXiv:1609.02907.

agate: Graph neural networks meet personalized pagerank,” 2018,
arXiv:1810.05997.

[26] X. Wang et al., “Heterogeneous graph attention network,” in Proc.

World Wide Web Conf ACM, 2019, pp. 2022–2032.

[27] K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are

graph neural networks?,” 2018, arXiv:1810.00826.

[28] F. Costa and K. De Grave, “Fast neighborhood subgraph pair
wise distance kernel,” in Proc. 26th Int. Conf. Mach. Learn. 2010,
pp. 255–262.

[29] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers:

Surpassing human-level performance on imagenet classification,”
in Proc. IEEE Int. Conf. Comput. Vis., 2015, pp. 1026–1034.

[30] R. Paulus, C. Xiong, and R. Socher, “A deep reinforced model for

abstractive summarization,” 2017, arXiv:1705.04304.

[31] Z. Lin et al., “A structured self-attentive sentence embedding,”

2017, arXiv: 1703.03130.

[32] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Sori
cut, “Albert: A lite bert for self-supervised learning of language
representations,” 2019, arXiv:1909.11942.

[33] Q.-W. Dong, X.-l. Wang, and L. Lin, “Application of latent seman
tic analysis to protein remote homology detection,” Bioinformatics,
vol. 22, no. 3, pp. 285–290, 2005.

[34] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for

image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2016, pp. 770–778.

[35] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” 2016,

arXiv:1607.06450.

[36] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation

by jointly learning to align and translate,” 2014, arXiv:1409.0473.

[37] I. Lee, J. Keum, and H. Nam, “DeepConv-DTI: Prediction of drug
target interactions via deep learning with convolution on protein
sequences,” PLoS Comput. Biol., vol. 15, no. 6, 2019, Art. no.
e1007129.

[38] H. Ozt [€] urk, A.€ Ozg [€] ur, and E. Ozkirimli, “DeepDTA: Deep drug–€

target binding affinity prediction,” Bioinformatics, vol. 34, no. 17,
pp. i821–i829, 2018.

[39] T. Nguyen, H. Le, and S. Venkatesh, “GraphDTA: Prediction of

drug–target binding affinity using graph convolutional networks,”
BioRxiv, p. 684662, 2019.

[40] X. Glorot and Y. Bengio, “Understanding the difficulty of training

deep feedforward neural networks,” in Proc. 13th Int. Conf. Artif.
Int. Statist., 2010, pp. 249–256.

[41] D. P. Kingma and J. Ba, “Adam: A method for stochastic opti
mization,” 2014, arXiv:1412.6980.

[42] B. Kuhlman and P. Bradley, “Advances in protein structure pre
diction and design,” Nat. Rev. Mol. Cell Biol., vol. 20, no. 11,
pp. 681–697, 2019.

[43] A. W. Senior et al., “Improved protein structure prediction using

potentials from deep learning,” Nature, vol. 577, no. 7792, pp. 706–
710, 2020.




[44] Z. Xia, L.-Y. Wu, X. Zhou, and S. T. Wong, “Semi-supervised

drug-protein interaction prediction from heterogeneous biological
spaces,” in Proc. BMC Syst. Biol., 2010, pp. 1–16.

[45] T. van Laarhoven, S. B. Nabuurs, and E. Marchiori, “Gaussian

interaction profile kernels for predicting drug–target interaction,”
Bioinformatics, vol. 27, no. 21, pp. 3036–3043, 2011.


Zhongjian Cheng is currently working toward
the graduation degree with the School of Computer Science and Technology, Central South
University, Changsha, China. His research interests include bioinformatics and data mining.


Cheng Yan received the PhD degree in computer
science and technology from Central South University, Changsha, China, in 2018. His research interests include bioinformatics and machine learning.


Fang-xiang Wu (Senior Member, IEEE) received
the BSc and the MSc degrees in applied mathematics, from the Dalian University of Technology,
Dalian, China, in 1990 and 1993, respectively, the
first PhD degree in control theory and its applications from Northwestern Polytechnical University,
Xi’an, China, in 1998, and the second PhD
degree in biomedical engineering from the University of Saskatchewan, Saskatoon, SK, Canada, in 2004. From 2004 to 2005, he was a
postdoctoral fellow with the Laval University Medical Research Center,
Quebec City, Canada. He is currently the professor with the College of
Engineering and the Department of Computer Science, University of
Saskatchewan. His research interests include artificial intelligence,
machine or deep learning, computational biology, health informatics,
medical image analytics, and complex network analytics. He is the editorial board member of several international journals, the guest editor of
numerous international journals, and the program committee chair or
member of many international conferences.


Jianxin Wang (Senior Member, IEEE) received
the BS and MS degrees in computer science and
application from the Central South University of
Technology, China, and the PhD degree in computer science and technology from Central South
University, China. He is currently, the dean and a
professor with the School of Computer Science
and Engineering, Central South University,
Changsha,China, and also a leader with Hunan
Provincial Key Lab on Bioinformatics, Central
South University, Changsha, China. He has authored and coauthored more than 200 papers in various International journals and refereed conferences. His research interests include algorithm
analysis and optimization, parameterized algorithm, bioinformatics, and
computer network. He is a senior member of the IEEE. He has been on
numerous program committees and NSFC review panels, and was the
editor for several journals including IEEE/ACM Trans. Computational
Biology and Bioinformatics, International Journal of Bioinformatics
Research and Applications, Current Bioinformatics, and Current Protein
& Peptide Science, Protein & Peptide Letters.


" For more information on this or any other computing topic,
please visit our Digital Library at www.computer.org/csdl.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:25:29 UTC from IEEE Xplore. Restrictions apply.


