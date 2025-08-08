[Neural Networks 165 (2023) 94–105](https://doi.org/10.1016/j.neunet.2023.05.039)


[Contents lists available at ScienceDirect](https://www.elsevier.com/locate/neunet)

# Neural Networks


[journal homepage: www.elsevier.com/locate/neunet](http://www.elsevier.com/locate/neunet)

# 3D graph neural network with few-shot learning for predicting drug–drug interactions in scaffold-based cold start scenario


Qiujie Lv [a], Jun Zhou [a], Ziduo Yang [a], Haohuai He [a], Calvin Yu-Chian Chen [a] [,] [b] [,] [c] [,] [∗]


a _School of Intelligent Systems Engineering, Shenzhen Campus of Sun Yat-sen University, Shenzhen, Guangdong 518107, China_
b _Department of Medical Research, China Medical University Hospital, Taichung, 40447, Taiwan_
c _Department of Bioinformatics and Medical Engineering, Asia University, Taichung, 41354, Taiwan_



a r t i c l e i n f o


_Article history:_
Received 7 February 2023
Received in revised form 15 May 2023
Accepted 19 May 2023
Available online 25 May 2023


_Keywords:_
3D
Graph neural network
Cold start
Few-shot learning
Drug–drug interactions


**1. Introduction**



a b s t r a c t


Understanding drug–drug interactions (DDI) of new drugs is critical for minimizing unexpected adverse
drug reactions. The modeling of new drugs is called a cold start scenario. In this scenario, Only a few
structural information or physicochemical information about new drug is available. The 3D conformation of drug molecules usually plays a crucial role in chemical properties compared to the 2D structure.
3D graph network with few-shot learning is a promising solution. However, the 3D heterogeneity
of drug molecules and the discretization of atomic distributions lead to spatial confusion in fewshot learning. Here, we propose a 3D graph neural network with few-shot learning, Meta3D-DDI, to
predict DDI events in cold start scenario. The 3DGNN ensures rotation and translation invariance by
calculating atomic pairwise distances, and incorporates 3D structure and distance information in the
information aggregation stage. The continuous filter interaction module can continuously simulate the
filter to obtain the interaction between the target atom and other atoms. Meta3D-DDI further develops
a FSL strategy based on bilevel optimization to transfer meta-knowledge for DDI prediction tasks
from existing drugs to new drugs. In addition, the existing cold start setting may cause the scaffold
structure information in the training set to leak into the test set. We design scaffold-based cold start
scenario to ensure that the drug scaffolds in the training set and test set do not overlap. The extensive
experiments demonstrate that our architecture achieves the SOTA performance for DDI prediction
under scaffold-based cold start scenario on two real-world datasets. The visual experiment shows that
Meta3D-DDI significantly improves the learning for DDI prediction of new drugs. We also demonstrate
how Meta3D-DDI can reduce the amount of data required to make meaningful DDI predictions.
© 2023 Elsevier Ltd. All rights reserved.



Drug–drug interaction (DDI) refers to the pharmacological action produced by the co-administration of two or more drugs.
When co-administered with another drug, the expected efficacy
of the drug may change significantly (Ryu, Kim, & Lee, 2018). The
survey shows that 67% of elderly Americans took 5 or more medications in 2010–2011, including prescription drugs, over-thecounter drugs, and dietary supplements (Qato, Wilder, Schumm,
Gillet, & Alexander, 2016), and significant increases in overall
prescription drug use and polypharmacy were observed (Kantor, Rehm, Haas, Chan, & Giovannucci, 2015). As the demand
of multiple drugs for disease treatment continues to increase,
understanding DDI is critical to minimize unexpected adverse


∗ Corresponding author at: School of Intelligent Systems Engineering, Shenzhen Campus of Sun Yat-sen University, Shenzhen, Guangdong 518107,
China.

_E-mail address:_ [chenyuchian@mail.sysu.edu.cn (C.Y. Chen).](mailto:chenyuchian@mail.sysu.edu.cn)


[https://doi.org/10.1016/j.neunet.2023.05.039](https://doi.org/10.1016/j.neunet.2023.05.039)
0893-6080/ © 2023 Elsevier Ltd. All rights reserved.



drug reactions and to maximize synergistic benefits when treating disease (Lv, Chen, He et al., 2023; Tatonetti, Ye, Daneshjou, &
Altman, 2012). However, the identification of DDI is highly limited because the experimental testing on a large number of drug
combinations is highly expensive and nearly impossible (Hussain
et al., 2020; Ren et al., 2022). The computer-aided DDI prediction
methods can be used as an alternative to alleviate this problem
due to the effective, fast, and low-cost (Lv et al., 2019).

Various machine learning methods for DDI prediction have
been developed, which can be further roughly classified into
feature-based methods (Deng et al., 2020; Ferdousi, Safdari, &
Omidi, 2017; Gottlieb, Stein, Oron, Ruppin, & Sharan, 2012; Zhang
et al., 2017), knowledge graph-based methods (Dai, Guo, Guo,
& Eickhoff, 2021; Huang, Xiao, Hoang, Glass, & Sun, 2020; Liu,
Huang, Qiu, Chen, & Zhang, 2019; Sridhar, Fakhraei, & Getoor,
2016; Zhang, Wang, Hu, & Sorrentino, 2015), graph-based methods (Nyamabo, Yu, Liu, & Shi, 2022; Nyamabo, Yu, & Shi, 2021;
Xu, Wang, Chen, Tao, & Zhao, 2019; Yang, Zhong, Lv, & Chen,
2022a; Yu, Zhao, & Shi, 2022), and text-based methods (Fatehifar


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



& Karshenas, 2021; He, Chen, & Yu-Chian Chen, 2022; Hong
et al., 2020; Huang et al., 2022; Zhu, Li, Lu, Zhou, & Qin, 2020).
These methods showed significant improvement on DDI prediction. However, the previous studies for DDI prediction extracted
2D structural information from molecular graphs constructed by
drugs SMILES (Weininger, 1988), such as which atom/group could
be connected to a double bond. These methods lack the ability
to learn the 3D structure of molecules, while the 3D-spatial
distribution of atoms is crucial for determining the atomic states
and interatomic forces (Liu, Wang, Liu et al., 2022; Zhang, Zhou,
Wu, & Gao, 2021). The knowledge of molecular 3D conformation
plays an important role in determining the physical, chemical and
biological activities of molecules (Méndez-Lucio, Ahmad, del RioChanona, & Wegner, 2021). There are stereoisomeric molecules
with the same molecular formula, where the atoms or groups
are connected to each other in the same order, but arranged
differently in space. Cis-platin is used to treat many cancers,
whereas trans-platin has no cytotoxic activity (Fang et al., 2022).

Existing methods generally use cross-validation to divide the
dataset according to a certain ratio (Deac, Huang, Veličković, Liò,
& Tang, 2019; Veličković et al., 2017). This setting is known as the
warm start setting or transductive setting (Sagawa & Hino, 2023).
Published baseline models for DDI prediction, SSI-DDI (Nyamabo
et al., 2021), SA–DDI (Yang et al., 2022a), etc., have achieved
state-of-the-art performance under warm start setting and can
serve as a powerful tool for DDI prediction. However, splitting the
dataset with random cross-validation approach results in training
set and test set sharing the same drug. This means that the model
may have seen all drugs. In testing, the proposed model simply
inferred interactions between drugs that have been seen. And the
drug information in the training set may leak into the test set,
causing the model to produce overly optimistic results (AltaeTran, Ramsundar, Pappu, & Pande, 2017; Lv, Chen, Yang et al.,
2023). Therefore, we need to ensure that the drugs in the test set
do not appear in the training set. This setting can be considered
as the cold start setting or inductive setting (Han, Kim, Han, Lee,
& Hong, 2023; Yang, Zhong, Zhao, & Chen, 2022b). Cold start
setting can evaluate the effectiveness of a model in predicting
interactions between new drugs and known drugs, or between
two new drugs (Cai et al., 2020). This corresponds to the fact
that researchers pay more attention to the accuracy of the model
prediction for new drugs in actual drug discovery (Dewulf, Stock,
& De Baets, 2021; Li, Zhang, Li, & Fu, 2020; Liu, Wang, Yu et al.,
2022; Shi, Mao, Yu, & Yiu, 2019). The cold start setting is a more
challenging evaluation scenario.

Few-shot learning (FSL) has demonstrated great potential in
computer vision, natural language processing, etc (Gao, Luo, Yang,
& Zhang, 2022; Guo et al., 2023; Hospedales, Antoniou, Micaelli,
& Storkey, 2021; Ju, Liu et al., 2023; Zhao, Lan, Huang, Ren, &
Yang, 2022). Each training task has few labeled and unlabeled
data to simulate test scenario (Vanschoren, 2018). FSL learns a
learning experience by systematically observing how a model
performs on a wide range of learning tasks, and then applies this
experience to test scenario with only few samples (Li, Xie, Zhang,
& Shi, 2023). The application of FSL coincides with the scenario
of new drug development. The DDI data of existing drug pairs are
relatively rich. However, for DDI prediction of new drugs, i.e. cold
start setting, there are usually only a few samples available. FSL
learns the meta-knowledge contained in the existing DDI data
of drug pairs, and then the model can be quickly transferred to
the DDI prediction of new drugs after several internal iterations.
The DDI prediction task in the cold start setting is basically a
FSL task. 3D graph neural network (3DGNN) has wide flexibility
and effectiveness in learning 3D structural data. Therefore, we
combined 3DGNN with FSL, and the proposed model has the
potential to achieve latent DDI prediction in cold start scenario.



However, 3DGNN with few-shot learning in DDI prediction
is still challenging. This challenge derives from the impact of
the complex 3D spatial structure of drug molecules (Schütt,
Arbabzadah, Chmiela, Müller and Tkatchenko, 2017; Schütt,
Sauceda, Kindermans, Tkatchenko, & Müller, 2018). This impact
can lead to spatial confusion in few-shot learning of DDI events.
Spatial confusion refers to confusion in the relative positions and
sequences between atoms in the drug molecule. When the local
structure of the drug molecule is rotated or translated, inputs,
such as the 3D coordinates of atoms and the relative positions
between atoms, also change in order and structure. This change
causes the corresponding change on the expected output of the
DDI prediction model, while the chemical feature of the drug and
its DDI events do not change (Schütt, Kindermans et al., 2017). In
cold start scenario, this spatial confusion exacerbates over-fitting
and reduces the adaptability of 3DGNN with few-shot learning.

Furthermore, only few experiments (Nyamabo et al., 2022,
2021; Yang et al., 2022a; Yu et al., 2022) were performed under
both warm and cold start conditions. Their cold start setting
keeps the drugs in the test set inaccessible in the training set.
We refer to this evaluation setting as drug-based cold start.
However, although the training set and test set are different
drugs in drug-based cold start setting, they may have the same or
similar scaffold. Scaffold is the core structure or basic framework
that constitutes the compound molecules, which can be modified by adding or replacing different functional groups to form
different compounds (Bajorath, 2017; Bemis & Murcko, 1996).
The different molecules originating from the same scaffold may
have similar properties. The distribution of molecular data points
exhibits an aggregation phenomenon in the same scaffold. Drugbased cold start setting does not consider scaffold structural
similarities in chemical space, which still leads to information
leakage between training data and test data under cold start
scenario. Drug-based cold start setting may cause the model to
produce overly optimistic results on the test set.

In this paper, we proposed a 3D graph network with few-shot
learning, Meta3D-DDI, to predict potential DDI events in scaffoldbased cold start scenario. Specifically, the proposed continuous
filter interaction module (CFIM) simulates the interactions between atoms at arbitrary positions in the drug, and introduces 3D
structure and distance information in the information aggregation
stage. To ensure the invariant description of the atomic distribution, the radial basis function calculates the pairwise distance
to predict the rotation, translation and indexing invariance of
the drug molecule. These designs improve the 3DGNN perception
ability on medicinal chemical environment and connectivity to
effectively alleviate the spatial confusion problem. In addition,
we propose a bilevel optimization strategy to selectively update parameters on different tasks. This strategy captures the
common meta-knowledge on existing drugs, and transfers the
meta-knowledge from existing drugs to new drugs. Finally, we
design scaffold-based cold start scenario, which prevents the
leakage of scaffold structure information to the test set. Scaffoldbased cold start scenario is defined as the scaffold of all drugs
in the test set is not visible in the training set. This is a more
realistic and challenging assessment scenario. The Meta3D-DDI
achieves the SOTA performance on multiple real datasets for DDI
prediction. Extensive experiments demonstrate the Meta3D-DDI
is a tool with great potential for low-data DDI prediction. To
summarize, the main contributions are described as follows:

1. We develop a competitive tool for DDI prediction in cold
start scenario. We also demonstrate how Meta3D-DDI can reduce
the amount of data required to make meaningful DDI predictions.

2. We design a 3D graph neural network, which utilizes continuous filters to continuously model on discretized atoms. It
constructs rotational and translational invariance to learn drug



95


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



representations that obey the fundamental symmetries of atomic
systems.

3. We propose a bilevel optimization strategy that captures
the common meta-knowledge on existing drugs, and transfers
the meta-knowledge from existing drugs to new drugs for DDI
prediction tasks.

4. We design scaffold-based cold start scenario, including the
cold start setting for a single drug and a pair of drugs, to correctly
assess whether the generalization performance of the DDI model
has been improved.


**2. Related work**


In this section, we review existing work including featurebased methods, knowledge graph-based methods, graph-based
methods, and text-based methods.

**Feature-based methods** typically rely on feature vectors
hand-crafted by domain experts or drug similarity assumptions.
Drug similarity hypothesis means that two similar drugs tend to
have common interactions. Zhang et al. (2017) collected multisource data such as substructure, target, enzyme and transporter,
and predicted DDI events through neighbor recommendation,
random walk and matrix perturbation. Gottlieb et al. (2012) calculated the similarity between the query drug pair and the drug
pair with known interactions, and used seven different drug–
drug similarity measures to predict DDI. Ferdousi et al. (2017)
utilized 5 binary vectors, including: carrier, transporter, enzyme,
target, and comprehensive vector, and then applied the selected
similarities and thresholds to identify DDI. Deng et al. (2020) used
four drug characteristics of chemical substructure, target, enzyme
and pathway to construct DNN sub-models, and then adopted
a joint DNN framework to learn modality representation and
predict DDI events. The disadvantage of feature-based methods
is that similar drugs do not necessarily have the same biological
activity (Nyamabo et al., 2021), and additional features, such as
targets, enzymes, pathways, transporters, etc., are not always
available for all drugs (Zhang, Leng, & Liu, 2020). Features of some
drugs, such as side effects, pathways, etc., are not available in
the early stages of drug development (Lv, Chen, Zhao, Zhong, &
Yu-Chian Chen, 2021).

**Knowledge graph-based methods** represent biomedical data
as graphs and use different graph-specific methods, such as label
propagation, matrix factorization, and graph auto-encoders, to
infer DDI. Zhang et al. (2015) used a label propagation approach
for DDI prediction on a high-order similarity-based network, and
proposed an integrative label propagation framework by integrating drug information from multiple sources. Liu et al. (2019)
used a multi-modal deep auto-encoder to learn a unified representation of drugs from multiple drug feature networks, then
adopted several operators on the learned drug embeddings to
represent drug–drug pairs for DDI prediction. Sridhar et al. (2016)
proposed a probabilistic approach for DDI prediction from a network of multiple drug–drug similarities and known interactions.
Dai et al. (2021) introduced an adversarial auto-encoder based
on wasserstein distance and gumbel-softmax relaxation for DDI
prediction tasks. Knowledge graph-based methods improve the
performance of DDI prediction models with external biomedical
knowledge (Huang et al., 2020). In the early stages of drug development, there is usually only chemical structural information or
few biological information (Ju, Gu et al., 2023). Knowledge graphbased methods require additional external biomedical knowledge
to build knowledge networks, and are not suitable for new drug
development with few samples.

**Graph-based methods** have been developed and are further
improving DDI prediction task. Yang et al. (2022a) proposed a
substructure-aware graph neural network, a message passing



neural network equipped with a novel substructure attention
mechanism and a substructure–substructure interaction module
for DDI prediction. Nyamabo et al. (2021) proposed substructure–
substructure interaction–drug–drug interaction, which uses the
co-attention mechanism to identify pairwise interactions between the respective substructures of two drugs. Xu et al. (2019)
proposed a multi-resolution receptive field based graph neural network, which leverages different-sized local features and
models interaction during the procedure of feature extraction to
predict structured entity interactions. Nyamabo et al. (2022) proposed a message passing neural network, which considers bonds
as gates that control the flow of message passing of GNN, thereby
delimiting the substructures in a learnable way. Yu et al. (2022)
designed a substructure-aware tensor neural network to learns a
triplets tensor, which characterizes a substructure–substructure
interaction space.

**Text-based methods** aim to automatically extract DDI information from text, or combine text with drug structures to predict
drug interactions. Hong et al. (2020) used latent tree learning
and self-attention to better represent semantic and syntactic information inside sentences. Zhu et al. (2020) embedded BioBERT
into BiGRU layer to obtain the vector representation of sentences,
and used Doc2Vec-encoded drug description documents as external knowledge. Huang et al. (2022) applied pre-trained BioBERT,
multihead self-attention mechanism and packed BiGRU to fuse
multiple semantic information, and combined with pre-trained
BioGPT-2 to generate meaningful text. In addition, the combination of drug text information and molecular structure information
is effective for improving performance of the model. He et al.
(2022) used 3D GNN and pre-trained text attention mechanism
to fuse text features, 3D structures and position information of
drug molecular entities. Asada, Miwa, and Sasaki (2021) used pretrained SciBERT to process input text and used GNN to learn the
structural representation of drugs.


**3. Methods**


_3.1. Problem formulations_


Our goal is to build a model that predicts potential DDI events
under cold start scenario, especially for new drug interactions
in new scaffold. For the drug pair ( _d_ _x_, _d_ _y_ ) of DDI query and an
interaction type _r_, we want to find a model _f_ _θ_ parameterized
by _θ_ to predict the probability of this interaction type between
two drugs. This DDI prediction task is considered as a binary
classification task.

Usually, the distribution of molecular data in different scaffolds is significantly different. In the scaffold-based cold start
setting, the model must be adapted to the drug molecule of the
new scaffold. New scaffold not seen in training can be seen by
the classifier as a new class, and data on new drugs give only
a few examples. According to the empirical risk minimization
theory (Wang, Yao, Kwok, & Ni, 2020), when only few labeled data
is available, the empirical risk _R_ ( _h_ _I_ ) is far from the satisfactory approximation of the expected risk _R_ ( _h_ ), and the obtained empirical
risk minimizer is unreliable. _h_ is the empirical risk minimizer, and
_I_ represents few labeled data.

The empirical risk minimization is closely related to sample
complexity (Wang et al., 2020). Sample complexity refers to the
number of training samples required to minimize the loss of
empirical risk _R_ ( _h_ _I_ ). The high sample complexity leads to the
inferior empirical risk minimizer (Guo, Xu et al., 2021). Therefore,
the key of predicting potential DDI events in the scaffold-based
cold start scenario is that the model reduces the requirement
of sample complexity by a clever design, thereby making _R_ ( _h_ _I_ )
approximates to the best _R_ ( _h_ ) in Eq. (1).


E [ _R_ ( _h_ _I_ → _fe_ _w_ ) − _R_ ( _h_ ) ] = 0 (1)



96


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_


**Fig. 1.** Overview of the Meta3D-DDI for DDI prediction in scaffold-based cold start scenario.The atom types and coordinates of two drugs were used as the input of
3DGNN. The matrix of atom type is initialized by the atom embedding layer and fed into CFIM. The 3D coordinates of atoms are used for distance-based sorting and
coordinate transformation. The proposed CFIM uses two branches for feature extraction and interaction of atom type and position information respectively. CFIM
consists of atomic layers, cfconv layers and residual connections. The CFIM is repeated three times for a single drug branch.



_3.2. 3D graph neural networks_


The proposed GNN architecture based on 3D structures aims
to simulate stable systems of drug molecules and learn representations that follow the physical laws of molecules. It reflects the
invariance of atom rotation, translation and indexing. Fig. 1 is an
overview of our 3DGNN model.

The distribution of atomic positions in drug molecules is often
discretization. Molecular representations of 3D structures often
contain a lot of redundant grid points, and do not located on a
regular grid. Therefore, we use the CFIM to continuously model
atoms with uneven spacing, which can simulate the interaction of atoms at arbitrary position in the molecule. The CFIM
contains continuous filter convolution and stacked multi-layer
perceptron. The continuous filtered convolution is a generalization of the commonly used discrete convolution layer (Schütt,
Kindermans et al., 2017). Stacked multi-layer perceptron is used
to perform nonlinear transformation of atomic embedding for
reconfiguration and optimization of feature representation.


_3.2.1. The continuous filter interaction module_

The proposed continuous filter interaction module (CFIM) is
used to improve the representation obtained by analyzing the
local environment of the atoms. This block is able to update the
representation according to the radial environment of each atom,
as shown in Fig. 1. The input of 3DGNN is a drug pair ( _d_ _x_, _d_ _y_ ),
where the drug molecule _d_ can be uniquely described by a group
of atoms _Z_ = ( _z_ 1 _, . . .,_ _z_ _n_ ) and the position _R_ = ( _r_ 1 _, . . .,_ _r_ _n_ ). First,
the representation of the atom _z_ _i_ is randomly initialized by the
atom embedding layer _x_ [0] _i_ [=] _[ az]_ _[i]_ [, where] _[ a]_ [ is an embedding matrix]
of atom type. Next, the CFIM is used to simulate the interaction
between atoms during the information aggregation stage.

For a group of atomic feature representation _x_ _[l]_ = ( _x_ _[l]_ 1 _[, . . .,]_ _[ x]_ _n_ _[l]_ [)]
at locations _R_ = ( _r_ 1 _, . . .,_ _r_ _n_ ) in drug molecule, the continuousfilter convolutional (cfconv) maps atomic position to corresponding value of the filter. The output _x_ _[l]_ _i_ [+] [1] of the convolution layer at
position _r_ _i_ is calculated by Eq. (2).



where ◦ represents the element-wise multiplication, _j_ represents
the remaining atoms outside of atom _i_, _n_ represents the number
of atoms, and _l_ represents the current _l_ th cfconv layer. We introduced residual connection in the interaction module. The middle
of cfconv layer and full connection layer is shifted non-linearity
softplus _ssp_ ( _x_ ) = _ln_ (0 _._ 5 _e_ _[x]_ + 0 _._ 5). These designs ensure that we
obtain the smooth potential energy surface of molecules, and
combine the 3D position information and the interaction between

atoms.


_3.2.2. Rotational and translational invariance_

When the local structure or atomic distribution of a drug
molecule is translated and rotated, it is crucial to maintain the
same atomic sequence. To satisfy the invariance of the 3D structure of drug molecules, we need to further combine chemical
knowledge and constraints to learn the molecular representation
that follows the basic symmetries of atomic systems.

We use the radial basis function as a transformation to calculate the distance. Specifically, for the target atom _z_ _i_ and the
remaining atoms _z_ _j_ in the drug molecule, we first compute the
distance between _z_ _i_ and the other atoms _z_ _j_ (Eq. (3)). Next, we
sort all _z_ _j_ according to the distance. Let _z_ _j_ [1] and _z_ _j_ [2] denote the
positions of the closest and second closest atoms to _z_ _i_ . Then we
rotate _z_ _j_ [1] [to the positive] _[ z]_ [-axis and] _[ z]_ _j_ [2] [to the] _[ yz]_ [-plane according]
to the rotation formula (Eq. (4)) (Koks, 2006; Zhang et al., 2021).
Finally, we calculate the new coordinates of the atoms.


_e_ _k_ ( _r_ _i_ − _r_ _j_ ) = _exp_ ( − _γ_ ( ∥ _r_ _i_ − _r_ _j_ ∥− _µ_ _k_ ) [2] ) (3)

_z_ _i_ [′] [=] _[ R]_ [2] _[R]_ [1] [(] _[z]_ _[i]_ [ −] [▽] _[z]_ [)] _[T]_ (4)


where _µ_ _k_ is the center between the zero and the distance cutoff.
_γ_ is the grid spacing and scaling parameter, which determines
the resolution of the filter. - _z_ is the coordinate increment, which
transform the origin to the 3D coordinates of the target node
_z_ _i_ . _R_ 1 is transform matrix that rotates _z_ _i_ [1] to [ 0 _,_ 0 _,_ | _z_ _i_ [1] [|]] _[T]_ [ .] _[ R]_ [2] [ is]
transform matrix that rotates _z_ _i_ [2] to yz-plane. The target atom
and the nearest two atoms form a unique three-dimensional
coordinate system, and other nodes can be transformed by the
same translation and rotation. This way ensures that the same
three-dimensional coordinate descriptor can be obtained even if



_x_ _[l]_ _i_ [+] [1] = ( _X_ _[l]_ ∗ _W_ _[l]_ ) _i_ =



_n_
∑ _x_ _[l]_ _j_ [◦] _[W]_ _[ l]_ [(] _[r]_ _[i]_ [ −] _[r]_ _[j]_ [)] (2)

_j_ = 0



97


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_


**Algorithm 1** Pseudocode of Meta3D-DDI for DDI prediction in
scaffold-based cold start scenario.
**Require:** drug pair ( _d_ _x_ _,_ _d_ _y_ _,_ _r_ );
**Ensure:** 3DGNN parameters _θ_, learning rate _α, β_ ;

1: Randomly initialize _θ_ ;

2: **while** not done **do**

3: Sample batch of tasks { _T_ _tr_ _[i]_ _s_ _[,]_ _[ T]_ _tr_ _[ i]_ _q_ [}] _i_ _[N]_ = 1 [∼] _[T]_ _[train]_ [;]

4: **for all** _T_ _tr_ _s_ **do**

5: Support set: sample _K_ examples;

6: Evaluate ∇ _θ_ _L_ _T_ _itrs_ [(] _[f]_ _[θ]_ [) by] _[ r]_ _support_ [′] [=] _[ f]_ _[θ]_ [(] _[d]_ _[x]_ _[,]_ _[ d]_ _[y]_ [);]

7: Compute _θ_ _i_ [′] [=] _[ θ]_ [ −] _[α]_ [∇] _[θ]_ _[ L]_ _T_ _trs_ _[i]_ [(] _[f]_ _[θ]_ [);]

8: Query set: Sample _L_ examples;

9: Evaluate ∇ _θ_ _L_ _T_ _itrq_ [(] _[f]_ _[θ]_ [′] [) by] _[ r]_ _query_ [′] [=] _[ f]_ _[θ]_ [′] [(] _[d]_ _[x]_ _[,]_ _[ d]_ _[y]_ [);]

10: **end for**

11: Update:

_N_ _B_
12: _θ_ ← _θ_ − _β_ ∇ _θ_ ∑ _i_ = 1 ∑ _b_ = 0 _[w]_ _[b]_ _[ L]_ _T_ _trq_ _[i]_ [(] _[f]_ _[θ]_ _i_ [′] _[b]_ [(] _[d]_ _[x]_ _[,]_ _[ d]_ _[y]_ [)] _[,]_ _[ r]_ [);]

13: **end while**

14: Sample batch of tasks ( _T_ _te_ _[i]_ _s_ _[,]_ _[ T]_ _[ i]_ _te_ _q_ [)] [ ∼] _[T]_ _[test]_ [ ;]



15: **for all** _T_ _te_ _s_ **do**

16: Support set: Sample K examples;

17: // Similar to the training phase

18: Evaluate and Compute adapted parameters with gradient
descent;

19: Update: _θ_ _i_ [∗] [=] [ Adam] _[ (]_ _[L]_ _[t]_ _[, θ)]_ [;]

20: Query set: Sample L examples;

21: _r_ _query_ [′] [=] _[ f]_ _[θ]_ _i_ [∗] [(] _[d]_ _[x]_ _[,]_ _[ d]_ _[y]_ [)]

22: **end for**


the input drug structure is rotated or translated. Obviously, rotation and translation invariance are guaranteed by distance-based
sorting and coordinate transformation.


_3.2.3. Drug–drug interaction prediction_

After aggregation of atomic level information and calculation
of invariance for atomic feature representation, we concatenate
the features of two drugs and calculate the final prediction. The
DDI prediction of _d_ _x_ and _d_ _y_ with interactions of type _r_ is given by
the joint probability (Eq. (5)).



_p_ ( _d_ _x_ _,_ _d_ _y_ _,_ _r_ ) = _σ_ [(] **M** **r** ( _x_ _[out]_ _x_ ∈ _d_ _x_ ∥ _x_ _[out]_ _y_ ∈ _d_ _y_ ) [)]



**Fig. 2.** Problem definition of S1 (green dashed line) and S2 (blue dashed line)
in scaffold-based cold start for DDI prediction. S1 refers to the interaction
between a new drug and an existing drug with different scaffold. S2 refers to the
interaction between new drug pairs of two different scaffolds. The gray dashed
lines indicate the interactions between drug pairs in different or same scaffold
set in the training set, which is used to simulate the interaction of existing drug
pairs.


and test set do not overlap. In scaffold-based cold start scenario,
there are also two tasks, as shown in Fig. 2. Cold start for a single
drug refers to the scaffold of one drug in the drug pair used for
DDI query is not accessible in the training set, denoted as S1 task;
Cold start for a pair of drugs refers to the scaffolds of both drugs
in the drug pair are not accessible in the training set, denoted as
S2 task. Specifically, we first obtain the scaffold information of all
drugs, and merge the drugs belonging to the same scaffold into a
set. Then, the training set, validation set and test set are divided
according to the scaffolds. Finally, drug pairs are generated by
mapping the drug sets in each scaffold according to the original
drug tuple. And the training set, validation set, S1 and S2 test set
are generated respectively.

Our scaffold-based cold start setting further prevents scaffold
structural information of the drug from leaking into the test set.
It has the ability to measure whether the existing DDI prediction
models actually improve generalization performance, and is a
more realistic and more challenging evaluation setting.


_3.4. Few-shot learning_


Few-shot learning (FSL) means learning a meta-knowledge
by systematically observing how the model performs in a wide
range of learning tasks (Vanschoren, 2018; Wang et al., 2020).
This meta-knowledge is especially helpful for solving few-shot
learning problems. FSL usually adopts episodes training strategy,
which is divided into two processes: meta-training and metatesting. Our FSL strategy learns the common knowledge in the
rich drug molecular data of the training set, and transfers the
knowledge to the prediction of potential DDI events of new drugs
under the scaffold-based cold start scenario. Algorithm 1 shows
the complete algorithm flow of FSL.


_3.4.1. Task definition_

FSL emphasizes the concept of task space, where both tasks
and data need to be sampled. In FSL for DDI prediction task, we
need to construct meta tasks in training set, validation set, S1 test
set, and S2 test set respectively.



_L_ = − [1]

| _M_ |



_N_
∑


_i_ = 1



( _y_ _i_ log( _p_ _i_ ) + (1 − _y_ _i_ ) log(1 − _p_ [′] _i_ [)] [)]



(5)


(6)



where _σ_ is the sigmoid function, ∥ represents concatenation, and
_M_ _r_ a learnable matrix representation of the interaction type _r_ . _x_ _[out]_

is the feature representation output by the last atomic layer. The
loss function is to minimize the cross-entropy, as Eq. (6). _p_ _i_ is used
for positive samples, _p_ [′] _i_ [is used for related negative samples.]


_3.3. The scaffold-based cold start setting_


The drug-based cold start scenario refers to the model needs
to make DDI interaction predictions for new drugs that are not in
the training set. The existing cold start setting ensures that there
are no structurally overlapping drugs in the test and training sets.
However, although the structures of the two drug molecules in
the training set and the test set are different, they may belong
to the same scaffold. The chemical properties of drug molecules
in the same scaffold may be the same or similar. Similar to the
warm start setting, the drug-based cold start setting cause some
scaffold structure information to leak into the test set.

To address the above problems, we design scaffold-based cold
start scenario to ensure that the drug scaffolds in the training set



98


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



Scaffold-based cold start setting is introduced in Section 3.3.
We obtained four subsets according to the scaffold division, training set, validation set, S1 test set, and S2 test set. Some scaffold
sets contain only a few drug molecules, or there is a similarity
and containment relationship between the scaffolds, so these
scaffold sets need to be merged together. Therefore, we used
Jaccard distance on binarized ECFP4 features to measure the
distance between any two drugs in each subset (Mayr et al.,
2018). A set of drugs that are close in distance is aggregated into
a meta task. Finally, we get 55, 14, 36, and 10 meta tasks in the
training set, validation set, S1 test set and S2 test set on DrugBank
dataset, respectively. We implement data aggregation function by
RDKit (Landrum et al., 2013) and SciPy (Virtanen et al., 2020), and
the number of meta tasks is the number of aggregated clusters.
Such task setting meets the requirements of rich tasks in the meta
training stage, thus extending the application of FSL algorithm to
DDI prediction tasks.


_3.4.2. Meta-training_

We use bilevel optimization (Hospedales et al., 2021; Wang
et al., 2020) to selectively update the parameters in each task to
obtain a suitable empirical risk minimizer. The outer optimization
computes the minimum loss for all training tasks to learn a
general meta-knowledge _w_, which pays more attention to the
future potential of the proposed model, as shown in Eq. (7). The
inner optimization learns the specificity of a single task based on
the meta-knowledge _w_ defined by the outer optimization, which
enables the model to quickly adapt to the DDI prediction task of
new drugs, as shown in Eq. (8).



calculate the gradient on the support set and update the model
parameters, as shown in Eq. (11).



1
_L_ _t_ =
| _M_ |



∑

( _d_ _x_ _,_ _D_ _y_ _,_ _r_ ) ∼ _D_ _[support]_ _test_ [(] _[i]_ [)]



( _y_ _i_ log( _p_ _i_ ) + (1 − _y_ _i_ ) log(1 − _p_ [′] _i_ [)] [)]



(11)



_w_ = arg min

_w_



_M_
∑ _L_ _meta_ ( _f_ ( _d_ _x_ _,_ _d_ _y_ ; _θ_ _[i]_ ( _w_ )) _, w_ ) (7)

_i_ = 1 ( _d_ _x_ _,_ _d_ _y_ _,_ _r_ ) ∼ _D_ _[query]_ [(] _[i]_ [)]



_s_ _._ _t_ _. θ_ _[i]_ ( _w_ ) = arg min _L_ _task_ ( _f_ ( _d_ _x_ _,_ _d_ _y_ ; _θ_ ) _, w_ ) (8)
_θ_ ( _d_ _x_ _,_ _d_ _y_ _,_ _r_ ) ∼ _D_ _[support]_ [(] _[i]_ [)]


where _L_ _meta_ and _L_ _task_ refer to the outer and inner objectives
respectively, and _i_ refers to the number of tasks. Specifically, for
the N-way K-shot setting, _N_ represents the number of classes in
an episode, and _K_ is the number of samples in the support set for
each class. We first construct multiple episodes for training, each
episodes containing a support set and a query set. We randomly
selected _N_ scaffold prediction tasks from the 55 training tasks
constructed in the previous section, and sampled _K_ drugs and _L_
drugs respectively as the support set and query set.

Next, we embed two drugs in the _i_ -th support set of a batch
into the 3DGNN, and perform one or more times gradient drops
to obtain the updated parameter _θ_ [′], as shown in Eq. (9). Each
task-specific parameter _θ_ [′] is temporarily cached. Then, we use
the Adam optimizer to update the model parameter _θ_ [′] on the
query set. Unlike (Finn, Abbeel, & Levine, 2017), we minimize the
weighted sum of the query set loss for all tasks after every step
towards a support set task, as shown in Eq. (10).


_θ_ _i_ [′] [=] _[ θ]_ [ −] _[α]_ [∇] _[θ]_ _L_ (9)
( _d_ _x_ _,_ _d_ _y_ _,_ _r_ ) ∼ _D_ _[support]_ [(] _[i]_ [)] [(] _[f]_ _[θ]_ [(] _[d]_ _[x]_ _[,]_ _[ d]_ _[y]_ [)] _[,]_ _[ r]_ [)]



_B_
∑ _b_ = 0 _w_ _b_ ( _d_ _x_ _,_ _d_ _y_ _,_ _r_ ) _L_ ∼ _D_ _[query]_ [(] _[i]_ [)] [(] _[f]_ _[θ]_ _i_ [′] _[b]_ [(] _[d]_ _[x]_ _[,]_ _[ d]_ _[y]_ [)] _[,]_ _[ r]_ [)] (10)



Adam _(_ _L_ _t_ _, θ)_ : _θ_ [∗] ← _θ_ − _α_ ∇ _θ_ _L_ _t_ ( _f_ _θ_ ) (12)


where _M_ is the number of drug pairs in support set, _p_ and _p_ [′] are
the predicted DDI interaction probability for positive and negative
samples, respectively. Finally, we hope that the prior model updated with the support set can perform well on the query set. For
the query set, we feed it into the Meta3D-DDI model _f_ _θ_ ∗, which
generates the predicted DDI score _r_ [′] = _f_ _θ_ ∗ ( _d_ _x_ _,_ _d_ _y_ ).


**4. Results and discussion**


In this section, we first describe the experimental setup, and
then conduct experiments on two public datasets (DrugBank and
Twosides) under scaffold-based cold start scenario to compare
performances of different models and show related analysis. Finally, we conduct two visualization experiments to rationalize
Meta3D-DDI.


_4.1. Datasets_


**DrugBank** sourced from FDA/Health Canada drug labels, is
a real resource containing drug data and drug target information (Wishart et al., 2018). The dataset contains 1706 drugs and
86 interaction types. Each type describes how one drug affects the
metabolism of another drug. 191,808 drug pairs with interactions
were experimentally measured. 99.87% of drug–drug pairs have
only one interaction, and each DDI pair is used as a positive
sample. In contrast to positive samples, negative samples are drug
pairs where a drug does not have this interaction type with other
drugs. Negative samples are not included in the dataset and are
randomly generated during training. A corresponding negative
sample for each DDI pair is generated using the strategy described
by Nyamabo et al. (2021) and Yang et al. (2022a).

**Twosides** is constructed after preprocessing the original TWOSIDES dataset (Tatonetti et al., 2012; Zitnik, Agrawal, & Leskovec,
2018), which contains 645 drugs with 963 interaction types and
4,576,287 DDI tuples. Different from DrugBank dataset, 73.27% of
drug pairs have multiple DDI types in Twosides dataset. There
may be interactions between the two drugs at the phenotypic
level rather than at the metabolic level. The negative samples are
generated by a procedure the same as the DrugBank dataset.


_4.2. Baselines_


We compare Meta3D-DDI with multiple baselines.
**SA-DDI** . Yang et al. (2022a) A substructure aware graph neural
network is used to capture substructures with irregular size and
shape, and to simulate substructure–substructure interaction.

**GMPNN-CS** . Nyamabo et al. (2022) This model treats bonds
as gates that control the message passing flow, thereby learning drug substructures of different sizes and shapes for DDI
prediction.

**SSI–DDI** . Nyamabo et al. (2021) It operates directly on the
raw molecular graph representation of a drug, breaking the DDI
prediction task between two drugs into identifying pairwise interactions between their respective substructures.

**GAT-DDI** . Veličković et al. (2017) The model assigns different
weights to different nodes in the neighborhood by stacking layers where nodes are able to participate in their neighborhood
features



_θ_ ← _θ_ − _β_ ∇ _θ_



_N_
∑


_i_ = 1



where _α_ and _β_ are inner learning rate and outer learning rate
respectively, and _w_ _b_ denotes the importance weight of the query
set loss at step _b_, which is used to compute the weighted sum.


_3.4.3. Meta-testing_

After training a 3DGNN meta-learning model from extensive
DDI prediction tasks, we use this model with meta-knowledge
for fine-tuning. Given _M_ meta test tasks in the test set, we first



99


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



**Table 1**
Performance comparison of Meta3D-DDI and baseline on DrugBank dataset
under S1 scaffold-based cold start setting. ‘‘0-shot’’ refers to few-shot learning
models without internal iteration, and their evaluation settings are the same as
other baselines.


ACC AUC F1 Prec Rec AP


SSI-DDI 0.6262 0.6776 0.6113 0.6366 0.5880 0.6749

SA-DDI 0.6210 0.6671 0.5781 0.6518 0.5193 0.6679

MR-GNN 0.6085 0.6497 0.5628 0.6373 0.5039 0.6492

GMPNN-CS 0.6236 0.6703 0.6308 0.6113 0.6532 0.6511

GAT-DDI 0.6167 0.6674 0.6306 0.6089 0.6556 0.6490

3DGT-DDI 0.6231 0.6398 0.6017 0.6434 0.5651 0.6297


**0-shot**


MAML 0.6933 0.7333 0.7001 0.7611 0.6667 0.7983

IterRefLSTM 0.6412 0.7164 0.6815 0.6321 0.7064 0.7153

Meta-MGNN 0.6733 0.7356 0.6231 0.6950 0.6497 0.7132

CHEF 0.6429 0.6889 0.6882 0.6719 **0.7120** 0.8046

**Meta3D-DDI** **0.7167** **0.8240** **0.7197** **0.7860** 0.6833 **0.8238**


**MR-GNN** . Xu et al. (2019) A multi-resolution based GNN architecture, which uses dual graph-state LSTM to summarize the
local features of each graph and extract the interactive features
between pairwise graphs.

**3DGT-DDI** . He et al. (2022) An attention-based 3D graph neural network framework, which combines the text features, 3D
structure and position information of drug molecular.

**MAML** . Finn et al. (2017) A meta learning baseline for DDI
prediction that builds a task-agnostic algorithm for FSL.

**IterRefLSTM** . Altae-Tran et al. (2017) An iterative refinement
long short-term memory architecture that uses GCN aggregate
embedding representations for low-data drug discovery

**Meta-MGNN** . Guo, Zhang et al. (2021) A pretrained graphbased meta-learning framework that introduces additional selfsupervised tasks, such as bond reconstruction and atom type prediction, co-optimized with molecular property prediction tasks.

**CHEF** . Adler et al. (2020) A cross-domain Hebbian Ensemble FSL that achieves representation fusion by an ensemble of
Hebbian learners acting on different layers of a deep neural
network.


_4.3. Implementation and metrics_


Meta3D-DDI is implemented using the pytorch framework and
uses the Adam optimizer with a 0.0001 learning rate for gradient
descent optimization. Cosine annealing scheduling is applied to
the optimizer of meta-model. The initialization matrix dimension
of the atom embedding layer is 128. Each DDI type is represented
by a 64-dimensional matrix. The grid spacing and scaling parameter _γ_ is 0 _._ 1 1 _/_ Å. 5 classes are selected in each episode, and
5 steps of gradient descent are performed in both the training
and testing stages. For each DDI test task, 300 independent runs
are performed based on different random seeds, and the average
values of metrics are reported in the experimental section.

The DDI event prediction tasks are binary classification tasks,
i.e., predicting whether a drug–drug interaction effect occurs (1)
or not occur (0). We adopt 6 widely used classification performance evaluation metrics to measure the performance of
Meta3D-DDI, i.e., accuracy (ACC), area under the ROC curve (AUC),
F1-score (F1), precision (Prec), recall (Rec), and average precision
(AP). The implementation of Meta3D-DDI is based on the public
code of MAML++ (Antoniou, Edwards, & Storkey, 2019), the source
code and data are available in https://github.com/lol88/Meta3D
DDI.



**Table 2**
Performance comparison of Meta3D-DDI and baseline on Twosides dataset under
S1 scaffold-based cold start setting. ‘‘0-shot’’ refers to few-shot learning models
without internal iteration, and their evaluation settings are the same as other
baselines.


ACC AUC F1 Prec Rec AP


SSI-DDI 0.5307 0.5471 0.5229 0.5317 0.5144 0.5288

SA-DDI 0.5573 0.4868 0.4143 0.6119 0.3132 0.5615

MR-GNN 0.4933 0.5196 0.5129 0.4946 0.5600 0.5642

GMPNN-CS 0.5210 0.5344 0.4808 0.5234 0.4458 0.5262

GAT-DDI 0.5083 0.5165 0.4063 0.5172 0.3458 0.5149

3DGT-DDI 0.5543 0.5812 0.3736 0.6295 0.2656 0.5861


**0-shot**


MAML 0.6107 0.6523 0.6322 0.6012 0.6668 0.6099

IterRefLSTM 0.5527 0.5680 0.5193 0.4517 0.6133 0.5943

Meta-MGNN 0.5482 0.6196 0.6362 0.5683 **0.7600** 0.5871

CHEF 0.5912 0.6418 **0.6517** 0.5836 0.7467 0.6156

**Meta3D-DDI** **0.6285** **0.6865** 0.6286 **0.6303** 0.6312 **0.6364**


**Table 3**
Performance comparison of Meta3D-DDI and FSL-based models on DrugBank
dataset when _shot_ _>_ 0 under S1 scaffold-based cold start setting. ‘‘1- _shot_ ’’
means that the FSL-based model uses 1 sample in the support set for internal
fine-tuning.


ACC AUC F1 Prec Rec AP


**1-shot**


MAML 0.7149 0.8093 0.7100 0.7050 0.7151 0.7996

IterRefLSTM 0.6883 0.7342 0.6957 0.7488 0.6467 0.7726

Meta-MGNN 0.7027 0.8087 0.7359 0.7156 **0.7482** 0.8145

CHEF 0.6723 0.7129 0.6357 0.7199 0.5794 0.7411

**Meta3D-DDI** **0.7467** **0.8116** **0.7351** **0.7701** 0.7032 **0.8216**


**3-shot**


MAML 0.7316 **0.8693** 0.7078 **0.8125** 0.6208 0.8544

IterRefLSTM 0.7462 0.8156 0.7314 0.7736 0.7067 0.8144

Meta-MGNN 0.7522 0.8462 0.7750 0.7611 **0.7985** 0.8377

CHEF 0.7483 0.8347 0.7114 0.7503 0.6945 0.8284

**Meta3D-DDI** **0.7794** 0.8685 **0.7681** 0.8098 0.7304 **0.8604**


**5-shot**


MAML 0.7733 0.8769 0.7586 0.7927 0.7467 0.8652

IterRefLSTM 0.7685 0.8391 0.7485 0.7758 0.7648 0.8354

Meta-MGNN 0.7822 0.8813 0.7518 **0.8218** 0.7402 0.8673

CHEF 0.7533 0.8427 0.7576 0.7362 0.7933 0.8405

**Meta3D-DDI** **0.8124** **0.8932** **0.8247** 0.7735 **0.8832** **0.8730**


**10-shot**


MAML 0.7927 0.8733 0.8008 0.7694 0.8412 0.8618

IterRefLSTM 0.8011 0.8693 0.8128 0.7679 0.8726 0.8669

Meta-MGNN 0.8223 **0.8951** 0.8197 **0.8193** 0.8347 0.8829

CHEF 0.7690 0.8511 0.7705 0.7556 0.7916 0.8379

**Meta3D-DDI** **0.8409** 0.8728 **0.8463** 0.8187 **0.8759** **0.9091**


**20-shot**


MAML 0.8159 0.8729 0.8083 0.8033 0.8205 0.8602

IterRefLSTM 0.8274 0.8964 0.8277 0.8071 0.8549 0.8922

Meta-MGNN 0.8374 0.8938 0.8265 **0.8271** 0.8625 0.8875

CHEF 0.8060 0.8911 0.8013 0.8150 0.8014 0.8909

**Meta3D-DDI** **0.8541** **0.9249** **0.8606** 0.8241 **0.9006** **0.9141**


_4.4. Performance comparison_


We proposed scaffold-based cold start setting to divide the
training and test sets, which further prevents information leakage. The S1 and S2 represent a single drug and a pair of drugs cold
start scenario, respectively. The size of S1 test set on DrugBank
and Twosides datasets is 35,509 and 885,261 respectively. We
evaluate the performance of Meta3D-DDI on different shot ( _shot_ =
0, 1, 3, 5, 10, 20) of support set. ‘‘0- _shot_ ’’ refers to few-shot learning models without fine-tuning, and their evaluation settings are
the same as other baselines. ‘‘1- _shot_ ’’ means that the FSL-based
model uses 1 sample in the support set for internal fine-tuning.

Tables 1 to 2 shows the performance comparison of Meta3DDDI and baseline methods on the DrugBank and Twosides dataset



100


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



**Table 4**
Performance comparison of Meta3D-DDI and FSL-based models on Twosides
dataset when _shot_ _>_ 0 under S1 scaffold-based cold start setting. ‘‘1- _shot_ ’’
means that the FSL-based model uses 1 sample in the support set for internal
fine-tuning.


ACC AUC F1 Prec Rec AP


**1-shot**


MAML 0.5532 0.6169 0.6389 0.5822 0.7291 0.6194

IterRefLSTM 0.6031 0.5902 0.6204 0.5825 0.6875 0.6062

Meta-MGNN 0.6166 0.6160 **0.6487** 0.5851 **0.7316** 0.6146

CHEF 0.5815 0.6116 0.5547 0.5157 0.6267 0.5916

**Meta3D-DDI** **0.6461** **0.7071** 0.6412 **0.6473** 0.6378 **0.6487**


**3-shot**


MAML 0.5839 0.6222 0.6079 0.5610 0.6875 0.6417

IterRefLSTM 0.5477 0.5893 0.5035 0.5469 0.4815 0.6088

Meta-MGNN 0.5817 0.6436 0.5960 0.5915 0.6133 0.6407

CHEF 0.5510 0.6071 0.6222 0.5566 **0.7042** 0.5875

**Meta3D-DDI** **0.6474** **0.7094** **0.6410** **0.6498** 0.6354 **0.6517**


**5-shot**


MAML 0.6087 0.6524 **0.6737** 0.6143 **0.7626** 0.6487

IterRefLSTM 0.6147 0.6276 0.6146 0.5618 0.6831 0.6329

Meta-MGNN 0.6119 0.6382 0.5494 0.5665 0.5673 0.6529

CHEF 0.5779 0.6222 0.5073 0.5476 0.5067 0.6121

**Meta3D-DDI** **0.6697** **0.7414** 0.6525 **0.6898** 0.6281 **0.6952**


**10-shot**


MAML 0.6310 0.6658 0.6889 0.6164 0.7862 0.6772

IterRefLSTM 0.5720 0.6400 0.6478 0.5897 0.7238 0.6417

Meta-MGNN 0.6159 0.6907 0.6014 0.6375 0.5733 0.6959

CHEF 0.5606 0.6418 0.6211 0.6032 0.6809 0.6635

**Meta3D-DDI** **0.7173** **0.7896** **0.7401** **0.6850** **0.8063** **0.7388**


**20-shot**


MAML 0.6528 0.6738 0.6739 0.5688 0.8249 0.7004

IterRefLSTM 0.6217 0.6613 0.7103 0.6157 0.8619 0.6912

Meta-MGNN 0.6334 0.7067 0.6615 0.6157 0.7244 0.7420

CHEF 0.5990 0.7227 0.6353 0.6278 0.6521 0.7098

**Meta3D-DDI** **0.7736** **0.8465** **0.7957** **0.7291** **0.8766** **0.8071**


**Table 5**
Performance comparison of Meta3D-DDI and baseline on DrugBank dataset
under S2 scaffold-based cold start setting.‘‘0-shot’’ refers to few-shot learning
models without internal iteration, and their evaluation settings are the same as
other baselines.


ACC AUC F1 Prec Rec AP


SSI-DDI 0.5130 0.5139 0.4501 0.5931 0.0826 0.5250

SA-DDI 0.5980 0.6296 0.5154 0.6487 0.4275 0.6442

MR-GNN 0.5532 0.5755 0.4231 0.5968 0.3278 0.5925

GMPNN-CS 0.5335 0.5538 0.5103 0.5565 0.4963 0.5516

GAT-DDI 0.5224 0.5707 0.5323 0.5114 0.5563 0.5523

3DGT-DDI 0.5879 0.6092 0.5254 0.6251 0.4532 0.6025


**0-shot**


MAML 0.5967 0.5947 0.5068 0.5033 0.4533 0.6447

IterRefLSTM 0.5732 0.5511 0.4476 0.4739 0.4472 0.5627

Meta-MGNN 0.5887 0.6062 **0.6514** 0.5981 **0.7356** 0.6455

CHEF 0.5761 0.6080 0.5600 0.5987 0.5467 0.6375

**Meta3D-DDI** **0.6067** **0.7324** 0.5277 **0.7539** 0.4933 **0.7478**


under S1 scaffold-based cold start scenario (0- _shot_ ). All the models were tested directly after the training. The FSL-based model
uses the same test setup as the other baselines. The experimental
results show that Meta3D-DDI have superior performance on the
0- _shot_ to demonstrate the effectiveness of the 3D graph network
with few-shot learning. Furthermore, existing baseline models
struggle on the Twosides dataset. This may be due to the fact
that the drug pairs in Twosides have multiple DDI types. The
FSL-based models generally perform better than other baselines,
demonstrating that FSL models are more generalizable and are a
promising algorithm.

Tables 3 to 4 shows the performance comparison of Meta3DDDI and FSL-based models on the DrugBank and Twosides dataset



**Table 6**
Performance comparison of Meta3D-DDI and baseline on Twosides dataset under
S2 scaffold-based cold start setting.‘‘0-shot’’ refers to few-shot learning models
without internal iteration, and their evaluation settings are the same as other
baselines.


ACC AUC F1 Prec Rec AP


SSI-DDI 0.5082 0.5110 0.4556 0.5102 0.4115 0.5086

SA-DDI 0.3998 0.2239 0.3344 0.4848 0.2707 0.3497

MR-GNN 0.4967 0.5037 0.4231 0.4851 0.4133 0.5036

GMPNN-CS 0.5067 0.5094 0.4458 0.5086 0.3967 0.5054

GAT-DDI 0.4860 0.4264 0.4801 0.5279 0.4935 0.5024

3DGT-DDI 0.5179 0.5267 0.2107 0.5808 0.1287 0.5183


**0-shot**


MAML 0.5503 0.5759 0.5664 0.5467 0.5885 0.5520

IterRefLSTM 0.5274 0.4987 0.5634 0.5270 0.6438 0.5246

Meta-MGNN 0.5618 0.5387 0.5651 0.4797 **0.7231** 0.5428

CHEF 0.5039 0.4747 0.5592 0.4826 0.6627 0.5203

**Meta3D-DDI** **0.5765** **0.6137** **0.5958** **0.5708** 0.6234 **0.5847**


when _shot_ _>_ 0 under S1 cold start scenario. After training, all FSLbased models are fine-tuned by randomly sampling _shot_ (1, 3, 5,
10, 20) samples in each test task. The performance of all models
gradually improves as _shot_ increases. It is consistent with the
fact that more labeled samples bring better model performance.
This shows that FSL can quickly adapt to new tasks of out-ofdomain with a few samples. The performance improvement of
Meta3D-DDI is most significant and it usually maintains stateof-the-art performance. In addition, although some FSL-based
models may have high accuracy or recall, their other metrics
are often very low. This indicates that the performance of these
methods is not comprehensive, and their prediction results for
positive and negative samples are unstable and unbalanced. In
general, the Meta3D-DDI shows better performance than all baselines on 6 metrics under S1 scaffold-based cold start scenario.
This high-performance show that the Meta3D-DDI guarantees the
invariant description of atomic distribution to effectively alleviate
the spatial confusion problem.

The size of S2 test set on DrugBank and Twosides datasets is
2277 and 52,009 respectively. Tables 5 and 6 shows the performance comparison of Meta3D-DDI and other baseline methods
on the DrugBank and Twosides dataset when _shot_ = 0 under
S2 scaffold-based cold start scenario. The results of the baseline
methods are basically similar to random guessing, such as 0.51–
0.62 for AUC, 0.42–0.53 for F1. All FSL-based models struggle on
this setting. This proves that S2 scaffold-based cold start setting
is the more challenging scenario. Although the performance of
Meta3D-DDI is similarly low, it still outperforms other baseline
methods.

Tables 7 to 8 shows the performance comparison of Meta3DDDI and FSL-based models on the DrugBank and Twosides dataset
when _shot_ _>_ 0 under S2 cold start scenario. Compared with
the FSL-based model, the Meta3D-DDI also achieves the best
performance on the S2 scaffold-based cold start scenario for a
pair of drugs. Especially with the increase of labeled samples _k_
in the support set, the performance of Meta3D-DDI is greatly
improved. However, it is worth noting that the performance of
all models is unsatisfactory under the scaffold-based cold start
setting for a pair of drugs. Therefore, it is still a challenge to
improve the generalization ability of DDI prediction models on
S2 scaffold-based cold start scenario.


_4.5. Ablation experiment_


Quantitative experiments evaluated the model architecture to
verify the effectiveness of each module in Meta3D-DDI, including
without few-shot learning (w/o FSL), without continuous filter interaction module (w/o CFIM), without rotational and translational



101


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



**Table 7**
Performance comparison of Meta3D-DDI and FSL-based models on DrugBank
dataset when _shot_ _>_ 0 under S2 scaffold-based cold start setting. ‘‘1- _shot_ ’’
means that the FSL-based model uses 1 sample in the support set for internal
fine-tuning.


ACC AUC F1 Prec Rec AP


**1-shot**


MAML 0.6086 0.6916 0.5817 0.6791 0.5214 0.6771

IterRefLSTM 0.5746 **0.6978** 0.6159 0.6187 **0.6153** 0.6776

Meta-MGNN 0.6029 0.6284 0.5462 0.6065 0.5356 0.6160

CHEF 0.5860 0.6427 0.5884 0.5857 0.6008 0.6611

**Meta3D-DDI** **0.6292** 0.6618 **0.6498** **0.6998** 0.6135 **0.6877**


**3-shot**


MAML **0.7032** 0.7431 0.7094 0.6880 0.7351 0.7343

IterRefLSTM 0.6380 0.7138 **0.7134** 0.6542 **0.8016** 0.6886

Meta-MGNN 0.6173 0.7316 0.7008 0.6626 0.7467 0.7407

CHEF 0.5866 0.7262 0.5652 0.6447 0.5069 0.7044

**Meta3D-DDI** 0.6981 **0.7516** 0.6910 **0.7078** 0.6750 **0.7563**


**5-shot**


MAML 0.6259 0.7644 0.6333 0.6610 0.6235 0.7634

IterRefLSTM 0.6286 0.7307 0.6801 0.6375 0.7467 0.7224

Meta-MGNN 0.6510 0.7547 0.6737 0.6386 **0.7497** 0.7528

CHEF 0.5989 0.6747 0.5834 **0.7195** 0.5067 0.7072

**Meta3D-DDI** **0.7140** **0.7760** **0.7169** 0.7098 0.7241 **0.7741**


**10-shot**


MAML 0.7011 0.7609 0.7256 0.6385 **0.8577** 0.7385

IterRefLSTM 0.6738 0.7271 0.6988 0.6374 0.7743 0.7414

Meta-MGNN 0.6825 0.7920 **0.7489** 0.7159 0.7815 0.7535

CHEF 0.6681 0.7280 0.6915 0.6794 0.7027 0.7072

**Meta3D-DDI** **0.7413** **0.7939** 0.7321 **0.7593** 0.7070 **0.8024**


**20-shot**


MAML 0.6925 0.7378 0.7440 0.7042 0.8123 0.7520

IterRefLSTM 0.6884 0.7520 0.7141 0.6552 **0.8161** 0.7589

Meta-MGNN 0.7023 0.7280 0.6404 0.6367 0.6503 0.7459

CHEF 0.6903 0.7083 0.6797 0.6418 0.7309 0.7266

**Meta3D-DDI** **0.7609** **0.8210** **0.7594** **0.7644** 0.7547 **0.8221**


**Fig. 3.** The ablation study confirmed the effectiveness of each module in
Meta3D-DDI.


invariance (w/o RTI), and ‘‘Full’’. ‘‘Full’’ indicates the complete
framework of Meta3D-DDI. ‘‘w/o’’ indicates without this module.
Fig. 3 shows the results of the quantitative analysis on the S1 task
in DrugBank. Experiments show that the proposed Meta3D-DDI
achieves a higher average ACC(0.7167), AUC(0.8240), F1(0.7197),
Prec(0.7860) and AP(0.8238) than other configurations. The performance drops of w/o CFIM and w/o RTI verifies the effectiveness
of our proposed CFIM and invariance. It is shown that proposed
3D-graph model enrich molecular embeddings and thus improve
predictions. Compared to ‘‘Full’’, the performance of w/o FSL
decreases significantly. FSL is crucial for proposed Meta3D-DDI
as FSL has a greater impact on model performance than other



**Table 8**
Performance comparison of Meta3D-DDI and FSL-based models on Twosides
dataset when _shot_ _>_ 0 under S2 scaffold-based cold start setting. ‘‘1- _shot_ ’’
means that the FSL-based model uses 1 sample in the support set for internal
fine-tuning.


ACC AUC F1 Prec Rec AP


**1-shot**


MAML 0.5511 0.5884 0.5375 0.5486 0.5874 0.5762

IterRefLSTM 0.6039 0.5404 0.5851 0.5242 0.6831 0.5666

Meta-MGNN 0.5649 0.5840 0.5660 0.5417 0.6036 0.5789

CHEF 0.5514 0.5698 **0.6220** 0.5406 **0.7308** 0.5887

**Meta3D-DDI** **0.5990** **0.6421** 0.6156 **0.5914** 0.6422 **0.6031**


**3-shot**


MAML 0.5856 0.5938 0.5564 0.5260 0.6263 0.6169

IterRefLSTM 0.5928 0.5822 0.5720 0.5512 0.6533 0.6050

Meta-MGNN 0.6127 0.6151 **0.6536** 0.5889 **0.7467** 0.5977

CHEF 0.5703 0.5920 0.5921 0.5568 0.6400 0.6084

**Meta3D-DDI** **0.6186** **0.6708** 0.6276 **0.6163** 0.6424 **0.6241**


**5-shot**


MAML 0.6057 0.6667 **0.6438** 0.6093 **0.6933** 0.6275

IterRefLSTM 0.5982 0.6498 0.5313 0.5619 0.5351 0.6127

Meta-MGNN 0.6437 0.6284 0.6172 0.5653 0.6923 0.6142

CHEF 0.5967 0.5867 0.5944 0.5603 0.6511 0.5969

**Meta3D-DDI** **0.6476** **0.7098** 0.6414 **0.6500** 0.6361 **0.6525**


**10-shot**


MAML 0.6490 0.6151 0.5892 0.6011 0.5733 0.6238

IterRefLSTM 0.5805 0.6080 0.6307 0.5379 **0.7716** 0.6131

Meta-MGNN 0.6318 0.6587 0.5962 0.5953 0.6113 0.6350

CHEF 0.6215 0.5991 0.6212 0.6031 0.6933 0.6127

**Meta3D-DDI** **0.6747** **0.7373** **0.6973** **0.6512** 0.7517 **0.6963**


**20-shot**


MAML 0.5932 0.6773 0.5805 0.5806 0.6133 0.6871

IterRefLSTM 0.6156 0.6373 0.6733 0.6257 0.7467 0.6236

Meta-MGNN 0.5758 0.6373 0.5999 0.5518 0.6833 0.6526

CHEF 0.5784 0.6231 0.5784 0.5736 0.6386 0.6301

**Meta3D-DDI** **0.7148** **0.7847** **0.7297** **0.6981** **0.7748** **0.7808**


modules. This provides further evidence of the effectiveness of
FSL in drug development with few samples. Therefore, the above
results can demonstrate the effectiveness of each configuration in
the proposed Meta3D-DDI.


_4.6. Necessity of scaffold-based cold start setting_


This experiment demonstrates that the scaffold-based cold
start setting is an evaluation setting to measure whether the
DDI model really improves the prediction performance and generalization ability. Fig. 4 shows baseline models achieve high
performance on the warm start setting (mean AUC and F1 are
0.97 and 0.93, respectively). However, the performance of these
methods is significantly decrease under the drug-based cold start
setting. The mean AUC and F1 are 0.86 and 0.75 for a single
drug, respectively. The mean AUC and F1 are 0.73 and 0.64 for
a pair of drugs, respectively. This indicates that it is difficult
for warm start setting to evaluate the generalization ability of
DDI prediction models. The cold start setting for a pair of drugs
cause more lower performance than cold start setting for a single
drug. This indicates the model produces unreliable results for the
interaction between the two new drugs.

The scaffold-based cold start setting further degrades the
model performance on both S1 and S2. The mean AUC and F1
on the single drug are 0.66 and 0.60, respectively. The mean
AUC and F1 on a pair of drugs are 0.57 and 0.49, respectively.
The results of the baseline methods on scaffold-based cold start
setting are basically similar to random guessing. This indicates
that the performance of baseline method is overly optimistic in
both warm start setting and drug-based cold start setting. This
phenomenon is consistent with the fact that scaffold-based cold



102


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_


**Fig. 4.** The performance comparison of the existing method under 5 test settings on DrugBank datasets, and its performance is significantly decreased sequentially.
The previous models achieves the lowest performance under scaffold-based cold start setting.


**Fig. 5.** Visualizations of embedded vectors generated by Meta3D-DDI on each
internal iteration during the test phase.



start setting absolutely prevents the structural information of
drugs from leaking into the test set. This setting is a more realistic and challenging. We hope that the scaffold-based cold start
setting brings a new evaluation setting for DDI prediction, which
better reflects generalization ability of model. We have published
AI-ready data that is reorganized according to scaffold-based cold
start setting.


_4.7. Visual explanations_


The interpretability of the proposed model is crucial for DDI
prediction. We utilize two visualization experiments to demonstrate the interpretability of Meta3D-DDI. We utilize the
t-distributed neighbor embedding (T-SNE) to demonstrate that
the DDI prediction obtained by Meta3D-DDI can distinguish between positive and negative samples. The experiment takes the
S1 test set of DrugBank as an example. First, we select the drug
pair sets corresponding to two types of DDI events. Then, the
T-SNE is used to reduce the high-dimensional hidden vectors
on the last layer to a 2D embedding space. This operation is
done at every step of gradient descent on the test phase. Finally,
we observe the distribution on two types of DDI events. Fig. 5
shows the distribution visualization of DDI event. The red and
blue represent the two different types, respectively. The triangle
and circle represent positive and negative samples, respectively.
Step 0 means that no support set is used for inner iteration. In
step 0, we can observe that the embedding vectors of positive
and negative examples generated by Meta3D-DDI are not clearly



**Fig. 6.** Visualization of the atom weights between alfentanil and the 8 drugs.


separated on the 2D mapping space, although there is a certain
degree of separation. After 1–2 iterations using the support set,
the separation degree of positive and negative examples becomes
clear, and the clustering phenomenon of the same class is exhibited. Continuing the internal iteration, we can clearly observe
that both positive and negative examples of both DDI event types
show a clear clustering phenomenon. These distribution changes
indicate that the Meta3D-DDI benefits from meta-knowledge to
quickly iterate and converge in the right direction.

We visualize the hidden vectors before merging drug information to show the weight of each atom in drug molecules.
The drug in the middle of Fig. 6 is alfentanil, which is a shortacting opioid anesthetic and analgesic derivative of fentanyl.
Surrounded by 8 drugs: ibuprofen, fenoprofen, loxoprofen, indomethacin, acemetacin, diclofenac, aceclofenac, and alclofenac
are non-steroidal anti-inflammatory drugs (NSAID) with analgesic
and antipyretic properties. Fig. 6 shows that Meta3D-DDI pays
more attention to acetic acid, propionic acid, benzene ring, and
oxhydryl, and less attention to CI heavy atom. Both ibuprofen,
fenoprofen, loxoprofen are NSAID of the propionic acid class,
while other 5 drugs are NSAID of acetic acid class. These drugs
exerts their pharmacological effects by inhibiting cyclooxygenase
(COX)-1 and − 2 involved in pain, fever, and inflammation. COX
is an enzyme involved in prostaglandin (mediators of pain and
fever) and thromboxane (stimulators of blood clotting) synthesis via the arachidonic acid pathway. Thromboxanes are potent



103


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



vasoconstrictors that is consistent in decreasing cerebral blood
flow and inhibiting CO 2 reactivity. Alfentanil depresses cardiac
and respiratory centers. The risk or severity of hypertension
can be increased when these 8 drugs is combined with alfentanil. According to the mechanism of action, we found that
acetic acid, propionic acid and benzene ring groups in COX inhibitors play an important role. This is consistent with the results
of our model visualization, which proves that Meta3D-DDI can
learn the structural properties of drug molecules close to human
understanding.


**5. Conclusion**


The baseline models have achieved state-of-the-art performance in warm start scenario. However, the performance of the
baseline method significantly decreased in scaffold-based cold
start scenario. And the baseline method ignores the 3D conformational knowledge of drug molecules. In response to these two
challenges, this work presented a 3D graph network with fewshot learning to predict potential DDI events in scaffold-based
cold start scenario. Meta3D-DDI learns molecular representations
that obey the fundamental symmetries of atomic systems by introducing the invariant description of the atomic distribution. The
continuous filter interaction module simulates the interactions
between atoms at arbitrary positions in the drug, and introduces
3D structure and distance information in the information aggregation stage. These designs improve the 3DGNN perception
ability on medicinal chemical environment and connectivity to
effectively alleviate the spatial confusion problem. We also design
a cold start setting based on scaffold, which further prevents
the leakage of drug structure information to the test set because
the scaffolds on the training set and test set do not overlap.
Meta3D-DDI architecture achieves SOTA performance for DDI
prediction under scaffold-based cold start scenario on two realworld datasets. The 3D graph network with few-shot learning is
a powerful tool to improve the generalization and interpretation
capability for DDI prediction of new drugs.


**CRediT authorship contribution statement**


**Qiujie Lv:** Software, Methodology, Writing – original draft,
Writing – review & editing. **Jun Zhou:** Software, Writing – original draft. **Ziduo Yang:** Software, Writing – original draft, Formal
analysis. **Haohuai He:** Software, Resources, Data curation. **Calvin**
**Yu-Chian Chen:** Conceptualization, Supervision, Writing – review
& editing.


**Declaration of competing interest**


The authors declare that they have no known competing financial interests or personal relationships that could have appeared
to influence the work reported in this paper.


**Data availability**


I am using publicly available data. Also, my code and public
data are included in the github link.


**Acknowledgments**


The authors thank the anonymous reviewers for their valuable
suggestions. This work was supported by the National Natural
Science Foundation of China (Grant No. 62176272), Research and
Development Program of Guangzhou Science and Technology
Bureau (No. 2023B01J1016), Key-Area Research and Development
Program of Guangdong Province (No. 2020B1111100001), and
China Medical University Hospital, Taiwan (DMR-112-085).



**References**


Adler, T., Brandstetter, J., Widrich, M., Mayr, A., Kreil, D., Kopp, M., et al. (2020).

Cross-domain few-shot learning by representation fusion. arXiv preprint
[arXiv:2010.06498.](http://arxiv.org/abs/2010.06498)
[Altae-Tran, H., Ramsundar, B., Pappu, A. S., & Pande, V. (2017). Low data drug](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb2)

discovery with one-shot learning. _[ACS Central Science](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb2)_, _3_ (4), 283–293.
Antoniou, A., Edwards, H., & Storkey, A. J. (2019). How to train your MAML. In _7th_

_International conference on learning representations, ICLR 2019, New Orleans,_
_la, USA, May 6-9, 2019_ [. OpenReview.net, URL https://openreview.net/forum?](https://openreview.net/forum?id=HJGven05Y7)
[id=HJGven05Y7.](https://openreview.net/forum?id=HJGven05Y7)
[Asada,](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb4) M., Miwa, M., & Sasaki, Y. (2021). Using drug descriptions and
[molecular structures for drug–drug interaction extraction from literature.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb4)
_Bioinformatics_, _[37](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb4)_ (12), 1739–1746.
[Bajorath, J. (2017). Computational scaffold hopping: cornerstone for the future](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb5)

[of drug design?](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb5)
[Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb6)

Molecular frameworks. _[Journal of Medicinal Chemistry](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb6)_, _39_ (15), 2887–2893.
[Cai, C., Wang, S., Xu, Y., Zhang, W., Tang, K., Ouyang, Q., et al. (2020).](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb7)

Transfer learning for drug discovery. _[Journal of Medicinal Chemistry](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb7)_, _63_ (16),
[8683–8694.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb7)
[Dai, Y., Guo, C., Guo, W., & Eickhoff, C. (2021). Drug–drug interaction pre-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb8)

[diction with wasserstein adversarial autoencoder-based knowledge graph](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb8)
embeddings. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb8)_, _22_ (4), bbaa256.
Deac, A., Huang, Y.-H., Veličković, P., Liò, P., & Tang, J. (2019). Drug-drug adverse

[effect prediction with graph co-attention. arXiv preprint arXiv:1905.00534.](http://arxiv.org/abs/1905.00534)
[Deng, Y., Xu, X., Qiu, Y., Xia, J., Zhang, W., & Liu, S. (2020). A multi-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb10)

[modal deep learning framework for predicting drug–drug interaction events.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb10)
_Bioinformatics_, _[36](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb10)_ (15), 4316–4322.
[Dewulf, P., Stock, M., & De Baets, B. (2021). Cold-start problems in data-driven](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb11)

[prediction of drug–drug interaction effects.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb11) _Pharmaceuticals_, _14_ (5), 429.
[Fang, X., Liu, L., Lei, J., He, D., Zhang, S., Zhou, J., et al. (2022). Geometry-enhanced](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb12)

[molecular representation learning for property prediction.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb12) _Nature Machine_
_Intelligence_, _4_ [(2), 127–134.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb12)
[Fatehifar, M., & Karshenas, H. (2021). Drug-drug interaction extraction using](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb13)

[a position and similarity fusion-based attention mechanism.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb13) _Journal of_
_[Biomedical Informatics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb13)_, _115_, Article 103707.
[Ferdousi, R., Safdari, R., & Omidi, Y. (2017). Computational prediction of drug-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb14)

[drug interactions based on drugs functional similarities.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb14) _Journal of Biomedical_
_Informatics_, _[70](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb14)_, 54–64.
[Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb15)

adaptation of deep networks. In _[International conference on machine learning](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb15)_
[(pp. 1126–1135). PMLR.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb15)
[Gao, F., Luo, X., Yang, Z., & Zhang, Q. (2022). Label smoothing and task-adaptive](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb16)

[loss function based on prototype network for few-shot learning.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb16) _Neural_
_Networks_, _[156](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb16)_, 39–48.
[Gottlieb, A., Stein, G. Y., Oron, Y., Ruppin, E., & Sharan, R. (2012). INDI: a](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb17)

[computational framework for inferring drug interactions and their associated](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb17)
recommendations. _[Molecular Systems Biology](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb17)_, _8_ (1), 592.
[Guo, S., Xu, L., Feng, C., Xiong, H., Gao, Z., & Zhang, H. (2021). Multi-level](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb18)

[semantic adaptation for few-shot segmentation on cardiac image sequences.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb18)
_[Medical Image Analysis](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb18)_, _73_, Article 102170.
[Guo, S., Zhang, H., Gao, Y., Wang, H., Xu, L., Gao, Z., et al. (2023). Survival](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb19)

[prediction of heart failure patients using motion-based analysis method.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb19)
_[Computer Methods and Programs in Biomedicine](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb19)_, Article 107547.
[Guo, Z., Zhang, C., Yu, W., Herr, J., Wiest, O., Jiang, M., et al. (2021). Few-shot](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb20)

[graph learning for molecular property prediction. In J. Leskovec, M. Grobel-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb20)
nik, M. Najork, J. Tang, L. Zia (Eds.), _[WWW ’21: The web conference 2021,](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb20)_
_[virtual event / Ljubljana, Slovenia, April 19-23, 2021](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb20)_ (pp. 2559–2567). ACM /
[IW3C2.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb20)
[Han, K., Kim, Y., Han, D., Lee, H., & Hong, S. (2023). TL-ADA: Transferable](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb21)

[loss-based active domain adaptation.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb21) _Neural Networks_, _161_, 670–681.
[He, H., Chen, G., & Yu-Chian Chen, C. (2022). 3DGT-DDI: 3D graph and text](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb22)

[based neural network for drug–drug interaction prediction.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb22) _Briefings in_
_Bioinformatics_, _[23](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb22)_ (3), bbac134.
[Hong, L., Lin, J., Li, S., Wan, F., Yang, H., Jiang, T., et al. (2020). A novel](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb23)

[machine learning framework for automated biomedical relation extraction](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb23)
[from large-scale literature repositories.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb23) _Nature Machine Intelligence_, _2_ (6),
[347–355.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb23)
[Hospedales, T., Antoniou, A., Micaelli, P., & Storkey, A. (2021). Meta-learning in](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb24)

neural networks: A survey. _[IEEE Transactions on Pattern Analysis and Machine](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb24)_
_Intelligence_, _44_ [(9), 5149–5169.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb24)
[Huang, L., Lin, J., Li, X., Song, L., Zheng, Z., & Wong, K.-C. (2022). EGFI: drug–](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb25)

[drug interaction extraction and generation with fusion of enriched entity](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb25)
and sentence information. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb25)_, _23_ (1), bbab451.
[Huang, K., Xiao, C., Hoang, T., Glass, L., & Sun, J. (2020). Caster: Predicting drug](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb26)

[interactions with chemical substructure representation. In](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb26) _Proceedings of the_
_[AAAI conference on artificial intelligence, Vol. 34](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb26)_ (pp. 702–709).
[Hussain, S., Anees, A., Das, A., Nguyen, B. P., Marzuki, M., Lin, S., et al.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb27)

[(2020). High-content image generation for drug discovery using generative](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb27)
adversarial networks. _[Neural Networks](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb27)_, _132_, 353–363.



104


_Q. Lv, J. Zhou, Z. Yang et al._ _Neural Networks 165 (2023) 94–105_



[Ju, W., Gu, Y., Luo, X., Wang, Y., Yuan, H., Zhong, H., et al. (2023). Unsuper-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb28)

[vised graph-level representation learning with hierarchical contrasts.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb28) _Neural_
_Networks_, _[158](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb28)_, 359–368.
[Ju, W., Liu, Z., Qin, Y., Feng, B., Wang, C., Guo, Z., et al. (2023). Few-shot molecular](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb29)

[property prediction via hierarchically structured learning on relation graphs.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb29)
_[Neural Networks](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb29)_, _163_, 122–131.
[Kantor, E. D., Rehm, C. D., Haas, J. S., Chan, A. T., & Giovannucci, E. L. (2015).](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb30)

[Trends in prescription drug use among adults in the United States from](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb30)
1999–2012. _Jama_, _[314](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb30)_ (17), 1818–1830.
[Koks, D. (2006).](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb31) _[Explorations in mathematical physics: The concepts behind an](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb31)_

_[elegant language](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb31)_ . Springer.
[Landrum, G., et al. (2013). Rdkit: A software suite for cheminformatics,](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb32)

[computational chemistry, and predictive modeling.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb32) _Greg Landrum_, _8_ .
[Li, Q., Xie, X., Zhang, J., & Shi, G. (2023). Few-shot human–object interaction](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb33)

[video recognition with transformers.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb33) _Neural Networks_, _163_, 1–9.
[Li, K., Zhang, Y., Li, K., & Fu, Y. (2020). Adversarial feature hallucination networks](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb34)

for few-shot learning. In _[Proceedings of the IEEE/CVF conference on computer](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb34)_
_[vision and pattern recognition](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb34)_ (pp. 13470–13479).
Liu, S., Huang, Z., Qiu, Y., Chen, Y.-P. P., & Zhang, W. (2019). Structural

network embedding using multi-modal deep auto-encoders for predicting
drug-drug interactions. In _2019 IEEE international conference on bioinformatics_
_and biomedicine_ [(pp. 445–450). http://dx.doi.org/10.1109/BIBM47256.2019.](http://dx.doi.org/10.1109/BIBM47256.2019.8983337)
[8983337.](http://dx.doi.org/10.1109/BIBM47256.2019.8983337)
Liu, Y., Wang, L., Liu, M., Lin, Y., Zhang, X., Oztekin, B., et al. (2022). Spherical

message passing for 3D molecular graphs. In _The tenth international confer-_
_ence on learning representations, ICLR 2022, virtual event, April 25-29, 2022_ .
[OpenReview.net, URL https://openreview.net/forum?id=givsRXsOt9r.](https://openreview.net/forum?id=givsRXsOt9r)
[Liu, Z., Wang, X.-N., Yu, H., Shi, J.-Y., & Dong, W.-M. (2022). Predict multi-type](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb37)

[drug–drug interactions in cold start scenario.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb37) _BMC Bioinformatics_, _23_ (1), 1–13.
[Lv, Q., Chen, G., He, H., Yang, Z., Zhao, L., Zhang, K., et al. (2023). Tcmbank-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb38)

[the largest TCM database provides deep learning-based Chinese-western](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb38)
medicine exclusion prediction. _[Signal Transduction and Targeted Therapy](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb38)_, _8_ (1),
[127.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb38)
[Lv, Q., Chen, G., Yang, Z., Zhong, W., & Chen, C. Y.-C. (2023). Meta learning with](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb39)

[graph attention networks for low-data drug discovery.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb39) _IEEE Transactions on_
_[Neural Networks and Learning Systems](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb39)_ .
[Lv, Q., Chen, G., Zhao, L., Zhong, W., & Yu-Chian Chen, C. (2021). Mol2Context-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb40)

[vec: learning molecular representation from context awareness for drug](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb40)
discovery. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb40)_, _22_ (6), bbab317.
[Lv, Q.-J., Chen, H.-Y., Zhong, W.-B., Wang, Y.-Y., Song, J.-Y., Guo, S.-D., et al.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb41)

[(2019). A multi-task group Bi-LSTM networks application on electrocardio-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb41)
gram classification. _[IEEE Journal of Translational Engineering in Health and](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb41)_
_Medicine_, _8_, 1–11.
[Mayr, A., Klambauer, G., Unterthiner, T., Steijaert, M., Wegner, J. K., Ceule-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb42)

[mans, H., et al. (2018). Large-scale comparison of machine learning methods](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb42)
[for drug target prediction on chembl.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb42) _Chemical Science_, _9_ (24), 5441–5451.
[Méndez-Lucio, O., Ahmad, M., del Rio-Chanona, E. A., & Wegner, J. K. (2021).](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb43)

[A geometric deep learning approach to predict binding conformations of](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb43)
bioactive molecules. _[Nature Machine Intelligence](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb43)_, _3_ (12), 1033–1039.
[Nyamabo, A. K., Yu, H., Liu, Z., & Shi, J.-Y. (2022). Drug–drug interaction](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb44)

[prediction with learnable size-adaptive molecular substructures.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb44) _Briefings in_
_Bioinformatics_, _[23](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb44)_ (1), bbab441.
[Nyamabo, A. K., Yu, H., & Shi, J.-Y. (2021). SSI–DDI: substructure–substructure](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb45)

[interactions for drug–drug interaction prediction.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb45) _Briefings in Bioinformatics_,
_22_ [(6), bbab133.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb45)
[Qato, D. M., Wilder, J., Schumm, L. P., Gillet, V., & Alexander, G. C. (2016). Changes](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb46)

[in prescription and over-the-counter medication and dietary supplement](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb46)
[use among older adults in the United States, 2005 vs 2011.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb46) _JAMA Internal_
_Medicine_, _[176](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb46)_ (4), 473–482.
[Ren, Z.-H., You, Z.-H., Yu, C.-Q., Li, L.-P., Guan, Y.-J., Guo, L.-X., et al. (2022).](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb47)

[A biomedical knowledge graph-based method for drug–drug interactions](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb47)
[prediction through combining local and global features with deep neural](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb47)
networks. _[Briefings in Bioinformatics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb47)_ .
[Ryu, J. Y., Kim, H. U., & Lee, S. Y. (2018). Deep learning improves prediction of](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb48)

[drug–drug and drug–food interactions.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb48) _Proceedings of the National Academy_
_of Sciences_, _115_ [(18), E4304–E4311.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb48)
Sagawa, S., & Hino, H. (2023). Cost-effective framework for gradual domain adap
tation with multifidelity. _Neural Networks_ [, URL https://www.sciencedirect.](https://www.sciencedirect.com/science/article/pii/S0893608023001703)
[com/science/article/pii/S0893608023001703.](https://www.sciencedirect.com/science/article/pii/S0893608023001703)
[Schütt, K. T., Arbabzadah, F., Chmiela, S., Müller, K. R., & Tkatchenko, A. (2017).](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb50)

[Quantum-chemical insights from deep tensor neural networks.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb50) _Nat. Commn._,
_8_ [(1), 1–8.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb50)



Schütt, K., Kindermans, P., Felix, H. E. S., Chmiela, S., Tkatchenko, A., &

Müller, K. (2017). SchNet: A continuous-filter convolutional neural network
for modeling quantum interactions. In I. Guyon, U. von Luxburg, S. Bengio,
H. M. Wallach, R. Fergus, S. V. N. Vishwanathan, & R. Garnett (Eds.),
_Advances in neural information processing systems 30: Annual conference on_
_neural information processing systems 2017, December 4-9, 2017, Long Beach,_
_CA, USA_ [(pp. 991–1001). URL https://proceedings.neurips.cc/paper/2017/hash/](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)
[303ed4c69846ab36c2904d3ba8573050-Abstract.html.](https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html)

[Schütt, K. T., Sauceda, H. E., Kindermans, P.-J., Tkatchenko, A., & Müller, K.-R.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb52)

[(2018). Schnet–a deep learning architecture for molecules and materials.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb52) _The_
_[Journal of Chemical Physics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb52)_, _148_ (24), Article 241722.
[Shi, J.-Y., Mao, K.-T., Yu, H., & Yiu, S.-M. (2019). Detecting drug communities](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb53)

[and predicting comprehensive drug–drug interactions via balance regularized](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb53)
[semi-nonnegative matrix factorization.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb53) _Journal of Cheminformatics_, _11_ (1),

[1–16.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb53)

[Sridhar, D., Fakhraei, S., & Getoor, L. (2016). A probabilistic approach for](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb54)

[collective similarity-based drug–drug interaction prediction.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb54) _Bioinformatics_,
_32_ [(20), 3175–3182.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb54)
[Tatonetti, N. P., Ye, P. P., Daneshjou, R., & Altman, R. B. (2012). Data-driven](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb55)

[prediction of drug effects and interactions.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb55) _Science Translational Medicine_,
_4_ [(125), Article 125ra31.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb55)
[Vanschoren, J. (2018). Meta-learning: A survey. arXiv preprint arXiv:1810.03548.](http://arxiv.org/abs/1810.03548)
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017).

[Graph attention networks. arXiv preprint arXiv:1710.10903.](http://arxiv.org/abs/1710.10903)
[Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Courna-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb58)

[peau, D., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb58)
computing in python. _[Nature Methods](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb58)_, _17_, 261–272.
[Wang, Y., Yao, Q., Kwok, J. T., & Ni, L. M. (2020). Generalizing from a few](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb59)

[examples: A survey on few-shot learning.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb59) _ACM Computing Surveys (Csur)_,
_53_ [(3), 1–34.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb59)
[Weininger, D. (1988). SMILES, a chemical language and information system.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb60)

[1. Introduction to methodology and encoding rules.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb60) _Journal of Chemical_
_[Information and Computer Sciences](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb60)_, _28_ (1), 31–36.
[Wishart, D. S., Feunang, Y. D., Guo, A. C., Lo, E. J., Marcu, A., Grant, J. R., et al.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb61)

[(2018). DrugBank 5.0: a major update to the DrugBank database for 2018.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb61)
_Nucleic Acids Research_, _[46](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb61)_ (D1), D1074–D1082.
Xu, N., Wang, P., Chen, L., Tao, J., & Zhao, J. (2019). Mr-gnn: Multi-resolution

and dual graph neural network for predicting structured entity interactions.
[arXiv preprint arXiv:1905.09558.](http://arxiv.org/abs/1905.09558)
[Yang, Z., Zhong, W., Lv, Q., & Chen, C. Y.-C. (2022a). Learning size-adaptive](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb63)

[molecular substructures for explainable drug–drug interaction prediction](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb63)
[by substructure-aware graph neural network.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb63) _Chemical Science_, _13_ (29),
[8693–8703.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb63)

[Yang, Z., Zhong, W., Zhao, L., & Chen, C. Y.-C. (2022b). MGraphDTA: deep](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb64)

[multiscale graph neural network for explainable drug–target binding affinity](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb64)
prediction. _[Chemical Science](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb64)_, _13_ (3), 816–833.
[Yu, H., Zhao, S., & Shi, J. (2022). STNN-DDI: a substructure-aware tensor neural](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb65)

[network to predict drug–drug interactions.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb65) _Briefings in Bioinformatics_, _23_ (4),
[bbac209.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb65)

[Zhang, W., Chen, Y., Liu, F., Luo, F., Tian, G., & Li, X. (2017). Predicting potential](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb66)

[drug-drug interactions by integrating chemical, biological, phenotypic and](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb66)
network data. _[BMC Bioinformatics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb66)_, _18_ (1), 1–12.
[Zhang, T., Leng, J., & Liu, Y. (2020). Deep learning for drug–drug interaction](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb67)

[extraction from the literature: a review.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb67) _Briefings in Bioinformatics_, _21_ (5),

[1609–1627.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb67)

[Zhang, P., Wang, F., Hu, J., & Sorrentino, R. (2015). Label propagation prediction](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb68)

[of drug-drug interactions based on clinical side effects.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb68) _Scientific Reports_, _5_ (1),

[1–10.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb68)

Zhang, B., Zhou, M., Wu, J., & Gao, F. (2021). Predicting material properties using

a 3D graph neural network with invariant local descriptors. arXiv preprint
[arXiv:2102.11023.](http://arxiv.org/abs/2102.11023)

[Zhao, J., Lan, L., Huang, D., Ren, J., & Yang, W. (2022). Heterogeneous pseudo-](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb70)

[supervised learning for few-shot person re-identification.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb70) _Neural Networks_,
_154_ [, 521–537.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb70)
[Zhu, Y., Li, L., Lu, H., Zhou, A., & Qin, X. (2020). Extracting drug-drug interactions](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb71)

[from texts with BioBERT and multiple entity-aware attentions.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb71) _Journal of_
_[Biomedical Informatics](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb71)_, _106_, Article 103451.
[Zitnik, M., Agrawal, M., & Leskovec, J. (2018). Modeling polypharmacy side effects](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb72)

[with graph convolutional networks.](http://refhub.elsevier.com/S0893-6080(23)00283-6/sb72) _Bioinformatics_, _34_ (13), i457–i466.



105


