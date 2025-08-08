[Journal of Biomedical Informatics 125 (2022) 103968](https://doi.org/10.1016/j.jbi.2021.103968)


Contents lists available at ScienceDirect

# Journal of Biomedical Informatics


[journal homepage: www.elsevier.com/locate/yjbin](https://www.elsevier.com/locate/yjbin)


Original Research

## An attentive joint model with transformer-based weighted graph convolutional network for extracting adverse drug event relation


Ed-drissiya El-allaly [a] [,] [*], Mourad Sarrouti [b] [,] [c], Noureddine En-Nahnahi [a], Said Ouatik El Alaoui [d ]


a _Laboratory of Informatics, Signals, Automatic, and Cognitivism (LISAC), Faculty of Sciences Dhar ELMehraz, Sidi Mohamed Ben Abdellah University, Fez, Morocco_
b _U.S. National Library of Medicine, National Institutes of Health, Bethesda, MD, USA_
c _Sumitovant Biopharma, NY, USA_
d _Laboratory of Engineering Sciences, National School of Applied Sciences, Ibn Tofail University, Kenitra, Morocco_



A R T I C L E I N F O


_Keywords:_
Adverse drug events
Weighted graph convolutional network
Joint learning
Transfer learning

Relation extraction

Natural language processing


**1. Introduction**



A B S T R A C T


Adverse drug event (ADE) relation extraction is a crucial task for drug safety surveillance which aims to discover
potential relations between ADE mentions from unstructured medical texts. To date, the graph convolutional
networks (GCN) have been the state-of-the-art solutions for improving the ability of relation extraction task.
However, there are many challenging issues that should be addressed. Among these, the syntactic information is
not fully exploited by GCN-based methods, especially the diversified dependency edges. Still, these methods fail
to effectively extract complex relations that include nested, discontinuous and overlapping mentions. Besides, the
task is primarily regarded as a classification problem where each candidate relation is treated independently
which neglects the interaction between other relations. To deal with these issues, in this paper, we propose an
attentive joint model with transformer-based weighted GCN for extracting **ADE Rel** ations, called ADERel. Firstly,
the ADERel system formulates the ADE relation extraction task as an N-level sequence labelling so as to model
the complex relations in different levels and capture greater interaction between relations. Then, it exploits our
neural joint model to process the N-level sequences jointly. The joint model leverages the contextual and
structural information by adopting a shared representation that combines a bidirectional encoder representation
from transformers (BERT) and our proposed weighted GCN (WGCN). The latter assigns a score to each de­
pendency edge within a sentence so as to capture rich syntactic features and determine the most influential edges
for extracting ADE relations. Finally, the system employs a multi-head attention to exchange boundary knowl­
edge across levels. We evaluate ADERel on two benchmark datasets from TAC 2017 and n2c2 2018 shared tasks.
The experimental results show that ADERel is superior in performance compared with several state-of-the-art
methods. The results also demonstrate that incorporating a transformer model with WGCN makes the pro­
posed system more effective for extracting various types of ADE relations. The evaluations further highlight that
ADERel takes advantage of joint learning, showing its effectiveness in recognizing complex relations.



Adverse drug events (ADEs) are largely defined as unexpected effects
caused by the drug such as overdoses, adverse drug reactions (ADR) and
drug interactions [1,2]. Generally, the clinical trials are not enough to
discover all potential ADEs as they require additional procedures such as
the large number of volunteer patients [3]. During post-marketing sur­
veillance, although spontaneous reporting systems (SRSs) are used to
alleviate the shortcoming of clinical trials, some ADEs are underreported




[4–6]. Therefore, aggregating knowledge about ADEs from supple­
mentary resources is critical for drug-safety monitoring which aims at
minimizing the costs and risks of undiscovered ADEs. In this context, the
application of natural language processing (NLP) systems to extract
ADEs from a variety of textual sources including drug labels, electronic
health records (EHRs) and biomedical literature, attracts a growing in­
terest for postmarket pharmacovigilance [7–9]. The process of ADEs
extraction from textual data includes two tasks: (1) ADE mention
extraction (ADE-ME) task which aims at recognizing a string of text




 - Corresponding author at: Laboratory of Informatics, Signals, Automatic, and Cognitivism (LISAC), Faculty of Sciences Dhar ELMehraz, Sidi Mohamed Ben
Abdellah University, Fez, Morocco.
_E-mail addresses:_ [eddrissiya.elallaly@usmba.ac.ma](mailto:eddrissiya.elallaly@usmba.ac.ma) (E.-d. El-allaly), [mourad.sarrouti@sumitovant.com](mailto:mourad.sarrouti@sumitovant.com) (M. Sarrouti), [noureddine.en-nahnahi@usmba.ac.ma](mailto:noureddine.en-nahnahi@usmba.ac.ma)
[(N. En-Nahnahi), ouatikelalaoui.said@uit.ac.ma (S. Ouatik El Alaoui).](mailto:ouatikelalaoui.said@uit.ac.ma)


[https://doi.org/10.1016/j.jbi.2021.103968](https://doi.org/10.1016/j.jbi.2021.103968)
Received 29 June 2021; Received in revised form 25 November 2021; Accepted 27 November 2021

Available online 4 December 2021
[1532-0464/© 2021 Elsevier Inc. This article is made available under the Elsevier license (http://www.elsevier.com/open-access/userlicense/1.0/).](http://www.elsevier.com/open-access/userlicense/1.0/)


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_



related to ADE components and (2) ADE relation extraction (ADE-RE)
task which aims at detecting relations between ADE mentions. In this
paper, we are interested in handling the second task where the mentions
are given from the gold set.
The ADE-RE task is the cornerstone of many NLP applications
including biomedical knowledge discovery [10] and question answering

[11,12]. Traditional approaches were primarily focused on machine
learning-based methods [13–15] which exploit various sets of features
and feed them into classifiers such as random forest and support vector
machines (SVM). However, these methods involve heavy handcrafted
features which are skill-dependent and labor intensive. Several studies
have successfully investigated deep neural networks for relation
extraction in general and specific domains, especially the combination
of recurrent neural networks (RNN) and bidirectional long short-term
memory network (Bi-LSTM) with attention mechanism [16–18]. Most
of them integrate dependency-based approaches as they involve rich
structural information from dependency graphs when they are incor­
porated into neural models. Recent works tended to explore the graph
convolutional network (GCN) [19] to improve the ability of relation
extraction task, specifically with respect to efficiency [20–22]. The GCN
can produce rich syntactic information of the graph as well as latent
feature representations of nodes. Formally, GCN transforms the de­
pendency graph of the input sentence into an undirected adjacency
matrix where the value 1 indicates that there is an edge between the two
corresponding words and 0, otherwise. Then, the obtained adjacency
matrix and words’ features are introduced to the graph convolution
operation to get high-level representation of the sentence.
However, despite the great success of GCN for relation extraction
task, some disadvantages remain. First, the label information of de­
pendency edges are different. For example, in Fig. 1, given the two edges
“appos” and “advmod” which are highlighted in yellow. The edge appos
(leflunomide, Pneumocystis) represents the appositional modifier rela­
tion between the head word “leflunomide” and the dependent word
“Pneumocystis”. Moreover, the edge advmod(Pneumocystis, especially)

“Pneumo­
is the adverbial modifier relation between the head word
cystis” and the dependent word “especially”. Despite the two edges
being different semantically, the GCN-based methods do not fully
involve this information to the adjacency matrix as they treat these
edges identically. Additionally, it is obvious that not all dependency
edges contribute equally for extracting ADE relationships. For instance,
given the two edges “det” and “prep” which are highlighted in black as
illustrated in Fig. 1, the edge det(setting, the) indicates the determinant
relation between the head word “setting” and the dependent word “the”.
Still, The edge prep(reported, in) represents the prepositional relation
between the head word “reported” and the dependent word “in”. The
semantic meaning of these two dependency edges shows that they have
less impact for determining the hypothetical relationship between the
two mentions “fatal” and “ ”.
leflunomide
Second, most of the standalone systems regard the relation extrac­
tion task as a classification problem where the instances for each
candidate pair are constructed from one sentence. For instance, given
the example (a) from Fig. 2, there are two relative instances: the first
instance is generated from the candidate relation between “Paradoxical
bronchospasm” and “Life threatening” while the second one is con­
structed from the candidate relation between “Paradoxical



bronchospasm” and “can”. However, the classification-based methods
treat the instances independently and do not take into account the
interaction between them which is crucial for determining relevant re­
lations. Third, there are many relations that include nested mentions,
which are embedded in other ones. In the example (b) from Fig. 2, the
severity mention “mild to moderate” contains two other mentions
“mild” and “moderate”. The three mentions have the effect relation with

the ADR mention “infections”. Further, there are other relations that
involve overlapping mentions (some tokens are shared by various
mentions). Consider the example (c) in Fig. 2, the two mentions “mild”
and “mild greater” share the common word “mild”. These mentions have

“ ”
the effect relation with the ADR mention aortic regurgitation .
In view of the above-mentioned drawbacks, we propose, in this
paper, an attentive joint model with transformer-based weighted GCN
for extracting ADE relations, called ADERel. Firstly, ADERel formulates
the ADE-RE task as an N-level sequence labelling. In fact, unlike con­
ventional relation classification-based methods, converting the task to a
unified sequence labelling offers significant advantage for dealing with
multi-head issue [23] and capturing more interaction between relations.
The proposed N-level sequence labelling is able to model complex re­
lations in different levels. Then, the ADERel system exploits our neural
joint model to process the N-level sequences jointly. The joint model
exploits a shared representation that incorporates the pre-trained lan­
guage models and our proposed weighted GCN, called WGCN. More
specifically, it employs BioBERT [24], a variant of bidirectional encoder
representations from transformers (BERT) pre-trained on biomedical
articles, to capture the contextualized representations and long-range
dependencies of input tokens. Second, the system adopts WGCN to
encode greater syntactic information from the dependency graph of the
input sentence. Due to the incoherence between words generated from
the dependency graph and the wordpiece unit of BioBERT, an alignment
procedure is used to resolve this issue. The WGCN’s adjacency matrix
assigns a score to each dependency edge within a sentence so as to
determine the edges that are the most influential for extracting ADE
relations. This allows WGCN to effectively leverage the diversified types
of dependency edges. Finally, it is obvious that each current level
heavily depends on the boundary information of its succeeding levels.
According to this point, ADERel adopts the multi-head attention [25] to
fully transmit boundary knowledge across levels. Experimental evalua­
tions show that ADERel achieves an F-score of 71.52% and 95.74% on

TAC 2017 [26] n2c2 2018 [27] shared tasks, respectively. ADERel is
superior in accuracy as compared with several state-of-the-art methods.
The contributions of this paper are summarized in the following points:


 - We develop ADERel, a neural joint system for extracting ADE re­

lations. It jointly learns N-level sequence labelling to effectively deal
with complex relations and integrate the interactions between
relations.

 - We propose a transformed-based weighted graph convolutional
network as a shared representation of ADERel to leverage richer
contextual and syntactical information.

 - We introduce a multi-head attention mechanism for exchanging
boundary knowledge across N-level sequences.

 - We conducted extensive experiments on TAC 2017 and n2c2 2018 to
demonstrate the effectiveness and the generalizability of ADERel.



**Fig. 1.** Example of a dependency graph for a sentence in TAC 2017 dataset noted with label of dependency edges.


2


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_


**Fig. 2.** Examples of ADE relations obtained from TAC 2017 dataset.



The remainder of the paper is organized as follows: Section 2 reviews
existing methods for extracting ADE relations from biomedical texts.
Section 3 describes our ADERel in detail. Section 4 reports and discusses
the obtained results of the proposed method. Finally, Section 5 con­
cludes the paper with a discussion of future works.


**2. Related work**


The most recent progress for extracting ADE relation from clinical
texts have been documented in several challenge workshops including:
the 2018 national NLP clinical challenge (n2c2) [27], medication,
indication and adverse drug events challenge (MADE 1.0) [28] and the
2017 text analysis conference (TAC) [29]. Such challenge has attracted
increasing attention from the NLP research community to address suc­
cessive shared tasks. For this purpose, TAC asked participants to assess
the state-of-the-art NLP systems for extracting relations between adverse
reactions and their related modified terms from drug labels. In this
context, IBM Research team [30] explored Bi-LSTM combined with
attention mechanism. Tao et al. [31] employed a regularized linear
regression model with rich features such as constituency, dependency
trees, lexicon and distances. Mart et al. [14] applied support vector
machines (SVMs) by combining lexical and semantic features. Gu et al.

[32] adopted an adversarially-trained piece-wise convolutional neural
network (CNN) to tackle the relation extraction task. Xu et al. [33]
developed a cascaded system based on Bi-LSTM and conditional random
fields (CRFs) networks for extracting entities and relations together.
The 2018 MADE 1.0 challenge further released an expert-curated
benchmark to promote research efforts in extracting medication and
ADE from electronic health records (EHRs) [28]. In this context,
Chapman et al. [13] adopted a two-stage approach based on random
forest by first identifying the ADE relations, and then assigning them the
relevant relation type. The two stages adopt the same features including
surface and candidate mentions. Dandala et al. [34] explored an
attention-based Bi-LSTM with medical domain ontology obtained from
unified medical language system (UMLS) concept unique identifier
(CUI) finder. Xu et al. [35] used SVM using C-Support Vector Classifier
by combining four features: bag of Words and entities, position and
distance. Magge et al. [36] adopted a random forest classifier with rich



handcrafted features such as entity types and number of words in en­
tities. Alimova et al. [37] also used a random forest classifier with dis­
tance, word, embedding and knowledge-based features.
Further studies on the 2018 n2c2 challenge were devoted to deep
and machine learning based methods for extracting ADEs and medica­
tion from EHRs [27]. For example, Wei et al. [38] proposed a joint
learning based on Bi-LSTM-CRF to identify both entities and relations.
They applied post-processing operations to fix some detected errors and
improve performance of the model. Peterson et al. [15] examined a
random forest classifier with candidate entities and syntactic features.
Christopoulou et al. [39] developed an ensemble method of weighted BiLSTM model and Walk-based model with majority voting technique for
intra-sentence. To enhance the performance of relation extraction task,
the authors proposed a separate model based on a transformer network
for inter-sentence relations. Yang et al. [40] explored a gradient boost­
ing classifier with semantic and context features. Chen et al. [41] pro­
posed an attention-based Bi-LSTM with only word embedding as feature.
IBM Research adopted a piece-wise Bi-LSTM combined with attention
mechanism. Kim et al. [42] applied SVM using distance, word embed­
ding and lexical features. Alimova et al. [43] investigated random forest
combined with word-based, distance-based and sentence embeddingbased features. GNTeam [44] explored a position-aware LSTM with
marker and word embeddings. Yang et al. [45] explored three
transformer-based models including BERT [46], RoBERTa [47], and
XLNet [48]. They adopted two strategies to handle inter-sentence re­
lations: unified model and distance-specific model.
Several studies have successfully explored the GCN-based methods
for relation extraction in the general domain. For instance, Zhang et al.

[21] proposed a contextualized GCN based on word embedding and BiLSTM for extracting relations from general texts. The authors integrated
a path-centric pruning strategy to retrieve relevant information from the
tree. Guo et al. [22] developed an attention-guided GCN. They incor­
porate a self-attention mechanism to selectively attend to the appro­
priate sub-structures that are beneficial for relation extraction. Li et al.

[49] proposed a GCN based-method with position attention and classi­
fication reinforcement modules. Inspired by the successes in the general
domain, recent advances on investigating GCN to the biomedical
domain have reached some success on relation extraction. For example,



3


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_



Park et al. [50] proposed a GCN based-method with Bi-LSTM and multihead attention for biomedical relation extraction. Their system takes as
input the word, POS, dependency and distance embeddings. Zhao et al.

[51] developed a combination of GCN and bidirectional gated recurrent
unit (Bi-GRU) with word and position embeddings as features.
Although these systems have proven to be quite successful for
improving relation extraction task, there are many challenging issues
that arise. First, the GCN-based methods do not fully benefit from the
syntactic information such as the diversified edge labels. Second, these
systems regard the relation extraction task as a classification problem
where the instances were treated independently. However, they neglect
the interaction between relations which are beneficial for extracting
ideal relations. Third, most of the aforementioned systems do not take
into account complex relations which contain at least one complex
mention (nested, discontinuous or overlapping).


**3. Method**


In this section, we describe our ADERel system. Fig. 3 shows the
overall structure of the proposed system and its main components.
Inspired by [52], ADERel formulates the ADE-RE task as an N-level
sequence labelling. It is obvious that each relation is made up of source
and target mentions. By adopting our N-level sequence labelling, we
assume that the source mention is given and we need to find its related
target attributes and relation types. The proposed system takes as input
the sequence of tokens describing the context of a given source mention
and ends up with N sequences of tags. Then, the ADERel system pro­
cesses the N sequences jointly. More specifically, a transformer-based
weighted graph convolutional network (TWGCN) is proposed as
shared representation to leverage richer contextual and syntactical in­
formation. TWGCN incorporates the pre-trained language model based
on transformer architecture and our proposed WGCN. Finally, multihead attention is applied to effectively transfer boundary information
across levels. We will provide the process of ADERel in detail in the
following sections.


_3.1. Pre-processing_


We perform two pre-processing operations: (1) sentence tokenization
using the GENIA Sentence Splitter [53] to segment the clinical texts into
a set of sentences, and (2) generating the dependency syntactic graph of
the sentences using Spacy toolkit. [1 ]


_3.2. N-level sequence labelling_


Generally, each ADE relation contains a dominant pair of target and
source mentions. For instance, in the TAC 2017 dataset, all relations
consist of ADR mentions as sources and other attributes as targets.
Inspired by [54–56,23], we expect that extracting all target mentions
and relation types for each source mention from a unified sequence
labelling can provide greater interaction between relations rather than
creating instances for each candidate pair independently. To this end,
we convert the ADE-RE task as an N-level sequence labelling by
modeling the complex relations in different levels where N indicates the
maximum overlapping and nested levels. Formally, let _T_ = { _t_ 1 _, t_ 2 _,_ … _, t_ _i_ _,_
… _, t_ _m_ } be a sentence consisting of _m_ tokens describing the context of a
given source mention. Our system aims at predicting _N_ sequences of
tags: _Y_ = {{ _y_ [1] 1 _[,][ y]_ [1] 2 _[,]_ [ …] _[,][ y]_ [1] _i_ _[,]_ [ …] _[,][ y]_ [1] _m_ [}] _[,]_ [ …] _[,]_ [ {] _[y]_ 1 _[N]_ _[,][ y]_ _[N]_ 2 _[,]_ [ …] _[,][ y]_ _[N]_ _i_ _[,]_ [ …] _[,][ y]_ _[N]_ _m_ [}}] where _y_ _[l]_ _i_
represents the _i_ -th tag at the _l_ -th level. To do so, we adopt the BIO
segment representation to tag each token within the context which re­
fers to the Beginning, Inside or Outside of the entity. We add two tags:
DB and DI to deal with discontinuous mentions. We generate the context



related to each source mention by replacing its position with a label
which is made up of two parts: the boundary tag (B, I, DB, DI) and the
type of the source mention. This allows the model to benefit from the
semantic information of the source mention so as to extract its related

attributes. Given the example (c) in Fig. 2, there are two source men­
tions: “aortic regurgitation” and “mitral regurgitation”, we create two
contexts for each mention as illustrated in Fig. 4. For each level, we
assign a tag to each token of the generated context. In addition to the
label (O), we define a new label which consists of three parts: the
boundary tag of the target attribute, the target type and the relation
type. For example, as shown in Fig. 4, the context of the source mention
“aortic regurgitation” is modeled with two levels. The first level tags the
tokens “mild” and “greater” as “ DB-Severity-Effect” and “DI-SeverityEffect”, respectively. The second level tags the token “mild” as “BSeverity-Effect”. This indicates that the mention “aortic regurgitation” is
related to “mild greater” and “mild” mentions with both effect relation.
The other mentions that are not related to “aortic regurgitation” like
“moderate greater” and “moderate” are labelled as (O) in the two levels.
The same procedure is applied to the source mention “mitral regurgi­
tation”. During the inference phase, the ADE-RE task is modelled by a
triple (Source, RelationType, Attribute). For instance, in Fig. 4, there are
four triples: (aortic regurgitation, Effect, mild greater), (aortic regurgi­
tation, Effect, mild), (mitral regurgitation, Effect, moderate greater) and
(mitral regurgitation, Effect, moderate).


_3.3. Transformer-based weighted graph convolutional network_


_3.3.1. BioBERT’s wordpiece alignment_
Recent advances in pre-trained contextual language models,
including BERT [46], have pushed new state-of-the-art performance for
several tasks in general and specific domains. BERT is based on multilayer bidirectional transformer architecture. This is equipped with the
multi-head attention mechanism to capture long-distance context
comprehension. BERT is pre-trained on two unsupervised tasks: masked
language model and next sentence prediction. It is further trained on two
general datasets including: BooksCorpus and text passages of English
Wikipedia. BioBERT [24], on the other hand, is a fine-tuned version of
BERT tailored to the biomedical domain. In addition to the general
domain corpus, it is pre-trained on PubMed Central full-text articles
(PMC) and PubMed abstracts (PubMed). Owing to the theoretical ben­
efits of BioBERT, this can allow the ADERel to better encode the sentence
as well as capture greater contextual and semantic relationships among
the contained words.

BioBERT splits the input context into wordpiece tokens. Therefore,
we need to map the dependency syntactic graph generated by spacy to
the wordpiece tokenization outcomes [57]. To do so, let _S_ = { _w_ 1 _,_ … _, w_ _i_ _,_
… _, w_ _n_ } be the word sequence generated by the parser. Let _T_ = { _t_ 1 _,_ … _, t_ _k_ _,_
… _, t_ _l_ _,_ … _, t_ _m_ } be the wordpiece tokens obtained by BioBERT. Each word
_w_ _i_ in _S_ has its corresponding subsequence in _T_ denoted by _T_ ( _w_ _i_ ) = { _t_ _k_ _,_ … _,_
_t_ _l_ }. We design the following rule for alignment procedure: if _r_ _ij_ repre­
sents the syntactic relation between _w_ _i_ and _w_ _j_, then the same relation _r_ _ij_
is mapped to any token in _T_ ( _w_ _i_ ) and _T_ ( _w_ _j_ ). As shown in Fig. 5b, BioBERT
splits the word “Fetal” into “Fe” and “##tal”. Thus, the BioBERT’s
wordpiece alignment represents the syntactic relation amod(Fetal,
harm) as amod(Fe, harm) and amod(##tal, harm). According to this
point, we extend the N-level sequence labelling by tagging the word­
piece tokens that start with “##” with the label (X) and the first
wordpiece token with the defined label.
The final hidden states of BioBERT _H_ _biobert_ = { _h_ 1 _, h_ 2 _,_ … _, h_ _m_ } are used
to encode the tokens _T_ as follows:


{ _h_ 1 _, h_ 2 _,_ … _, h_ _m_ } = _BioBERT_ ({ _t_ 1 _, t_ 2 _,_ … _, t_ _m_ }) (1)



_3.3.2. Weighted graph convolutional network_
Unlike the conventional GCN-based methods, where the dependency

1
[https://spacy.io/.](https://spacy.io/) edges were treated identically, we expect that the proposed WGCN has


4


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_


**Fig. 3.** The main architecture of our proposed ADERel system.


**Fig. 4.** Examples of N-level sequence labelling.


**Fig. 5.** Example of BioBERT’s wordpiece alignment.



the ability to determine which edge labels of the sentence are the most
influential so as to extract relevant ADE relations as well as model rich
syntactic information. Let _G_ be the graph obtained after alignment



which consists of _m_ nodes and _p_ edges. Firstly, we need to transform _G_ to

a weighted adjacency matrix _A_ with _m_ × _m_ nodes. To do so, let _W_ _[dep]_ ∈



5


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_



R _[d]_ [×|] _[V]_ _[dep]_ [|] be the embedding matrix of dependency tags, where _d_ is the
dependency embedding dimension and | _V_ _[dep]_ | is the size of dependency
relations’ vocabulary. _W_ _[dep ]_ is obtained by training word2vec on the
entire datasets so as to better estimate the dependency vectors with low
frequency. We assign a score function to each edge based on the de­
pendency vectors of the head and dependent nodes. Concretely, if _r_ _k_
represents the dependency edge between the nodes _i_ and _j_, the element
_A_ _ij_ is computed by using Eq. 2:


_i_ _, w_ _[dep]_ _j_ )
_A_ _ij_ = _[dot]_ [(] _[w]_ _[dep]_ _d_ (2)


where _w_ _[dep]_ _i_ ∈ R _[d ]_ and _w_ _[dep]_ _j_ ∈ R _[d ]_ represent the dependency-level embed­

dings of the nodes i and j, respectively. dot denotes the dot-product
operation on two vectors. Besides, the element _A_ _ji_ is equal to _A_ _ij_ by
considering that the generated graph is undirected. We further add a
self-loop to each node in order to learn the node information itself by

adding the identity matrix _I_ _m_ to the weighted adjacency matrix: _A_ [̂] =
_A_ + _I_ _m_ . To avoid vanishing and exploding gradient, we normalize the

weighted adjacency matrix _A_ [̂] as follow:


_A_ ̃ = _D_ [1] _[/]_ [2] _AD_ ̂ [1] _[/]_ [2] (3)


where _D_ = [∑] _[m]_ _j_ =1 _[A]_ [̂] _[ij ]_ [is the degree matrix. The WGCN takes as input the ]

weighted adjacency matrix _A_ [̃] and the feature matrix _H_ _biobert_ . Then, the
convolution operation of WGCN is computed by Eq. 4:


_H_ _wgcn_ = _σ_ ( _AH_ [̃] _biobert_ _W_ + _b_ ) (4)


where _W_ and _b_ are the weight matrix and bias term, respectively. _σ_ is the
ReLU nonlinear activation function.


_3.4. Multi-head attention across levels_


_H_ _wgcn_ ∈ R _[m]_ [×] _[d]_ _[wgcn ]_ represents the shared representation to all levels
where _d_ _wgcn_ is the length of WGCN. However, each current level heavily
depends on the boundary information of its succeeding levels, which is
intuitive to the complex relation nature. We thus adopt the self-attention
mechanism to fully exchange this information across levels. The essence
of self-attention is computing the representation of the sequence by
relating different positions of the text sequence. The boundary infor­
mation of words can be obtained by self-attention since it captures the
word dependencies inside the sentences and assigns the attention score
between each word. In order to learn important features from different
representation aspects, we adopt the multi-head attention which applies
the self-attention mechanism multiple times in parallel.
Let h be the number of heads. Different learnable weight matrices
( _W_ _[Q]_ _i_ [∈] [R] _[d]_ _[wgcn]_ [×] _[d]_ _[q]_ _[,]_ _[W]_ _i_ _[K]_ [∈] [R] _[d]_ _[wgcn]_ [×] _[d]_ _[k]_ _[,]_ _[W]_ _[V]_ _i_ [∈] [R] _[d]_ _[wgcn]_ [×] _[d]_ _[v]_ [) are applied to the hidden ]

states of the previous level _H_ [[] _[l]_ [−] [1][]] to obtain the query (Q), key (K) and
value (V) for each head _i_ as shown in Eq. 5:


( _Q_ _i_ _, K_ _i_ _, V_ _i_ ) = ( _H_ [[] _[l]_ [−] [1][]] _W_ _[Q]_ _i_ _[,][ H]_ [[] _[l]_ [−] [1][]] _[W]_ _[K]_ _i_ _[,][ H]_ [[] _[l]_ [−] [1][]] _[W]_ _[V]_ _i_ [)] (5)


where computed based on the scaled dot-product attention. More specifically, _d_ _q_ = _d_ _k_ = _d_ _v_ = _d_ _wgcn_ _/h_ . After that, the attention weight is
the dot-product of the key and query is computed to get the attention
score, then scaled by the root square of _d_ _q_ . The attention weight is ob­
tained by applying the softmax function to its corresponding attention
score and the value V. The obtained matrix is computed as follows:


_head_ _i_ = _softmax_ ( _[Q]_ _[i]_ _[K]_ ~~̅̅̅~~ _i_ _[T]_ ~~**̅**~~ ~~_V_~~ _i_ ) (6)
~~√~~ _d_ _q_


Finally, the current level _H_ [[] _[l]_ []] takes as input the concatenation of the h
times attention and once again linear projections. This is expressed by
Eq. 7:



_H_ [[] _[l]_ []] = _concat_ ( _head_ 1 _,_ … _, head_ _h_ ) _W_ _[o]_ (7)


where _W_ _[o]_ ∈ R _[d]_ _[wgcn]_ [×] _[d]_ _[wgcn ]_ is the weight matrix to be learned. The initial
level _H_ [[][0][]] takes _H_ _wgcn_ as input.


_3.5. Output layer and training objective_


The ADERel system predicts the tags sequence for each level by using
the softmax function as shown in Eq. 8, which encodes the probability
distribution over the total number of tags _N_ 1 .


_P_ _l_ ( _y_ _[l]_ = _j_ | _t_ ) = _softmax_ ( _WH_ [[] _[l]_ []] + _b_ ) (8)


where _W_ ∈ R _[N]_ [1] [×] _[d]_ _[wgcn ]_ and _b_ ∈ R _[N]_ [1 ] are the weight and bias terms,
respectively, which are used for affine transformation. ADERel is jointly
trained to optimize the categorical cross-entropy loss function for each

9:
level defined by Eq.



_m_
ℒ _l_ = − ∑

_t_ =1



_N_ 1
∑

_j_ =1



̂ _y_ _[l]_ _t_ _j_ _[log]_ [(] _[P]_ _[l]_ [(] _[y]_ _[l]_ [ =] _[ j]_ [|] _[t]_ [))] (9)



where ̂ _y_ _[l]_ _t_ _j_ [is the ground-truth label at ] _[l]_ [-th level. The final loss of ADERel ]

is the weighted sum of all levels’ losses formulated as follows:



_N_
_L_ _total_ = ∑

_l_ =1



_w_ _[l]_ ℒ _l_ (10)



where _w_ _[l ]_ is the weight parameter per level, such that all losses are
approximately at the same scale. The tag with the maximum probability
is selected as the best tag of the corresponding token.


**4. Experimental results and discussion**


_4.1. Datasets_


In our experiments, we evaluated the proposed ADERel system on
two datasets: TAC 2017 and n2c2 2018. The statistics of the two datasets

are shown in Table 1.


 - **TAC 2017** [26]: contains 200 drug labels which were split into 101
for training and 99 for test set. The dataset defines three relation
types: Hypothetical, Effect and Negated.


**Table 1**

Statistics of TAC 2017 and n2c2 2018 datasets.


Dataset Type training test Description


TAC Effect 1454 1181 connects an ADR mention with

2017 Severity one
Hypothetical 1611 1486 links the ADR with Factor,
DrugClass or Animal mentions
Negated 163 288 connects the ADR with Factor or
Negation mentions


n2c2 ADE-Drug 1107 733 Relationship between the ADE
2018 mention and drug name
Strength-Drug 6702 4244 links a drug strength with its

name

Dosage-Drug 4225 2695 connects a drug dosage with its

name

Duration- 643 426 links a duration with the drug
Drug name.
Frequency- 6310 4034 links a frequency with the drug
Drug name.
Form-Drug 6654 4374 connects a drug form and its

name

Route-Drug 5538 3546 Relationship between the route
and the drug name
Reason-Drug 5169 3410 links a drug reason with its name



6


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_




 - **N2c2 2018** [27]: provides 505 discharge summaries extracted from
MIMIC-III. It is divided into two parts: training set and test set with
303 and 202 of clinical notes, respectively. The dataset includes eight
relation types: ADE-Drug, Dosage-Drug, Strength-Drug, DurationDrug, Frequency-Drug, Route-Drug, Form-Drug and Reason-Drug. In
this dataset, we excluded the inter-sentential relations which account
approximately 6% on the training set and 7% on the test set.


_4.2. Evaluation metrics_


To evaluate the performance of ADERel, we used the official scripts
provided by the 2018 n2c2 and 2017 TAC challenges. The microaverage Precision (P), micro-average Recall (R) and micro-average F1score (F1) have been employed by both challenges as the main metrics to
all relation types. For system ranking, the exact matching score and
lenient matching score are used as the primary metrics on the 2017 TAC
and 2018 n2c2 challenges, respectively. To report the statistical signif­
icant results, we adopted the approximate randomization test [58].


_4.3. Implementation details_


Our experiments were carried out on Google Collaboratory tool. [2 ] To
validate the model hyper-parameters, we used 10% of the training set as
the validation set by adopting stratified sampling split. The best models
were selected based on the highest performance of the validation set.
The dimension of dependency embedding and the maximum sequence
length were set at both 200. The hidden unit number of WGCN was the
same as the BioBERT dimension. The number of levels and multi-heads

was set at both 2 on the TAC 2017 dataset. As the complex relations do
not exist on the n2c2 2018 dataset, we set the number of levels to 1.
Consequently, the attention used across levels was not applied in this
case. The Adam algorithm was used to optimize the training of our
system with learning rate, weight decay and batch size being 5 _e_ [−] [5], 0.05
and 8. Dropout was applied to 0.3 in the BioBERT, WGCN, and attention
layers. ADERel yields optimal performance for 20 and 10 epochs on TAC
2017 and n2c2 2018 datasets, respectively. The ADERel source code is
[available at https://github.com/drissiya/ADERel.](https://github.com/drissiya/ADERel)


_4.4. Effects of contextualized representations_


We conducted several experiments to demonstrate the potential
impact the contextualized embeddings have on the overall performance
of the proposed ADERel system. We explore the following pre-trained
language models:



(the original BERT vocabulary) and scivocab (the vocabulary built
using SentencePiece on the scientific corpus). We adopt the scivocab
version recommended by the authors.


Table 2 shows the reported results. As can be seen from the table, the
BioBERT model exhibits better performance than other models on both
datasets. On the TAC 2017 dataset, BioBERT brings an average
improvement of up to 9.02%, 10.53% and 9.82% in terms of P, R and F1,
respectively. This difference is statistically significant at _p <_ 0 _._ 01 across
all models with P-value of 9 _._ 99 _e_ [−] [05 ] and 0 _._ 004 for BERT and SciBERT,
respectively. The main improvement is observed on extracting Negated,
Effect and Hypothetical relations by an average of 9.06%, 10.57% and
10.44% in terms of F1, respectively. On the n2c2 2018 dataset, BioBERT
increases the overall performance by 1.8%, 2.49% and 2.15% in terms of
P, R and F1, respectively. The increased performance is statistically
significant at _p <_ 0 _._ 01 across BERT with P-value of 9 _._ 99 _e_ [−] [05] . The F1 of
BioBERT is 0.11%, 0.05%, 0.52%, 0.56%, 0.85% and 0.43% higher than
SciBERT on the extraction of ADE-Drug, Dosage-Drug, Frequency-Drug,
Form-Drug, Route-Drug and Reason-Drug relations, respectively.
Theoretically, the performance of ADERel heavily depends on the
quality and size of corpora on which the BERT-based models were pretrained. Indeed, the BioBERT model was trained on 7.8B tokens from
general and biomedical domains compared to other models (SciBERT
and BERT was pre-trained on 3.17B and 3.3B tokens, respectively).
Thus, BioBERT has the ability to deal with the shift of word distribution
between specific domain corpus and general domain corpus. As a result,
the BioBERT model makes the proposed system more effective for
extracting various types of ADE relations.
Moreover, we observed that the average improvement between
BERT and BioBERT is very large on the TAC 2017 dataset compared with
the n2c2 2018 one. This is likely caused by the highly correlated re­
lations within the dataset. To see how much this factor causes perfor­
mance instability, we trained every model five times with various
random seeds. Then, we reported the F1-score on each dataset and
computed their means and standard deviations. Fig. 6 shows the ob­
tained results. It can clearly be seen from the figure that the largest
variance happens on the TAC 2017 dataset irrespective of the used
model in comparison with the n2c2 2018 dataset. It means that there are
many relations on the TAC 2017 dataset that correlate with each other.
In other words, the change of prediction on one relation affects the
predictions on all the correlated relations causing the instability and
inconsistency in the final performance of different runs.


_4.5. Effects of weighted graph convolutional network_




- **BERT** [46]: there are two released models: BERT-Large and BERTBase with 340 million and 110 million total parameters, respec­ To evaluate the effectiveness of the proposed WGCN, we compared it
tively. BERT-Large contains 24 transformer layers with 1024 hidden with conventional GCN. Compared with WGCN, if there is a dependency
states and 16 self-attention heads, while BERT-Base has 12 trans­ edge between two nodes _i_ and _j_, then the element _A_ _ij_ of the adjacency
former layers with 768 hidden states and 12 self-attention heads. Due matrix in GCN takes the value 1. As shown in Table 3, WGCN out­
to the computational complexity limitations of BERT-Large, our performs GCN on TAC 2017 dataset by an average of 2.03% in terms of
system adopts the base version of BERT. F1. The improvement is mainly observed on extracting Hypothetical and

- **BioBERT** [24]: There are five versions of released pre-trained Effect relations by 2.09% and 2.31% in terms of F1, respectively. WGCN
weights: BioBERT-Base v1.0 (pre-trained for 200 K and 270 K steps performs slightly better than GCN on the n2c2 dataset as it increases the
on PubMed and PMC, respectively), BioBERT-Base v1.0 (pre-trained F1 by only 0.1%. The main improvement is detected on extracting ADEfor 270 K steps on PMC), BioBERT-Base v1.0 (pre-trained for 200 K Drug, Form-Drug and Route-Drug by an average of 0.79%, 0.56% and
steps on PubMed), BioBERT-Large v1.1 (pre-trained for 1 M steps on 0.46% in terms of F1, respectively. As expected, WGCN leads to better
PubMed) and BioBERT-Base v1.1 (pre-trained for 1 M steps on results due to the following reasons. First, as the source and target
PubMed). In our work, we adopt the latest version BioBERT-Base mentions can be located close to each other in clinical texts, WGCN can

v1.1.

effectively identify the most influential edges which is vital for dealing

- **SciBERT** [59]: is pre-trained on full-text papers from Semantic with these complicated cases. Fig. 7 illustrates the weight distribution of
Scholar (82% from the biomedical domain and 18% from the com­ the adjacency matrix of WGCN and GCN on an example from TAC 2017
puter science domain). It has two versions of vocabulary: basevocab dataset to find out how they work and which way they take the most

advantages of the edges information. It can clearly be seen from the
figure that the edges pobj(of, seizures) and acl(seizures, associated) have
2
[https://colab.research.google.com/.](https://colab.research.google.com/) greater weights than the edges aux(increase, may) and det(risk, the).


7


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_


**Table 2**

Effects of contextualized embeddings on TAC 2017 and n2c2 2018 datasets.


Dataset Type BioBERT SciBERT BERT


P R F1 P R F1 P R F1


TAC 2017 Effect 68.15 66.38 67.25 56.17 54.98 55.57 57.85 55.56 56.68

Hypothetical 80.08 79.53 79.81 79.89 77.45 78.65 69.91 68.84 69.37
Negated 69.58 68.62 69.10 74.90 67.93 71.25 69.06 53.10 60.04

Overall 72.96 70.14 71.52 68.69 64.51 66.54 63.94 59.61 61.70


n2c2 2018 ADE-Drug 84.46 78.58 81.41 84.54 78.31 81.30 71.04 63.57 67.10
Strength-Drug 98.88 97.69 98.28 98.81 97.88 98.34 97.10 97.05 97.08
Dosage-Drug 95.52 96.44 95.97 95.54 96.29 95.92 93.59 94.77 94.17
Duration-Drug 91.61 89.67 90.63 92.72 89.67 91.17 87.39 73.24 79.69
Frequency-Drug 98.52 95.51 96.99 98.03 94.97 96.47 98.01 94.17 96.06
Form-Drug 97.65 95.77 96.70 97.13 95.18 96.14 97.34 94.65 95.98
Route-Drug 96.99 96.28 96.63 95.90 95.66 95.78 95.32 94.87 95.10
Reason-Drug 93.93 90.82 92.35 93.01 90.85 91.92 90.88 85.57 88.14

Overall 96.63 94.86 95.74 96.17 94.57 95.36 94.83 92.37 93.59


**Fig. 6.** The mean and standard deviation of the overall F1-score on TAC 2017 and n2c2 2018 datasets over each transformer-based models.


**Table 3**

Comparison between WGCN and GCN on TAC 2017 and n2c2 2018 datasets.


Dataset Type WGCN GCN


P R F1 P R F1


TAC 2017 Effect 68.15 66.38 67.25 65.84 64.06 64.94

Hypothetical 80.08 79.53 79.81 77.99 77.45 77.72
Negated 69.58 68.62 69.10 69.42 69.66 69.54

Overall 72.96 70.14 71.52 70.93 68.11 69.49


n2c2 2018 ADE-Drug 84.46 78.58 81.41 84.02 77.49 80.62
Strength-Drug 98.88 97.69 98.28 98.30 98.37 98.34
Dosage-Drug 95.52 96.44 95.97 95.68 96.96 96.31
Duration-Drug 91.61 89.67 90.63 93.87 89.91 91.85
Frequency-Drug 98.52 95.51 96.99 98.44 95.59 96.99
Form-Drug 97.65 95.77 96.70 97.09 95.22 96.14
Route-Drug 96.99 96.28 96.63 95.75 96.59 96.17
Reason-Drug 93.93 90.82 92.35 94.08 90.94 92.48

Overall 96.63 94.86 95.74 96.30 94.99 95.64



The edge pobj(of, seizures) indicates the object of a preposition relation
between the head word “of ” and the dependent word “seizures”. Still,
the edge acl(seizures, associated) represents the adnominal clause be­
tween the head word “seizures” and the dependent word “associated ”.
This shows that the edges pobj and acl have more impact for extracting
the Hypothetical relation between seizures and risk mentions, as shown
semantically. In contrast to the proposed WGCN, the conventional GCN
predicts incorrect relations since it treats all edges in the sentence



identically. This verifies our hypothesis that assigning a weight to each
edge in the sentence is crucial for extracting ideal ADE relations. Second,
WGCN benefits from BioBERT embedding used as initial hidden states of
graph’s nodes. Existing studies on GCN-based methods largely concern
the utilization of word embedding-based methods for graph construc­
tion. However, they fail to capture the context information which is very
important for understanding the meaning of the sentence to some extent.
The integration of contextual embedding from BioBERT in a feature


8


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_


**Fig. 7.** The weight distribution of the adjacency matrix of WGCN and GCN.



based approach helps WGCN to improve the performance of ADERel by
involving word disambiguation and sequence order implicitly. This al­
lows the system to be more effective for encoding and learning both
contextual and syntactic information into each level without computa­
tionally expensive fine-tuning of BioBERT.


_4.6. Effects of segment representation_


Several segment representations have been employed to label tokens
in the sentence. BIO and BILOU are the most popular segments used for
biomedical named entity recognition task. BIO tags each token as either
the beginning, inside or outside of target mention. BILOU further dis­
tinguishes the last token of multi-token target mention as well as unit
target mention. However, choosing the ideal segment is a complex
problem. Thus, we examined the performance of the proposed ADERel
regardless of the chosen segment. According to the Table 4, the BIO
segment achieves better results compared to the BILOU one on both
datasets. It increases the F1 by an average of 1.3% and 0.5% on the TAC
2017 and n2c2 2018 datasets, respectively. On the TAC 2017 dataset,
The P, R and F1 of BIO exceed those of the BILOU for Effect relation by
5.2%, 5.8% and 5.51%, respectively. On the n2c2 2018 dataset, the F1 of
BIO 0.27%, 2.73%, 0.09% and 0.55% is higher than that of BILOU for
extracting ADE-Drug, Form-Drug, Route-Drug, Reason-Drug relations,
respectively. Although the BILOU segment is more expressive for



capturing fine-grained distinctions of entity components compared to
the BIO segment, it brings performance degradation due to the following
factors. First, the BIO segment outperforms the BILOU one in terms of
computational complexity as it requires fewer labels. The total number
of labels needed to annotate the datasets using BIO and BILOU is
_N_ 1 = 2 +4*| _A_ |*| _R_ | and _N_ 1 = 2 + 7*| _A_ |*| _R_ |, respectively, where | _R_ | is the
length of relation types and | _A_ | is the length of attribute types. It is
obvious that the labels’ number of the BILOU segment is much higher
than that of the BIO segment. Thus, the labels’ distribution becomes very
unbalanced under the BILOU segment which leads to the sparsity
problem and increases the complexity of ADERel. Second, The BILOU
segment suffers from inconsistent label assignment where the same
word that appears in different contexts is assigned with different labels.
For instance, given the two following contexts: “Warfarin 3 mg alter­
nating with 6 mg PO daily” and “ TACROLIMUS - (Dose adjustment - no
new Rx) - 1 mg Capsule - 3”, the token “3” is assigned with “B-Strength”
and “U-Dosage” with regard to the two contexts, respectively. Thus, the
system correctly identifies the Strength-Drug relation between “3 mg”
and “Warfarin” mentions, but fails to identify the Dosage-Drug relation
between “3” and “TACROLIMUS” mentions. This makes BILOU not

much effective for distinguishing ADE relations from common texts. As a
result, the BIO segment is the best choice for the ADERel system.



**Table 4**

Comparison between BIO and BILOU segments on TAC 2017 and n2c2 2018 datasets.


Dataset Type BIO BILOU


P R F1 P R F1


TAC 2017 Effect 68.15 66.38 67.25 62.95 60.58 61.74

Hypothetical 80.08 79.53 79.81 81.46 79.60 80.52
Negated 69.58 68.62 69.10 77.61 69.31 73.22

Overall 72.96 70.14 71.52 72.21 68.34 70.22


n2c2 2018 ADE-Drug 84.46 78.58 81.41 85.16 77.49 81.14
Strength-Drug 98.88 97.69 98.28 98.86 97.90 98.38
Dosage-Drug 95.52 96.44 95.97 95.69 97.11 96.39
Duration-Drug 91.61 89.67 90.63 94.80 89.91 92.29
Frequency-Drug 98.52 95.51 96.99 98.84 95.34 97.06
Form-Drug 97.65 95.77 96.70 97.26 90.90 93.97
Route-Drug 96.99 96.28 96.63 97.09 96.00 96.54
Reason-Drug 93.93 90.82 92.35 93.87 89.82 91.80

Overall 96.63 94.86 95.74 96.73 93.82 95.25


9


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_



_4.7. Effects of attention mechanism_


To better understand the behaviors of attention mechanism trans­

mitted between levels on the TAC 2017 dataset, we performed a com­
parison between multi-head attention and self-attention. We further
evaluated the order of levels by adopting two directions:


 Outside-in order: we first extract the outermost entities in the first
level, then we identify the inner ones in the succeeding levels. As
shown in Fig. 4, the attribute “mild” is embedded into the attribute
“mild greater”. We first identify the relation (aortic regurgitation,
Effect, mild greater) in the first level, then we extract the relation
(aortic regurgitation, Effect, mild) in the second level.

 Inside-out order: we first identify the innermost entities in the first
level, then we extract the outer ones in the next levels. For example,
we first extract the relation (aortic regurgitation, Effect, mild) in the
first level, then we identify the relation (aortic regurgitation, Effect,
mild greater) in the second level.


Table 5 presents the experimental results. According to the table, it is
notable that multi-head attention with outside-in order achieves better

performance in comparison with self-attention since it brings an average
enhancement of up to 1.06%, 1.57% and 1.33% in terms of P, R and F1,
respectively. The F1 of multi-head attention is 2.25% and 0.45% higher
on the extraction of Effect and Hypothetical relations, respectively.
Indeed, in contrast to the self-attention, the multi-head attention learns
relevant information from different representation subspaces at
different positions which makes it more effective for exchanging
boundary information between levels. For example, given the following
sentence: “Grade 3 or 4 late-onset neutropenia (onset at least 42 days
after last treatment dose)”, there are two relations: (late-onset neu­
tropenia, Effect, Grade 3) and (late-onset neutropenia, Effect, Grade 4).
The ADERel system effectively extracts the two relations by using multihead attention. However, under the self-attention mechanism, it fails to
correctly extract the second relation that appears in the second level as it
misses the right boundary “4” of the attribute “Grade 4”. In addition, we
observed that the outside-in order performs better than inside-in order
on both attention mechanisms (up to 2.54% and 0.5% in terms of the
overall F1 for multi-head attention and self-attention, respectively). This
indicates that extracting the non-embedded entities in the first level
allows the system to transmit sufficient information about the innermost
entities in the succeeding levels. For instance, in the sentence: “As an

– ”
adverse reaction, grade 3 4 thrombocytopenia was reported, the first
level contains the relation (thrombocytopenia, Effect, grade 3) while the
second level has the relation (thrombocytopenia, Effect, grade 4). By
adopting the inside-out order, the system extracts incorrect relation in
the second level (thrombocytopenia, Effect, grade 3–4) which serves as
false-positive of ADERel. Consequently, the outside-in order is the best
choice for our system.


_4.8. Ablation study_


To investigate the impact of each component of ADERel, we con­
ducted an ablation study on the TAC 2017 dataset as follows:


**Table 5**

Effect of attention mechanism across levels on TAC 2017 dataset.




 - **BioBERT** : the shared representation has only the BioBERT model
without attention across levels.

 - **BioBERT þ WGCN** : the shared representation has the combination
of BioBERT and WGCN without attention across levels.

 - **BioBERT þ Attention** : the shared representation has only BioBERT
with attention across levels.

 - **BioBERT þ WGCN þ Attention** : the shared representation has the
combination of BioBERT and WGCN with attention across levels.


The experimental results are shown in Table 6. The results showed
that combining BioBERT and WGCN improves the performance of
ADERel on TAC 2017 dataset, where the P, R and F1 increase by 1.29%,
1.8% and 1.57%, respectively. It means that leveraging contextual and
structural information together is of considerable significance. The
increased performance is not significant on the n2c2 2018 dataset as the
F1 is improved by only 0.2%. This suggests that generating dependency
syntactic graph with existing general-purpose NLP toolkits such as spacy
are not easily adapted to the clinical texts due to their complex nature.
Moreover, adding multi-head attention across levels brings an average
improvement of up to 0.64% in terms of F1 on TAC 2017 dataset. This
result demonstrates that the multi-head attention has significant impact
for dispatching boundary information between levels. Finally, the
combination of BioBERT, WGCN and multi-head attention leads ADERel
to achieve better results as it improves the P, R and F1 by 1.32%, 2.84%
and 2.12%, respectively. This indicates that all components of ADERel
are complementary to each other for extracting complex ADE relations.
Additional results are described and analyzed in the supplementary
material.


_4.9. Performance comparison with state-of-the-art systems_


We compared the performance of ADERel with its best-performing
model to the top systems participating in the 2018 n2c2 challenge.
The UTH system [38] developed a joint learning system based on BiLSTM-CRF to identify the entities and relations together. Yang et al.

[45] explored various transformer-based models where the best per­
formance is achieved by XLNet. The VA system [15] employed a random
forest with various features. The NaCT system [39] proposed an
ensemble method of deep neural networks with majority voting tech­
nique. The UFL system [40] adopted a machine learning method based
on SVM. The MDQ system [41] proposed a Bi-LSTM with attention
mechanism. It can be seen from the Table 7 that our model outperforms


**Table 6**

Ablation study on TAC 2017 and n2c2 2018 datasets.


Type TAC 2017 n2c2 2018


P R F1 P R F1


BioBERT 71.64 67.30 69.40 95.87 95.21 95.54

BioBERT + WGCN 72.93 69.10 70.97 96.63 94.86 95.74

BioBERT + Attention 71.34 68.80 70.04 – – –

BioBERT + WGCN + 72.96 70.14 71.52 – – –

Attention


“–” indicates that the attention is not applied when the number of levels equals

to 1.



Type Multi-head attention Self-attention


Outside-in Inside-out Outside-in Inside-out


P R F1 P R F1 P R F1 P R F1


Effect 68.15 66.38 67.25 63.26 60.39 61.79 65.87 64.15 65.00 65.62 64.93 65.27

Hypothetical 80.08 79.53 79.81 79.70 77.38 78.52 80.17 78.56 79.36 78.35 77.59 77.96
Negated 69.58 68.62 69.10 77.29 66.90 71.72 76.78 70.69 73.61 73.55 70.00 71.73

Overall 72.96 70.14 71.52 71.47 66.65 68.98 71.90 68.57 70.19 71.13 68.30 69.69


10


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_



**Table 7**

Performance comparison with state-of-the-art systems on n2c2 2018 dataset.


Models P (%) R (%) F1 (%)


UTH [38] – – 96.30
Yang et al. [45] – – 96.10
VA [15] – – 95.3

NaCT [39] 94.63 94.80 94.72

UFL [40] 96.23 93.00 94.59

MDQ [41] 94.55 94.29 94.42

ADERel 96.63 94.86 95.74


“–” indicates that the results are not provided.


the second ranking system by an average of 0.44% in terms of F1. The
combination between WGCN and BioBERT can not only automatically
encode the feature representation of the sentence but also improves the
overall performance, as shown experimentally. ADERel further im­
proves the P, R and F1 by 2.08%, 0.57% and 1.32% in comparison with
the MDQ system. Indeed, these systems neglect the correlations between
relations as they treat each one independently. This shows that con­
verting the task to the sequence labelling problem is effective for dealing
with this issue. Yang et al. [45] performs better than ADERel, owing to
the integration of inter-sentence relations where the candidate mentions
are located in different sentences. Beside, although UTH outperforms
ADERel, the increased performance is mainly caused by applying postprocessing operations. The system designed some rules to fix the er­
rors generated by the model and improve its performance (the F1 of UTH
without post-processing operations was 93.99). This confirms the ben­
efits of all components of ADERel for extracting relevant relations. on
the other hand, we remedied two prevailing challenges: (1) there are
some relations that occur with a low frequency words (such as the ADEDrug and Reason-Drug relations, where the best system obtained only
79.46% and 75.79%, respectively, in terms of F1 without postprocessing operations) and (2) the polysemy issue where a disease can
be either a reason or ADE for drugs. The ADERel system has the ability to
solve these problems for the following reasons. First, the ADERel adopts
the WordPiece tokenization to split the low frequency and unseen words
into pieces which alleviate the out-of-vocabulary issue. Second, the deep
structure of transformer-based models allows ADERel to distinguish
between various meanings of the concepts by generating different rep­
resentations of the sentences according to the surrounding contexts.


_4.10. Error analysis_


We carried out an error analysis to gain further insights about the
best model of ADERel. The most common errors on both TAC 2017 and

n2c2 datasets were resulted from incorrect attributes that are not linked

to the source mentions. For example, given the sentence from the TAC
2017 dataset: “Hepatic steatosis associated with JUXTAPID may be a
risk factor for progressive liver disease, including steatohepatitis and
cirrhosis (5.1).”, there is an Hypothetical relation between the source
mention “steatohepatitis” and the target mention “risk”. However,
ADERel was wrongly extracted the relation between “steatohepatitis”
and “may”. In addition, we observed that another frequent kind of error
on the n2c2 dataset was caused by inter-sentential relationships which
serve as false-negative of ADERel. In the sentence: “ Hyponatremia due
to trileptal/HCTZ …You were admitted to the hospital because your
Sodium level was low.”, our model does not take into account the ADEDrug relation between “ Sodium level was low” and “ trileptal” due to
the long distance between the two mentions.


_4.11. Limitations_


Although its effectiveness, the ADERel system has limitations which
point out here. First, generating dependency syntactic graphs that are
able to understand the linguistic structure of clinical texts has not been
fully explored in our work. In fact, there are several NLP toolkits for



dependency parsing that have widely been used by the large NLP
community including spaCy, UDify, Flair and Stanford CoreNLP since
they are easy to use and provide high performance. However, they are
not easily adapted to the clinical texts due to their complex nature.
Consequently, generating dependency syntactic graphs by clinical NLP
toolkits has the potential to improve the performance of the ADERel
system. Second, our system does not integrate various heterogeneous
graphs extracted from medical sources such as DrugBank. This could be
beneficial for leveraging syntactic and semantic information of the GCNbased methods.


**5. Conclusion and future works**


In this paper, we proposed an attentive joint model with transformerbased weighted graph convolutional network for extracting ADE re­
lations, called ADERel. First, ADERel transformed the task into N-level
sequence labelling. This had the ability to cover various kinds of re­
lations by modeling the complex ones in different levels. It further in­
tegrated more interaction between relations which is crucial for
extracting relevant relations. Second, the proposed model processed
jointly the N sequences by adopting a combination of BioBERT and
WGCN as a shared representation. Compared with GCN-based methods,
WGCN assigned a score to each dependency edge to measure its
importance for extracting ideal relations within the sentence. This made
WGCN more effective for capturing richer syntactic information. Third,
the ADERel model adopted the multi-head attention across levels. This
effectively dispatched boundary information between levels. ADERel
proved its generalizability and effectiveness on 2017 TAC and 2018
n2c2 challenges as it provided increased performance with several stateof-the-art methods. The experimental results demonstrated that our
model effectively combined the advantage of contextual and syntactic
information to further improve model performance. In future works, we
plan to explore medical knowledge graphs extracted from different
sources like DrugBank and UMLS to better model the relationships be­
tween ADE mentions. The integration of different heterogeneous graphs
to the GCN-based methods would provide supplementary information
that will preserve the semantic and syntactic representations of the
model. We believe such an approach might improve the accuracy of
ADE-RE task. We also intend to extend our model to a document-level

based method for dealing with inter-sentential relations.


**CRediT authorship contribution statement**


**Ed-drissiya El-allaly:** Conceptualization, Methodology, Software,
Formal analysis, Investigation, Visualization, Writing – original draft,
Writing – review & editing. **Mourad Sarrouti:** Conceptualization,
Methodology, Validation, Writing – review & editing. **Noureddine En-**
**Nahnahi:** Supervision, Validation. **Said Ouatik El Alaoui:** Supervision,
Validation.


**Declaration of Competing Interest**


The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence
the work reported in this paper.


**Appendix A. Supplementary material**


Supplementary data associated with this article can be found, in the
[online version, at https://doi.org/10.1016/j.jbi.2021.103968.](https://doi.org/10.1016/j.jbi.2021.103968)


**References**


[1] S. Bayer, C. Clark, O. Dang, J. Aberdeen, S. Brajovic, K. Swank, L. Hirschman,
R. Ball, ADE eval: An evaluation of text processing systems for adverse event
extraction from drug labels for pharmacovigilance, Drug Saf. 44 (2020) 83–94,
[https://doi.org/10.1007/s40264-020-00996-3.](https://doi.org/10.1007/s40264-020-00996-3)



11


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_




[2] C.Y. Lee, Y.-P.P. Chen, Machine learning on adverse drug reactions for
[pharmacovigilance, Drug Discov. Today 24 (2019) 1332–1343, https://doi.org/](https://doi.org/10.1016/j.drudis.2019.03.003)
[10.1016/j.drudis.2019.03.003.](https://doi.org/10.1016/j.drudis.2019.03.003)

[3] Y. Ji, H. Ying, P. Dews, A. Mansour, J. Tran, R.E. Miller, R.M. Massanari,
A potential causal association mining algorithm for screening adverse drug
reactions in postmarketing surveillance, IEEE Trans. Inf. Technol. Biomed. 15
[(2011) 428–437, https://doi.org/10.1109/titb.2011.2131669.](https://doi.org/10.1109/titb.2011.2131669)

[4] E. Russo, C. Palleria, C. Leporini, S. Chimirri, G. Marrazzo, S. Sacchetta, L. Bruno,
R. Lista, O. Staltari, A. Scuteri, F. Scicchitano, Limitations and obstacles of the
spontaneous adverse drugs reactions reporting: Two challenging case reports,
[J. Pharmacol. Pharmacotherap. 4 (2013) 66, https://doi.org/10.4103/0976-](https://doi.org/10.4103/0976-500x.120955)
[500x.120955.](https://doi.org/10.4103/0976-500x.120955)

[5] Y. Luo, W.K. Thompson, T.M. Herr, Z. Zeng, M.A. Berendsen, S.R. Jonnalagadda,
M.B. Carson, J. Starren, Natural language processing for EHR-based
[pharmacovigilance: A structured review, Drug Saf. 40 (2017) 1075–1089, https://](https://doi.org/10.1007/s40264-017-0558-6)
[doi.org/10.1007/s40264-017-0558-6.](https://doi.org/10.1007/s40264-017-0558-6)

[6] E. El-allaly, M. Sarrouti, N. En-Nahnahi, S.O.E. Alaoui, An adverse drug effect
mentions extraction method based on weighted online recurrent extreme learning
[machine, Comput. Methods Programs Biomed. 176 (2019) 33–41, https://doi.org/](https://doi.org/10.1016/j.cmpb.2019.04.029)
[10.1016/j.cmpb.2019.04.029.](https://doi.org/10.1016/j.cmpb.2019.04.029)

[7] C.Y. Lee, Y.-P.P. Chen, Prediction of drug adverse events using deep learning in
pharmaceutical discovery, Briefings in Bioinformatics doi:10.1093/bib/bbaa040.

[8] E. El-allaly, M. Sarrouti, N. En-Nahnahi, S.O.E. Alaoui, A LSTM-based method with
attention mechanism for adverse drug reaction sentences detection, in: Advances in
Intelligent Systems and Computing, Springer International Publishing, 2020,
[pp. 17–26, https://doi.org/10.1007/978-3-030-36664-3_3.](https://doi.org/10.1007/978-3-030-36664-3_3)

[9] E. El-allaly, M. Sarrouti, N. En-Nahnahi, S.O.E. Alaoui, Adverse drug reaction
mentions extraction from drug labels: An experimental study, in: Advances in
Intelligent Systems and Computing, Springer International Publishing, 2019,
[pp. 216–231, https://doi.org/10.1007/978-3-030-11884-6_21.](https://doi.org/10.1007/978-3-030-11884-6_21)

[10] C. Quirk, H. Poon, Distant supervision for relation extraction beyond the sentence
boundary, in: Proceedings of the 15th Conference of the European Chapter of the
Association for Computational Linguistics: Long Papers, vol. 1, Association for
[Computational Linguistics, 2017, https://doi.org/10.18653/v1/e17-1110.](https://doi.org/10.18653/v1/e17-1110)

[11] M. Sarrouti, S.O.E. Alaoui, A passage retrieval method based on probabilistic
information retrieval model and UMLS concepts in biomedical question answering,
[J. Biomed. Inform. 68 (2017) 96–103, https://doi.org/10.1016/j.jbi.2017.03.001.](https://doi.org/10.1016/j.jbi.2017.03.001)

[12] M. Sarrouti, A. Lachkar, A new and efficient method based on syntactic
dependency relations features for ad hoc clinical question classification, Int. J.
[Bioinform. Res. Appl. 13 (2) (2017) 161, https://doi.org/10.1504/](https://doi.org/10.1504/ijbra.2017.10003490)
[ijbra.2017.10003490.](https://doi.org/10.1504/ijbra.2017.10003490)

[13] A.B. Chapman, K.S. Peterson, P.R. Alba, S.L. DuVall, O.V. Patterson, Detecting
adverse drug events with rapidly trained classification models, Drug Saf. 42 (2019)
[147–156, https://doi.org/10.1007/s40264-018-0763-y.](https://doi.org/10.1007/s40264-018-0763-y)

[14] J.L. Martínez, I. Segura-Bedmar, P. Martínez, A. Carruana, A. Naderi, C. Polo, Mcuc3m participation at tac 2017 adverse drug reaction extraction from drug labels,
in: proceedings of the Text Analysis Conference (TAC 2017), 2017.

[15] K. Peterson, J. Shi, H. Eyre, H. Lent, K. Grave, J. Shao, S. Nag, O. Patterson, J.F.
Hurdle, Hybrid models for medication and adverse drug events extraction, 2019.

[16] D. Huang, Z. Jiang, L. Zou, L. Li, Drug-drug interaction extraction from biomedical
literature using support vector machine and long short term memory networks, Inf.
[Sci. 415–416 (2017) 100–109, https://doi.org/10.1016/j.ins.2017.06.021.](https://doi.org/10.1016/j.ins.2017.06.021)

[17] P. Zhou, W. Shi, J. Tian, Z. Qi, B. Li, H. Hao, B. Xu, Attention-based bidirectional
long short-term memory networks for relation classification, in: Proceedings of the
54th Annual Meeting of the Association for Computational Linguistics (Volume 2:
[Short Papers), Association for Computational Linguistics, 2016. https://doi.org/1](https://doi.org/10.18653/v1/p16-2034)
[0.18653/v1/p16-2034.](https://doi.org/10.18653/v1/p16-2034)

[18] W. Zheng, H. Lin, L. Luo, Z. Zhao, Z. Li, Y. Zhang, Z. Yang, J. Wang, An attentionbased effective neural model for drug-drug interactions extraction, BMC
[Bioinformatics 18(1). https://doi.org/10.1186/s12859-017-1855-x.](https://doi.org/10.1186/s12859-017-1855-x)

[19] T.N. Kipf, M. Welling, Semi-supervised classification with graph convolutional
networks, in: 5th International Conference on Learning Representations, ICLR
2017, Toulon, France, April 24–26, 2017, Conference Track Proceedings, 2017.

[20] C. Park, J. Park, S. Park, AGCN: Attention-based graph convolutional networks for
[drug-drug interaction extraction, Expert Syst. Appl. 159 (2020) 113538, https://](https://doi.org/10.1016/j.eswa.2020.113538)
[doi.org/10.1016/j.eswa.2020.113538.](https://doi.org/10.1016/j.eswa.2020.113538)

[21] Y. Zhang, P. Qi, C.D. Manning, Graph convolution over pruned dependency trees
improves relation extraction, in: Proceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing, Association for Computational
[Linguistics, 2018. https://doi.org/10.18653/v1/d18-1244.](https://doi.org/10.18653/v1/d18-1244)

[22] Z. Guo, Y. Zhang, W. Lu, Attention guided graph convolutional networks for
relation extraction, in: Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics, Association for Computational Linguistics, 2019.
[https://doi.org/10.18653/v1/p19-1024.](https://doi.org/10.18653/v1/p19-1024)

[23] E. El-allaly, M. Sarrouti, N. En-Nahnahi, S.O.E. Alaoui, MTTLADE: A multi-task
transfer learning-based method for adverse drug events extraction, Inform. Process.
[Manage. 58 (2021) 102473, https://doi.org/10.1016/j.ipm.2020.102473.](https://doi.org/10.1016/j.ipm.2020.102473)

[24] J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C.H. So, J. Kang, BioBERT: a pre-trained
biomedical language representation model for biomedical text mining,
[Bioinformatics. https://doi.org/10.1093/bioinformatics/btz682.](https://doi.org/10.1093/bioinformatics/btz682)

[25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, L. Kaiser, I.
Polosukhin, Attention is all you need, ArXiv abs/1706.03762.

[26] D. Demner-Fushman, S.E. Shooshan, L. Rodriguez, A.R. Aronson, F. Lang, W.
Rogers, K. Roberts, J. Tonning, A dataset of 200 structured product labels
[annotated for adverse drug reactions, Sci. Data 5. https://doi.org/10.1038/sdata](https://doi.org/10.1038/sdata.2018.1)
[.2018.1.](https://doi.org/10.1038/sdata.2018.1)




[27] S. Henry, K. Buchan, M. Filannino, A. Stubbs, O. Uzuner, n2c2 shared task on
adverse drug events and medication extraction in electronic health records, J. Am.
[Med. Inform. Assoc. 27 (2019) (2018) 3–12, https://doi.org/10.1093/jamia/](https://doi.org/10.1093/jamia/ocz166)
[ocz166.](https://doi.org/10.1093/jamia/ocz166)

[28] A. Jagannatha, F. Liu, W. Liu, H. Yu, Overview of the first natural language
processing challenge for extracting medication, indication, and adverse drug
events from electronic health record notes (MADE 1.0), Drug Saf. 42 (2019)
[99–111, https://doi.org/10.1007/s40264-018-0762-z.](https://doi.org/10.1007/s40264-018-0762-z)

[29] K. Roberts, D. Demner-Fushman, J.M. Tonning, Overview of the tac 2017 adverse
reaction extraction from drug labels track, in: proceedings of the Text Analysis
Conference (TAC 2017), 2017.

[30] B. Dandala, D. Mahajan, M.V. Devarakonda, Ibm research system at tac 2017:
Adverse drug reactions extraction from drug labels, in: proceedings of the Text
Analysis Conference (TAC 2017), 2017.

[31] C. Tao, K. Lee, M. Filannino, K. Buchan, K. Lee, T.R. Arora, J. Liu, O. Farri, O. [¨]
Uzuner, Extracting and normalizing adverse drug reactions from drug labels, in:
proceedings of the Text Analysis Conference (TAC 2017), 2017.

[32] X. Gu, C. Ding, S.K. Li, W. Xu, Bupt-pris system for tac 2017 event nugget detection,
event argument linking and adr tracks, in: proceedings of the Text Analysis
Conference (TAC 2017), 2017.

[33] J. Xu, H.-J. Lee, Z. Ji, J. Wang, Q. Wei, H. Xu, Uth-ccb system for adverse drug
reaction extraction from drug labels at tac-adr 2017, in: proceedings of the Text
Analysis Conference (TAC 2017), 2017.

[34] B. Dandala, V. Joopudi, M. Devarakonda, Adverse drug events detection in clinical
notes by jointly modeling entities and relations using neural networks, Drug Saf. 42
[(2019) 135–146, https://doi.org/10.1007/s40264-018-0764-x.](https://doi.org/10.1007/s40264-018-0764-x)

[35] D. Xu, V. Yadav, S. Bethard, Uarizona at the made1.0 nlp challenge., in:
Proceedings of machine learning research Medication and Adverse Drug Event
Detection Workshop, 2018, pp. 57–65.

[36] A. Magge, M. Scotch, G. Gonzalez-Hernandez, Clinical ner and relation extraction
using bi-char-lstms and random forest classifiers, in: Proceedings of machine
learning research Medication and Adverse Drug Event Detection Workshop, 2018.

[37] I. Alimova, E. Tutubalina, A comparative study on feature selection in relation
extraction from electronic health records, in: Data Analytics and Management in
Data Intensive Domains: I International Conference DADID/RCDL, vol. 2523 of
CEUR Workshop Proceedings, CEUR-WS.org, 2019, pp. 34–45.

[38] Q. Wei, Z. Ji, Z. Li, J. Du, J. Wang, J. Xu, Y. Xiang, F. Tiryaki, S. Wu, Y. Zhang,
C. Tao, H. Xu, A study of deep learning approaches for medication and adverse
drug event extraction from clinical text, J. Am. Med. Inform. Assoc. 27 (2019)
[13–21, https://doi.org/10.1093/jamia/ocz063.](https://doi.org/10.1093/jamia/ocz063)

[39] F. Christopoulou, T.T. Tran, S.K. Sahu, M. Miwa, S. Ananiadou, Adverse drug
events and medication relation extraction in electronic health records with

ensemble deep learning methods, J. Am. Med. Inform. Assoc. 27 (2019) 39–46,
[https://doi.org/10.1093/jamia/ocz101.](https://doi.org/10.1093/jamia/ocz101)

[40] X. Yang, J. Bian, R. Fang, R.I. Bjarnadottir, W.R. Hogan, Y. Wu, Identifying
relations of medications with adverse drug events using recurrent convolutional
neural networks and gradient boosting, J. Am. Med. Inform. Assoc. 27 (2019)
[65–72, https://doi.org/10.1093/jamia/ocz144.](https://doi.org/10.1093/jamia/ocz144)

[41] L. Chen, Y. Gu, X. Ji, Z. Sun, H. Li, Y. Gao, Y. Huang, Extracting medications and
associated adverse drug events using a natural language processing system
combining knowledge base and deep learning, J. Am. Med. Inform. Assoc. 27 (1)
[(2020) 56–64, https://doi.org/10.1093/jamia/ocz141.](https://doi.org/10.1093/jamia/ocz141)

[42] Y. Kim, S.M. Meystre, Ensemble method-based extraction of medication and
related information from clinical texts, J. Am. Med. Inform. Assoc. 27 (2019)
[31–38, https://doi.org/10.1093/jamia/ocz100.](https://doi.org/10.1093/jamia/ocz100)

[43] I. Alimova, E. Tutubalina, Multiple features for clinical relation extraction: A
[machine learning approach, J. Biomed. Inform. 103 (2020) 103382, https://doi.](https://doi.org/10.1016/j.jbi.2020.103382)
[org/10.1016/j.jbi.2020.103382.](https://doi.org/10.1016/j.jbi.2020.103382)

[44] M. Belousov, N. Milosevic, G.A. Alfattni, H. Alrdahi, G. Nenadic, Gnteam at n2c2
2018 track 2: An end-to-end system to identify ade, medications and related
entities in discharge summaries, 2019.

[45] X. Yang, Z. Yu, Y. Guo, J. Bian, Y. Wu, Clinical relation extraction using
transformer-based models, ArXiv abs/2107.08957.

[46] J. Devlin, M.-W. Chang, K. Lee, K. Toutanova, Bert: Pre-training of deep
bidirectional transformers for language understanding, ArXiv abs/1810.04805.

[47] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L.
Zettlemoyer, V. Stoyanov, Roberta: A robustly optimized bert pretraining
approach, arXiv preprint arXiv:1907.11692.

[48] Z. Yang, Z. Dai, Y. Yang, J.G. Carbonell, R. Salakhutdinov, Q.V. Le, Xlnet:
Generalized autoregressive pretraining for language understanding., in: CoRR, Vol.
abs/1906.08237, 2019.

[49] Z. Li, Y. Sun, J. Zhu, S. Tang, C. Zhang, H. Ma, Improve relation extraction with
[dual attention-guided graph convolutional networks, Neural Comput. Appl. htt](https://doi.org/10.1007/s00521-020-05087-z)
[ps://doi.org/10.1007/s00521-020-05087-z.](https://doi.org/10.1007/s00521-020-05087-z)

[50] C. Park, J. Park, S. Park, AGCN: Attention-based graph convolutional networks for
[drug-drug interaction extraction, Expert Syst. Appl. 159 (2020) 113538, https://](https://doi.org/10.1016/j.eswa.2020.113538)
[doi.org/10.1016/j.eswa.2020.113538.](https://doi.org/10.1016/j.eswa.2020.113538)

[51] D. Zhao, J. Wang, H. Lin, Z. Yang, Y. Zhang, Extracting drug-drug interactions with
hybrid bidirectional gated recurrent unit and graph convolutional network,
[J. Biomed. Inform. 99 (2019) 103295, https://doi.org/10.1016/j.jbi.2019.103295.](https://doi.org/10.1016/j.jbi.2019.103295)

[52] E. El-allaly, M. Sarrouti, N. En-Nahnahi, S.O.E. Alaoui, DeepCADRME: A deep
neural model for complex adverse drug reaction mentions extraction, Pattern
[Recogn. Lett. 143 (2021) 27–35, https://doi.org/10.1016/j.patrec.2020.12.013.](https://doi.org/10.1016/j.patrec.2020.12.013)

[53] R. Sætre, K. Yoshida, A. Yakushiji, Y. Miyao, Y. Matsubayashi, T. Ohta, Akane
system: Protein-protein interaction 1 akane system: Protein-protein interaction



12


_E.-d. El-allaly et al._ _Journal of Biomedical Informatics 125 (2022) 103968_



pairs in the biocreative 2 challenge, ppi-ips subtask, in: the Second BioCreative
Challenge Evaluation Workshop, 2007.

[54] J. Chen, J. Gu, Jointly extract entities and their relations from biomedical text,
[IEEE Access 7 (2019) 162818–162827, https://doi.org/10.1109/](https://doi.org/10.1109/access.2019.2952154)
[access.2019.2952154.](https://doi.org/10.1109/access.2019.2952154)

[55] L. Luo, Z. Yang, M. Cao, L. Wang, Y. Zhang, H. Lin, A neural network-based joint
learning approach for biomedical entity and relation extraction from biomedical
[literature, J. Biomed. Inform. 103 (2020) 103384, https://doi.org/10.1016/j.](https://doi.org/10.1016/j.jbi.2020.103384)
[jbi.2020.103384.](https://doi.org/10.1016/j.jbi.2020.103384)

[56] Z. Li, Z. Yang, Y. Xiang, L. Luo, Y. Sun, H. Lin, Exploiting sequence labeling
framework to extract document-level relations from biomedical texts, BMC
[Bioinformat. 21. https://doi.org/10.1186/s12859-020-3457-2.](https://doi.org/10.1186/s12859-020-3457-2)




[57] F. Meng, J. Feng, D. Yin, S. Chen, M. Hu, A structure-enhanced graph convolutional
network for sentiment analysis, in: Findings of the Association for Computational
Linguistics: EMNLP 2020, Association for Computational Linguistics, 2020, pp.
[586–595. https://doi.org/10.18653/v1/2020.findings-emnlp.52.](https://doi.org/10.18653/v1/2020.findings-emnlp.52)

[58] E.S. Edgington, Approximate randomization tests, J. Psychol. 72 (1969) 143–149,

[https://doi.org/10.1080/00223980.1969.10543491.](https://doi.org/10.1080/00223980.1969.10543491)

[59] I. Beltagy, K. Lo, A. Cohan, SciBERT: A pretrained language model for scientific
text, in: Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on Natural
[Language Processing (EMNLP-IJCNLP), 2019. https://doi.org/10.18653/v1/d19-1](https://doi.org/10.18653/v1/d19-1371)
[371.](https://doi.org/10.18653/v1/d19-1371)



13


