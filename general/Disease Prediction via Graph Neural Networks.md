This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                                Biomedical and Health Informatics
            JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          1
                     Disease Prediction via Graph Neural Networks
                                     Zhenchao Sun, Hongzhi Yin, Hongxu Chen, Tong Chen, Lizhen Cui, Fan Yang
               Abstract—With  the  increasingly  available  electronic  medical                  formulated as learning a classiﬁer that infers the prediction
            records (EMRs), disease prediction has recently gained immense                       results from EMRs [2], [3]. For example, Palaniappan and
            research  attention,  where  an  accurate  classiﬁer  needs  to  be                  Awang applied a series of data mining techniques, namely
            trained  to  map  the  input  prediction  signals  (e.g.,  symptoms,                 Decision Trees [4], Naive Bayes [5] and Neural Networks [6],
            patient demographics, etc.) to the estimated diseases for each pa-                   to build a heart disease prediction system [7]. With the power
            tient. However, existing machine learning-based solutions heavily
            rely on abundant manually labeled EMR training data to ensure                        of  Convolutional  Neural  Networks  (CNNs),  Suo  et  al.  [2]
            satisfactory  prediction  results,  impeding  their  performance  in                 ﬁrstly identiﬁed the similarity between patients based on their
            the  existence  of  rare  diseases  that  are  subject  to  severe  data             EMRs, and then performed personalized disease predictions.
            scarcity. For each rare disease, the limited EMR data can hardly                     Ma  et  al.  [3]  further  incorporated  discrete  prior  medical
            offer sufﬁcient information for a model to correctly distinguish                     knowledge into CNNs to improve the prediction performance.
            its identity from other diseases with similar clinical symptoms.
            Furthermore,  most  existing  disease  prediction  approaches  are                      Nevertheless, training these models requires a large amount
            based on the sequential EMRs collected for every patient and are                     of  EMR  data  with  respect  to  each  particular  disease,  hin-
            unable to handle new patients without historical EMRs, reducing                      dering existing models from generating accurate predictions
            their real-life practicality.                                                        when there are no sufﬁcient disease-speciﬁc EMR records,
               In  this  paper,  we  introduce  an  innovative  model  based  on
            Graph Neural Networks (GNNs) for disease prediction, which                           e.g., predicting rare diseases. Rare diseases have a common
            utilizes  external  knowledge  bases  to  augment  the  insufﬁcient                  characteristic of low prevalence and perception, while both
            EMR data, and learns highly representative node embeddings                           the treatments and related medical research are inadequate [8].
            for patients, diseases and symptoms from the medical concept                         This leads to critical challenges in the clinical diagnosis of a
            graph and patient record graph respectively constructed from the                     rare disease. According to the fact that up to 8% of the human
            medical knowledge base and EMRs. By aggregating information                          population is affected by rare disease [9], the average time it
            from  directly  connected  neighbor  nodes,  the  proposed  neural
            graph encoder can effectively generate embeddings that capture                       takes to achieve a correct diagnosis for a rare disease case can
            knowledge  from  both  data  sources,  and  is  able  to  inductively                be as much as 4.8 years1. The difﬁculty of identifying rare
            infer the embeddings for a new patient based on the symptoms                         diseases is mainly associated with the high diversity of such
            reportedinher/hisEMRstoallowforaccuratepredictiononboth                              diseases (more than 6,000 types discovered) and the lack of
            general diseases and rare diseases. Extensive experiments on a                       clinical experience [9], [10]. As a result, the majority of these
            real-world EMR dataset have demonstrated the state-of-the-art
            performance of our proposed model.                                                   patients are suffering from long-term illness, despair, and even
               Index Terms—Disease  Prediction,  Big  Data  Health  Applica-                     wrong treatments caused by misdiagnosis. Therefore, on top
            tions, Data Mining, Graph Embedding                                                  of the conventional disease prediction, accurately predicting
                                                                                                 rare  diseases  at  an  early  stage  will  help  patients  receive
                                        I.  INTRODUCTION                                         prevention treatments in a timely manner, thus signiﬁcantly
                                                                                                 increasing their survival rates and minimizing the harm from
                   S  a  widely-used  data  management  scheme,  electronic                      such diseases. At the same time, most existing models make
            Amedical  records  (EMRs)  are  used  to  store  the  rich                           predictions in a sequential manner, where historical EMRs for
            clinical data collected from different patients’ visits to hos-                      a patient is an indispensable part of the model input. Con-
            pitals. Recently, with the prosperous advances in information                        sequently, these approaches are incompatible to new patients
            technology and machine learning, the sheer volume of EMRs                            that are unseen during training, rendering them unable to make
            is  becoming  more  manageable,  and  analyzing  EMRs  with                          predictions for new patients.
            machine learning and data mining techniques is becoming an                              Despite  the  importance  of  a  machine  learning  model’s
            emerging research direction to fulﬁll the goal of improving                          sensitivity to rare diseases in the disease prediction task, as
            health care services [1].                                                            suggested  by  the  name,  it  is,  however,  extremely  difﬁcult
               An important application of machine learning in healthcare                        to  collect  abundant  EMR  data  on  these  rare  diseases  to
            is disease prediction that aims to predict whether a patient                         train  a  robust  and  reliable  classiﬁer.  Moreover,  in  EMRs,
            suffers from a certain disease, where the task is commonly                           some  rare  diseases  may  develop  symptoms  similar  to  the
                                                                                                 ones  of  common  diseases,  which  offer  counterfeit  signals
               Z. Sun and L. Cui are with the School of Software, Shandong University,           and are prone to be misclassiﬁed. In this regard, instead of
            Jinan, China. E-mail: zhenchao.sun@mail.sdu.edu.cn; clz@sdu.edu.cn.                  solely relying on the EMRs, a classiﬁer should be able to
               H. Yin and T. Chen are with the School of Information Technology &                fully utilize various external information sources to guarantee
            Electrical Engineering, The University of Queensland, Brisbane, Australia.
            E-mail: h.yin1@uq.edu.au, tong.chen@uq.edu.au.                                       the prediction performance in the presence of rare diseases.
               H. Chen is with the School of Computer Science, University of Technology          Fortunately, in addition to EMRs, a wide range of medical
            Sydney, Australia. E-mail: hongxu.chen@uts.edu.au.
               F.  Yang  is  with  the  School  of  Public  Health  &  Institute  for  Medical
            Dataology, Shandong University, Jinan, China. E-mail: fanyang@sdu.edu.cn.              1https://globalgenes.org/rare-facts/
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                         Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                              Biomedical and Health Informatics
            JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          2
            knowledge  has  been  standardized  by  reputable  institutions                          outperforms all baseline models in both common disease
            and  published  as  well-organized  ontologies,  including  the                          and rare disease prediction tasks.
            International  Classiﬁcation  of  Diseases  (ICD)2,  the  Human
            Phenotype Ontology (HPO)3 and Orphanet4. Such information                                                    II.  RELATED WORK
            sources are essentially structured knowledge bases containing
            veriﬁed and valuable disease-related metadata (i.e., attributes),                     The  creation  and  adoption  of  electronic  medical  records
            which  bring  immense  potentials  for  improving  the  disease                    (EMRs) have ignited widespread interest and opened up abun-
            prediction accuracy in practice.                                                   dant opportunities for clinical and translational research [14],
               To this end, in light of the availability of both the public                    thus  motivating  further  studies  on  the  prediction  of  risks,
            diseaseknowledgebasesandEMRs,werespectivelyformulate                               diagnoses, and diseases. Choi et al. [15] proposed GRaph-
            two information sources as a medical concept graph and a                           based Attention Model (GRAM) that supplements EMRs with
            patient record graph, and introduce a novel graph embedding-                       hierarchical  information  extracted  from  medical  ontologies.
            based model for disease prediction. Both constructed graphs                        Choi et al. [16] also developed the REverse Time AttentIoN
            are heterogeneous, where the medical concept graph links dis-                      model  (RETAIN)  for  utilizing  the  EMR  data  to  improve
            eases with related symptoms and patient record graph extracts                      both the accuracy and interpretability of predictive models.
            connections between patients and observed symptoms from                            However, using Recurrent Neural Networks (RNNs) as the
            the EMRs. Speciﬁcally, by investigating a disease’s/patient’s                      main building block, such approaches suffer from a severe per-
            associations to different symptoms, we build a novel disease                       formance drop when the length of the sequences becomes too
            prediction model upon Graph Neural Networks (GNNs) [11]                            large for RNNs to learn long-range dependencies. To address
            to encode the information of different symptoms, users and                         this issue, Ma et al. [17] proposed Dipole to predict patients’
            diseases into compact but representative latent vectors (a.k.a.,                   future health information, which employs bidirectional RNNs
            embeddings). As such, the probability of observing an active                       to memorize all information of both long-term and short-term
            patient-disease relationship (i.e., predicting a disease type for                  patient  status,  and  leverages  three  attention  mechanisms  to
            a patient) can be easily inferred via the similarity between the                   measure the contributions of different visits to the prediction.
            embeddings of a disease and the target patient.                                    However, as typical variants of deep neural networks, these
               In  order  to  overcome  the  shortage  of  EMR  data  when                     approaches invariably require a large amount of training data
            learning  to  predict  rare  diseases,  our  model  gains  external                to  learn  the  complex  non-linear  functions  and  data-driven
            medical knowledge on both the diseases and symptoms by                             patterns for accurate prediction. Consequently, these methods
            aggregating the information of their connected neighbors from                      tend to underperform in EMR-related prediction tasks when
            the  medical  concept  graph.  Hence,  in  the  prediction  stage,                 sufﬁcient EMR data is unavailable. To address this issue, Ma
            given an arbitrary patient that is a new visitor to the hospital                   et al. [18] presented an end-to-end model named KAME to
            (the patient’s EMRs are unseen during training), our model                         exploit external medical knowledge to improve the accuracy
            can effectively generate the patient’s embedding by merging                        of  diagnosis  prediction.  One  recent  work  [19]  proposed  a
            the  learned  latent  representations  of  symptoms  reported  in                  meta-learningframeworkforclinicriskpredictionwithlimited
            her/his EMRs. In addition, as our model is subsumed under a                        patient  record  data,  which  transfers  knowledge  from  other
            generic graph embedding framework that is not restricted to                        closely related but information-intensive disease domains.
            speciﬁc information aggregation schemes, we further explore                           Among various EMR-related tasks, the disease prediction
            twodistinctneighborhoodinformationaggregators,namelythe                            task is designed to predict whether a patient suffers from a
            Graph Attention Networks (GATs) [12] and Graph Isomorphic                          certain  disease  based  on  the  historical  EMR  data.  Disease
            Networks (GINs) [13] via comparative studies.                                      prediction  is  formulated  as  a  classiﬁcation  task,  where  a
               The contributions of this paper are summarized as follows.                      wide range of traditional classiﬁcation approaches have been
               •We identify the challenges of predicting both common                           applied  to  solve  it.  For  instance,  Palaniappan  and  Awang
                  and  rare  diseases  based  on  EMRs,  and  introduce  a                     applied  Decision  Trees,  Naive  Bayes  and  Neural  Network,
                  systematic  solution  by  fusing  expert  knowledge  with                    and  introduced  a  system  for  heart  disease  prediction  [7].
                  machine learning techniques.                                                 Moreover,  with  the  recent  advances  in  deep  attention  net-
               •We propose a novel graph embedding-based model for                             works and graph neural networks [20]–[25], there has been
                  disease prediction. The model inductively learns embed-                      an  increasing  amount  of  applications  in  disease  prediction
                  dings from the medical concept graph and patient record                      tasks.  Suo  et  al.  [2]  used  CNNs  to  perform  personalized
                  graph respectively extracted from the external knowledge                     disease predictions by identifying similar patients based on
                  base  and  EMR  data,  while  being  able  to  handle  new                   their historical EMRs. Also, Ma et al. [3] incorporated discrete
                  patients and identify highly relevant symptoms to support                    prior medical knowledge into a CNN-based model to improve
                  accurate disease prediction.                                                 the  prediction  performance.  Unfortunately,  existing  models
               •Extensive experiments on real-world EMR datasets have                          mainly focus on general diseases with inherently high volume
                  been conducted, and the results suggest that our model                       of relevant EMRs for training, and cannot adapt to cases of
                                                                                               rare diseases due to higher data scarcity and more intricate
               2http://www.who.int/                                                            relationships between symptoms and diagnoses.
               3https://https://hpo.jax.org/app/                                                  Finding the right treatments for patients is a primary beneﬁt
               4https://www.orpha.net/                                                         of accurate disease prediction results, but rare diseases have
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                        Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                               Biomedical and Health Informatics
            JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          3
            Fig. 1.  Building the medical concept graph and patient record graph. (a) shows three medical concepts. Each medical concept consists of one speciﬁc disease
            and several symptoms. (b) shows the medical concept graph extracted from (a). (c) shows three patient EMRs. Each EMR consists of the patient’s identity
            and several symptoms. (d) shows a patient record graph extracted from (c).
            been rather difﬁcult to be identiﬁed among a large number of                       all the symptoms occurred in EMRs (V′S⊆VS  in our case),
            possible diagnoses. In the context of rare disease prediction,                     andEPS   represents all observed edges between patient nodes
            machine learning techniques have recently started to demon-                        and symptom nodes. An example of a patient record graph
            strate more advantageous performance in terms of analyzing                         is illustrated in Fig.1(c)-(d). Each recordp∈VP   is assigned
            the latent patterns within EMRs. For example, Garg et al. [26]                     a multi-hot encoding cp∈{0, 1}|VM|indicating the diseases
            targeted a speciﬁc rare disease called cardiac amyloidosis and                     this patient has (corresponding indexes are marked as 1).
            successfully automated the process of identifying potential pa-                        Problem 1: (Rare) Disease Prediction. Given a medical
            tients with bootstrap machine learning algorithms. Along this                      concept graphC= (V  M∪VS,EMS   ), a patient record graph
            line of research, MacLeod et al. [27] used self-reported behav-                    P= (VP∪V′S,EPS  ) and the corresponding labels, our goal is
            ioral data to distinguish people with rare diseases from people                    to learn a Graph Convolutional Network-based model, which
            with more common chromic illnesses, while Hare et al. [28]                         is able to predict the diseases for each new patientp/∈VP .
            used pattern recognition ensembles to improve the accuracy                         Apart from general disease prediction, we will additionally
            of identiﬁed rare disease patients. Additionally, genomic data                     perform rare disease prediction by evaluating the prediction
            was  studied  in  [29],  where  a  method  adopting  imbalance-                    performance exclusively on patients who are diagnosed with
            aware  learning  strategies  with  a  resampling  algorithm  was                   rare diseases in the real clinical dataset.
            proposed for predicting rare and common diseases. In general,
            most studies make predictions based on longitudinal historical                     B. Neural Graph Encoder
            patient records, which means that they can only serve patients                         We use Fig.2 to demonstrate the workﬂow of our proposed
            whose historical EMRs are used for model training.                                 framework for disease prediction. Our model takes the medical
            A. Deﬁnitions       III.  THE PROPOSED METHOD                                      concept graphCand patient record graphPas its input, then
                                                                                               embeds every node in both graphs by aggregating the infor-
               Deﬁnition 1: Medical Concept Graph. For an arbitrary                            mation from their sampled neighbors. Eventually, for a given
            disease,  its  associated  medical  concepts  include  its  name,                  patient, we can form a vector representation by fusing the
            diagnostic symptoms, and the category it belongs to. As the                        learned embeddings of symptoms described in her/his EMRs.
            available medical concepts vary in different public medical                        Then, by measuring the closeness between the embeddings
            knowledge bases, without loss of generality, we only consider                      of the patient and any disease, we can eventually estimate
            a common concept, i.e., symptoms in this paper. We represent                       the  likelihood  of  diagnosing  patientp with  diseasem.  In
            these medical concepts as a medical concept graph, denoted                         this section, we ﬁrst describe a graph encoder model, which
            byC=  (VM ∪VS,EMS   ),  whereVM    is  the  node  set  of                          is  responsible  for  producing  a  low-dimensional  embedding
            diseases,  andVS   is  the  node  set  for  symptoms  extracted                    z∈R d  for each node in an arbitrary graph.
            from medical knowledge bases, andEM    is the edge set. If                             Inspired  by  recent  advances  in  graph  convolutional  net-
            a symptoms∈VS  is associated tom∈VM  , then there is an                            works [11], we deﬁne the calculation of embeddings of each
            edge between two types of nodes. The construction process                          node (a disease, a symptom, or a patient) via an aggregation
            of the medical concept graph is depicted in Fig.1(a)-(b). Each                     scheme on the features of its directly connected neighbors.
            diseasem∈VM    has a|VM|-dimensional one-hot encoding                              The deﬁned aggregation function takes into account the ﬁrst-
            em   ={0,  1}|VM|with 1  at them-th position as its unique                         order neighbors of a node and applies the same transformation
            indentiﬁer. At the same time, each disease also has a binary                       across  all  nodes  in  the  graph.  In  this  way,  each  node  in
            labelrm∈{0, 1} indicating whether it is a rare disease or not                      the graph deﬁnes its own ﬁled of computational input, but
            (rm  = 1    if true and vice versa).                                               different  nodes’  computational  procedures  reuse  the  same
               Deﬁnition 2: Patient Record Graph. Similar to the med-                          set of parameters that deﬁne how information is shared and
            ical concept graph, the key information for disease prediction                     propagated. This setting makes efﬁcient use of information
            can  be  represented  as  a  patient  record  graph  denoted  by                   shared across regions in the graph, and allows embeddings to
            P= (VP∪V′S,EPS  ).Pis a graph-structured representation of                         be generated for previously unseen nodes during training, e.g.,
            EMRs,whereVP  isthenodesetofallpatientsandV′S  contains                            a newly joined patient in the EMR dataset.
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                        Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                                 Biomedical and Health Informatics
            JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          4
            Fig. 2.  The workﬂow of our graph neural network-based model for disease prediction.
                To  begin  with,  for  a  graphG= {C,P},  we  uniformly                           C. Varying Graph Encoder Kernels
            represent a disease, symptom, or patient node asv∈Gto                                    To  fully  investigate  the  effectiveness  of  different  neural
            be succinct. Then, at thel-th information propagation layer,                          architectures in disease prediction, we introduce two variants
            the embedding hlv  of nodevis calculated as:                                          of the graph encoder. In this section, we replace the graph
                     hlN(v) = AGGREGATE   ({h   l−1v ′ ,∀v′∈N(v)})                                encoder  kernels  described  in  Eq.1  with  two  widely-used
                     hlv = σ(W l·[h l−1v   ;hlN(v  )])                                         (1)GNNs. We adopt two encoder architectures: the Graph Atten-
                                                                                                  tion Network [12] and the Graph Isomorphic Network [13].
            where Wl is the weight matrix to be learned at thel-th layer,                         Both variants also follow the paradigm of neighborhood-based
            hl−1v      is nodev’s embedding at the previous layer, and we                         information aggregation, but each encoder employs a speciﬁc
            denote the total layer size asL. We use    [·;·]   to represent                       message  passing  rule  focusing  on  different  nuances  of  the
            the concatenation of two vectors, and useN(v)  to denote                              graph structural information.
            the set of evenly sampled neighbor nodes ofv. Note that for                              Graph Attention Networks (GATs). GATs apply attention
            l= 0  , the node embedding h0v∈R d  is initialized via either                         mechanisms to selectively encode the information from neigh-
            randmonized values or side information from the data (subject                         bors according to their importance to the target nodev. This
            to availability). For instance, given a patient node, with the                        is achieved by taking a weighted sum of the representations
            available  patient  demographics  and  medical  proﬁles  in  the                      of allv’s neighbor nodes:
            EMR data, then h0v  will be initialized as a real-valued dense                                                hlv =   ∑         αv ′vMhl−1v ′                            (4)
            feature vector, and each digit in h0v  represents the observed                                                        v ′∈N(v  )
            value of a feature dimension (e.g., age). hlN(v  ) is the synergic                    where M is the transformation weight matrix, andαv ′v  is the
            representation resulted from the aggregation function, which                          attentive weights indicating the importance of neighbor node
            is designed to aggregate the embeddings of nodev’s neighbors                          v′∈N(v)  when calculating hlv. Eachav ′v  is computed via
            at the(l−1)-th layer. σis a non-linear activation function (e.g.,                     the following attention network:
            tanh), and the aggregator can be chosen as mean, max pooling,                         αv ′v =          exp(LeakyReLU∑(a                     T [Nh l−1v ||Nh  l−1v ′  ]))
            RNNs, etc. By default, we deploymean(·)              in our model for                               k∈N(v) exp(LeakyReLU(a                     T [Nh l−1v ||Nh  l−1k   ]))     (5)
            information aggregation.                                                              withaprojectionvectoraandtheweightmatrixN.Essentially,
                Then, we take a normalization step before reaching the ﬁnal                       the learned attentive weights allows the aggregator to lay more
            embedding for all nodes at the last layerL:                                           emphasis on neighbor nodes having more contributions to the
                                   hv =      hLv                                                  message passing process, thus being able to generate highly
                                           ||h  Lv||2,∀v∈G                            (2)         expressive node embeddings.
            In this paper, it is worth mentioning that the representations                           Graph Isomorphic Networks (GINs). GINs are claimed
            learned for the symptom nodesp in both the medical con-                               effective   in   representing   isomorphic   and   non-isomorphic
            cept  graphCand  patient  record  graphP share  the  same                             graphs with discrete attributes. The computation process of
            embedding  space.  That  is  to  say,  for  each  symptomp,  its                      itsl-th layer is deﬁned as:
            embeddings  remain  the  same  in  both  graphs,  thus  serving                                               hlv = MLP ( ∑             hl−1u   )                         (6)
            as  an  effective  bridge  between  patient  and  disease  nodes                                                               u∈N(v  )
            from different graphs. Meanwhile, as different types of nodes,                        whereMLP   is a multi-layer perceptron. In contrast to other
            i.e., diseases, patients, and symptoms are learned from three                         graph encoder kernels which combine the information from
            separate embedding spaces, we further align their contexts by                         both  nodev itself  and  its  neighbors,  a  GIN-based  graph
            projecting all node embeddings onto the same space, followed                          encoder  forms  the  embedding  forv purely  based  on  the
            by a non-linear activation:                                                           embeddings of neighbor nodes.
                                   zv = σ(Wh v),∀v∈G                            (3)               D. Graph Decoder for Disease Prediction
            where W is the learnable projection weight, and zv  is the ﬁnal                          Disease Prediction for Patients. The purpose of the graph
            embedding for nodev.                                                                  decoder in our model is to translate the information contained
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                         Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                               Biomedical and Health Informatics
            JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          5
            in the symptom, disease, and patient node embeddings into                          the patient record graph, we deﬁne the combined loss function
            predictions of possible diseases associated to a given patient.                    as the following:
            In summary, given the embedding zp  of a patientp, the graph                                                     L=LM +LP                              (11)
            decodermapsittoavectorizedoutputˆcp∈{0, 1}|C|whichap-                              which can be easily optimized via Stochastic Gradient Descent
            proximates this patient’s multi-hot disease label cp∈{0, 1}|C|(SGD) algorithms.
            (i.e., the ground truth in patient record graphP). Speciﬁcally,                                                IV.  EXPERIMENTS
            in this decoding process, every element inˆcp  is computed via:                       In what follows, we present detailed experimental analysis
                            ˆcp,n  =   sigmoid(z              Tp Q  n ),∀m   n∈VM                  (7)on our proposed disease prediction model. To support easy im-
            where ˆcp,n   is then-th element in   ˆcp, whilen≤|VM|is used                      plementation, we have released the source code of our model
            for indexing all the diseasesm. The closer    ˆcp,n∈ˆcp  is to                     at: https://github.com/zhchs/Disease-Prediction-via-GCN.
            1, the more likely patient p is diagnosed with diseasemn .                         A. Datasets
            Q∈R|VM|×d      carries the corresponding regression weights
            for all diseases, and Q  n   is then-th column of it. To train                        Inourexperiments,weutilizetheProprietaryEMRdataset
            our model, we quantify the prediction error via the following                      for constructing the patient record graph, which is our private
            negative log likelihood loss function:                                             real-world patient clinical record dataset collected from local
                                              |VM|∑                                            hospitals.Itcontains806patients,while451amongthemwere
                                  LP=−              cp,n  log(  ˆcp,n )                       (8)diagnosed with at least one rare disease. Each patient has an
                                              n=1                                              average of 1.49 diagnosed diseases. The main statistics of the
               Handling New Patients. With a fully trained neural graph                        Proprietary EMR dataset are shown in Table I.
            encoder, the message passing schema and the latent correla-                                                           TABLE I
            tionsbetweendiseasesandsymptomsareuncovered.However,                                                MAJOR STATISTICS OF THE EMR DATASET.
            compared with existing patient nodes inP, before we can                                            number of patients                         806
                                                                                                               number of patients with rare diseases      451
            predict the diseases for a newly joined patientp, we need                                          number of rare diseases                     71
            to ﬁrst infer her/his embedding vectorzp. In this regard, our                                      number of symptoms                         131
            approach shows its advantage in inductively generating node                                        average number of diseases                 1.49
                                                                                                               maximum number of diseases                  5
            representations.  The  intuition  is  that  the  knowledge  mined                     Besides, to formulate external medical knowledge into the
            from the medical concept graphCabout all symptom nodes                             medical concept graph, we choose Human Phenotype Ontol-
            is  stored  in  the  learned  parameters  (e.g.,  weight  matrices),               ogy (HPO) in our experiments. HPO provides an ontology of
            which can be additionally applied to a newly arrived patient                       medically relevant phenotypes, disease-phenotype annotations,
            node connecting to several well-represented symptom nodes.                         and the corresponding algorithms. The HPO is mainly used
            In short, based on the symptoms reported in a new patient’s                        for computational deep phenotyping, precision medicine, as
            EMRs, our model can effectively produce an expressive rep-                         well as the integration of clinical data into translational re-
            resentation for the patient. To be speciﬁc, with the weight                        search[30].Itcurrentlycontainsover13,000termsarrangedin
            matrices{W  l}Ll=1    for the aggregation function at each layer,                  a directed acyclic graph and are connected by “is-a” (subclass-
            we consolidate the trained aggregators and apply them to the                       of)  edges,  such  that  a  term  represents  a  more  speciﬁc  or
            newlyaddedpatients.Oneofthecrucialfeaturesofthedeﬁned                              limited instance of its parent term(s). The annotation ﬁle of
            aggregation  scheme  is  that  the  calculation  of  embeddings                    the HPO contains manual and semi-automated annotations of
            of  a  node  only  relies  on  its  ﬁrst-order  neighbors,  making                 OMIM,            Orphanet, and                DECIPHER entries. Here we mark the
            the embeddings of a newly added patient easily computable                          diseases annotated in the Orphanet database as rare diseases.
            by  aggregating  the  embeddings  of  its  neighbor  symptoms                      We  extract  71  diseases  with  Orphanet  annotation  and  669
            according to the graph encoder deﬁned in Eq.1.                                     linked phenotypes for our experiments.
               Supplementary Node Classiﬁcation Task. To thoroughly
            learn the embeddings and network parameters, we leverage the                       B. Baseline Methods
            available disease label information to design a supplementary                         We compare our model with three well-established classi-
            task of node classiﬁcation. Speciﬁcally, with the embedding                        ﬁers, namelySupport Vector Machine (SVM) [31], Decision
            zm   for each diseasem, we decode the latent representation                        Tree  (DT)  [4]  and  Random  Forest  (RF)  [32],  as  well  as
            to approximate its one-hot label cm   deﬁned in the medical                        fourstate-of-the-artgraphembedding-basedmodels,whichare
            concept graphC. Similar to Eq.7, the estimated label ˆcm   of                      introduced below.
            disease identity for a given embedding zm   is as follows:                            DeepWalk  [33]  is  an  approach  for  learning  latent  node
                           ˆcm,n   =   sigmoid(z              Tm G  n ),∀mn∈VM                 (9)representations  in  a  graph.  It  learns  node  embeddings  by
            where G∈R d×|VM|is a trainable weight matrix, and G  n  is                         maximizing  the  co-occurrence  probability  of  nodes  on  the
            then-th column of  G. We deﬁne the following negative log                          sequences generated by random walks.
            likelihood for this supplementary node classiﬁcation task:                            LINE [34] is a graph embedding method that is well suited
                                             |VM|∑                                             to  heterogeneous  graphs.  It  has  an  objective  function  that
                                 LM =−             cm,n   log(  ˆcm,n  )                    (10)preserves both the ﬁrst-order and second-order proximities.
                                              n=1                                                 SDNE [35] is able to map graph-structured data to a highly
               Loss Function. To retain both the external knowledge in the                     non-linear latent space to preserve the both the global and
            medical concept graph and patients diagnostic information in                       local network structures and is robust to data sparsity.
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                        Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                                Biomedical and Health Informatics
            JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          6
                                                                                         TABLE II
                                     OVERALL PREDICTION PERFORMANCE ON ALL DISEASES. ENTRIES IN BOLD FACE ARE THE BEST RESULTS.
                  Method                 K=1                         K=2                          K=3                          K=4                         K=5
                               F1     Recall  Precision     F1    Recall   Precision     F1    Recall   Precision    F1     Recall  Precision     F1    Recall   Precision
                    SVM       0.430   0.373      0.508    0.428    0.507     0.370     0.370   0.560      0.277     0.355   0.656      0.244    0.321    0.703     0.208
                     DT       0.393   0.348      0.450    0.410    0.506     0.345     0.362   0.569      0.266     0.302   0.580      0.205    0.271    0.617     0.174
                     RF       0.414   0.361      0.483    0.427    0.517     0.364     0.386   0.596      0.285     0.343   0.645      0.233    0.299    0.671     0.193
                 DeepWalk     0.343   0.289      0.421    0.379    0.443     0.331     0.368   0.565      0.273     0.344   0.631      0.237    0.299    0.653     0.194
                   LINE       0.345   0.290      0.426    0.404    0.473     0.353     0.378   0.580      0.281     0.354   0.652      0.243    0.314    0.678     0.204
                   SDNE       0.409   0.350      0.492    0.440    0.518     0.382     0.423   0.638      0.317     0.372   0.687      0.255    0.337    0.732     0.219
                 Struc2Vec    0.284   0.245      0.339    0.284    0.341     0.244     0.266   0.409      0.197     0.240   0.451      0.163    0.246    0.547     0.159
                    Ours      0.406   0.347      0.488    0.457    0.547     0.393     0.427   0.648      0.318     0.379   0.702      0.259    0.346    0.759     0.224
               Struc2Vec [36] uses a hierarchy to measure node similar-                             WhenK>   1, SVM, DT and RF obtain similar results
            ities at different scales and constructs a multi-layer graph to                     with  F1  scores  over  0.41  whenK   =  2     and  around  0.37
            encode structural contexts into node embeddings.                                    whenK  = 3. For the graph embedding-based models, SDNE
            C. Experimental Settings and Evaluation Protocols                                   performs the best among other graph embedding-based models
                                                                                                and it is comparable to our model. As an extension to the
               We evaluate our model in terms of performance on both the                        LINE  model,  SDNE  focuses  more  on  the  ﬁrst-order  and
            general disease prediction and rare disease prediction. For our                     second-order proximity between nodes, making it able to learn
            model, we set the learning rate and batch size respectively to                      highly representative node embeddings. Also, Struc2Vec did
            0.3     and 200, the aggregator layer size  L = 1. The dimensions                   not perform as good as other models, and it is possibly due to
            ofinitialization(i.e.,h0v)andoutput(h Lv )embeddingsaresetto                        the fact that Struc2Vec mainly models the structural similarity
            10,  000    and 1, 000, respectively. For each node, we uniformly                   between the nodes instead of the neighbor nodes’ features,
            sample 5  of its neighbors for information aggregation (i.e.,                       which are rather important when inferring the representation
            |N(v)|= 5). In classic baselines, i.e., SVM, DT and RF, we                          of a disease from its related symptoms.
            takesymptomsineachpatient’sEMRsastheinputfeaturesfor                                    It is worth noting that our model can scale up to more
            classiﬁcation. In all graph embedding-based peer methods, we                        complex EMR datasets with heterogeneous information. As
            transform the EMRs into a patient record graph as their input.                      symptoms and diagnosed diseases are two fundamental types
            We randomly split the patients in the Proprietary EMR dataset                       of  clinical  information  available  in  almost  all  EMRs,  our
            with a ratio of 7:3     for training and evaluation, respectively.                  proposed model is compatible and can easily generalize to
               We utilize three widely-used metrics, namely recall,         preci-              other EMR datasets containing such information. Furthermore,
            sion, and        F1 score of the top-K         diseases in each patient’s           as described in Section III-B, by transforming auxiliary side
            prediction  result.  In  short,  recall  reﬂects  how  accurately  a                information  into  initial  node  features,  our  model  can  fully
            model can predict the right disease for a patient, precision                        incorporate the available knowledge from complex EMR data
            indicates  how well  a  model distinguishes  the  true  diseases                    to achieve optimal performance.
            from the false ones, while F1 is the trade-off between two                          E. Rare Disease Prediction
            terms by taking the harmonic mean of recall and precision.
            Here we chooseK  ={1,  2, 3, 4, 5} based on the average and                             In this section, we exclusively evaluate the prediction per-
            maximum number of diagnosed diseases in the dataset.                                formance on rare diseases. We apply the same training settings
            D. Overall Prediction Effectiveness                                                 described  in  Section  IV-C  and  only  make  predictions  on
                                                                                                patients diagnosed with at least on rare disease in the test
               We showcase all models’ prediction results on the general                        set. The performance of all models in this task is reported in
            disease prediction task (i.e., considering all diseases in the                      Table III. Similar to the general disease prediction task, our
            dataset) in Table II. Apparently, in most cases (K   = 2,  3, 4, 5),                model outperforms all baselines in this task forK>  1. Also,
            our  model  outperforms  all  state-of-the-art  baselines  by  a                    whenK  = 2    orK  = 3, every model reaches its highest
            signiﬁcant margin. This veriﬁes our model’s effectiveness in                        F1 score. Generally, the results indicate that our model can
            inductively representing a disease by aggregating the learned                       capture the latent relationship between symptoms and diseases
            representations of its neighbor nodes (i.e., diseases). With an                     to better distinguish the types of rare diseases.
            increasingK , the general trend shows an increasing recall and                          Compared  with  the  overall  performance,  the  results  of
            a decreasing precision, whilst all models achieved the highest                      most models in rare disease prediction slightly drops. It is
            F1 score whenK  = 2    orK  = 3  . It is because that, when                         reasonable because the information and number regarding rare
            more possible diseases are predicted, more actually diagnosed                       diseases  and  diagnosed  patients  are  insufﬁcient  for  models
            diseases will be covered in the result, but the percentage of                       to  thoroughly  learn  the  patterns.  In  terms  ofK   =  1  ,  the
            correct results also drops in the prediction.                                       traditional methods like the RF method yields better perfor-
               In the case ofK  = 1, the SVM performs the best. However,                        mancethanallgraphembedding-basedmodels,includingours.
            as each patient is diagnosed with an average of more than                           Though graph embedding-based method shows advantageous
            one disease, the setting of generating only one prediction for                      performance whenK>  1  on general disease prediction, it
            each patient can be a mismatch for the real situations and                          is interesting that the traditional machine learning methods
            misses important opportunities for identifying other diseases.                      can perform better on rare disease prediction whenK  grows,
            Furthermore, in real-world scenarios, it is more practical and                      especially whenK  = 2. One possible reason might be that
            realistic  if  a  disease  prediction  model  can  provide  a  few                  the  graphs  constructed  from  our  EMR  dataset  is  not  large
            possible results to assist the doctors with accurate diagnoses.                     enough  for  models  to  capture  the  relation  between  nodes.
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                         Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                                 Biomedical and Health Informatics
            JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          7
                                                                                           TABLE III
                                          PREDICTION PERFORMANCE ON RARE DISEASES. ENTRIES IN BOLD FACE ARE THE BEST RESULTS.
                  Method                  K=1                          K=2                          K=3                          K=4                           K=5
                                F1    Recall   Precision     F1     Recall   Precision    F1     Recall   Precision     F1    Recall   Precision     F1    Recall   Precision
                    SVM       0.332    0.271      0.430     0.344   0.377      0.316     0.303   0.419      0.237     0.337    0.571     0.239     0.306    0.618      0.204
                     DT       0.374    0.321      0.447     0.410   0.486      0.355     0.386   0.577      0.289     0.328    0.601     0.226     0.286    0.618      0.186
                     RF       0.388    0.329      0.474     0.433   0.499      0.382     0.404   0.591      0.307     0.343    0.621     0.237     0.318    0.670      0.209
                 DeepWalk     0.196    0.148      0.289     0.336   0.364      0.311     0.330   0.470      0.254     0.308    0.520     0.219     0.273    0.546      0.182
                    LINE      0.172    0.126      0.272     0.360   0.392      0.333     0.348   0.493      0.269     0.313    0.531     0.221     0.289    0.575      0.193
                   SDNE       0.264    0.209      0.360     0.383   0.421      0.351     0.388   0.543      0.301     0.341    0.584     0.241     0.318    0.637      0.212
                 Struc2Vec    0.123    0.092      0.184     0.184   0.200      0.171     0.182   0.248      0.143     0.182    0.307     0.129     0.199    0.393      0.133
                    Ours      0.329    0.267      0.430     0.442   0.503      0.395     0.408   0.578      0.316     0.375    0.652     0.263     0.336    0.681      0.223
            In this situation, directly using the symptoms as the feature
            vectors and utilize traditional classiﬁers can already provide
            good prediction performance.
            F. Impact of Hyperparameters
                We further study our model’s sensitivity to different values
            of key hyperparameters, namely the initialization embedding
            dimension of h0v, the output embedding dimension of zv  (i.e.,
            d), and the aggregator layer sizeL. Speciﬁcally, we test the
            overall disease prediction performance with different hyper-
            parameters on the full EMR dataset following the evaluation
            protocols as in Section IV-C. We report the F1 scores with
            K  = 2,  3, 5  for demonstration.
                Initialization    Embedding    Dimension.   In   this   test,
            we    vary    the    initialization    embedding    dimension    in
            {500,      1,000,       2,000,       4,000,       6,000,       8,000,       10,000}            and   record
            the  performance  of  our  model  accordingly  in  Fig.3(a).  In                      Fig. 3.   Results obtained with: (a) different initial embedding dimensions;
            general, the higher the dimension is, the better performance                          (b) different output embedding dimensions; and (c) different aggregator layer
                                                                                                  sizes. (d) is the average batch time cost with different aggregator layer sizes.
            will be achieved. The most obvious performance gain from                              may infuse noise into node embeddings and lead to inferior
            higher  initialization  embedding  dimension  is  observed  at                        performance. On the other hand, the time cost increases when
            K  = 2    andK  = 3. When the dimension exceeds     2, 000,                           L increases from 1  to 4  because it takes more computational
            the  performance  of  our  model  becomes  stable.  Notably,                          steps to generate the embedding for each target node, but our
            though the values of node embedding vectors are randomly                              proposed graph neural network is highly efﬁcient withL = 1
            initialized,  it  is  the  carrier  of  the  graph  topology  structure               as the average running time per batch is less than 0.1s.
            information,  i.e.,  the  crucial  disease-symptom  relationships,                    G. Comparing Different Graph Encoder Kernels
            and  higher  embedding  dimension  means  that  more  latent
            structural information is passed into our model.                                          In graph neural networks, the selection of an appropriate
                Output Embedding Dimension. We also explore the effect                            kernel function for information aggregation is largely associ-
            ofdifferentoutembeddingdimensionsfromthegraphencoder.                                 ated with the characteristics of the data. As we have described
            The  embedding  dimension  of  the  encoder  is  adjusted  in                         in Section III-C, we utilize two GNN-based variants as the
            {50,    100,    200,    400,    600,    800,    1,000}, and Fig.3(b) shows cor-       graph encoder kernel in our model. In particular, we choose
            responding results. A dramatical performance boost appears                            GATandGINastwovariantkernelsformodellingourmedical
            as the embedding dimension increases from 50   to 100. We                             concept graph and patient record graph, because GAT focuses
            can observe that when the dimension of embedding is over                              on the most relevant parts of the input to make decisions [12],
            100,  the  performance  increase  becomes  negligible,  which                         whileGINgeneralizestheWeisfeiler-Lehmantestandishence
            indicates  that  our  proposed  model  can  preserve  the  node                       able to produce discriminative node embeddings [13].
            and structural information well for disease prediction with a                             The comparison results are illustrated in Table IV, where
            relatively compact output dimension.                                                  we use “overall” and “rare” to mark the prediction results in
                Aggregator Layer SizeL. Our model’s performance with                              general and rare disease prediction tasks, respectively. Both
            layer sizeL∈{1, 2, 3, 4} is reported in Fig.3(c). The em-                             GAT and GIN are deployed with a single-layer structure and
            bedding dimension of each layer is 1,000 and the number of                            the same hyperparameters introduced in Section IV-C. In our
            sampled neighbors is 5. As increasing the layer size directly                         assumption,  utilizing  complex  kernel  methods  can  improve
            affects the running time of our model, we also report the time                        the model’s expressiveness for disease prediction. However, in
            cost per batch in Fig.3(d). On one hand, the results show that                        our experiments, both GAT and GIN kernels are not as good
            adding extra deep layers in our graph neural network does                             as our default mean aggregator function. Firstly, an obvious
            not incur further performance gain. As our constructed bipar-                         performance  drop  is  observed  with  the  GIN  kernel.  One
            tite graphs contain only patient-disease and disease-symptom                          possible reason is that, GIN neglects the important information
            links, so settingL = 1    is already sufﬁcient for the model to                       from  the  target  nodev itself  when  generating  embedding
            learn representative node embeddings for disease prediction,                          hlv.  Secondly,  GAT  kernel  shows  slightly  lower  prediction
            while  propagating  more  information  with  additional  layers                       accuracy  than  our  default  model,  and  the  cause  is  largely
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                         Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3004143, IEEE Journal of
                                                                                          Biomedical and Health Informatics
              JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS                                                                                                                                                                                          8
                                                                                                    TABLE IV
                               PREDICTION PERFORMANCE WITH DIFFERENT GRAPH ENCODER KERNELS. ENTRIES IN BOLD FACE ARE THE BEST RESULTS.
                     Kernel                     K=1                              K=2                             K=3                             K=4                              K=5
                                      F1     Recall    Precision      F1     Recall    Precision      F1      Recall   Precision       F1     Recall    Precision      F1     Recall    Precision
                 GAT (overall)      0.394    0.330       0.488      0.411     0.479       0.360      0.385    0.574       0.289      0.352    0.646       0.242      0.310     0.672       0.202
                 GIN (overall)      0.274    0.233       0.331      0.308     0.362       0.269      0.276    0.423       0.205      0.270    0.515       0.183      0.238     0.531       0.154
                 Ours (overall)     0.406    0.347       0.488      0.457     0.547       0.393      0.427    0.648       0.318      0.379    0.702       0.259      0.346     0.759       0.224
                   GAT (rare)       0.287    0.223       0.404      0.361     0.394       0.333      0.352    0.491       0.275      0.346    0.606       0.242      0.292     0.582       0.195
                   GIN (rare)       0.131    0.102       0.184      0.206     0.221       0.193      0.204    0.289       0.158      0.191    0.348       0.132      0.197     0.406       0.130
                  Ours (rare)       0.329    0.267       0.430      0.442     0.503       0.395      0.408    0.578       0.316      0.375    0.652       0.263      0.336     0.681       0.223
              related to its excessive model parameters and the insufﬁciency                                 [11]  W. L. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation
              of the available EMR data. GAT introduces substantially more                                         learning on large graphs,” in NIPS, 2017, pp. 1025–1035.
              parameters, making it prone to overﬁtting and require a lot                                    [12]  P. Veliˇckovi´c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Ben-
                                                                                                                   gio,“Graphattentionnetworks,” arXivpreprintarXiv:1710.10903,2017.
              more training data and iterations to effectively optimize all                                  [13]  K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph
              modelparametersandthoroughlycapturethecomplexpatterns                                                neural networks?” in ICLR, 2019.
              within the data. In contrast, our model only aggregates the                                    [14]  P. B. Jensen, L. J. Jensen, and S. Brunak, “Mining electronic health
                                                                                                                   records: towards better research applications and clinical care,” Nature
              information of ﬁrst-order neighbors, so it can directly learn the                                    Reviews Genetics, vol. 13, no. 6, p. 395, 2012.
              disease-symptom relationships to ensure better performance.                                    [15]  E. Choi, M. T. Bahadori, L. Song, W. F. Stewart, and J. Sun, “Gram:
                                                                                                                   graph-based attention model for healthcare representation learning,” in
                                                                                                                   SIGKDD, 2017, pp. 787–795.
                                              V.  CONCLUSION                                                 [16]  E. Choi, M. T. Bahadori, J. Sun, J. Kulas, A. Schuetz, and W. Stewart,
                 In this work, we present a GNN-based model for disease                                            “Retain: An interpretable predictive model for healthcare using reverse
                                                                                                                   time attention mechanism,” in NIPS, 2016, pp. 3504–3512.
              prediction with EMRs, which novelly leverages the external                                     [17]  F.Ma,R.Chitta,J.Zhou,Q.You,T.Sun,andJ.Gao,“Dipole:Diagnosis
              graph-structured medical knowledge to learn the latent node                                          prediction in healthcare via attention-based bidirectional recurrent neural
              embeddings, thus enabling accurate disease prediction for new                                        networks,” in SIGKDD, 2017, pp. 1903–1911.
              patients under sparse training data in an inductive manner. The                                [18]  F.  Ma,  Q.  You,  H.  Xiao,  R.  Chitta,  J.  Zhou,  and  J.  Gao,  “Kame:
                                                                                                                   Knowledge-based attention model for diagnosis prediction in health-
              experimental results on our real-world EMR dataset shows                                             care,” in CIKM, 2018, pp. 743–752.
              promising effectiveness of our proposed model, especially in                                   [19]  X. S. Zhang, F. Tang, H. H. Dodge, J. Zhou, and F. Wang, “Metapred:
              multi-label  disease  classiﬁcation  settings.  To  conclude,  our                                   Meta-learning for clinical risk prediction with limited patient electronic
                                                                                                                   health records,” in SIGKDD, 2019, pp. 2487–2495.
              model  offers  an  intuitive  yet  accurate  solution  to  disease                             [20]  H. Chen, H. Yin, W. Wang, H. Wang, Q. V. H. Nguyen, and X. Li,
              prediction, tackling the data scarcity problem and the hardship                                      “Pme: projected metric embedding on heterogeneous networks for link
              in diagnosing rare diseases at the same time.                                                        prediction,” in SIGKDD, 2018, pp. 1177–1186.
                                                                                                             [21]  Y.  Wang,  H.  Yin,  H.  Chen,  T.  Wo,  J.  Xu,  and  K.  Zheng,  “Origin-
                                                                                                                   destination matrix prediction via graph convolution: a new perspective
                                           ACKNOWLEDGMENT                                                          of passenger demand modeling,” in SIGKDD, 2019, pp. 1227–1235.
                                                                                                             [22]  T. Chen, H. Yin, H. Chen, R. Yan, Q. V. H. Nguyen, and X. Li, “Air:
                 This  work  is  supported  by  NSFC  No.91846205,  Na-                                            Attentional intention-aware recommender systems,” in ICDE, 2019, pp.
              tional Key R&D Program No.2017YFB1400100, Innovation                                                 304–315.
              Method Fund of China No.2018IM020200, and Shandong Key                                         [23]  H.  Chen,  H.  Yin,  T.  Chen,  W.  Wang,  X.  Li,  and  X.  Hu,  “Social
                                                                                                                   boosted  recommendation  with  folded  bipartite  network  embedding,”
              R&D   Program   No.2018YFJH0506,   No.2019JZZY011007.                                                TKDE, 2020.
              This work is also supported by Australian Research Council                                     [24]  H. Chen, H. Yin, T. Chen, Q. V. H. Nguyen, W.-C. Peng, and X. Li,
              (Grant No.DP190101985).                                                                              “Exploiting centrality information with graph convolutions for network
                                                                                                                   representation learning,” in ICDE, 2019, pp. 590–601.
                                                                                                             [25]  H. Chen, H. Yin, X. Sun, T. Chen, B. Gabrys, and K. Musial, “Multi-
                                                 REFERENCES                                                        level graph convolutional networks for cross-platform anchor link pre-
               [1]  R. Hillestad, J. Bigelow, A. Bower, F. Girosi, R. Meili, R. Scoville, and                      diction,” SIGKDD, 2020.
                    R.Taylor,“Canelectronicmedicalrecordsystemstransformhealthcare?                          [26]  R. P. Garg, S. Dong, S. J. Shah, and S. R. Jonnalagadda, “A bootstrap
                    potential health beneﬁts, savings, and costs,” Health affairs, vol. 24,                        machine learning approach to identify rare disease patients from elec-
                    no. 5, pp. 1103–1117, 2005.                                                                    tronic health records,” CoRR, vol. abs/1609.01586, 2016.
               [2]  Q. Suo, F. Ma, Y. Yuan, M. Huai, W. Zhong, A. Zhang, and J. Gao,                         [27]  H.  MacLeod,  S.  Yang,  K.  Oakes,  K.  Connelly,  and  S.  Natarajan,
                    “Personalized disease prediction using a cnn-based similarity learning                         “Identifying rare diseases from behavioural data: A machine learning
                    method,” in BIBM.    IEEE Computer Society, 2017, pp. 811–816.                                 approach,” in CHASE.    IEEE Computer Society, 2016, pp. 130–139.
               [3]  F. Ma, J. Gao, Q. Suo, Q. You, J. Zhou, and A. Zhang, “Risk prediction                   [28]  T. Hare, P. Sharan, E. J. Kleczyk, and D. Evans, “Improving accuracy in
                    on electronic health records with prior medical knowledge,” in SIGKDD,                         rare disease patient identiﬁcation using pattern recognition ensembles,”
                    2018, pp. 1910–1919.                                                                           Journal of the Pharmaceutical Management Science Association, 2018.
               [4]  J. Quinlan, “Simplifying decision trees,”  International Journal of Man-                 [29]  M. Schubach, M. Re, P. N. Robinson, and G. Valentini, “Imbalance-
                    Machine Studies, vol. 27, no. 3, pp. 221–234, 1987.                                            aware  machine  learning  for  predicting  rare  and  common  disease-
               [5]  H. Zhang, “The optimality of naive bayes,” AA, vol. 1, no. 2, p. 3, 2004.                      associated non-coding variants,” Scientiﬁc Reports, 2017.
               [6]  Y. LeCun, Y. Bengio, and G. Hinton, “Deep learning,” nature, vol. 521,                   [30]  S. K¨ohler, N. A. Vasilevsky et al., “The human phenotype ontology in
                    no. 7553, pp. 436–444, 2015.                                                                   2017,” Nucleic acids research, vol. 45, no. D1, pp. D865–D876, 2016.
               [7]  S.  Palaniappan  and  R.  Awang,  “Intelligent  heart  disease  prediction               [31]  C. Cortes and V. Vapnik, “Support-vector networks,” Machine Learning,
                    system using data mining techniques,” in IEEE/ACS International Con-                           vol. 20, no. 3, pp. 273–297, Sep 1995.
                    ference on Computer Systems and Applications, 03 2008, pp. 108–115.                      [32]  T. K. Ho, “Random decision forests,” in  ICDAR, 1995, pp. 278–282.
               [8]  R. C. Griggs, Batshaw et al., “Clinical research for rare disease: oppor-                [33]  B. Perozzi, R. Al-Rfou, and S. Skiena, “Deepwalk: Online learning of
                    tunities, challenges, and solutions,” Molecular genetics and metabolism,                       social representations,” in SIGKDD, 2014, pp. 701–710.
                    vol. 96, no. 1, pp. 20–26, 2009.                                                         [34]  J. Tang, M. Qu, M. Wang, M. Zhang, J. Yan, and Q. Mei, “Line: Large-
               [9]  H.J.Dawkins,R.Draghia-Aklietal.,“Progressinrarediseasesresearch                                scale information network embedding,” in WWW, 2015, pp. 1067–1077.
                    2010–2016: an irdirc perspective,” Clinical and translational science,                   [35]  D. Wang, P. Cui, and W. Zhu, “Structural deep network embedding,” in
                    vol. 11, no. 1, p. 11, 2018.                                                                   SIGKDD, 2016, pp. 1225–1234.
              [10]  S.  N.  Wakap,  D.  M.  Lambert  et  al.,  “Estimating  cumulative  point                [36]  L. F. Ribeiro, P. H. Saverese, and D. R. Figueiredo, “struc2vec: Learning
                    prevalenceof rarediseases:analysis oftheorphanet database,”European                            node representations from structural identity,” in SIGKDD, 2017, pp.
                    Journal of Human Genetics, vol. 28, no. 2, pp. 165–173, 2020.                                  385–394.
      2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                            Authorized licensed use limited to: University of Exeter. Downloaded on July 16,2020 at 06:03:54 UTC from IEEE Xplore.  Restrictions apply.

