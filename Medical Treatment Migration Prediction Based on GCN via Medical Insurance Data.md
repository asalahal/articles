This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3008493, IEEE Journal of
                                                                               Biomedical and Health Informatics
                                                                                                                                                                              1
                Medical Treatment Migration Prediction based on
                                         GCN via Medical Insurance Data
                                          Yongjian Ren, Yuliang Shi, Kun Zhang, Zhiyong Chen, Zhongmin Yan
               Abstract—Nowadays,  prediction  for  medical  treatment  mi-                    treatment graph to make predictions of the medical treatment
            gration  has  become  one  of  the  interesting  issues  in  the  ﬁeld             migration behavior. The medical treatment graph covers the
            of  health  informatics.  This  is  because  the  medical  treatment               entities  of  patients,  hospitals,  diseases,  and  medical  items
            migration behavior is closely related to the evaluation of regional                (such as medicines). In addition, a patient’s medical record
            medical  level,  the  rational  use  of  medical  resources,  and  the             (identiﬁed by the record ID) is modeled as an event entity in
            distribution of medical insurance. Therefore, a prediction model
            for medical treatment migration based on medical insurance data                    the graph.
            is  introduced  in  this  paper.  First,  a  medical  treatment  graph                Recently, graph representation learning and application has
            is  constructed  based  on  medical  insurance  data.  The  medical                gained  increasing  attention  [1],  [2],  [3].  This  is  caused  by
            treatment graph is a heterogeneous graph, which contains entities                  its intuitive and ﬂexible data organization and presentation,
            such  as  patients,  diseases,  hospitals,  medicines,  hospitalization            that  is,  to  describe  complex  facts  through  triples.  An  in-
            events,  and  the  relations  between  these  entities.  However,  ex-
            isting graph neural networks are  unable to capture the time-                      teresting  development  of  representation  learning  for  graph
            series relationships between event-type entities. To this end, a                   structure  data  is  the  emergence  of  GCN  [4],  in  which  the
            prediction model based on Graph Convolutional Network (GCN)                        information propagation of neighboring nodes is proved to be
            is proposed in this paper, namely, Event-involved GCN (EGCN).                      effective for graph neural networks according to Chebyshev
            The proposed model aggregates conventional entities based on                       ﬁrst-order approximation of the eigenvalues of the Laplace
            attention  mechanism,  and  aggregates  event-type  entities  based
            on a gating mechanism similar to LSTM. In addition, jumping                        matrix. Subsequently, graph neural networks that can repre-
            connection is deployed to obtain the ﬁnal node representation. In                  sent multiple types of relationships have been proposed. R-
            order to obtain embedded representations of medicines based on                     GCN  [5]  is  a  GCN  used  on  the  knowledge  graph,  which
            external information (medicine descriptions), an automatic en-                     realizes the representation learning of the relationship between
            coder capable of embedding medicine descriptions is deployed in                    multiple relationship types and entities. The model takes into
            the proposed model. Finally, extensive experiments are conducted
            on a real medical insurance data set. Experimental results show                    account the central nodes, relationships and neighbors when
            that our model’s predictive ability is better than the best models                 performing graph convolution operations. GCN-ED [6] uses
            available.                                                                         graph convolutional networks to learn syntactic information,
               Index Terms—GCN, deep learning, medical treatment migra-                        and adds weight adjustment factors based on neural networks
            tion prediction, health.                                                           to the convolution operation. KGCN-sum [7] and PinSage [8]
                                                                                               are GCN-based recommendation system models. KGCS-sum
                                        I.  INTRODUCTION                                       gives neighbors different weights when aggregating neighbor
                NChina,medicaltreatmentmigrationreferstothebehavior                            information,  while  PinSage  model  uses  a  two-layer  neural
            Iof  seeking  medical  treatment  outside  the  insured  areas                     network  in  the  convolutional  layer  and  combines  neighbor
            by an insured person. The occurrence of such behavior may                          selection based on importance.
            reﬂect the imbalance of therapeutic level in the region, and                          Although a graph neural network model usually follows the
            may affect the rational use of medical resources and the dis-                      principle of disorder of node neighbors, some existing work
            tributionstrategyofmedicalinsurance.Therefore,thebehavior                          has conducted beneﬁcial research on the ordering of nodes.
            prediction of medical treatment migration has become one of                        GraphSage  [9]  explores  a  variety  of  neighbor  aggregation
            the interesting issues in the ﬁeld of health informatics. In this                  methods including LSTM based aggregation. After performing
            paper, we construct the medical insurance data as a medical                        the node representation of the graph through classic GCN,
                                                                                               DGCNN [10] sorts the nodes based on SoftPooling to obtain
               Yongjian  Ren  is  with  School  of  Software,  Shandong  University,  Jinan,   the graph-level output. PATCHY-SAN [11] sorts the neighbors
            China, e-mail: ryjsdu@outlook.com.                                                 according to the label of each node, and then expresses the
               Yuliang  Shi  is  with  School  of  Software,  Shandong  University,  Jinan,    ordered neighbors as grid-structured data so that the converted
            China,  and  works  at  Dareway  Software  Co.,  Ltd,  Jinan,  China,  e-mail:     data can be learned through CNN convolution kernel.
            shiyuliang@sdu.edu.cn, and he is corresponding author.
               Kun  Zhang  is  with  School  of  Information  Science  and  Engineering,          The combination of medical data and graph model provides
            University  of  Jinan,  250022,  China;  Shandong  Provincial  Key  Laboratory     thepossibilitytodigdeepintothecomplexhiddeninformation
            of Network-based Intelligent Computing, Jinan, 250022, China; School of            in medical data [12], [13], [14], [15]. It is worth noting that
            Software, Shandong University, Jinan, China; and works at Dareway Software
            Co., Ltd, Jinan, China, e-mail: kunzhangcs@126.com.                                in the medical treatment graph constructed based on medical
               Zhiyong Chen is with School of Software, Shandong University, Jinan,            insurance  data,  the  order  of  event  entities  has  important
            China, e-mail: chenzy@sdu.edu.cn.                                                  practical signiﬁcance for patients. Therefore, the EGCN model
               Zhongmin Yan is with School of Software, Shandong University, Jinan,
            China, e-mail: yzm@sdu.edu.cn.                                                     is proposed in this paper. The model aggregates the regular
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                    Authorized licensed use limited to: Uppsala Universitetsbibliotek. Downloaded on July 20,2020 at 06:48:41 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3008493, IEEE Journal of
                                                                             Biomedical and Health Informatics
                                                                                                                                                                          2
            neighbors of the node based on the attention mechanism, and
            deals with the neighbors with a sequence relationship (i.e.,
            medical  treatment  events)  based  on  the  gating  mechanism.
            In addition, jumping connections [10], [16], [17] that have
            been proven to effectively improve the performance of GNN
            models  are  deployed.  Due  to  the  long-tailed  nature  of  the
            data,  most  medicines  appear  less  frequently,  which  is  not
            conducive to statistical learning of the model. Therefore, we
            introduce external information in the form of text, that is, the
            medicine description which contains the medicine name, the
            composition of medicine, the interaction of the drug, and the
            possible impact on body organs. The unsupervised automatic
            encoder KATE [18] is used to pre-train medicine descriptions                                  Fig. 1: Information propagation in GCN.
            to realize the semantic embedding of the medicines. Finally,
            extensive experiments were conducted based on real medical
            insurance data.                                                                  method usually needs to deﬁne the meta paths. However, pre-
               The main contributions of this paper are as follows:                          deﬁning meta paths not only brings extra manual workload
               •We construct a medical treatment graph, which models                         but also limits the model’s expressive ability.
                  a patient’s medical treatment event as an entity. In addi-                    GCN is a model for representation learning of graph struc-
                  tion, the graph also contains entities of patient, disease,                turedatabasedonconvolutionoperation[4],whichrealizesthe
                  hospital, and the relationship between these entities. The                 information propagation between adjacent nodes according to
                  medical treatment graph describes the patient’s historical                 the ﬁrst-order approximation of Chebyshev polynomial. The
                  medical records, and also enables the patient to explicitly                core idea of GCN is that nodes exchange information with
                  ﬁnd other patients who are similar (e.g., those suffering                  their neighbors and iteratively obtain information about their
                  from the same disease).                                                    local networks [4], [9], [28]. Speciﬁcally, each node can get
               •Weproposeagraphneuralnetworkmodelthatcanhandle                               information from its neighbors and propagate its information
                  evententities.Wheniteratingtherepresentationofanode,                       to its neighbors. When all the nodes in the graph complete
                  it ﬁrst performs attention-based aggregation on regular                    the information exchange with their neighbors, one iteration
                  neighbors,  then  performs  gating-based  aggregation  on                  is completed. AfterK   iterations, each node is ﬁnally able
                  event  neighbors,  and  ﬁnally  updates  the  node  repre-                 to obtain the information of theK -neighborhood, as shown
                  sentation based on the aggregation results. In addition,                   in Figure 1. The computational cost of GCN is proportional
                  the  model  uses  jumping  connections  to  generate  the                  to the number of relationships in the graph and the number
                  ﬁnal node representation. And an unsupervised automatic                    of iterations of information exchange. Once emerged, GCN
                  encoder is employed to realize the semantic embedding                      was quickly applied to many ﬁelds [29], [30] including trafﬁc
                  of the external information of medicines.                                  prediction [31], EEG emotion recognition [32], protein family
               The rest of this paper is as follows. In Section II, related                  classiﬁcation [33] ,and fraud detection [34].
            work is introduced. In Section III, ﬁrst, the construction of                       When implementing the GCN model using a deep neural
            the  medical  treatment  graph  is  introduced  in  detail.  Then,               network, each iteration will produce a new neural network
            the  graph  neural  network  model  proposed  in  this  paper  is                layerforembeddedrepresentationsofentities.Inthefollowing
            presented in detail. Experimental results are shown in Section                   sections of this paper, we use the concept of layer to describe
            IV and the paper is concluded in Section V.                                      the embedded representation of nodes and edges.
                                      II.  RELATED WORK                                      B. GCNs in the medical ﬁeld
            A. The GCN Model                                                                    Seg-GCRN  is  a  model  using  segmentation  graph  convo-
                                                                                             lution and RNN to classify relationships in clinical records
               At  ﬁrst,  graph  representation  learning  was  mostly  mod-                 [35]. Seg-GCRN ﬁrst converts the clinical records in natural
            eled on triplets [19], [20], [21]. They are difﬁcult to mine                     language into data in graph form, and then uses the graph con-
            the  information  implied  by  the  network  topology  in  graph                 volutional layer to learn syntactic dependency information and
            structure data. Subsequently, the models that using Convolu-                     uses bi-LSTM to learn word sequence information. The model
            tional Neural Network (CNN) [22], [23] or Recurrent Neural                       uses word embedding and sentence syntax dependencies and
            Network  (RNN)  [24],  [25],  [26],  [27]  to  model  the  path,                 does not require manual feature engineering.
            and then combining the weighted paths to obtain the hidden                          InceptionGCN is a GCN for disease prediction [36]. The
            information of the complex connections is proposed. However,                     novelty  of  the  model  lies  in  geometric  inception  modules,
            considering all possible paths leads to a high computational                     which  are  capable  of  capturing  intra-graph  and  inter-graph
            cost. In a graph, the number of paths starting from a given                      structural heterogeneity during convolutions.
            node and having a length ofK  will explode with the increase                        Decagon  is  a  method  of  simulating  the  side  effects  of
            ofK . Therefore, the path-based graph representation learning                    multiple  drugs  [37].  This method  constructs  a  multi-modal
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                   Authorized licensed use limited to: Uppsala Universitetsbibliotek. Downloaded on July 20,2020 at 06:48:41 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3008493, IEEE Journal of
                                                                                     Biomedical and Health Informatics
                                                                                                                                                                                           3
             map including protein-protein interactions, drug-protein target                                  TABLE I: Triplets in the medical treatment graph.
             interactions,anddrug-druginteractions.Inthegraph,eachside                                   Head type      Relation type      Tail type      Meanings
             effect is represented by a speciﬁc type of edge. The authors                                Person           Happen               Event           Thepersonhasamedicalevent
                                                                                                         Event             Get                      Disease       Thediseaserelatedtotheevent
             used  Decagon  to  predict  the  exact  side  effects  of  a  given                         Event             In hospital          Hospital      The hospital visited
             combination of drugs.                                                                       Event             Take                    Item             Medical  items  used,  such  as
                Intheliterature[38],theauthorsconductedacomprehensive                                                                                    medicines
             assessment of the proposed general framework which utilizing                               TABLE II: Entity attributes in the medical treatment graph.
             imaging and non-imaging information and used the framework                                  Node          Attributes
             topredictdiseaseprogressioninautismspectrumdisordersand                                     Event          Occurrence time, total cost, hospital stay, etc.
             Alzheimer’s disease. The framework exploits GCNs where its                                  Person        Gender, date of birth, etc.
             nodes are associated with imaging-based feature vectors while                               Disease       2-week mortality, 1-year mortality, etc.
             phenotypic information is integrated as edge weights.                                       Hospital      Hospital level (implies the hospital’s medical level)
                                                                                                         Item            -
                In  the  literature  [39],  the  authors  capture  the  molecular
             characteristics of breast cancer in quantitative and qualitative
             evaluation through integrated GCN and relational networks.                                   We construct the medical treatment graph (which is a direct-
                In the medical treatment graph studied in this paper, the                              ed heterogeneous graph) based on the medical insurance data.
             sequence information of event nodes has important practical                               The medical insurance data is stored in a relational database.
             signiﬁcance for the patient node. However, the above medical-                             A medical record represents a medical treatment event, which
             related graph models follow the disorder principle of node                                has an event ID. A medical record records the patient ID,
             neighbors, which cannot properly meet our needs. Therefore,                               disease ID, hospital ID, and medical item IDs related to the
             a graph neural network model that deploys attention-based ag-                             medical treatment event. Therefore, we can directly extract
             gregation for conventional neighbors and gating-mechanism-                                entities from the medical records based on the entity ID and
             based  aggregation  for  event  neighbors  is  proposed  in  this                         obtain the relationship between them. Forms of triplets in the
             paper.                                                                                    medical treatment graph are shown in Table I. Then, we are
                                                                                                       abletoconstructthemedicaltreatmentgraphbasedonmedical
                                              III.  METHOD                                             insurance data, as shown in Figure 2. In addition, the medical
                In this section, we ﬁrst introduce the structure and char-                             insurance data contains attribute information of entities. The
             acteristics of the medical treatment graph. Subsequently, the                             attributes of various types of entities are shown in Table II.
             proposed EGCN model is introduced in detail. What is special                              For example, based on the time of the event and the date of
             about  EGCN  is  that  it  considers  the  time  series  of  event-                       birth of the patient, we can calculate the age of the patient at
             type entities. Besides, each EGCN layer contains two neural                               the time the medical event occurred. Therefore, the medical
             network layers, namely neighbor aggregation layer and feature                             treatment graph is a graph with node attribute information.
             extraction layer.
                                                                                                       B. EGCN
             A. The Heterogeneous Medical Treatment Graph                                                 1) EGCN layer: In the medical treatment graph, the neigh-
                                                                                                       borsoftheeventnodesareallregularnodes,andtheneighbors
                                                                                                       of the regular nodes can be regular nodes or event nodes or
                                                                                                       both. Without losing generality, Figure 3 shows the situation
                                                                                                                         TABLE III: Summary of notations.
                                                                                                         Notations       Meanings
                                                                                                         ∗                  Math multiplication
                                                                                                         ·                  Multiply corresponding position elements
                                                                                                         ⊙                  Concatenation
                                                                                                         v∈N (u)   N (u)      is the neighbor set of entityu. Entity vis one ofu’s
                                                                                                                        neighbors.
                                                                                                         e∈E(u)    E(u)      is the event neighbor list of entityu, whose element
                                                                                                                        order is the order of occurrence of events. Entityeis the
                                                                                                                        incident neighbor ofu.
                                                                                                         hku                   Embedded representation of entityu at layerk
                                                                                                         ikv,ike              Information passed by neighborvoreto entityu
                                                                                                         Iku                    Information passed by all neighbors to entityu
                                                                                                         τ(·)                   Activation function
                                                                                                         W kr                 Feature transformation matrix of relationrin thekth layer
                                                                                                         Akr                  Attention parameter matrix of relationrin layerk
                                                                                                         Fkr                   Parameter matrix of forgetting gate with relationrin the
                                                                                                                        kth layer
                                                                                                         αkv                   Entityu’s attention to neighbor v
                                                                                                         fke                   Output vector of the forget gate of neighbore
                              Fig. 2: The medical treatment graph.                                       W k2                  Parameter matrix of the feature extraction layer of thek-th
                                                                                                                        EGCN layer
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                     Authorized licensed use limited to: Uppsala Universitetsbibliotek. Downloaded on July 20,2020 at 06:48:41 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3008493, IEEE Journal of
                                                                                 Biomedical and Health Informatics
                                                                                                                                                                                   4
            Fig. 3: EGCN model. EGCN has two major characteristics: (1) Each layer of the model contains 2 sub-neural network layers:
            neighbor aggregation layer and feature extraction layer. (2) The model performs attention-based aggregation and gating-based
            aggregation on the regular-type neighbors and event-type neighbors, respectively. Without losing generality, the neighbor
            aggregation layer described in the ﬁgure includes the aggregation of regular neighbors and the aggregation of event neighbors.
            First, attention-based aggregation is performed on regular neighbors. Then, the aggregation based on the gating mechanism is
            performed on the event neighbors. Finally, the new node representing is obtained via the feature extraction layer.
            where  the  node  neighbor  contains  both  regular  nodes  and                       matrixW kr .
            event nodes. As shown in Figure 3, the EGCN layer has two                                First, aggregate the information of regular neighbors based
            characteristics:                                                                      on the attention mechanism:
                •A  two-layer  neural  network  is  used  to  extract  node                                                 ikv = τ((h  ku⊙hkv)∗W kr )
                   features. An EGCN layer contains two sub-layers: the                                                     skv = τ((h  ku⊙hkv)∗Akr)
                   neighbor  aggregation  layer  and  the  feature  extraction
                   layer.  One  iteration  of  node  representation  contains  2                                            αkv =          exp(s       kv)∑                     (1)
                   feature  transformations.  One  is  the  transformation  of                                                 v∈N     (u  ) exp(s       kv)
                   neighbor features at the neighbor aggregation layer, and                                                 Iku←  ∑           αkv∗ikv
                   the other is the transformation of aggregated features at                                                        v∈N     (u)
                   the feature extraction layer.
                •Deploy attention-based aggregation and gating-based ag-                          whereskv  represents the score of nodev, and generates the
                   gregation for regular neighbors and event neighbors, re-                       attention to nodevvia the softmax function.
                   spectively. If a node has regular neighbors, the model ﬁrst                       Then, nodeuaggregates the information of its event neigh-
                   aggregates the information of the regular neighbors based                      bors: 
                   on the attention mechanism. Then, similar to LSTM, the                                      ike = tanh((h           ku⊙hke)∗W kr )
                   model  uses  input  gates  and  forget  gates  to  aggregate                                fke  = sigmoid((h         ku⊙hke)∗Fkr )                          (2)
                   the information of event neighbors according to the time                                
                   sequence of event neighbors.                                                                Iku← (1−fke )∗Iku  + fke∗ike,fore∈E(u )
                Before discussing the details of the proposed EGCN model,                            Now, we can get the iteration formula of the representation
            notations and their semantics used in this paper are shown in                         of nodeu:
            Table III.                                                                                                   Ik +1u     = τ((h  ku⊙Iku )∗W k2 )                       (3)
                The model extracts features through a neural network layer
            based on the tripletT =<node,relationship,neighbor>.It should be noted that although the medical treatment graph
            The speciﬁc form is that the input is the concatenation of the                        is represented in the form of a directed graph, information
            node representation and the neighbor representation, and the                          is transmitted in both directions. The forward and reverse of
            relationship of typeris represented by a separate parameter                           the same relationship have different semantics. Therefore, the
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                    Authorized licensed use limited to: Uppsala Universitetsbibliotek. Downloaded on July 20,2020 at 06:48:41 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3008493, IEEE Journal of
                                                                                 Biomedical and Health Informatics
                                                                                                                                                                                  5
            forward and backward propagation of information is regarded                          TABLE  IV:  Convolution  operations  of  baselines.  For  com-
            as ﬂowing on different relationship types, that is, for the same                     parison,  the  convolution  operation  in  each  layer  of  each
            type of relationship, the forward and backward propagation of                        model is split into 3 sub-steps: (1) Obtain the informationiv
            information works based on different parameter matrices.                             propagatedbyneighborvtonodeu;(2)Nodeuaggregatesthe
                2) embedding layer:  In the embedding layer, the entityu                         information propagated by its neighbors to get the aggregated
            in  the  medical  treatment  graph  is  transformed  into  a  low-                   informationIu ; (3) Nodeu updates its representation vector
            dimensional dense vectorh0u :                                                        to gethk +1u     . The meanings of notations are shown in Table
                                           h0u  = W  0∗xu                                  (4)   III.
                                                                                                   Models             Convolution operation
                               ′                          ′
            wherexu  = xu⊙xpu  is a vector.xu  is the one-hot representa-                                           ikv  = hkv
            tion of entityu.xpu  is the normalized attribute vector of entity                      basic GCN        Ik +1u       = aggregatek ({i  kv|∀v∈N (u)})
            u.W e  is the parameter matrix of the embedding layer.                                                  hk +1u       = τ(W   k∗(h ku⊙Ik +1u      ))
                Due to the long-tailed nature, most medicines used to treat                                         ikv  = W kr∗(h ku⊙hkv )
            diseases  appear  less  frequently,  which  is  not  conducive  to                     R-GCN            Ik +1u       = ∑v∈N     (u) ikv
            statistical learning of the model. Fortunately, the medicine de-                                        hk +1u       = τ(Ik +1u      )
            scription contains information about medicine name, medicine                                            ikv  = W kr∗hkv
            ingredients,  drug  interactions,  and  the  effects  on  the  body                    GCN-ED           Ik +1u       = ∑v∈N     (u) fkv·ikv , andfkv  = sigmoid(h kv∗Fkr )
            organs. This textual information can be used to pre-train to                                             is a gating mechanism.
            obtain the semantic embedding of medicines. Therefore, we                                               hk +1u       = τ(Ik +1u      )
            deployed the unsupervised automatic encoder KATE [18] to                                                ikv  = W kr∗hkv
            achieve the embedding representations of medicines. KATE’s                             KGCN-sum         Ik +1u       = aggregatek ({i  kv|∀v∈N (u)}), based on user-
            encoder reads and encodes a medicine description, and the                                               relation scores.
            decoder reconstructs the description in text form according to                                          hk +1u       = τ(W   k∗(h ku  + Ik +1u      ))
            the embedding code. After the training, the embedded code                                               ikv  = τ(W   kr∗hkv )
            obtained by the KATE encoder is the embedded representa-                               PinSage          Ik +1u       = aggregatek ({i  kv|∀v∈N (u)}), based on element-
                                                                                                                    wise mean, max pooling or weighted pooling.
            tion of the medicine. The embedded code not only allows                                                 hk +1u       = τ(W   k∗(h ku⊙Ik +1u      ))
            a  less  frequent  medicine  to  obtain  a  reasonable  embedded
            representation but also reduces the dimension of the medicine
            representation, so as not to make our model too complicated.                         in  different  hospitals,  so  that  we  can  more  fully  describe
                3) output layer:  In the output layer, jumping connections                       the patient’s historical behavior. We performed 6 experiments
            are deployed. Assuming that there areK  convolutional layers,                        on each model to obtain its predictions. Experiments were
            thenhfinalu       = h1u⊙h2u⊙...⊙hKu   is the ﬁnal representation of                  performed  by  randomly  selecting  1000  patients  each  time.
            entityu.Weuseafullyconnectedneuralnetworkastheoutput                                 Each experiment involved approximately 4,300 entities and
            layer  to  predict  the  patient’s  medical  behavior  of  medical                   7,600 relationships in the graph. Among them, there are about
            treatment migration:                                                                 100 medical institution entities and about 340 disease entities.
                                 ˆyu  = sigmoid(W         out∗hfinalu      )                     (5)
            where ˆyu  represents the predicted value of the entityu, and                        B. Baselines and Experimental Settings
            W out    is  the  weight  parameter  of  the  output  layer  neural                      We use the LSTM model which is suitable for time series
            network. Classiﬁcation cross-entropy is used to calculate the                        data and ﬁve GCN models for comparative experiments, as
            cost.                                                                                shown in Table IV.
                As shown in Figure 2, when predicting whether the hos-                               LSTM is a classic deep learning model for time sequence
            pitalization  event  (for  example,e5 )  will  lead  to  a  medical                  data. The model effectively mitigates the problem of gradient
            treatmentmigration,wecanusethepatient’sattributeinforma-                             disappearance in long-term dependencies through forgetting
            tion (p 2 ) and his historical medical information (e4 ), as well                    gates, input gates, and output gates. GCN [4] mainly considers
            as information of events similar to this hospitalization (e3 ,                       the structural information of the local network where the node
            which involves the same disease ase5 ). That is, the model                           is located but does not consider the relationship type between
            can comprehensively consider the time-series information of                          the two nodes. R-GCN [5] is a GCN-based model proposed
            related events and the information of similar entities to make                       in  2017  for  knowledge  graphs.  GCN-ED  [6]  is  a  GCN-
            predictions.                                                                         based model for knowledge graphs with weight adjustment
                                                                                                 factors added. KGCN-sum [7] is a model for a large-scale
                                        IV.  EXPERIMENTS                                         recommendation system with aggregation operations based on
            A. Data Set                                                                          neighbor-relational  scores.  PinSage  [8]  is  a  model  for  the
                                                                                                 recommendationsystem,whoseconvolutionoperationconsists
                We  conducted  extensive  experiments  on  a  real  medical                      of a 2-layer activation function combined with importance-
            insurance  data  set.  The  medical  insurance  data  set  comes                     based neighbor selection.
            from the Z-city medical insurance institution in China, and                              Each patient is a sample. Note that the history behaviors of
            contains  the  complete  medical  records  of  the  same  patient                    medical treatment migration of a patient are known, and the
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                    Authorized licensed use limited to: Uppsala Universitetsbibliotek. Downloaded on July 20,2020 at 06:48:41 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3008493, IEEE Journal of
                                                                               Biomedical and Health Informatics
                                                                                                                                                                             6
                               TABLE V: Comparison results.                                        TABLE VI: Effect of the number of convolution layer.
                                    Models                 Accuracy                                                     Layer num          Accuracy
                                    LSTM              0.716±0.055                                                            2             0.795±0.041
                                    GCN                0.674±0.061                                                           3             0.820±0.034
                                    R-GCN            0.734±0.057                                                             4             0.781±0.021
                                    GCN-ED          0.780±0.031
                                    KGCN-sum      0.720±0.037
                                    PinSage            0.756±0.073                             D. Inﬂuence of the Number of Convolution Layer
                                    EGCN              0.820±0.034
                                                                                                  In this experiment, we explored the impact of the number of
            migration state of the last medical treatment is set to don’t                      convolution layers on EGCN. Although the more layers, the
            know (the corresponding value in the input vector is set to                        more likely it is to produce abstract features, the experimental
            0.5), which needs to be predicted by the model. In the LSTM                        results show that the predictive ability of the model does not
            model, a series of medical records of each patient constitute                      necessarily increase with the increase of the number of deep
            an input sequence. Each step of the input sequence is a vector                     neural network layers. The EGCN with 3 convolutional layers
            describing the state of the patient including information such                     achieved the best predictions when using EGCN for medical
            as gender, age, current income, disease-related attributes such                    treatment migration predictions, as shown in Table VI.
            as 1-year mortality rate, and hospital-related attributes such as                  E. Convergence
            hospital level. In the KGCN-sum and PinSage models, user-                             Finally, the convergence of the proposed model is studied.
            relationship-based  scoring  or  random  walk-based  weighted                      In Figure 5, the convergence about cost and prediction accu-
            pooling is deployed to aggregate neighbor information. While                       racy of the training set and test set is presented. From the
            in our experiments, we did not calculate the scores of node-                       perspective of cost and prediction accuracy, the results of the
            relationships  for  the  medical  treatment  graph  but  assigned                  training set and test set gradually stabilize with the increase of
            weights to different neighbors based on the attention mech-                        the number of iterations. It can be seen that when the number
            anism. The baselines and the proposed EGCN adopt the same                          of iterations reaches 400, the performance of the proposed
            parameter settings. The number of convolutional layers of the                      model has stabilized.
            graph neural network is set to 3, and the dimension of node
            representations is 20.τ(·) =  tanh(·). The ratio of the training
            set  to  the  test  set  is  4:  1.  For  each  model,  we  conducted
            experiments via 5-fold cross-validation.
            C. Comparison of Experimental Results
               Inthissection,theexperimentalresultsofmedicalmigration
            prediction are discussed. As can be seen from Table V, the
            prediction accuracy of the proposed model is higher than the                                   (a) Example 1                           (b) Example 2
            best model available. In addition, GCN-ED and PinSage show
            a good predictive ability, indicating that the gating mechanism                                          Fig. 5: EGCN convergence.
            and the two-layer activation function play an important role
            in the behavior prediction of medical treatment migration.
                                                                                                                           V.  CONCLUSION
                                                                                                  This paper presents a predictive model for the medical treat-
                                                                                               ment migration task based on medical insurance data. First,
                                                                                               we constructed a medical treatment graph based on medical
                                                                                               insurance data. The medical treatment graph is a heteroge-
                                                                                               neous graph, which establishes a close relationship between
                                                                                               various entities and ﬂexibly expresses the rich information of
                                                                                               the patient’s medical treatment events. Then, to enable the
                                                                                               graph neural network to capture sequence information of event
                                                                                               entities, a new graph neural network model, named EGCN,
                                                                                               was proposed. The EGCN aggregates regular neighbors based
                                                                                               on the attention mechanism, and aggregates event neighbors
                                                                                               based  on  the  gating  mechanism.  In  addition,  a  pre-trained
                           Fig. 4: The ROC of different models.                                automatic encoder is deployed to optimize the model to solve
                                                                                               the problem of the low frequency of some medicines. The
               Further, Figure 4 shows the ROC curves for each model.                          automatic encoder can map the medicine description in text
            For each model, the results of an experiment that was closest                      form to an embedded representation in vector form. Finally,
            to its average prediction accuracy were selected. As can be                        we conducted a comprehensive experiment on real medical
            seen from the ROC curve, EGCN shows the best predictive                            insurance  data.  Experimental  results  show  that  EGCN  has
            power.                                                                             better prediction ability than existing models.
     2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                    Authorized licensed use limited to: Uppsala Universitetsbibliotek. Downloaded on July 20,2020 at 06:48:41 UTC from IEEE Xplore.  Restrictions apply.

This article has been accepted for publication in a future issue of this journal, but has not been fully edited. Content may change prior to final publication. Citation information: DOI 10.1109/JBHI.2020.3008493, IEEE Journal of
                                                                                           Biomedical and Health Informatics
                                                                                                                                                                                                          7
                                            ACKNOWLEDGMENT                                                     [20]  M.  Nickel,  K.  Murphy,  V.  Tresp,  and  E.  Gabrilovich,  “A  review  of
                  This work was supported by the National Key Research                                               relational machine learning for knowledge graphs,” Proceedings of the
                                                                                                                     IEEE, vol. 104, no. 1, pp. 11–33, 2016.
              and Development Plan of China (No.2018YFC0114709), the                                           [21]  H.  Cai,  V.  W.  Zheng,  and  K.  C.  Chang,  “A  comprehensive  survey
              Natural Science Foundation of Shandong Province of China                                               of  graph  embedding:  Problems,  techniques,  and  applications,”  IEEE
              for Major Basic Research Projects (No. ZR2017ZB0419), the                                              Transactions on Knowledge and Data Engineering, vol. 30, no. 9, pp.
                                                                                                                     1616–1637, 2018.
              Taishan Industrial Experts Program of Shandong Province of                                       [22]  T. Dettmers, P. Minervini, P. Stenetorp, and S. Riedel, “Convolutional
              China (No.tscy20150305).                                                                               2d  knowledge  graph  embeddings,”  in  AAAI  Conference  on  Artiﬁcial
                                                                                                                     Intelligence, (AAAI), 2018, pp. 1811–1818.
                                                                                                               [23]  B. Shi and T. Weninger, “Open-world knowledge graph completion,”
                                                  REFERENCES                                                         in AAAI Conference on Artiﬁcial Intelligence, (AAAI), 2018, pp. 1957–
                                                                                                                     1964.
                [1]  W. L. Hamilton, R. Ying, and J. Leskovec, “Representation learning                        [24]  B. Nie and S. Sun, “Knowledge graph embedding via reasoning over
                     on graphs: Methods and applications,” IEEE Data Engineering Bulletin,                           entities,  relations,  and  text,”  Future  Generation  Computer  Systems,
                     vol. 40, no. 3, pp. 52–74, 2017.                                                                vol. 91, no. 2019, pp. 426–433, 2019.
                [2]  H.  Cai,  V.  W.  Zheng,  and  K.  C.  Chang,  “A  comprehensive  survey                  [25]  A.  Neelakantan,  B.  Roth,  and  A.  McCallum,  “Compositional  vector
                     of  graph  embedding:  Problems,  techniques,  and  applications,”  IEEE                        space models for knowledge base completion,” in Annual Meeting of
                     Transactions on Knowledge and Data Engineering, vol. 30, no. 9, pp.                             the Association for Computational Linguistics and International Joint
                     1616–1637, 2018.                                                                                Conference on Natural Language Processing of the Asian Federation of
                [3]  Z. Zhang, P. Cui, and W. Zhu, “Deep learning on graphs: A survey,”                              Natural Language Processing, (ACL), 2015, pp. 156–166.
                     CoRR, vol. abs/1812.04202, 2018.                                                          [26]  A.  McCallum,  A.  Neelakantan,  R.  Das,  and  D.  Belanger,  “Chains
                [4]  T. N. Kipf and M. Welling, “Semi-supervised classiﬁcation with graph                            of  reasoning  over  entities,  relations,  and  text  using  recurrent  neural
                     convolutional networks,” in International Conference on Learning Rep-                           networks,” in Conference of the European Chapter of the Association
                     resentations, (ICLR), 2017.                                                                     for Computational Linguistics, (EACL), 2017, pp. 132–141.
                [5]  M. S. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov,                     [27]  K. Toutanova, V. Lin, W. Yih, H. Poon, and C. Quirk, “Compositional
                     and M. Welling, “Modeling relational data with graph convolutional                              learning of embeddings for relation paths in knowledge base and text,”
                     networks,” in International Conference on the Semantic Web, (ESWC),                             in Annual Meeting of the Association for Computational Linguistics,
                     2018, pp. 593–607.                                                                              (ACL), 2016, pp. 1434–1444.
                [6]  T. H. Nguyen and R. Grishman, “Graph convolutional networks with                          [28]  W. L. Hamilton, R. Ying, and J. Leskovec, “Representation learning
                     argument-aware pooling for event detection,” in AAAI Conference on                              on graphs: Methods and applications,” IEEE Data Engineering Bulletin,
                     Artiﬁcial Intelligence, (AAAI), 2018, pp. 5900–5907.                                            vol. 40, no. 3, pp. 52–74, 2017.
                [7]  H. Wang, M. Zhao, X. Xie, W. Li, and M. Guo, “Knowledge graph                             [29]  Y. Cao, M. Fang, and D. Tao, “BAG: bi-directional attention entity graph
                     convolutional networks for recommender systems,” in The World Wide                              convolutional  network  for  multi-hop  reasoning  question  answering,”
                     Web Conference, (WWW), 2019, pp. 3307–3313.                                                     in Conference of the North American Chapter of the Association for
                [8]  R.  Ying,  R.  He,  K.  Chen,  P.  Eksombatchai,  W.  L.  Hamilton,  and                        Computational Linguistics: Human Language Technologies, (NAACL-
                     J. Leskovec, “Graph convolutional neural networks for web-scale rec-                            HLT), 2019, pp. 357–362.
                     ommender  systems,”  in  ACM  SIGKDD  International  Conference  on                       [30]  C. Zhuang and Q. Ma, “Dual graph convolutional networks for graph-
                     Knowledge Discovery & Data Mining, (KDD), 2018, pp. 974–983.                                    based semi-supervised classiﬁcation,” in International World Wide Web
                [9]  W. L. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation                             Conference, (WWW), 2018, pp. 499–508.
                     learning on large graphs,” in Annual Conference on Neural Information                     [31]  B.  Yu,  H.  Yin,  and  Z.  Zhu,  “Spatio-temporal  graph  convolutional
                     Processing Systems, 2017, pp. 1025–1035.                                                        networks: A deep learning framework for trafﬁc forecasting,” in Inter-
              [10]  M. Zhang, Z. Cui, M. Neumann, and Y. Chen, “An end-to-end deep                                   national Joint Conference on Artiﬁcial Intelligence, (IJCAI), 2018, pp.
                     learning architecture for graph classiﬁcation,” in AAAI Conference on                           3634–3640.
                     Artiﬁcial Intelligence, (AAAI), 2018, pp. 4438–4445.                                      [32]  X. Wang, T. Zhang, X. Xu, L. Chen, X. Xing, and C. L. P. Chen,
              [11]  M. Niepert, M. Ahmed, and K. Kutzkov, “Learning convolutional neural                             “EEG emotion recognition using dynamical graph convolutional neural
                     networks for graphs,” in International Conference on Machine Learning,                          networks and broad learning system,” in IEEE International Conference
                     (ICML), 2016, pp. 2014–2023.                                                                    on Bioinformatics and Biomedicine, (BIBM), 2018, pp. 1240–1244.
              [12]  Y. Shen, K. Yuan, J. Dai, B. Tang, M. Yang, and K. Lei, “KGDDS: A                          [33]  D. Zhang and M. R. Kabuka, “Protein family classiﬁcation with multi-
                     systemfordrug-drugsimilaritymeasureintherapeuticsubstitutionbased                               layer graph convolutional networks,” in IEEE International Conference
                     on knowledge graph curation,” Journal of Medical Systems, vol. 43,                              on Bioinformatics and Biomedicine, (BIBM), 2018, pp. 2390–2393.
                     no. 4, pp. 92:1–92:9, 2019.                                                               [34]  J. Wang, R. Wen, C. Wu, Y. Huang, and J. Xion, “Fdgars: Fraudster
              [13]  K. Lei, K. Yuan, Q. Zhang, and Y. Shen, “Medsim: A novel semantic                                detectionviagraphconvolutionalnetworksinonlineappreviewsystem,”
                     similarity measure in bio-medical knowledge graphs,” in Internation-                            in International World Wide Web Conference, (WWW), 2019, pp. 310–
                     al Conference on Knowledge Science, Engineering and Management,                                 316.
                     (KSEM), 2018, pp. 479–490.                                                                [35]  Y. Li, R. Jin, and Y. Luo, “Classifying relations in clinical narratives
              [14]  X. Sun, Y. Man, Y. Zhao, J. He, and N. Liu, “Incorporating description                           using segment graph convolutional and recurrent neural networks (Seg-
                     embeddings into medical knowledge graphs representation learning,” in                           GCRNs),” JAMIA, vol. 26, no. 3, pp. 262–268, 2019.
                     International Conference on Human Centered Computing, (HCC), 2018,                        [36]  A.  Kazi,  S.  Shekarforoush,  S.  A.  Krishna,  H.  Burwinkel,  G.  Vivar,
                     pp. 188–194.                                                                                    K. Kort¨um, S. Ahmadi, S. Albarqouni, and N. Navab, “InceptionGCN:
              [15]  T. Ruan, Y. Huang, X. Liu, Y. Xia, and J. Gao, “Qanalysis: a question-                           Receptive ﬁeld aware graph convolutional network for disease predic-
                     answer driven analytic tool on knowledge graphs for leveraging elec-                            tion,” in International Conference on Information Processing in Medical
                     tronic medical records for clinical research,” BMC Medical Informatics                          Imaging, (IPMI), 2019, pp. 73–85.
                     and Decision Making, vol. 19, no. 1, pp. 82:1–82:13, 2019.                                [37]  M. Zitnik, M. Agrawal, and J. Leskovec, “Modeling polypharmacy side
              [16]  K. Xu, C. Li, Y. Tian, T. Sonobe, K. Kawarabayashi, and S. Jegelka,                              effects  with  graph  convolutional  networks,”  Bioinformatics,  vol.  34,
                     “Representation learning on graphs with jumping knowledge networks,”                            no. 13, pp. i457–i466, 2018.
                     in International Conference on Machine Learning, (ICML), 2018, pp.                        [38]  S. Parisot, S. I. Ktena, E. Ferrante, M. C. H. Lee, R. Guerrero, B. Glock-
                     5449–5458.                                                                                      er,  and  D.  Rueckert,  “Disease  prediction  using  graph  convolutional
              [17]  K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph                               networks:  Application  to  autism  spectrum  disorder  and  alzheimer’s
                     neural networks?” in International Conference on Learning Represen-                             disease,” Medical Image Analysis, vol. 48, pp. 117–130, 2018.
                     tations, (ICLR), 2019.                                                                    [39]  S. Rhee, S. Seo, and S. Kim, “Hybrid approach of relation network and
              [18]  Y. Chen and M. J. Zaki, “KATE: k-competitive autoencoder for text,” in                           localized graph convolutional ﬁltering for breast cancer subtype clas-
                     ACM SIGKDD International Conference on Knowledge Discovery and                                  siﬁcation,” in International Joint Conference on Artiﬁcial Intelligence,
                     Data Mining, (KDD), 2017, pp. 85–94.                                                            (IJCAI), 2018, pp. 3527–3534.
              [19]  Q. Wang, Z. Mao, B. Wang, and L. Guo, “Knowledge graph embed-
                     ding: A survey of approaches and applications,” IEEE Transactions on
                     Knowledge and Data Engineering, vol. 29, no. 12, pp. 2724–2743, 2017.
      2168-2194 (c) 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.
                       Authorized licensed use limited to: Uppsala Universitetsbibliotek. Downloaded on July 20,2020 at 06:48:41 UTC from IEEE Xplore.  Restrictions apply.

