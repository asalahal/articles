Li and Nabavi
 RESEARCH
A Multimodal Graph Neural Network Framework
for Cancer Molecular Subtype Classification
Bingjun Li and Sheida Nabavi                       *
*Correspondence:
sheida.nabavi@uconn.edu                                   Abstract
Department of Computer Science
and Engineering, University of                            Background:          The recent development of high-throughput sequencing creates a
Connecticut, Storrs, US                                   large collection of multi-omics data, which enables researchers to better
Full list of author information is                        investigate cancer molecular profiles and cancer taxonomy based on molecular
available at the end of the article
                                                          subtypes. Integrating multi-omics data has been proven to be effective for
                                                          building more precise classification models. Most current multi-omics integrative
                                                          models use either an early fusion in the form of concatenation or late fusion with
                                                          a separate feature extractor for each omic, which are mainly based on deep neural
                                                          networks. Due to the nature of biological systems, graphs are a better structural
                                                          representation of bio-medical data. Although few graph neural network (GNN)
                                                          based multi-omics integrative methods have been proposed, they suffer from
                                                          three common disadvantages. One is most of them use only one type of
                                                          connection, either inter-omics or intra-omic connection; second, they only
                                                          consider one kind of GNN layer, either graph convolution network (GCN) or
                                                          graph attention network (GAT); and third, most of these methods have not been
                                                          tested on a more complex classification task, such as cancer molecular subtypes.
                                                          Results:      In this study, we propose a novel end-to-end multi-omics GNN
                                                          framework for accurate and robust cancer subtype classification. The proposed
                                                          model utilizes multi-omics data in the form of heterogeneous multi-layer graphs,
                                                          which combine both inter-omics and intra-omic connections from established
                                                          biological knowledge. The proposed model incorporates learned graph features
                                                          and global genome features for accurate classification. We test the proposed
                                                          model on the Cancer Genome Atlas (TCGA) Pan-cancer dataset and TCGA
                                                          breast invasive carcinoma (BRCA) dataset for molecular subtype and cancer
                                                          subtype classification, respectively. The proposed model shows superior
                                                          performance compared to four current state-of-the-art baseline models in terms
                                                          of accuracy, F1 score, precision, and recall. The comparative analysis of
                                                          GAT-based models and GCN-based models reveals that GAT-based models are
                                                          preferred for smaller graphs with less information and GCN-based models are
                                                          preferred for larger graphs with extra information.
                                                          Keywords:         graph attention network; multi-omics integration; cancer subtype;
                                                          molecular subtype
                                                      Background
                                                      The fast-growing high-throughput sequencing technology has made DNA and RNA
                                                      sequencingmoreefficientandaccessible,resultinginalargecollectionofmulti-omics
                                                      data which makes molecular profiling possible. Due to the heterogeneity in cancer
                                                      and the complexity of the biological processes, employing multi-omics sequencing
                                                      data are crucial to more accurate cancer classification and tumor profiling. Many
                                                      researchers have proposed methods that incorporate multi-omics data for either

Li and Nabavi                                                                                                                                                                                    Page 2 of 18
                         cancer type classification or cell type clustering [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].
                         These methods show that utilizing multi-omics data improves performance, and
                         provides a better understanding of the key pathophysiological pathways across dif-
                         ferent molecular layers [12]. A typical multi-omics data generated from DNA and
                         RNA sequencing usually consists of mRNA expression, microRNA (miRNA) ex-
                         pression, copy number variation (CNV), and DNA methylation [13]. The difference
                         in data distributions across each omic, and the complex inter-omics and intra-omic
                         connections (certain omic can act as a promotor or suppressor to genes) add more
                         challenges to developing an integrative multi-omics classification method for cancer
                         molecular subtypes.
                             Recent studies have shown that cancer taxonomy based on molecular subtypes
                         canbecrucialforprecisiononcology[13,14].Anaccuratecancermolecularsubtype
                         classifier is crucial for early-stage diagnosis, prognosis, and drug development. Tra-
                         ditionalcancertaxonomyisbasedonitstissueorigin.In2014,TheCancerGenome
                         Atlas (TCGA) Research Network proposed a new clustering method for cancers
                         based on their integrated molecular subtypes that share mutations, copy-number
                         alterations, pathway commonalities, and micro-environment characteristics instead
                         of their tissue of origin [13]. They found 11 subtypes from 12 cancer types. In 2018,
                         they applied the new taxonomy method to 33 cancer types and found 28 molec-
                         ular subtypes [15]. The new cancer taxonomy provides a better insight into the
                         heterogeneous nature of cancer.
                             With the recent development in deep learning models, data-driven models benefit
                         from the powerful feature extraction capability of deep learning networks in many
                         fields [16, 17, 18, 19, 20, 21, 22]. Most multi-omics integrative models employ an
                         early fusion approach that aggregates multi-omics data (mainly by concatenation)
                         and then applies a deep neural network as a feature extractor; or a late fusion ap-
                         proachthatfirstextractsfeaturesfromeachomicbydeepneuralnetworksandthen
                         aggregates extracted features as inputs to the classification network. For efficient
                         implementation of multi-omics integrative models, convolutional neural networks
                         (CNNs) are widely used [23].
                             Traditional deep neural networks are based on the assumption that the inner
                         structureofthedataisinEuclideanspace[24].Becauseofthecomplexinteractions
                         across many biological processes, such data structure is not a proper representa-
                         tion of bio-medical data, and researchers proposed graph-based data structures to
                         tackle this limitation. In 2016, a graph convolution network (GCN), ChebNet, was
                         proposed [16]. It uses the Chebyshev polynomial as the localized learning filter to
                         extract thegraph feature representation. In2017, Petar Velickovic et al.proposed a
                         graphattentionnetwork(GAT)that overcomes GCN’s disadvantage ofdependence
                         on the Laplacian eigenbasis [25]. GAT uses masked self-attention layers to enable
                         nodes to attend over their neighborhoods’ features [25]. With the recent growing
                         interestinthegraphneuralnetwork,manygraph-basedclassificationmethodshave
                         been proposed in the bio-medical field.
                             To utilize the power of graph-structured data, Ramirez et al. proposed a GCN
                         method to use intra-omic connections, protein-protein interaction networks, and
                         gene co-expression networks. The model achieves a 94.71% classification accuracy
                         for 33 cancer types and normal tissue on TCGA data [26]. To use the intra-omic

Li and Nabavi                                                                                                                                                                                    Page 3 of 18
                         connection across multiple omics, Wang et al. proposed MOGONET, a late-fusion
                         GCN-basedmethodthatintegratesmulti-omicsdataforbio-medicaldataclassifica-
                         tion.Andtheyachieve80.61%accuracyonbreastcancersubtypeclassificationwith
                         BRCAdataset[5].TocompensateforthelimitationofGCN,thatitonlyextractslo-
                         calrepresentationonthegraph,Lietal.proposedaparallel-structuredGCN-based
                         method that utilizes a gene-based prior knowledge graph for cancer molecular sub-
                         type classification [1]. There are also other ways to structure the graph. Wang et
                         al. proposed a GCN-based method that uses a KNN-generated cell-cell similarity
                         graph for single-cell sequencing data classification [27].
                            Since the introduction of GAT in 2017, it has gained more and more interest.
                         Shanthamalluetal.proposedaGAT-basedmethod,GrAMME,withtwovariations
                         thatuseasupra-graphapproachandlate-fusionapproachtoextractfeaturesfroma
                         multi-layergraphwithintra-omicconnectionsonlyforclassificationinsocialscience
                         andpoliticalsciencedatasets [28].Ontheotherhand,Kaczmareketal.proposeda
                         multi-omics graph transformer to utilize an inter-omics connection only graph, the
                         miRNA-gene target network, for cancer classification on 12 cancer types from the
                         TCGA data [7].
                            There are three common disadvantages of these approaches. First, most of them
                         consideronlyonekindofconnectionsintheirmodel,eitherinter-omicsorintra-omic
                         connections.Theydonotaimtoutilizebothinter-omicsandintra-omicconnections
                         for more effective feature extraction. Second, they only consider one kind of GNN
                         models, either GCN or GAT. We find that GAT and GCN have their strength in
                         differentscenariosasshowninourexperiments.Differentgraphlayersarepreferred
                         for different tasks even with datasets in a similar domain. Third, most of these
                         methods have not been tested on a more complex classification task. They are used
                         for classification based on the cell-of-origin taxonomy such as cancer type classifica-
                         tion and have not been applied to a more complex classification task such as cancer
                         molecular subtype classification, which is more useful for diagnosis, prognosis, and
                         treatment. Inspired by our previous work on the cancer molecular subtype clas-
                         sification based solely on intra-omic connections, we aim to develop a multi-omics
                         integrativeframeworkthatexploitsthepowerfuldataaggregationpropertyofGCN
                         or GAT models (depending on the situation) and utilizes both the intra-omic net-
                         work and the inter-omics network for more precise classification.
                            Our goal is to build an accurate, robust, and efficient multi-omics integrative pre-
                         dictive model to classify these cancer molecular subtypes. In this work, we propose
                         ageneralframeworkthatcanbeusedwithanygraphneuralnetworksasthefeature
                         extractor, incorporate both gene-based and non-gene-based prior biological knowl-
                         edge(primarilymiRNA),andlearnaknowledgegraphconsistingofbothintra-omic
                         andinter-omicsconnections.Weapplytheproposedmodeltoclassifycancermolec-
                         ular subtypes and breast cancer molecular subtypes. We choose breast cancer as it
                         is one of the most common and lethal cancers with a large number of samples in
                         TCGA. It can be categorized into four major molecular subtypes based on the gene
                         expression of the cancer cells, and breast cancer subtypes have significant impacts
                         on the patient’s survival rates  [29]. Our experimental results show the proposed
                         method outperforms both the graph-based and CNN-based state-of-the-art meth-
                         ods.

Li and Nabavi                                                                                                                                                                                    Page 4 of 18
                            Our contributions in this study are i) a novel generalized GNN-based multi-omics
                         integrative framework for cancer molecular subtype classification, ii) a supra-graph
                         approach that can incorporate both intra-omic and inter-omics prior biological
                         knowledge in the form of graphs, iii) a representation of multi-omics data in the
                         form of heterogeneous multi-layer graph, and iv) a comparative analysis of GCN
                         andGATbasedmodelsatdifferentcombinationsofomicsanddifferentgraphstruc-
                         tures.
                         Method and Materials
                         The overview of the proposed framework structure is shown in Figure 1. The in-
                         put data for the proposed framework is shown as a graph structure on the leftmost
                         side.Thedataconsistsofthreeomics,mRNAexpression(orangeboxes),copynum-
                         ber variation (CNV) (yellow boxes), and miRNA (green boxes). The details of the
                         network structure are discussed in the following Network Section. The proposed
                         framework consists of 4 major modules: Module 1) a linear dimension-increase neu-
                         ral network, Module 2) a graph neural network (GNN), Module 3) a decoder, and
                         Module 4) a shallow parallel network. Any kind of graph neural network can be
                         used in Module 2. In this study, we focus on graph convolutional network (GCN)
                         and graph attention network (GAT), which are two major kinds of GNN. Experi-
                         ments about the effect of the decoder and the shallow parallel network modules are
                         discussed in our ablation study.
                         Network
                         Webuildaheterogeneousmulti-layergraphbasedonthepriorbiologicalknowledge,
                         i.e. gene-gene interaction (GGI) network from BioGrid and miRNA-gene target
                         networkfrommiRDB[30,31].Inspiredbythemeta-pathandsupra-graphapproach
                         for the multi-layered network models [32, 28], we build a supra-graph with miRNA-
                         miRNAmeta-paths.AmiRNA-miRNAmeta-pathisdefinedasiftwomiRNAnodes
                         are connected to the same gene node from the GGI network and miRNA-gene
                         network. An example of how we construct the supra-graph is shown in Figure 2.
                         Meta-paths are shown as dotted lines in the figure.
                            The adjacencymatrix of the supra-graph is an (                         N  +  M  )× (N  +  M  ) matrix,where
                         N   isthenumber ofgenesand               M   isthenumberofmiRNA.Everynodeinthegraph
                         isassumedtobeself-connected,thusthediagonalelementsoftheadjacencymatrix
                         in the study are 1. The adjacency matrix of the supra-graph is shown in Equation
                         (1).
                                                      "                                           #
                                   A  Supra    =        A  gene  − gene     A  gene  − mi            ,                                                        (1)
                                                          A  Tgene  − mi      A  mi − mi ,
                         where    A  gene  − gene  ∈ R N × N ,A  gene  − mi  ∈ R N × M  , and   A  mi − mi  ∈ R M × M  .
                            We also construct four different kinds of graphs other than supra-graph in our
                         ablationstudyandapplythemtofiveinputcombinationsofomics:mRNA,miRNA,
                         mRNA + miRNA, mRNA+CNV, mRNA + MiRNA + CNV, to test the effect of
                         thedifferentgraphsonthemodelperformance.Thefourdifferentgraphsaredefined
                         as follows.

Li and Nabavi                                                                                                                                                                                    Page 5 of 18
                             Figure 1: The overall structure of the proposed model has four major mod-
                             ules shown as dotted grey rectangles. The input graph consists of inter-omics
                             (rededges),intra-omic(blueedges)edgesandmiRNA-miRNAmeta-path(black
                             dashededges),andthreeomicsdata,mRNA(orangeboxes),CNV(yellowboxes),
                             andmiRNA(greenboxes)isshownastheleftmostside.Module1consistsoftwo
                             parallel linear dimension-increase layers for gene-based nodes and miRNA-based
                             nodes. The upgraded graph shown in the middle is obtained by feeding the node
                             attributes from the input graph through module 1, where the dark orange boxes
                             are the updated gene-based node attributes and the dark green boxes are the
                             updated miRNA-based node attributes. Module 2 consists of two graph neural
                             network layers, which can be any graph neural networks. The output of module
                             2 is then fed into a max pooling layer and then a transformation layer to obtain
                             the learned graph representation (blue boxes). Module 3 consists of a decoder to
                             reconstruct the graph representation back to the input graph node attributes.
                             Module 4 consists of a shallow fully connected network that takes the updated
                             node attributes as the input. The output of the parallel network (grey cubes) is
                             then concatenated with the learned graph representation, and passes through a
                             classification layer for the classification task.

Li and Nabavi                                                                                                                                                                                    Page 6 of 18
                            Figure2:Theoverallgraph,supra-graph,isconstructedfromthreedifferentomic
                            dataontheleft-handsideandtwopriorknowledgegraphsontheright-handside.
                            mRNA (orange table) and CNV (yellow table) data are considered gene-based,
                            whichhavethesamedimension.miRNA(greentable)datahasthesamenumber
                            of rows but different feature lengths for each sample.
                            Only Gene-based Nodes                 : When the input combination of omics is mRNA or
                         mRNA+mRNA+CNV (                     M    = 0), the graph is built with the GGI network,                      A   =
                         A  gene  − gene  ∈ R N × N .
                            Only miRNA-based Nodes                     : When the input combination of omics is miRNA
                         (N   = 0), the graph is built with only miRNA meta-path network,                               A  =  A  mi − mi  ∈
                         R M × M  .
                            Only Intra-class Edges              : The graph only contains GGI network and miRNA
                         meta-path network.
                                                      "                                         #
                                   A  Supra    =        A  gene  − gene      0 N,M                 ∈ R (N  + M   )× (N  + M   ).                              (2)
                                                              0 M,N         A  mi − mi
                            Only Inter-class Edges              : The graph only contains miRNA-gene target network.
                                                      "                                         #
                                   A  Supra    =             IN,N        A  gene  − mi             ∈ R (N  + M   )× (N  + M   ).                              (3)
                                                        A  Tgene  − mi       IM,M
                            The input graph is denoted as a tuple                 G  = (  V,E,  X  V ), where     V   is the set of
                         nodes,    E  is the set of edges, and         x V  is the node attributes. The prior knowledge is
                         incorporated into the model through the supra-graph defined above. In the supra-
                         graph, nodes consist of both gene-based nodes and miRNA-based nodes, and edges
                         are assigned by the adjacency matrix. Each gene-based node has a node attribute
                         of a vector consisting of both gene expression and CNV data,                            x v∈V gene   ∈ R 2. Each
                         miRNA-basednodehasanodeattributeasascalar,                             x v∈V miRNA    ∈ R .Thegene-based
                         nodes and miRNA-based nodes are fed through a linear dimension-increase layer,

Li and Nabavi                                                                                                                                                                                    Page 7 of 18
                         denoted as Module 1 in Figure 1 to achieve the same node attribute dimension,
                         X ′V  ∈ R (N  + M   )× F , where     F  is the increased node attribute dimension.
                         Graph Neural Network: Convolution-based
                         Asmentionedbefore,anygraphneuralnetworkcanbeusedintheGNNmodule.We
                         use ChebNet [16] to implement the GCN in this study. The supra-graph adjacency
                         matrix introduced in the previous network section is first Laplacian normalized to
                         L  as expressed in Equation (4).
                                   L  =  I +  D − 1/2AD     1/2,                                                                               (4)
                         where     I ∈  R (N  + M   )× (N  + M   )  is an identity matrix, and the degree matrix                     D   ∈
                         R (N  + M   )× (N  + M   )  is a diagonal matrix. The eigen decomposition form of                          L  can be
                         obtained as
                                   L  =  U Λ U  T ,                                                                                              (5)
                         where    U  =( u 1,u 2,...,u n )isamatrixof       n orthonormaleigenvectorsof              L ,therefore
                         UU    T  =  I. And    Λ  =  diag  (λ 1,λ 2,...,λn ) is the eigenvalue matrix [16].
                            After transforming the graph on the Fourier domain, the learning filter can be
                         approximated by a            K  th -order Chebshev polynomial. The convolution on the graph
                         by such localized learning filter,               h (Λ ) can be expressed in Equation (6).
                                                                              K − 1X                                  K − 1X
                                   y =  U h (Λ )U  T X  j =  U                        β kT k(˜Λ )U  T X  j =                  β kT k(˜LX   j),                  (6)
                                                                              k =1                                     k =0
                         where    X  j ∈  R (N  + M   )× F   is the features of        j-th sample,        ˜L   = 2  L /λ  max   −  I, and
                         T k(˜L ) = 2   ˜L T k− 1(˜L ) −  T k− 2(˜L ) with    T 0(˜L ) =   I and    T 1(˜L ) =    ˜L . K   is a hyper-
                         parameter, where           K   = 5 in our study. A max-pooling layer with                        p = 8 is used to
                         reduce the number of nodes and one layer of fully connected network is used to
                         transform the learned local feature representation to a vector of length 64 for each
                         sample,    θ 1 ∈ R 64 .
                         Graph-Neural Network: Attention-based
                         GATaimstosolvetheproblemofGCN’sdependenceonLaplacianeigenbasisofthe
                         graph adjacency matrix [25]. The updated node attributes are first passed through
                         a linear transformation by a learnable weight, denoted as                          W   ∈ R F   ′× F , where     F  is
                         the updated node attribute dimension and                         F ′is the intended output dimension for
                         this GAT layer. Then, the self-attention coefficients for each node can be calculated
                         as Equation (7).
                                   eij =  a(W  x i,W  x j),                                                                                 (7)
                         where     eij  represents the importance of node                   j  to node     i and   x i,x j  are the node
                         attributesfornode          i,j.Suchattentionscoreisonlycalculatedfor                     j ∈  NB   (i),where
                         NB   (i) is all the first-order neighbor nodes around node                         i. The method normalizes

Li and Nabavi                                                                                                                                                                                    Page 8 of 18
                         the attention score by a softmax layer of                     eij and uses LeakyReLU as the activation
                         function as express in Equation (8).
                                    α ij =           exp(LeakyReLU(            ⃗a T [W  x i||W  x j]))P
                                                    k∈NB    (i) exp(LeakyReLU(            ⃗a T [W  x i||W  x k]))                                  (8)
                         The output for each node can be expressed as Equation (9).
                                    x ′i =  σ (  X              α ijW  x j).                                                                           (9)
                                                  j∈NB    (i)
                         A multi-head attention mechanism is used to stabilize the attention score. In our
                         study,thenumberofheadsis8.SimilartotheGCN-basedGNNmodule,theoutput
                         is then passed through a max-pooling layer and a transformation layer to obtain
                         the local graph representation,             θ 1 ∈ R 64 .
                         Decoder & Shallow Parallel Network
                         As shown in Figure 1, the decoder is a two-layer fully connected network that is
                         used to reconstruct the node attributes on the input graph. To compensate the
                         localization property of either GCN or GAT layer in the GNN module, we use a
                         parallel shallow fully connected network. Since the prior knowledge graphs have
                         many limitation [1], we may neglect some global patterns in the data when extract-
                         ing features based on the graph structure only. A shallow two-layer fully connected
                         network is able to learn the global features of the data while ignoring the actual
                         innerstructureofthedata.Thesetwomoduleshelptheframeworktobetterextract
                         theoverallsamplefeaturerepresentation.Theeffectofincludingvs.excludingthese
                         two modules is discussed in detail in the Ablation Study Section.
                            Theinputoftheparallelnetworkistheupdatednodeattributes,                             X ′V  ∈ R (N  + M   )× F
                         and the output global representation of the sample,                       θ 1  is in the same dimension as
                         the local feature representation from the GNN module,                          θ 2 ∈ R 64 . θ 1  and   θ 2  are
                         then concatenated and passed through a classification layer for prediction.
                         Loss Function
                         In the proposed framework, we define the loss function                             L  as a linear combination
                         of three loss functions in Equation (10).
                                    L  =  λ 1L ent  +  λ 2L recon   +  λ 3L reg ,                                                             (10)
                         where     λ 1, λ 2  and   λ 3  are linear weights,         L ent  is the standard cross-entropy loss for
                         the classification results,           L recon    is the mean squared error for the reconstruction
                         loss when the decoder is included, and                     L reg   is the squared        l2  norm of the model
                         parameters to penalize the number of parameters to avoid overfitting.                                      L recon    is
                         defined as
                                    L recon   = X          (x j −  ˆx j)2,                                                                           (11)
                                                       j

                    Li and Nabavi                                                                                                                                                                                    Page 9 of 18
                                              where    x j is the flattened feature vector of               j-th sample and          ˆx j is the corresponding
                                              reconstructed vector. We denote                 W   all  as the vector consists of all parameters in
                                              the model and the           L reg  is defined as
                                                         L reg  =   X               w  2.                                                                                    (12)
                                                                       w ∈W  all
                                              Results and Discussion
                                              We apply the proposed model to two different classification problems. The first
                                              is cancer molecular subtype classification on the TCGA Pan-cancer dataset and
                                              the second is breast cancer subtype classification on the TCGA breast invasive
                                              carcinoma (BRCA) dataset [15, 33].
                                              Data and Experiment Settings
                                              The TCGA Pan-cancer RNA-seq data, CNV data, miRNA data, and molecular
                                              subtype labels are obtained from the University of California Santa Cruz’s Xena
                                              website [34]. We only keep samples that have all three omics data and molecular
                                              subtype labels, and collect 9,027 samples in total. We use 17,946 genes that are
                                              common in both the gene expression data and the CNV data, and 743 miRNAs.
                                              The total number of molecular subtypes is 27 and there is a clear imbalance among
                                              these 27 classes as shown in Figure 3. All samples from class 24 are excluded from
                                              the study due to the lack of miRNA data. For BRCA subtype classification, there
                                              are 981 samples in total with 4 subtypes as shown in Table 1. For the experiments
                                              on both datasets, 80% of the data is used for training, 10% is used for validation,
                                              and 10% is used for testing. All classes are present in the test set.
                                                                            Table 1: Number of Cases in Each BRCA Subtype
                                                  All expression values are normalized within their own omics. We select the top
                                              700 genes ranked by gene expression variances across the samples, and the top 100
                                              miRNAs by miRNA expression variance. Results are averaged from five individual
                                              trials. The details of the model structure and hyperparameters are disclosed in the
                                              appendix. The model is implemented using Pytorch Geometric Library.
                                              Baseline Models
                                              We selected four state-of-the-art models [1, 7, 26, 28] as baseline models to eval-
                                              uate the performance of the proposed approach. These four baseline models are
                                              implemented within the proposed framework in two forms, one is with the original
                                              structure, and the other is with some modifications to accommodate the multi-
                                              omics data. The details of all graph-based baseline implementation configurations
                                              are shown in Table 2. We also included a fully-connected neural network (FC-NN)
                                              asaEuclidean-basedbaselinemodel.Conventionalmachinelearningmethods,such
                                              asRandomForestandSVMarenotincludedinthescopeofthisstudybecausethey
                                              do not scale well to the multi-omics data as mentioned in our previous work [1].
BRCA Subtypes     Counts
LumA                    529
LumB                    197
Basal                     175
Her2                      80

                       Li and Nabavi                                                                                                                                                                                  Page 10 of 18
                                                      Figure 3: The number of cases in each molecular subtypes is shown. All samples
                                                      from class 24 are excluded due to lack of miRNA data.
                                                   Table 2: Configurations of Baseline Models on Omics, Graph Structure, GNN Lay-
                                                   ers, and Regularizaiton Modules
                                                   Fully-connected Neural Network (FC-NN)
                                                   The FC-NN is one of the widely used deep learning model for data in Euclidean
                                                   space. The implemented structure is the same as the parallel structure. The input
                                                   data is passed through a dimension-increase layer and then flattened. The flattened
                                                   data is passed through three hidden layers and a softmax layer for classification.
                                                   GCN Models by Ramirez et. al.
                                                   The GCN model on cancer type classification is designed for gene expression data
                                                   with intra-omic connections only [26]. The implementation of the original structure
                                                   and the modified structure is a GCN model with no regularization modules.
                                                   Multi-omics GCN Models by Li et al.
                                                   The multi-omics GCN model on cancer molecular subtype classification is designed
                                                   for gene expression and CNV data with intra-omic connections only [1]. The im-
                                                   plementation of both structures is a GCN model with a decoder and a parallel
                                                   structure as shown in Table 2.
Model                                                    Omics                               Graph                   GNNLayer                   Module
                                            mRNA    CNV    miRNA                 Intra-omic    Inter-omic           GCN    GAT           Decoder    Parallel
GCN(Original)[26]                              ✓          –          –                 ✓               –              ✓         –             –            –
GCN(Modified)                                  ✓      ✓      ✓                         ✓               –              ✓         –             –            –
Multi-omicsGCN(Original)[1]                    ✓      ✓           –                    ✓               –              ✓         –            ✓       ✓
Multi-omicsGCN(Modified)                       ✓      ✓      ✓                         ✓               –              ✓         –            ✓       ✓
GrAMME(Modified)[28]                           ✓      ✓      ✓                         –          ✓                    –     ✓                –            –
Multi-omicsGAT(Original)[7]                    ✓          –      ✓                     ✓               –               –     ✓                –            –
Multi-omicsGAT(Modified)                       ✓      ✓      ✓                         ✓               –               –     ✓                –            –

                     Li and Nabavi                                                                                                                                                                                  Page 11 of 18
                                                GrAMME
                                                Since GrAMME is not designed for cancer type classification [28], we modified the
                                                original structure for multi-omics data. GrAMME is designed for a GAT model
                                                with intra-omic connections only. The implementation is a GAT model with no
                                                regularization modules.
                                                Multi-omics GAT by Kaczmarek et al.
                                                The multi-omics graph transformer on 12 cancer type classification is designed for
                                                gene expression and miRNA data with inter-omics connections only [7]. As shown
                                                in Table 2, the main difference between multi-omics GAT and GrAMME is the
                                                construction of the graph.
                                                Performance on Classification
                                                Table 3: Results of the Proposed and Baseline Models with 700 Genes for Molecu-
                                                lar Subtype Classification on the TCGA Pan-cancer Dataset And Cancer Subtype
                                                Classificaiton on the TCGA BRCA Dataset
                                                    For both classification tasks, the results of the proposed model and the baseline
                                                models are shown in Table 3. The proposed model with GAT layers outperforms
                                                all the baseline models for both tasks in all four metrics and the proposed model
                                                with GCN layers achieves third for the pan-cancer classification, and second for the
                                                breast cancer subtype classification. For the task of pan-cancer molecular subtype
                                                classification,theadditionalomicdatainthemodifiedstructureimprovethemodel
                                                performance in all three cases of the baseline model with the original structure vs.
                                                the baseline model with the modified structure. For the same task, the multi-omics
                                                GCN model with the decoder and parallel structure shows superior performance
                                                among all the baseline models that utilize GCN layers. And GrAMME, which uti-
                                                lizes intra-omic connections, performs better than GAT models that utilize inter-
                                                omicsconnections.GrAMMEisthebest-performingoneamongthebaselinemodels
                                                for the pan-cancer task. Overall, we see the proposed model achieves the best per-
                                                formance for the classification task on the complex pan-cancer molecular subtype
                                                classification in all four metrics and we can conclude that more omics improve the
                                                performanceofmodels,andthemodelswithmorerestrictionmodulesorGATlayers
                                                have better performance.
                                                    Forbreastcancersubtypeclassification,theoveralltrendisslightlydifferentfrom
                                                that in the previous task. In most cases of including more omics, the performancePan-cancerBRCA
Model                                               Accu.    1                   F1                      Accu.    1                   F1
Proposed w/ GAT                                     83.9%   ±  1.4%    0.84    ±  0.01                   86.4%   ±  1.9%    0.87    ±  0.02
Proposed w/ GCN                                     81.2%   ±  0.6%      0.81     ±  0.01                83.8%   ±  0.9%      0.84     ±  0.01
FC-NN                                               78.4%   ±  0.8%      0.75     ±  0.02                80.8%   ±  1.1%      0.80     ±  0.02
GCN (Original) [26]                                 77.6%   ±  0.9%      0.76     ±  0.02                82.8%   ±  1.2%      0.84     ±  0.01
GCN (Modified)                                      78.5%   ±  1.2%      0.77     ±  0.02                81.8%   ±  1.4%      0.82     ±  0.01
Multi-omics GCN (Original) [1]                      78.6%   ±  0.9%      0.78     ±  0.01                81.8%   ±  1.1%      0.82     ±  0.01
Multi-omics GCN (Modified)                          80.2%   ±  0.8%      0.79     ±  0.01                82.8%   ±  0.9%      0.83     ±  0.01
GrAMME (Modified) [28]                              81.4%   ±  1.3%      0.81     ±  0.03                82.8%   ±  1.6%      0.84     ±  0.03
Multi-omics GAT (Original) [7]                      76.3%   ±  1.2%      0.76     ±  0.02                81.8%   ±  1.3%      0.82     ±  0.02
Multi-omics GAT (Modified)                          79.7%   ±  1.3%      0.79     ±  0.02                82.8%   ±  1.4%      0.84     ±  0.02
The bold font indicates the highest values and the values after                      ±  sign are the standard deviations.
1 Accu. stands for Accuracy.

                     Li and Nabavi                                                                                                                                                                                  Page 12 of 18
                                                of the models shows little or no improvement. We believe it is due to the nature of
                                                breast cancer taxonomy. The subtype is based on the expression level of multiple
                                                proteins. Thus, it makes the breast cancer subtype to be more closely related to the
                                                gene expression omic than the pan-cancer molecular subtype does. Such character-
                                                istic of the breast cancer subtype makes the model only using gene expression data
                                                perform very well such as the original GCN model. However, the proposed model
                                                still outperforms any baseline models by a large margin in all four metrics.
                                                Ablation Study
                                                We conduct an ablation study to evaluate the effects of different numbers of genes,
                                                different training set splits, different combinations of modules within the model,
                                                and different combination of omics and graphs on the performance of the proposed
                                                model.
                                                Different Numbers of Genes
                                                Table 4: Results of the Proposed Model and Baseline Models with 300 and 500
                                                Genes for Molecular Subtype Classification Using the TCGA Pan-cancer Dataset
                                                    We trained the proposed model and all baseline models at the 300 and 500 genes
                                                for pan-cancer molecular subtype classification and 300, 500, 1000, 2000, and 5000
                                                genes for breast cancer subtype classification. The limitation of the test scope on
                                                pan-cancer classification is due to the computation constraints caused by its large
                                                number of samples. As shown in Table 4, increasing the number of gene nodes im-
                                                proves the performance of all models. FC-NN model demonstrates great improve-
                                                ment in performance as the number of genes increases. And the proposed model
                                                with the GAT layer outperforms the baseline models at both numbers of genes.
                                                    The accuracy and F1 scores of the proposed model and the baseline models for
                                                BRCAsubtypeclassificationareshowninFigure4.TheproposedmodelwithGAT
                                                performs best when the number of genes is smaller than 1000 and the proposed
                                                model with GCN performs best when the number of genes is larger than 1000. The
                                                proposed GAT-based model yields the best result with an accuracy of 88.9% and
                                                an F1 score of 0.89 when using 700 genes; and the proposed GCN-based model
                                                yields the best result with an accuracy of 90.1% and an F1 score of 0.90 when
                                                using 5000 genes. The detailed results are shown in the supplementary file. The
                                                performance of the proposed model with GAT deteriorates beyond 1,000 genes, but
                                                the performance of the proposed model with GCN continues to rise as the number300500
Model                                            Accu.    1                    F1                       Accu.                     F1
Proposed w/ GAT                                  77.6%   ±  1.6%     0.76     ±  0.02                   81.6%   ±  1.2%     0.80     ±  0.01
Proposed w/ GCN                                  75.8%    ±  1.1%       0.74     ±  0.02                80.0%    ±  1.2%       0.79     ±  0.02
FC-NN                                            65.9%    ±  1.3%       0.59     ±  0.04                77.5%    ±  1.4%       0.74     ±  0.02
GCN (Original)                                   74.5%    ±  1.6%       0.72     ±  0.05                76.1%    ±  1.3%       0.73     ±  0.03
GCN (Modified)                                   75.5%    ±  1.4%       0.72     ±  0.03                77.9%    ±  1.1%       0.77     ±  0.02
Multi-omics GCN (Original)                       76.4%    ±  1.3%          0.76  ±  0.03                77.4%    ±  1.3%       0.77     ±  0.03
Multi-omics GCN (Modified)                       77.4%    ±  1.3%          0.76  ±  0.02                78.2%    ±  1.2%       0.75     ±  0.02
GrAMME (Modified)                                77.4%    ±  1.5%          0.76  ±  0.02                79.6%    ±  1.4%       0.79     ±  0.02
Multi-omics GAT (Original)                       73.4%    ±  1.8%       0.71     ±  0.04                75.1%    ±  1.5%       0.74     ±  0.04
Multi-omics GAT (Modified)                       75.8%    ±  1.5%       0.74     ±  0.04                77.4%    ±  1.3%       0.74     ±  0.02
The bold font indicates the highest values and the values after                       ±  sign are the standard deviations.
1 Accu. stands for Accuracy.
2 Prec. stands for Precision.

Li and Nabavi                                                                                                                                                                                  Page 13 of 18
                                Figure 4: Performance of the Proposed Models and Baseline Models with Differ-
                                ent Numbers of Genes on BRCA Dataset
                                     (a) The accuracy of the proposed model with GAT (blue solid line) or GCN (or-
                                     ange solid line) and baseline models (dashed line) are plotted against different
                                     numbers of genes (300, 500, 700, 1000, 2000, and 5000) for BRCA subtype clas-
                                     sification.
                                     (b) The F1 scores of the proposed model with GAT (blue solid line) or GCN
                                     (orange solid line) and baseline models (dashed line) are plotted against different
                                     numbers of genes (300, 500, 700, 1000, 2000, and 5000) for BRCA subtype clas-
                                     sification.

                     Li and Nabavi                                                                                                                                                                                  Page 14 of 18
                                                of genes grows beyond 1,000 genes. All GAT-based baseline models show similar
                                                deterioration around 1000 genes. We think the high computation cost of the GAT-
                                                based model can cause it to perform worse on a large graph than on a small graph.
                                                Overall, we can conclude that the proposed model with GCN layers scales better
                                                than that with GAT layers at a large number of genes.
                                                    In the process of testing the models on a large graph, we also find that a GAT-
                                                based model is more stable on a smaller learning rate compared to a GCN-based
                                                model.WebelieveitiscausedbyGAT’shighcomputationcostssinceahighlearning
                                                rate may cause the model to be stuck in a local optimum.
                                                    Overall, we see the proposed model achieves the best performance and scales well
                                                withalargernumberofgenes.Wecanalsoconcludethatmoregenesandmoreomics
                                                mostly improve the performance of models, the models with more modules have
                                                better performance, and GAT-based models perform better with smaller graphs
                                                while GCN-based models scale better at larger graphs.
                                                Different Training Set Split
                                                To examine the performance of the proposed model on a complex dataset with a
                                                smaller training set, we tested the model on the Pan-cancer dataset using three
                                                different training set splits. This approach was taken to mimic situations where
                                                only a smaller labeled dataset is available in the real world. The training set splits
                                                were set at 70%, 60%, and 50%, with corresponding testing set splits of 20%, 30%,
                                                and 40%. Throughout these tests, the validation set split was consistently kept at
                                                10%.
                                                          Table 5: Proposed Model with Different Training-validation-testing Split
                                                    As shown in Table 5, the proposed model with the GAT layer exhibits a slight
                                                performancedeteriorationat70%and60%trainingsetsplits.However,itdisplaysa
                                                morepronounceddeclineinclassificationaccuracyat50%.Incontrast,theproposed
                                                model with the GCN layer demonstrates consistent and robust performance across
                                                all three training-validation-testing splits. However, its classification accuracy is
                                                lowerthanthatofthemodelwiththeGATlayerat70%and60%trainingsetsplits.
                                                Therefore, we can conclude that the proposed model with the GAT layer achieves
                                                superiorperformancecomparedtothemodelwiththeGCNlayerwhenthetraining
                                                set is relatively small. However, the model with the GCN layer outperforms at a
                                                very small training set (50%). Overall, the proposed model with the GCN layer
                                                offers more robust classification performance with smaller training sets.
                                                Different Combinations of Modules
                                                To examine the effect of different modules within the proposed model, we test
                                                threedifferentvariantsoftheproposedmodelforthePan-cancermolecularsubtype
                                                classification. All variants of the proposed model are trained with all three omics
                                                                               Training Set Ratio
Model                                        70%                                     60%                                     50%
Proposed w/ GAT              82   .5%  ±  1 .5%    0      .82 ±  0 .02    79      .9%  ±  4 .0%    0      .78 ±  0 .06    74      .2%  ±  7 .5%    0      .71 ±  0 .10Accu.    1               F1                Accu.      1               F1                Accu.      1               F1
Proposed w/ GCN              77   .9%  ±  1 .2%    0      .76 ±  0 .02    76      .7%  ±  0 .4%    0      .75 ±  0 .01    77      .3%  ±  2 .5%    0      .76 ±  0 .03
The values after      ±  sign are the standard deviations.
1 Accu. stands for Accuracy.

                        Li and Nabavi                                                                                                                                                                                  Page 15 of 18
                                                     data at 300, 500, and 700 genes. The proposed model without the decoder acts as a
                                                     parallel structured GNN model, the proposed model without the parallel structure
                                                     acts as a graph autoencoder model, and the proposed model without both the
                                                     decoder and the parallel structure acts as a graph-classification GNN model.
                                                     Table 6: Results of the Variants of the Proposed Model for Molecular Subtype
                                                     Classification Using the TCGA Pan-cancer Dataset.
                                                        As shown in Table 6, models without the parallel structure perform poorly com-
                                                     pared to those without the decoder at any number of genes in general. It shows
                                                     that the parallel structure plays an important role in feature extraction, which also
                                                     demonstrates the benefit of including both local features and global features. When
                                                     the graph size is small (300 genes), the model without the decoder and the parallel
                                                     structureperformsmorepoorlycomparedtothosewitheithercomponent.However,
                                                     when the graph size is large enough (500 genes and 700 genes), the model without
                                                     the decoder and the parallel structure performs relatively the same compared to
                                                     those with either of the component. We believe the extra information in the large
                                                     graph compensates for the loss in performance caused by the exclusion of either the
                                                     decoder or the parallel structure.
                                                     Different Combination of Omics and Graphs
                                                     Table 7: Results of the Proposed Model on Different Combinations of Omics and
                                                     Networks at 500 Genes Using the TCGA Pan-cancer Dataset.
                                                        Totesttheeffectofdifferentchoicesofomicsanddifferentgraphs,wegeneratefive
                                                     differentcombinationsofomics.ThefivecombinationsofomicsaremRNA,miRNA,
                                                     mRNA + CNV, mRNA + miRNA, and mRNA + CNV + miRNA. For mRNA +
                                                     miRNA and mRNA + CNV + miRNA, two different variants of graphs are also
                                                     tested. All models are conducted for Pan-cancer molecular subtype classification,
                                                     and trained with 500 genes except for only miRNA omic, which contains only 100
                                                     miRNA nodes. As shown in Table 7, the best-performing setting is mRNA + CNV
              Data                   Network                                      GAT                                                   GCN
                                                         300         Accu.    5                  F1500                     Accu.    5                  F1700
GNNLayers(Module)mRNA        1               Intra-omic       3Accu.   1               F177.0%   ±  1.9%      0.75    ±  0.03Accu.   1               F176.1%   ±  0.9%     0.73    ±  0.01Accu.   1               F1
GAT(NoDecoder)miRNA       2               Intra-omic       476.3%  ±  1.6%   0.76   ±  0.0374.0%   ±  0.4%      0.70    ±  0.0178.2%  ±  1.2%   0.77   ±  0.0168.2%   ±  4.1%     0.63    ±  0.0480.2%  ±  1.2%   0.79   ±  0.01
GCN(NoDecoder)mRNA+CNV               1          Intra-omic       375.3%  ± 1.2%     0.74   ± 0.0279.1%   ±  1.4%      0.77    ±  0.0376.8%  ± 0.8%     0.75   ± 0.0177.1%   ±  0.7%     0.76    ±  0.0179.3%  ± 0.8%     0.78   ± 0.01
GAT(NoParallel)                         75.4%  ± 1.8%     0.73   ± 0.0376.1%   ±  1.6%      0.73    ±  0.0376.1%  ± 1.7%     0.73   ± 0.0275.4%   ±  0.7%     0.73    ±  0.0179.8%  ± 1.3%     0.78   ± 0.02
GCN(NoParallel)mRNA+miRNA                       Inter-omic73.5%  ± 1.2%     0.72   ± 0.0275.4%  ± 1.2%     0.73   ± 0.01      76.7%  ± 0.8%     0.75   ± 0.01
GAT(NoDecoder&Parallel)                 74.9%  ± 1.4%     0.73   ± 0.02Intra-omic77.3%   ±  1.6%      0.75    ±  0.0376.4%  ± 0.9%     0.74   ± 0.0176.8%   ±  0.7%     0.74    ±  0.0180.1%  ± 0.8%     0.79   ± 0.01
GCN(NoDecoder&Parallel)                 73.1%  ± 1.2%     0.73   ± 0.0280.3%   ±  1.6%      0.80    ±  0.0275.6%  ± 0.8%     0.73   ± 0.0177.4%   ±  0.6%     0.74    ±  0.0177.3%  ± 0.02%   0.76    ± 0.01
Theboldfontindicatesthehighestvaluesandthevaluesafter                 ± signarethestandarddeviations.mRNA+CNV+miRNA                        Inter-omicIntra-omic80.5%   ±  1.2%    0.80    ±  0.0278.2%   ±  0.6%        0.75  ±  0.01
1 Accu.standsforAccuracy.
The bold font indicates the highest values and the values after                     ±  sign are the standard deviations.
 1 Data contains no miRNA-based nodes, so only 500 gene nodes in the graph
 2 Data contains no gene-based nodes, so only 100 miRNA nodes in the graph
 3 The graph contains only gene-gene connections.
 4 The graph contains only miRNA-miRNA meta-path connections.
 5 Accu. stands for accuracy.

Li and Nabavi                                                                                                                                                                                  Page 16 of 18
                         + miRNA with intra-omic edges for both GAT-based and GCN-based models. The
                         worst-performingsettingismiRNA,whichhasthesmallestgraphsizeandinforma-
                         tion.ModelsonmRNA+CNVperformbetterthanthoseonmRNA+miRNA,but
                         addingmiRNAtomRNA+CNV(mRNA+CNV+miRNAsetting)stillimproves
                         themodelperformance.Modelswithintra-omicgraphperformsslightlybetterthan
                         models with inter-omics graph. The performance difference across different settings
                         is the same for both GAT-based and GCN-based models.
                         Conclusion
                         In this study, we propose a novel end-to-end multi-omics GNN framework for ac-
                         curate and robust cancer subtype classification. The proposed model utilizes multi-
                         omics data in the form of a heterogeneous multi-layer graph, which is the supra-
                         graph built from GGI network, miRNA-gene target network, and miRNA meta-
                         path. While GNNs have been previously employed for genomics data analysis, our
                         model’s novelty lies in the utilization of a heterogeneous multi-layer multiomics
                         supra-graph. The supra-graph not only incorporates inter-omics and intra-omic
                         connections from established biological knowledge but also integrates genomics,
                         transcriptomics, and epigenomics data into a single graph, providing a novel ad-
                         vancement in cancer subtype classification. The proposed model outperforms all
                         four baseline models for cancer molecular subtype classification. We do a thorough
                         comparative analysis of GAT and GCN-based models at different numbers of gene
                         settings, different combinations of omics, and different graphs.
                            Comparingtheproposedmodeltothebaselinemodels,itachievesthebestperfor-
                         manceforcancermolecularsubtypeclassificationandBRCAsubtypeclassification.
                         The proposed model with GAT layers performs better than that with GCN layers
                         at smaller-size graphs (smaller than 1,000 genes). However, the performance of the
                         GAT-based model deteriorates as the size of the graph grows beyond a certain
                         threshold. On the other hand, the performance of the GCN-based model contin-
                         ues to improve as the size of the graph grows. Therefore, we can conclude that a
                         GAT-basedmodelismoresuitableonasmallergraph,whereithasahigherfeature
                         extraction ability and its computation cost isn’t that high yet.
                            By studying the effect of different modules within the proposed model and dif-
                         ferent combinations of omics, we find the addition of a decoder and the parallel
                         structure, and including other omics improves the performance of the proposed
                         model. The benefit of using parallel structure outweighs that of decoder, especially
                         onsmaller-sizegraphs,andthebenefitofaddingCNVishigherthanthatofadding
                         miRNA. We also find that using a graph with only intra-omic edges yields a better
                         performance than using a graph with only inter-omics edges, which agrees with the
                         results from the previous study [7].
                            The proposed model also has some limitations. We investigate only two well-
                         established and widely adopted GNN models. New models are emerging with the
                         recent blooming of studies in GNN models. As the size of the graph grows or more
                         omics are added, GAT-based models become more sensitive to parameters and take
                         a much longer time to train. It is our future research direction to overcome such
                         limitations.Theproposedmodelforcancersubtypeclassificationdependsonlabeled
                         data, which is costly to annotate and difficult to obtain in the real world. Exploring

Li and Nabavi                                                                                                                                                                                  Page 17 of 18
                                   unsupervised learning for cancer subtype detection is also a direction we aim to
                                   pursue in our future research.
                                        In summary, incorporating gene-based and non-gene-based omic data in the form
                                   of a supra-graph with inter-omics and intra-omic connections improves the cancer
                                   subtype classification. The GAT-based model is preferable on smaller graphs than
                                   theGCN-basedmodel.GCN-basedmodelispreferablewhendealingwithlargeand
                                   complex graphs.
                                   Declarations
                                   Acknowledgements
                                   Not applicable.
                                   Funding
                                   This work is supported by the National Science Foundation (NSF) under grant No. 1942303, PI: Nabavi.
                                   Abbreviations
                                   Not applicable.
                                   Availability of data and materials
                                   TCGA Pan-cancer dataset and TCGA BRCA dataset are both obtained from Xena database
                                   (https://xenabrowser.net), the detailed link for TCGA Pan-cancer dataset is
                                   (https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%
                                   2Fxena.treehouse.gi.ucsc.edu%3A443                               ) and the detailed link for TCGA BRCA dataset is
                                   (https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)&removeHub=https%3A%2F%
                                   2Fxena.treehouse.gi.ucsc.edu%3A443                               ) [34]. The code for the proposed method can be found at our Github
                                   repository (https://github.com/NabaviLab/Multimodal-GNN-for-Cancer-Subtype-Clasification).
                                   Ethics approval and consent to participate
                                   Not applicable.
                                   Competing interests
                                   The authors declare that they have no competing interests.
                                   Consent for publication
                                   Not applicable.
                                   Authors’ contributions
                                   B.L. obtained the TCGA data and network data. B.L. designed the new method and analyzed the results. B.L. and
                                   S.N. drafted the manuscript and revised the manuscript together. Both authors have approved the final manuscript.
                                   Author details
                                   Department of Computer Science and Engineering, University of Connecticut, Storrs, US.
                                   References
                                     1.Li, B., Wang, T., Nabavi, S.: Cancer molecular subtype classification by graph convolutional networks on
                                            multi-omics data. Proceedings of the 12th ACM Conference on Bioinformatics, Computational Biology, and
                                            Health Informatics, BCB 2021                   1  (2021). doi:10.1145/3459930.3469542
                                     2.Zhang, X., Zhang, J., Sun, K., Yang, X., Dai, C., Guo, Y.: Integrated multi-omics analysis using variational
                                            autoencoders: Application to pan-cancer classification. Proceedings - 2019 IEEE International Conference on
                                            Bioinformatics and Biomedicine, BIBM 2019, 765–769 (2019). doi:10.1109/BIBM47256.2019.8983228
                                     3.Yang, B., Zhang, Y., Pang, S., Shang, X., Zhao, X., Han, M.: Integrating multi-omic data with deep subspace
                                            fusion clustering for cancer subtype prediction. IEEE/ACM Transactions on Computational Biology and
                                            Bioinformatics         XX   , 1–1 (2019). doi:10.1109/tcbb.2019.2951413
                                     4.Sharifi-Noghabi, H., Zolotareva, O., Collins, C.C., Ester, M.: Moli: Multi-omics late integration with deep neural
                                            networks for drug response prediction. Bioinformatics                              35 , 501–509 (2019). doi:10.1093/bioinformatics/btz318
                                     5.Wang, T., Shao, W., Huang, Z., Tang, H., Zhang, J., Ding, Z., Huang, K.: Mogonet integrates multi-omics
                                            data using graph convolutional networks allowing patient classification and biomarker identification. Nature
                                            Communications             12 , 3445 (2021). doi:10.1038/s41467-021-23774-w
                                     6.Ma, T., Zhang, A.: Integrate multi-omics data with biological interaction networks using multi-view
                                            factorization autoencoder (mae). BMC Genomics                                20 , 1–11 (2019). doi:10.1186/s12864-019-6285-x
                                     7.Kaczmarek, E., Jamzad, A., Imtiaz, T., Nanayakkara, J., Renwick, N., Mousavi, P.: Multi-omic graph
                                            transformers for cancer classification and interpretation. Pacific Symposium on Biocomputing. Pacific
                                            Symposium on Biocomputing                     27 , 373–384 (2022)
                                     8.Lotfollahi, M., Litinetskaya, A., Theis, F.J.: Multigrate : single-cell multi-omic data integration, 1–5 (2022).
                                            doi:10.1101/2022.03.16.484643
                                     9.Huang, Z., Zhan, X., Xiang, S., Johnson, T.S., Helm, B., Yu, C.Y., Zhang, J., Salama, P., Rizkalla, M., Han,
                                            Z., Huang, K.: Salmon: Survival analysis learning with multi-omics neural networks on breast cancer. Frontiers
                                            in Genetics       10 , 1–13 (2019). doi:10.3389/fgene.2019.00166

Li and Nabavi                                                                                                                                                                                  Page 18 of 18
                                     10.Bai, J., Li, B., Nabavi, S.: Semi-supervised classification of disease prognosis using cr images with clinical data
                                              structured graph. In: Proceedings of the 13th ACM International Conference on Bioinformatics, Computational
                                              Biology and Health Informatics, pp. 1–9 (2022)
                                     11.Chai, H., Zhou, X., Zhang, Z., Rao, J., Zhao, H., Yang, Y.: Integrating multi-omics data through deep learning
                                              for accurate cancer prognosis prediction. Computers in biology and medicine                                           134  , 104481 (2021)
                                     12.Heo, Y.J., Hwa, C., Lee, G.H., Park, J.M., An, J.Y.: Integrative multi-omics approaches in cancer research:
                                              From biological networks to clinical subtypes. Molecules and Cells                                    44 , 433–443 (2021).
                                              doi:10.14348/molcells.2021.0042
                                     13.Hoadley, K.A., Yau, C., Wolf, D.M., Cherniack, A.D., Tamborero, D., Ng, S., Leiserson, M.D., Niu, B.,
                                              McLellan, M.D., Uzunangelov, V.,                      et al. : Multiplatform analysis of 12 cancer types reveals molecular
                                              classification within and across tissues of origin. Cell                         158  (4), 929–944 (2014)
                                     14.Mateo, J., Steuten, L., Aftimos, P., Andr´e, F., Davies, M., Garralda, E., Geissler, J., Husereau, D.,
                                              Martinez-Lopez, I., Normanno, N.,                      et al. : Delivering precision oncology to patients with cancer. Nature
                                              Medicine       28 (4), 658–665 (2022)
                                     15.Hoadley, K.A., Yau, C., Hinoue, T., Wolf, D.M., Lazar, A.J., Drill, E., Shen, R., Taylor, A.M., Cherniack, A.D.,
                                              Thorsson, V.,         et al. : Cell-of-origin patterns dominate the molecular classification of 10,000 tumors from 33
                                              types of cancer. Cell           173  (2), 291–304 (2018)
                                     16.Defferrard, M., Bresson, X., Vandergheynst, P.: Convolutional neural networks on graphs with fast localized
                                              spectral filtering. Advances in Neural Information Processing Systems, 3844–3852 (2016)
                                     17.Zou, J., Huss, M., Abid, A., Mohammadi, P., Torkamani, A., Telenti, A.: A primer on deep learning in
                                              genomics. Nature genetics                51 (1), 12–18 (2019)
                                     18.He, S., Pepin, L., Wang, G., Zhang, D., Miao, F.: Data-driven distributionally robust electric vehicle balancing
                                              for mobility-on-demand systems under demand and supply uncertainties. In: 2020 IEEE/RSJ International
                                              Conference on Intelligent Robots and Systems (IROS), pp. 2165–2172. IEEE
                                     19.Wang, T., Li, B., Nabavi, S.: Single-cell rna sequencing data clustering using graph convolutional networks. In:
                                              2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), pp. 2163–2170 (2021). IEEE
                                     20.Shi, C., Emadikhiav, M., Lozano, L., Bergman, D.: Constraint learning to define trust regions in
                                              predictive-model embedded optimization. arXiv preprint arXiv:2201.04429 (2022)
                                     21.He, S., Han, S., Miao, F.: Robust electric vehicle balancing of autonomous mobility-on-demand system: A
                                              multi-agent reinforcement learning approach. In: 2023 IEEE/RSJ International Conference on Intelligent Robots
                                              and Systems (IROS), pp. 5471–5478 (2023). IEEE
                                     22.Wang, K., Lozano, L., Bergman, D., Cardonha, C.: A two-stage exact algorithm for optimization of neural
                                              network ensemble. In: Integration of Constraint Programming, Artificial Intelligence, and Operations Research:
                                              18th International Conference, CPAIOR 2021, Vienna, Austria, July 5–8, 2021, Proceedings 18, pp. 106–114
                                              (2021). Springer
                                     23.Nicora, G., Vitali, F., Dagliati, A., Geifman, N., Bellazzi, R.: Integrated multi-omics analyses in oncology: a
                                              review of machine learning methods and tools. Frontiers in oncology                                      10 , 1030 (2020)
                                     24.Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Philip, S.Y.: A comprehensive survey on graph neural networks.
                                              IEEE transactions on neural networks and learning systems (2020)
                                     25.Velickovi´c, P., Cucurull, G., Casanova, A., Romero, A., Li`o, P., Bengio, Y.: Graph attention networks. arXiv,
                                              1–12 (2017)
                                     26.Ramirez, R., Chiu, Y.-C., Hererra, A., Mostavi, M., Ramirez, J., Chen, Y., Huang, Y., Jin, Y.-F.: Classification
                                              of cancer types using graph convolutional neural networks. Frontiers in physics                                           8  (2020)
                                     27.Wang, T., Bai, J., Nabavi, S.: Single-cell classification using graph convolutional networks. BMC bioinformatics
                                              22 (1), 1–23 (2021)
                                     28.Shanthamallu, U.S., Thiagarajan, J.J., Song, H., Spanias, A.: Gramme: Semisupervised learning using
                                              multilayered graph attention models. IEEE Transactions on Neural Networks and Learning Systems                                                            31 ,
                                              3977–3988 (2020). doi:10.1109/TNNLS.2019.2948797
                                     29.Onitilo, A.A., Engel, J.M., Greenlee, R.T., Mukesh, B.N.: Breast cancer subtypes based on er/pr and her2
                                              expression: comparison of clinicopathologic features and survival. Clinical medicine & research                                                  7(1-2), 4–13
                                              (2009)
                                     30.Oughtred, R., Rust, J., Chang, C., Breitkreutz, B.J., Stark, C., Willems, A., Boucher, L., Leung, G., Kolas, N.,
                                              Zhang, F., Dolma, S., Coulombe-Huntington, J., Chatr-Aryamontri, A., Dolinski, K., Tyers, M.: The BioGRID
                                              database: A comprehensive biomedical resource of curated protein, genetic, and chemical interactions. Protein
                                              Sci  30 (1), 187–200 (2021)
                                     31.Chen, Y., Wang, X.: mirdb: an online database for prediction of functional microrna targets. Nucleic acids
                                              research     48 (D1), 127–131 (2020)
                                     32.Lee, B., Zhang, S., Poleksic, A., Xie, L.: Heterogeneous multi-layered network model for omics data integration
                                              and analysis. Frontiers in Genetics                  10 , 1–11 (2020). doi:10.3389/fgene.2019.01381
                                     33.13, B..W.H..H.M.S.C.L...P.P.J..K.R., data analysis: Baylor College of Medicine Creighton Chad J. 22 23
                                              Donehower Lawrence A. 22 23 24 25, G., for Systems Biology Reynolds Sheila 31 Kreisberg Richard B. 31
                                              Bernard Brady 31 Bressler Ryan 31 Erkkila Timo 32 Lin Jake 31 Thorsson Vesteinn 31 Zhang Wei 33
                                              Shmulevich Ilya 31, I.,            et al. : Comprehensive molecular portraits of human breast tumours. Nature                                          490  (7418),
                                              61–70 (2012)
                                     34.Goldman, M.J., Craft, B., Hastie, M., Repeˇcka, K., McDade, F., Kamath, A., Banerjee, A., Luo, Y., Rogers,
                                              D., Brooks, A.N.,           et al. : Visualizing and interpreting cancer genomics data via the xena platform. Nature
                                              biotechnology         38 (6), 675–678 (2020)
                                     Additional Files
                                     Additional file 1: Detailed Results and Model Settings

