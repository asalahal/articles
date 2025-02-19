Since January 2020 Elsevier has created a COVID-19 resource centre with
free information in English and Mandarin on the novel coronavirus COVID-
   19. The COVID-19 resource centre is hosted on Elsevier Connect, the
                company's public news and information website.
     Elsevier hereby grants permission to make all its COVID-19-related
research that is available on the COVID          -19 resource centre - including this
   research content - immediately available in PubMed Central and other
publicly funded repositories, such as the WHO COVID database with rights
for unrestricted research re-use and analyses in any form or by any means
    with acknowledgement         of the original source. These permissions are
 granted for free by Elsevier       for as long as the COVID       -19 resource centre
                                     remains active.

                                                                                                  Contents lists available at            ScienceDirect
                                                                                  ArtificialIntelligenceInMedicine
                                                                                   journal homepage:               www.elsevier.com/locate/artmed
Researchpaper
Deepvariationalgraphautoencodersfornovelhost-directedtherapyoptions
againstCOVID-19
SumantaRay              a,b,SnehalikaLall           e,AnirbanMukhopadhyay                         d,SanghamitraBandyopadhyay                              e,
AlexanderSchönhuth                     c,∗
a Department of Computer Science and Engineering, Aliah University, New Town, Kolkata, India
b Health Analytics Network, PA, USA
c Genome Data Science, University of Bielefeld, Bielefeld, Germany
d Department of Computer Science and Engineering, University of Kalyani, Kalyani, India
e Machine Intelligence Unit, Indian Statistical Institute, Kolkata, India
A R T I C L E   I N F O                                                              A B S T R A C T
Datasetlink:      https://github.com /sum antaray/C                                  TheCOVID-19pandemichasbeenkeepingaskingurgentquestionswithrespecttotherapeuticoptions.Existing
ovid19                                                                               drugs that can be repurposed promise rapid implementation in practice because of their prior approval.
Keywords:                                                                            Conceivably, there is still room for substantial improvement, because most advanced artificial intelligence
COVID-19                                                                             techniques for screening drug repositories have not been exploited so far. We construct a comprehensive
VariationalgraphautoEncoder                                                          network by combining year-long curated drug–protein/protein–protein interaction data on the one hand, and
Node2Vec                                                                             most recent SARS-CoV-2 protein interaction data on the other hand. We learn the structure of the resulting
Molecularinteractionnetwork                                                          encompassing molecular interaction network and predict missing links using variational graph autoencoders
Hostdirectedtherapy                                                                  (VGAEs), as a most advanced deep learning technique that has not been explored so far. We focus on hitherto
                                                                                     unknown links between drugs and human proteins that play key roles in the replication cycle of SARS-CoV-2.
                                                                                     Thereby, we establish novel host-directed therapy (HDT) options whose utmost plausibility is confirmed by
                                                                                     realistic simulations. As a consequence, many of the predicted links are likely to be crucial for the virus to
                                                                                     thrive on the one hand, and can be targeted with existing drugs on the other hand.
1. Introduction                                                                                                                            design crucially depends on the availability of such 3D structures.
                                                                                                                                           Note further that because the virus hijacks the host cell machinery
      The ongoing pandemic of COVID-19 (Coronavirus Disease-2019),                                                                         for replication through interactions of viral with human proteins, a
caused by SARS-CoV-2, an enveloped, single-stranded RNA virus [                                     1],                                    comprehensive understanding of the interactions between viral and
has led to more than a million deaths so far and keeps asking urgent                                                                       human proteins is essential [              2,3].
questions. Accepting the challenge, researchers have been relentlessly                                                                           In that quest for therapy options, the most advanced artificial in-
searching for possible therapeutic strategies in the last few months.                                                                      telligence based approaches may mean a massive boost with respect to
However, still, no truly reliable remedy has been showing on the                                                                           drug repurposing screens. However, despite their promising potential,
horizon.
      Repurposing  drugs  refers  to  screening  databases  for  molecules                                                                 many state-of-the-art AI – in particular deep neural network – based
whose risks have been found to be manageable in prior applications                                                                         approaches have not yet been explored so far.
on the one hand, and that can be shown to target proteins that are                                                                               Here, we do exactly this. We combine existing, year-long curated
crucial for SARS-CoV-2 to replicate and thrive on the other hand. If                                                                       and approved molecular (drug/human protein) interaction data with
not even representing viable cures by themselves, repurposed drugs                                                                         most recent experimental interaction screens (yielding new, and so
have the potential to mitigate the severity of the pandemic for the                                                                        faronlyinsufficientlyexploredSARS-CoV-2–humanproteininteraction
time being. The fact that the majority of 3D structures of the SARS-                                                                       data). In this, we design an experimental setup that enables us to ex-
CoV-2 proteins has remained unknown so far, corroborates the need                                                                          ploit most advanced (and hitherto unexplored) deep variational graph
for artificial intelligence based screens of molecular interaction data                                                                    autoencoder techniques for generating novel therapy options.
thatrelatewithCOVID-19further,becauseexperimental,denovodrug
   ∗  Corresponding author.
       E-mail addresses:          sumanta.ray@aliah.ac.in               (S. Ray),     alexander.schoenhuth@uni-bielefeld.de                        (A. Schönhuth).
https://doi.org/10.1016/j.artmed.2022.102418
Received 21 August 2021; Received in revised form 22 March 2022; Accepted 2 October 2022

S. Ray et al.
      In detail, we first learn the structure of the comprehensive drug–                                                                      As for (3)–(5), we will demonstrate how to adapt most advanced
human–virus molecule interaction network while encoding the net-                                                                        deep learning based techniques to learn and exploit molecular interac-
work. We then predict links between human proteins that are crucial                                                                     tion network data. By this, we are able to predict new links between
forthevirustoreplicateontheonehand,andexistingdrugsasaresult                                                                            drugsandproteinsatutmostaccuracy.Thespectrumofdrugswereveal
of decoding the autoencoder representation of our network, although                                                                     is fairly broad in terms of mechanism of action. We are therefore
these links were not part of the original network. As a result, we do                                                                   convinced that several drugs we suggest have solid potential to be
not only predict drugs to act against COVID-19, but we also identify                                                                    amenable to developing successful HDTs against COVID-19.
the human proteins that when blocked lead to disruption of the viral
replicationcycle,whichfostersthebiomolecularunderstandingofhow                                                                          2. Materials and methods
viral replication can be controlled.
      Notethatwesuggesttoblockhumanproteinssuchthatthereplica-                                                                                In the following, we will first describe the workflow of our analysis
tion machinery of the virus falls apart. However, we do not suggest to                                                                  pipeline and the basic ideas that support it.
target viral proteins themselves. The justification is that viral proteins,                                                                   First, we raise a novel network by combining well-established and
when being targeted, tend to elicit resistance-inducing mutations, such                                                                 most recent resources into an overarching, comprehensive interac-
that the virus rapidly adapts to the (rather simpleminded) attack. In                                                                   tion network that puts drugs, human and SARS-CoV-2 proteins into
comparison to viral proteins, human proteins acquire mutations at                                                                       encompassing context.
rates that are smaller by orders of magnitudes. This renders human                                                                            We then carry out a simulation study that proves that our AI
proteins substantially more sustainable therapy options when estab-                                                                     supported pipeline predicts missing links in the encompassing drug–
lishing actionable drug targets [4]. This explains why we focus on the                                                                  human protein–SARS-CoV-2–protein network at utmost accuracy. With
corresponding host-directed therapy (HDT) options here. In summary,                                                                     this,weprovideevidenceforourpredictionstoreflecttrueinteractions
we suggest drugs that have the potential to be rapidly integrated in                                                                    between molecular interfaces, at utmost likelihood.
clinicalpractice(thankstorepurposing)andthattheviruscannoteasily                                                                              Subsequently, in our real experiments, we predict links between
escape (thanks to serving HDT based strategies).                                                                                        drugs on the one hand, and SARS-Cov-2-associated human proteins
      Our combination of drug repurposing and HDT based on screening                                                                    on the other hand to be missing. Corroborated by our simulations, a
molecular interaction data is further supported by prior work that has                                                                  large fraction (if not possibly even the vast majority) of predictions
been describing unprecedented opportunities lately [5]. Examples of                                                                     establish true molecular interactions, potentially actionable in HDT
pathogensthatweretreatedearlierareDengue[6],HIV[7],Ebola[8],                                                                            based strategies.
next to various other, non-viral diseases.                                                                                                    Finally, we inspect the postulated mechanism of action of the sug-
      Asforrelatedwork,ahandfulofresearchgroupshavebeentryingto                                                                         gested drugs in the frame of several diseases, including the closely
suggest drugs to be repurposed so as to counteract the spread of SARS-                                                                  related SARS-CoV (‘‘SARS-classic’’) and MERS-CoV, documenting the
CoV-2 in the human body based on exploiting network resources since                                                                     plausibility of our predictions.
the outbreak of COVID-19. Zhou et al. made the first attempt through
anintegrativenetworkanalysis[9],followedbyLietal.whocombined                                                                            2.1. Workflow
network data with a comparative analysis on the gene sequences of
different viruses [10].                                                                                                                       SeeFig.1fortheworkflowofouranalysispipeline.Wewilldescribe
      Only shortly thereafter, however, Gordon et al. generated a map                                                                   all important steps in the paragraphs of this subsection.
that juxtaposes SARS-CoV-2 proteins with human proteins based on
affinity-purification mass spectrometry (AP-MS) screens in pioneering                                                                   2.1.1. Raising a comprehensive interaction network
work [11], closely followed by Dick et al. who, in independent work,                                                                          See A & B inFig.1. We have combined well-established drug–
identified  high  confidence  interactions  between  human  and  SARS-                                                                  gene interaction and human interactome data (compiled from eight,
CoV-2 proteins using sequence-based PPI predictors (a.k.a. PIPE4 &                                                                      year-long curated, much refined, well-established publicly accessible
SPRINT) [12]. Recently, Sadegh et al. developed CoVex to visually
exploretheSARS-CoV-2hostinteractomeandrepurposabledrugsinan                                                                             resources) with the SARS-CoV-2–human protein-interaction network
onlineinteractiveplatform[13].Giuliaetal.[14],proposedanetwork                                                                          publishedonlyafewweeksago.Theintegratednetworkhasfourtypes
similarity based approach to prioritize drug molecules associated with                                                                  of nodes:
COVID-19. Gysi et al. [15] integrate artificial intelligence, network                                                                         (1)   SARS-CoV-2 proteins,              (2)   SARS-CoV-2-associated host proteins
diffusion, and network proximity measure to rank the existing drugs                                                                     (CoV-host),        (3)   human proteins other than (2) and                     (4)   drugs. This
for their expected efficacy against SARS-CoV-2.                                                                                         means that we put drugs, human proteins and SARS-CoV-2 proteins
      Both of the studies [11,12] provide crucial data, because only                                                                    into a context that is as comprehensive as currently possible. Still, it
because of the two studies we are able to link existing (long term                                                                      is highly likely, however, that links are missing. Because many such
curated and highly reliable) drug–protein and human protein–protein                                                                     missing links reflect drugs that can be repurposed, it remains to set up
interaction data with the SARS-CoV-2 proteins, just as was possible for                                                                 an AI approach that can predict such links.
the above-mentioned diseases earlier [6,7,16,17].
      Still, however, the exploitation of the novel data, in combination                                                                2.1.2. AI model first stage — Node2Vec
with year-long established, refined and curated interaction data using                                                                        See C inFig.1. For the link prediction machinery to work, we
most advanced AI techniques needed to be brought into effect. As a                                                                      operate in two stages. First, we employ Node2Vec [18], as a network
brief summary of our contributions:                                                                                                     embedding strategy that extracts node features from the integrated
      (1)   We link existing high-quality, long-term curated and refined,                                                               network. Formally, Node2Vec converts the adjacency matrix that rep-
large scale drug/protein–protein interaction data with                                                                                  resents the network into a fixed-size, low-dimensional latent feature
      (2)  molecularinteractiondataonSARS-CoV-2itself,raisedrecently                                                                    space. As elements of this space, nodes correspond to feature vectors.
in literature,                                                                                                                          Thereby, Node2Vec aims at preserving the properties of the nodes
      (3)   exploit the resulting overarching network using an advanced                                                                 relative to their surroundings in the network. For efficiency reasons,
AI supported techniques (namely variational graph autoencoder based                                                                     Node2Vec makes use of a sampling strategy. The result of this step
techniques)                                                                                                                             is a feature matrix (   𝐹) where rows refer to nodes and columns refer
      (4)  for repurposing drugs in the fight against SARS-CoV-2                                                                        to the inferred network features (See supplementary text for detailed
      (5)  in the frame of HDT based strategies.                                                                                        descriptions).

S. Ray et al.
Fig. 1.     Overall workflow of the proposed method: The three networks SARS-CoV-2–host PPI, human PPI, and drug–target network (Panel-A) are mapped by their common interactors
to form an integrated representation (Panel-B). The neighborhood sampling strategy Node2Vec converts the network into fixed-size low dimensional representations that perverse
the properties of the nodes belonging to the three major components of the integrated network (Panel-C). The resulting feature matrix (F) from the node embeddings and adjacency
matrix (A) from the integrated network are used to train a VGAE model, which is then used for prediction (Panel-D).
2.1.3. AI model second stage — variational graph autoencoders (VGAE)                                                                          Table 1
      SeeB,C&Din        Fig.  1.Inthenextstep,weemployvariationalgraph                                                                        Average AUC and AP across the last 10 training epochs of FastGAE. Validation AUC
autoencoders (VGAE), as a most recent graph neural network based                                                                              and AP for different numbers       𝑁𝑆 of sampling nodes are reported.
techniquethatwasshowntopredictmissinglinksinnetworksatutmost                                                                                    𝑁𝑠                  Average performance on validation set
accuracy[     19 ].VGAEsrequiretheoriginalgraph(codedasitsadjacency                                                                                                    AUC (    % )                      AP (   % )                         Training time (in sec)
matrix  𝐴) and, optionally, a feature matrix       𝐹 that annotates the nodes                                                                   7000               89.21         ±  0.02               85.32        ±  0.02               1587
of the network with helpful additional information. Often,             𝐹 does not                                                               5000               89.17         ±  0.03               85.30        ±  0.04               1259
necessarily refer to the topology of the network itself. Here, however,                                                                         3000               88.91         ±  0.10               85.02        ±  0.04               1026
we do make use of the feature matrix         𝐹 that was inferred from     𝐴 by                                                                  2500               88.27         ±  0.15               84.88        ±  0.13               998
                                                                                                                                                1000               86.69         ±  0.17               83.58        ±  0.19               816
Node2Vec. We found that using        𝐹 aided in raising prediction accuracy
substantially, despite    𝐹 only being an alternative representation of          𝐴.
The explanation for this is that      𝐹 consists of knowledge obtained using
Node2Vec, which, as being complementary to VGAEs, indeed reveals                                                                              explicit before, the existence is implied by the topological constraints
additional information. Our pipeline thus unifies the virtues of both                                                                         the comprehensive network imposes on such links to exist or not.
VGAE and Node2Vec. See Section                      3-C and     Fig.   2  for corresponding                                                   Our model thus predicts both drugs and proteins: repurposing these
experiments.                                                                                                                                  drugs leads to targeting the matching proteins. See                           Fig.  1  for the total
                                                                                                                                              workflow we just described.
2.1.4. Predicting missing links
      See D in     Fig.  1. After training the VGAE, we predict links in the en-                                                              2.1.5. Addressing computation time
compassing drug–human–virus interaction network that had remained                                                                                   The biomolecular interaction networks one needs to consider for
to be missing. For this, we make use of the decoding part of the                                                                              successful drug repurposing, i.e. standard protein–protein and drug–
VGAE, which re-raises the interaction network based on the latent                                                                             proteininteractionnetworks,consistofhundredsofthousandsofnodes,
representation the encoder had computed. Re-raising the network re-                                                                           and are too large for standard implementations of VGAEs to deal with.
sults in edges between nodes that although not having been explicit                                                                           This renders advanced, runtime friendly implementations of VGAEs
before, are imperative to exist relative to the encoded version of the                                                                        crucial ingredients of our workflow. Most recent progress on that topic
network. Thereby, one predicts links between drugs and SARS-CoV-                                                                              by Salha et al. (published Feb 5, 2020), running under the name
2-associated human proteins in particular. Although not having been                                                                           FastGAE[      20 ]providesthelastkeyelementforourapproachtoworkin

S. Ray et al.
                                                Fig. 2.     Performance of the model (AUC on the validation set) with and without using feature matrix (F).
practice.FastGAEreliesonastrategybywhichtorepeatedlysubsample                                                             where,  𝑁𝑁𝑐 is the number of SARS-CoV-2 proteins.         𝑁𝐷𝑇 is the number
nodes from large graphs, and train VGAEs on the resulting subgraphs,                                                      of drug targets, whereas      𝑁𝑁𝑇  and 𝑁𝐷 represent the number of CoV-
and subsequently to join the resulting autoencoders in a consistent                                                       host and drugs nodes, respectively. The total number of edges is given
manner. In experiments (seeTable1), we determined 5000 nodes as                                                           by:
an optimal size for a subsample.                                                                                          𝑚 = |𝐸1|+ |𝐸2|+ |𝐸3|,                                                                    (5)
2.2. Sampling strategy and feature matrix generation                                                                      where,  𝐸1 representsinteractionsbetweenSARS-CoV-2andhumanhost
                                                                                                                          proteins,  𝐸2 is the number of interactions among human proteins, and
     We have utilized           Node2vec       [18], an algorithmic framework for                                         𝐸3  represents the number of interactions between drugs and human
learning continuous feature representations for nodes in networks. It                                                     host proteins.
maps the nodes to a low-dimensional feature space that maximizes the                                                      2.3.2. Feature matrix preparation
likelihood of preserving network neighborhoods.                                                                                 The neighborhood sampling strategy is used to compute feature
     The principle of feature learning framework in a graph can be de-                                                    representations for all nodes. A flexible biased random walk procedure
scribed as follows: Let     𝐺 = (𝑉,𝐸) be a graph, where     𝑉 represents a set                                            isemployedtoexploretheneighborhoodofeachnode.Arandomwalk
of nodes, and   𝐸 represents the set of edges. The feature representation                                                 in a graph  𝐺 can be described as the probability
of nodes (   |𝑉|) is given by a mapping function:        𝑓 ∶ 𝑉 → 𝑅𝑑, where
𝑑 specifies the feature dimension. Alternatively,          𝑓 may be considered                                            𝑃(𝑎𝑖= 𝑥∣𝑎𝑖−1   = 𝑣) = 𝜋(𝑣,𝑥),                                                           (6)
as a node feature matrix of a dimension of                |𝑉|× 𝑑. For each node,                                          where,  𝜋(𝑣,𝑥)  is the transition probability between nodes          𝑣 and 𝑥,
𝑣∈ 𝑉, a network neighborhood       𝑁𝑁𝑆(𝑣)⊂𝑉 of node  𝑣is defined by                                                       where      (𝑣,𝑥) ∈ 𝐸 and 𝑎𝑖is the 𝑖th node in the walk of length       𝑙. The
employing a neighborhood sampling strategy            𝑆. The sampling strategy                                            transition probability is given by       𝜋(𝑣,𝑥) = 𝑐𝑝𝑞(𝑡,𝑥)×𝑤𝑣𝑥, where  𝑡is the
can be sketched as an interpolation between breadth-first search and                                                      previous node of    𝑣in the walk,   𝑤𝑣𝑥is the (static) weight attached to
depth-first search [18], with objective function                                                                          theedge      (𝑣,𝑥) and 𝑝,𝑞arethetwoparametersthatguidethewalk.The
      ( ∑                                       )                                                                         coefficient  𝑐𝑝𝑞(𝑡,𝑥) is given by
max𝑓           log𝑃(𝑁𝑁𝑆(𝑣) ∣𝑓(𝑣))                                                                            (1)
         𝑣∈𝑉                                                                                                                              ⎧⎪⎪⎨⎪⎪⎩
                                                                                                                                             1∕𝑝   distance(t ,x) = 0
This maximizes the likelihood of observing a network neighborhood                                                         𝑐𝑝𝑞(𝑡,𝑥) =         1       distance(t  ,x) = 1                                                                (7)
𝑁𝑁𝑆(𝑣) for a node   𝑣given its feature representation       𝑓(𝑣). The prob-
ability of observing a neighborhood node          𝑛𝑖∈ 𝑁𝑁𝑆(𝑣) given  𝑓(𝑣) is                                                                  1∕𝑞   distance(t ,x) = 2
𝑃(𝑁𝑁𝑆(𝑣) ∣𝑓(𝑣)) =    ∏                         𝑃(𝑛𝑖∣𝑓(𝑣)).                                        (2)                     where     distance( 𝑡,𝑥) representsthedistanceoftheshortestpathbetween
                                                                                                                          nodes  𝑡and node   𝑥. The process of feature matrix       𝐹𝑛×𝑑 generation
                                𝑛𝑖∈𝑁𝑁𝑆(𝑣)                                                                                 is governed by the Node2vec algorithm. It starts from every node,
where  𝑛𝑖refers to the  𝑖th neighbor of node     𝑣as part of  𝑁𝑁𝑆(𝑣). Last,                                               simulating   𝑟randomwalksoffixedlength       𝑙.Ineverystepofawalkthe
the conditional probability      𝑃(𝑛𝑖∣𝑓(𝑣)) of a neighborhood node      𝑛𝑖∈                                               transition probabilities     𝜋(𝑣,𝑥) govern the sampling. In each iteration,
𝑁𝑁𝑆(𝑉) giventheoriginalnode     𝑣iscomputedasthesoftmaxthescalar                                                          generated  walks  are  added  to  a  list  of  walks.  Each  random  walk
product of their feature vectors       𝑓(𝑣) and 𝑓(𝑛𝑖)                                                                     forms a sentence which is ultimately used by word2vec [21], a well-
                                                                                                                          known algorithm that takes a set of sentences (walks), and outputs an
𝑃(𝑛𝑖∣𝑓(𝑣)) =   𝑒𝑥𝑝(𝑓(𝑣)  ⋅𝑓(𝑛𝑖))∑                                                                                         embedding for each word. The log-likelihood in Eq.(1)is optimized in
                          𝑢∈𝑉𝑒𝑥𝑝(𝑓(𝑢)  ⋅𝑓(𝑣))                                                  (3)                        the Optimization step by using stochastic gradient descent algorithm
2.3. Drug–SARS-CoV-2 link prediction                                                                                      on a two-layer Skip-gram neural network model used by word2vec.
                                                                                                                          2.3.3. Link prediction
2.3.1. Adjacency matrix preparation                                                                                             We utilize scalable and fast variational graph autoencoder (FastV-
     In this work, we consider an undirected graph           𝐺  =  (𝑉,𝐸) with                                             GAE) [20] to reduce the computational time of VGAE in networks that
|𝑉|= 𝑛nodes and     |𝐸|= 𝑚 edges. Let  𝐴 be the binary adjacency matrix                                                   are as large as ours. The adjacency matrix          𝐴 and the feature matrix
of𝐺.Here 𝑉 consistsofSARS-Cov-2proteins,CoV-hostproteins,drug–                                                            𝐹 are fed into the encoder of FastVGAE as input. The encoder uses a
target proteins and drugs. The matrix (        𝐴) contains a total of    𝑛= 16444                                         graph convolution neural network (GCN) on the entire graph to create
nodes given as:                                                                                                           the latent representation
𝑛= |𝑁𝑁𝑐|+ |𝑁𝐷𝑇|+ |𝑁𝑁𝑇|+ |𝑁𝐷|,                                                (4)                                          𝑍 = 𝐺𝐶𝑁 (𝐴,𝐹)                                                                               (8)

S. Ray et al.
The encoder works on the full adjacency matrix            𝐴. After encoding,                                              the 𝑑′-dimensional distribution              . Both 𝜇𝑖and 𝜎𝑖reflect the output of
one samples subgraphs, and decoding is performed on the sampled                                                           two graph convolutional networks (GCN) that share parameters in the
subgraphs.                                                                                                                first layer. For further details, see [19]. In that sense,           𝑧𝑖reflects to be
     The mechanism of the decoder of FastVGAE slightly differs from                                                       sampled from the Gaussian distributions that are learned by two GCNs
that of a traditional VGAE. For each subsample of graph nodes               𝑉𝑠,                                           that partially share parameters.
it regenerates an adjacency matrix         ̂𝐴. For subsampling graph nodes,                                               Decoder     .  The decoder is a generative model that seeks to reconstruct
it makes use of a technique that determines the nodes from which                                                          the graph, as represented by its adjacency matrix           𝐴 from the latent
to reconstruct the adjacency matrix in each iteration. Therefore, each                                                    variables  𝑧𝑖. The result is an estimate      ̂𝐴 of the adjacency matrix that is
node is assigned with a probability                                                                                       supposed to match the original        𝐴 as well as possible. The probability
𝑝(𝑖) =   𝑑(𝑖)𝛼∑                                                                                                           that 𝐴𝑖𝑗is one (that is there is an edge between node          𝑖and 𝑗), given
             𝑗∈𝑉𝑑(𝑗)𝛼                                                                            (9)                      the embedding vector      𝑍, evaluates as
where  𝑑(𝑖) is the degree of node     𝑖, and 𝛼is the sharpening parameter,                                                𝑝(𝐴𝑖,𝑗= 1 ∣ 𝑧𝑖,𝑧𝑗) = Sigmoid(   𝑧𝑇
where in our study     𝛼= 2  . Nodes are then selected during subsampling                                                                                                𝑖𝑧𝑗),                                                (13)
according to their probabilities       𝑝𝑖until the subsampled nodes amount                                                thattheapplicationofasigmoidfunctiontothescalarproductof              𝑧𝑖and
to |𝑉𝑠|= 𝑛𝑠, the prescribed number of sampling nodes.                                                                     𝑧𝑗.Theobjectivefunctionofthevariationalgraphautoencoder(VGAE)
     Thedecoderreconstructsthesmallermatrix,           ̂𝐴𝑠ofdimension   𝑛𝑠×𝑛𝑠                                             reads as
insteadofdecodingthemainadjacencymatrix           𝐴.Thedecoderfunction                                                    𝐶𝑉𝐺𝐴𝐸
follows the following equation:
̂𝐴𝑠(𝑖,𝑗) = Sigmoid(   𝑧𝑇                                                                                                  = 𝐸𝑞(𝑍∣𝐴,𝐹)[log 𝑝(𝐴 ∣𝑍)]− 𝐷𝐾𝐿(𝑞(𝑍 ∣𝐴,𝐹) ∥𝑝(𝑍))                          (14)
                               𝑖  ⋅𝑧𝑗),∀(𝑖,𝑗) ∈ 𝑉𝑠×𝑉𝑠                                     (10)                            where  𝐷𝐾𝐿(. ∥ .)  reflects Kullback–Leibler divergence and          𝑝(𝑍)  is
where  𝑧𝑖,𝑧𝑗reflecttherepresentationsofnodes        𝑖,𝑗,ascomputedbythe                                                   the prior distribution that governs the latent variables            𝑍.𝐶𝑉𝐺𝐴𝐸  is
encoder, see(8). At each training iteration a different subgraph (                𝐺𝑠) is                                  maximizedusingstochasticgradientdescent;fordetailsseeagain[19].
drawn using the sampling method.
     After training the model, the drug–CoV-host links are predicted                                                      3.2. Practical implementation: FastGAE
using the equation
𝑝(𝐴𝑖𝑗= 1 ∣ 𝑧𝑖,𝑧𝑗) = Sigmoid(   𝑧𝑇𝑖𝑧𝑗),                                                (11)                                     We utilize FastGAE, a fast version of VGAE, for the implementa-
                                                                                                                          tion of variational graph autoencoding in practice. Note that while
where  𝐴𝑖𝑗= 1   reflects a link between nodes       𝑖and 𝑗to exist, where   𝑖                                             encoding is feasible in practice also for large networks, decoding is
and 𝑗further reflect human proteins that interact with SARS-CoV-2 on                                                      not. Therefore, FastGAE is identical to the original VGAE during the
the one hand and drugs on the other hand (recalling that we would                                                         encoding phase. Decoding, however, is computationally too expensive
like to predict links between human proteins that when targeted lead                                                      if the underlying graph, and hence its adjacency matrix is too large.
to the replication machinery of the virus falling apart). For each such                                                   Notethatherethenumberofnodescorrespondstothenumberofdrugs,
combinationofnodesthemodelcomputestheprobabilitybasedonthe                                                                human and SARS-CoV-2 proteins together, which clearly exceeds the
logistic sigmoid function.                                                                                                limits of the original VGAE.
                                                                                                                               To resolve the issue, FastGAE randomly samples subgraphs               𝐺𝑆,
3. Formal details and background                                                                                          referringtosmallersetsofnodes       𝑆⊂𝑉 ofsize 𝑁𝑆,andreconstructsthe
                                                                                                                          corresponding submatrices       𝐴𝑆. FastGAE proceeds in several iterations
3.1. Variational graph autoencoder                                                                                        in each of which a different subset of nodes          𝑆 of size 𝑁𝑆 is sampled.
                                                                                                                          Thedecodingstepthenestimatesanadjacencymatrix             𝐴𝑆 whoseentries
     The Variational Graph Autoencoder (VGAE) is a framework for                                                          refer only nodes from     𝑆. The submatrices     𝐴𝑆  resulting from single
unsupervised learning on graph-structured data [19]. This model uses                                                      iterations are eventually combined into an overarching matrix               ̃𝐴, as
latent variables and is effective in learning interpretable latent repre-                                                 an approximation of the matrix        ̂𝐴 that gets reconstructed as a whole
sentations for undirected graphs. The VGAE consists of two subnet-                                                        in the decoding phase of the original VGAE. The justification is that
works that are stacked onto another: (1) Encoder and (2) Decoder.                                                         depending on the number of iterations and the size            𝑁𝑆 of a sample,   ̃𝐴
First, a graph convolution networks (GCN) based encoder [19] maps                                                         was shown to be a highly accurate estimate of the original             𝐴 [20].
the nodes into a low-dimensional embedding space. Subsequently, a                                                              For evaluating the effects of including a feature matrix            𝐹  that
decoder attempts to reconstruct the original graph structure from the                                                     reflects an alternative representation of the original adjacency matrix
encoder representations. Both models are jointly trained to optimize                                                      𝐴, we evaluated the performance of the model with and without
the quality of the reconstruction from the embedding space, in an                                                         including  𝐹.Fig.2shows the average performance of the model on
unsupervised way. The encoder and the decoder are described in the                                                        validation sets with and without        𝐹 as input for different numbers of
following.                                                                                                                sampling nodes. The average AUC, and AP scores are reported for 50
Encoder     .  TheencoderconsistsofaGraphConvolutionNetwork(GCN)                                                          complete runs. FromFig.2, it is evident that including                𝐹 as a feature
thattakestheadjacencymatrix       𝐴 andthefeaturerepresentationmatrix                                                     matrix enhances the model’s performance markedly. As mentioned in
𝐹 as input. The encoder generates a        𝑑′-dimensional latent variable      𝑧𝑖                                         Results, the explanation for this – at first glance surprising – effect
for each node   𝑖∈ 𝑉, with  |𝑉|= 𝑛, where  𝑑′ ≤𝑛. Let 𝑍 = (𝑧𝑖) represent                                                  is the complementarity of the methods Node2Vec (which generates                 𝐹
allsuchlatentvariables.Theprobabilitytogenerateaparticularchoice                                                          independently of VGAE) and VGAE. The integration of              𝐹  evidently
of𝑍 is given by the formula                                                                                               leads to synergetic effects between Node2Vec and VGAE.
                      |𝑣|∏                                                                                                4. Result
𝑞(𝑍 ∣𝐴,𝐹) =                𝑞(𝑧𝑖∣𝐴,𝐹),                                                         (12)
                      𝑖=1                                                                                                 4.1. Dataset preparation
which assumes conditional independence between the              𝑧𝑖 given the
adjacencymatrix    𝐴 (thatis,thegraph)andtheNode2Vecbasedfeature                                                               We have utilized three categories of interaction datasets: human
representation   𝐹 of the graph. The probability       𝑟(𝑧𝑖|𝐴,𝐹) follows a nor-                                           protein–proteininteractomedata,SARS-CoV-2–hostproteininteraction
mal distribution,          (𝑧𝑖|𝜇𝑖,diag( 𝜎2𝑖)) where  𝜇𝑖and    diag( 𝜎2𝑖) parameterize                                    data, and drug–host interaction data.

S. Ray et al.
                          Table 2
                           Description of data sets.
                            Index      Dataset Category                Dataset                                                          #Edges         #Nodes
                            1            Human PPI                                 CCSB [29]                                                     13944                                         4303HPRD [27]                                                    39240                                         9617
                            2            SARS-CoV-2–Host PPI                       Gordon et al [11]                                         332               27 (#SARS-CoV-2)      332 (#Host)Dick et al [12]                                             261               6 (#SARS-CoV-2)        202 (#Host)
                                                                                  DrugBank (v4.3) [22]
                            3            Drug–target interaction                  ChEMBL [23]                                                      1788407      1307 (# Drug)            12134 (# Host-target)
                                                                                  Therapeutic Target Database (TTD) [24]
                                                                                  PharmGKB database
SARS-CoV-2-host interaction data                     .  We have taken SARS-CoV-2–host                                                  (𝑉,̃𝐸) where  𝑉 is the same set of molecules as before, while the edges
interaction information from two recent studies by Gordon et al. and                                                                    ̃𝐸 lackanyinteractionofthetypewearelookingfor,asjustdescribed.
Dick et al. [11,12]. In [11], 332 high confidence interactions be-                                                                     Note that removing             all  such edges creates a particularly challenging
tween SARS-CoV-2 and human proteins are predicted using affinity                                                                       scenario(incomparisonto,forexample,onlyremovingselectedsubsets
purification mass spectrometry (AP-MS). In [12], 261 high-confidence                                                                   of such edges).
interactions are identified using sequence-based PPI predictors (PIPE4                                                                       We then ran our pipeline on       ̃𝐺, yielding an adjacency matrix       ̃𝐴 and
& SPRINT).                                                                                                                             afeaturematrix    ̃𝐹 (resultingfromrunningNode2Vecon          ̃𝐺)fortraining
Drug–host interactome data                .  The drug–target interaction information                                                   the VGAE. We used the resulting decoder for predicting missing edges.
hasbeencollectedfromfivedatabases:DrugBankdatabase(v4.3)[22],                                                                          In the evaluation of this experiment, we focused on the edges that we
ChEMBL[23]database,TherapeuticTargetDatabase(TTD)[24],Phar-                                                                            had removed before, because these are known to be true.
mGKB database, and IUPHAR/BPS Guide to PHARMACOLOGY [25].                                                                                    Evaluatingtheresultingpredictionsconfirmedthatpredictingmiss-
Thetotalnumberofdrugsanddrug–hostinteractionsusedinthisstudy                                                                           ing edges by means of our pipeline operates at utmost performance
are   1309    and    1788407     , respectively.                                                                                       (ROC–AUC:           93.56 ± 0 .01   AP:    90.88 ± 0 .02   averaged across 100 runs);
                                                                                                                                       werecallthatweconsideredthemostchallengingscenarioconceivable
Thehumanprotein–proteininteractome                        .  Wehavebuiltacomprehen-                                                    (it is conceivable that performance rates increase when re-integrating
sive list of human PPIs from two datasets:                                                                                             existing edges, because the model profits from the additional struc-
     (1)CCSBhumanInteractomedatabaseconsistingof7000genes,and                                                                          ture provided during training). The purpose of these simulations is to
13944 high-quality binary interactions [26]                                                                                            point out that the following results are trustworthy; note that in the
     (2) The Human Protein Reference Database [27] which consists of                                                                   following we do make use of all the edges that we removed in the
8920 proteins and 53184 PPIs.                                                                                                          simulations described here, which provides the additional structure, as
     The summary of all the datasets is provided inTable2. The CMAP                                                                    just explained.
database [28] is used to annotate the drugs according to their usage                                                                         To choose the correct size of sampling node (          𝑁𝑆) in the decod-
with respect to different diseases.                                                                                                    ing stage of FastGAE, we tested the model performance for different
                                                                                                                                       numbers of   𝑁𝑆 and kept track of the corresponding performance (area
4.2. Advantages of including a feature matrix         𝐹                                                                                under the ROC curve (AUC), average precision (AP) score) and model
                                                                                                                                       training time) in the frame of a train-validation-test split at propor-
     For evaluating the effects of including a feature matrix            𝐹  that                                                       tions 8:1:1.Table1shows the performance of the model for sampled
reflects an alternative representation of the original adjacency matrix                                                                subgraph sizes    𝑁𝑆  =   7000, 5000, 3000, 2500 and 1000. For 5000
𝐴, we evaluated the performance of the model with and without                                                                          samplednodes,themodel’sperformanceissufficientlygoodconcerning
including  𝐹.Fig.2shows the average performance of the model on                                                                        its training time and validation-AUC and -AP score. The average test
validation sets with and without        𝐹 as input for different numbers of                                                            ROC–AUC and AP score of the model for           𝑁𝑠 =  5000 are       88.53 ± 0 .03
sampling nodes. The average AUC, and AP scores are reported for 50                                                                     and    84.44±0 .04 .
complete runs. FromFig.2, it is evident that including                𝐹 as a feature
matrix enhances the model’s performance markedly. As mentioned in                                                                      4.4. Drug–CoV-host interaction prediction
Results, the explanation for this—at first glance surprising—effect is
the complementarity of the methods Node2Vec (which generates                 𝐹                                                               The overall number of possible links between drugs and CoV-host
independently of VGAE) and VGAE. The integration of              𝐹  evidently                                                          proteins amounts to 332                ×  1302 (CoV-host           ×  drugs). While many such
leads to synergetic effects between Node2Vec and VGAE.                                                                                 links make part of the network, the majority of such possible links
                                                                                                                                       does not make part of the network. We refer to all such links that do
4.3. Predicting missing links: Validation                                                                                              not make part of network as ‘‘non-edges’’. Any such ‘‘non-edge’’ that is
                                                                                                                                       predicted to exist at sufficiently high probability is a prediction for an
     Let 𝐺 = (𝑉,𝐸) be the entire drug–human–virus interaction network                                                                  interactionofadrugwithaCoV-hostprotein.Wetrainthemodelonthe
in the following, where nodes       𝑣  ∈ 𝑉  represent molecules (drugs,                                                                whole network reflected by adjacency matrix           𝐴 and feature matrix     𝐹
proteins).  Edges           (𝑢,𝑣)   ∈  𝐸  between  molecules      𝑢,𝑣reflect  known                                                    (the latter computed by Node2Vec). The trained model is then applied
interactions where we are interested in the case of           𝑢being a drug and                                                        to the ‘‘non-edges’’ to discover the most probable missing drug–Cov–
𝑣being a human protein that was found to interact with SARS-CoV-2                                                                      host interactions.Fig.3(a), Panel-A shows the heatmap of probability
proteinsrecently[11,12].Thegoalofthisstudyistopredictsuchedges                                                                         scores between predicted drugs and CoV-host proteins. We identified
(𝑢,𝑣) to exist with great probability, despite not making explicit part of                                                             692links,connecting92drugswith78CoV-hostproteinswhoseproba-
the network   𝐺.                                                                                                                       bilitytoexistexceededathresholdof0.8.Aswewillillustratefurtherin
     For approving and corroborating the quality of the predictions                                                                    thefollowing,thepredictedCoV-hostproteinsareinvolvedindifferent
of such virtual, non-explicit edges in the following, we designed the                                                                  pathways that are crucial for viral infections (Supplementary Table 3).
following canonical experiment. We first removed                             all existing edges                                        We further used a Weighted bipartite clustering algorithm [30] for an-
(𝑢,𝑣) ∈ 𝐸 between drugs and SARS-CoV-2-associated human proteins                                                                       alyzingthebipartitegraphwhosepartitionsconsistofdrugsontheone
(‘‘CoV-host proteins’’) from      𝐺, resulting in an interaction network        ̃𝐺 =                                                   hand, and CoV-host proteins on the other hand further. Application of

S. Ray et al.
                                                                                                   Fig. 3.     Drug–CoV-host predicted interaction.
the algorithm results in 4 bipartite modules (Panel-A                           Fig.  3(a)): B1 (11                                       several antibiotics (Anisomycin and Midecamycin in B1; Puromycin,
drugs,28CoV-host),B2(4drugs,41CoV-host),B3(71rugsand4CoV-                                                                                 Demeclocycline, Dirithromycin, Geldanamycin, and Chlortetracycline
host), and B4 (6 drugs and 5 CoV-host). Panels B–D in                              Figs.   3(b),  3(c),                                   in B3), anti-cancer drugs (Doxorubicin, Camptothecin) and other drugs
and   3(d)showthe network diagram offourbipartite modules. Of note,                                                                       (Lobeline and Ambroxol in B3) have a variety of therapeutic uses,

S. Ray et al.
                                                                                                                     Fig. 3.     (continued     ).
including bronchitis, pneumonia, and respiratory tract infections [                                 31 ]                                perform a weighted clustering (using clusterONE [                           32 ]) on this net-
which provides further evidence of the reasonability of our results.                                                                    work, resulting in several quasi-bicliques (shown in Panels B–E of
      High-confidence interactions (exceeding a probability threshold of                                                                Fig.  4(b)  )
0.9) are further shown in              Fig.   4(a)  , Panel A. To highlight some re-                                                          We matched our predicted drugs with the drug list recently pub-
purposable drug combination and their predicted CoV-host target, we                                                                     lished by Zhou et al. [          9] and found six common drugs: Mesalazine,

S. Ray et al.
Fig. 4.     Predicted interactions for probability threshold: 0.9.  (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this
article.)
Vinblastine,Menadione,Medrysone,Fulvestrant,andApigenin.Among                                                                               4.5. Repurposable drugs for SARS-CoV-2
them, Apigenin has a known effect on the antiviral activity together                                                                              Here we showcased some repurposable drugs that have promi-
with quercetin, rutin, and other flavonoids [                      33 ]. Mesalazine is also                                                 nent literature-reported antiviral evidence, especially for two other
proventobeextremelyeffectiveinthetreatmentofotherviraldiseases                                                                              coronaviruses SARS-CoV and MERS-CoV. Some drugs are directly as-
like influenza A/H5N1 virus [                34 ].                                                                                          sociated with the treatment of SARS-CoV-2 as well. The details of the

S. Ray et al.
predicted drugs and their uses are given in supplementary text and                                                                      5. Discussion
Supplementary Table-2.
Topoisomerase inhibitors              .  Topoisomerase Inhibitors such as Camp-                                                               In this work, we have successfully generated a list of candidate
tothecin, Daunorubicin, Doxorubicin, Irinotecan and Mitoxantrone are                                                                    drugs that can be repurposed to counteract SARS-CoV-2 infections. As
in the list of predicted drugs. The anticancer drug camptothecin (CPT)                                                                  novelties, we have integrated the most recently published SARS-CoV-
and its derivative Irinotecan have a potential role in antiviral activ-                                                                 2 interaction data into well-established network resources to raise an
ity [35]. Daunorubicin (DNR) is demonstrated as an inhibitor of HIV-1                                                                   encompassing network putting drugs, viral and human proteins into a
                                                                                                                                        comprehensivecontext.Further,toexploitthisnovelnetwork,wehave
virus replication in human host cells [36]. The anticancer antibiotic                                                                   made use of the most recent and advanced deep learning methodology
Doxorubicin was previously identified as a selective inhibitor of                                 in-                                   that addresses learning and exploiting network data, establishing an-
vitro  DengueandYellowFevervirus[37].Mitoxantroneshowsantiviral                                                                         other novelty. Experiments validate that our predictions are of utmost
activityagainstthehumanherpessimplexvirus(HSV1)byreducingthe                                                                            accuracy, which confirms the quality of the novel interactions between
transcription of viral genes in many human cells that are essential for                                                                 drugs and virus related proteins that we suggest.
DNA synthesis [38].                                                                                                                           The recent publication of two novel SARS-CoV-2–human protein
Histone  deacetylases  inhibitors  (HDACi)                       .  Our  predicted  drug  list                                          interaction resources [11,12] has unlocked enormous possibilities in
(supplementarytable-2)containstwoHDACi:ScriptaidandVorinostat.                                                                          studying the mechanisms that drive virulence and pathogenicity of
Both drugs can be used to achieve latency reversal in the HIV-1 virus                                                                   SARS-CoV-2. Only now sufficiently systematical and accurate, AI sup-
safelyandrepeatedly[39].AsymptomaticpatientsinfectedwithSARS-                                                                           ported drug repurposing strategies for fighting COVID-19 have become
CoV-2 are of significant concern as they are more vulnerable to infect                                                                  conceivable.
large population than symptomatic patients. Moreover, in most cases                                                                           To the best of our knowledge, we have raised such a systematic
(99-percentile), patients develop symptoms after an average of 5–14                                                                     approach of utmost accuracy with an advanced AI boosted model for
days, which is longer than the incubation period of SARS and MERS.                                                                      the first time. We have integrated the new SARS-CoV-2 protein inter-
To this end, HDACi may serve as good candidates for recognizing and                                                                     actiondataintowellestablished,carefullycuratedresources,capturing
clearing the cells in which SARS-CoV-2 latency has been reversed.                                                                       hundreds of thousands of approved interfaces between molecules that
                                                                                                                                        reflectdrugsorhumanproteins.Asaresult,wehavebeenabletoraise
HSP inhibitor        .  Heat shock protein 90 (HSP) is described as a crucial                                                           a comprehensive drug–human–SARS-CoV-2 network that reflects the
host factor in the life cycle of several viruses that includes an entry in                                                              latest state of the art with respect to the interactions that it displays.
the cell, nuclear import, transcription, and replication [40,41]. HSP90                                                                       This new network already establishes a novel resource in its own
is also shown to be an essential factor for SARS-CoV-2 envelop (E)                                                                      right. For exploiting it, we have opted for using variational graph
protein [42]. In [43], HSP90 is described as a promising target for                                                                     autoencoders (VGAE), which have been most recently presented as the
antiviral drugs. The predicted drug list contains three HSP inhibitors:                                                                 state of the art in analyzing large network datasets, and which allow
Tanespimycin,Geldanamycin,anditsderivativeAlvespimycin.Thefirst                                                                         to predict links that are missing in the network whose structure and
two have a substantial effect in inhibiting the replication of Herpes                                                                   the rules that underlie the interplay of links it has ‘‘learned’’ at utmost
Simplex Virus and Human enterovirus 71 (EV71), respectively. Re-                                                                        accuracy. Note that FastGAE, the practical implementation of VGAEs
cently in [44], Geldanamycin and its derivatives are proposed to be                                                                     that enables us to analyze networks of sufficiently large sizes, was
an effective drug in the treatment of COVID-19.                                                                                         presented only very recently [20] as well, pointing out the timeliness
                                                                                                                                        of our study yet again.
Antimalarial agent, DNA-inhibitor, DNA methyltransferase/synthesis                                                                            Simulation experiments, reflecting scenarios where links known to
inhibitor    .  Inhibiting DNA synthesis during viral replication is one of                                                             exist are predicted upon their artificial removal, have pointed out that
the critical steps in disrupting the viral infection. The list of predicted                                                             our approach operates with utmost accuracy.
drugscontainssixsuchsmallmolecules/drugs,viz.,Niclosamide,Azac-                                                                               Encouraged by these simulations, we predicted links to be missing
itidine, Anisomycin, Novobiocin, Primaquine, Menadione, and Metron-                                                                     without prior removal of links. Our predictions have revealed 692
idazole(seesupplementarytext).RecentlyHydroxychloroquine(HCQ),                                                                          highconfidenceinteractionsbetweenhumanproteinsthatareessential
aderivativeofCQ,hasbeenevaluatedtoefficientlyinhibitSARS-CoV-2                                                                          for the virus on the one hand and 92 drugs on the other hand; note
infection     in vitro   [45]. Therefore, another anti-malarial aminoquino-                                                             that we had been emphasizing host-directed therapy (HDT) strategies,
lin drug Primaquine may also contribute to the attenuation of the                                                                       which explains why we have focused on the type of interaction just
inflammatory response of COVID-19 patients. Primaquine is already                                                                       described.WerecallthatthecombinationofHDTanddrugrepurposing
establishedtobeeffectiveinthetreatmentofPneumocystis-pneumonia                                                                          promisestoyielddrugsthatnotonlyenableacceleratedusage,butalso
(PCP) [46].                                                                                                                             guarantee a sufficiently high degree of sustainability.
                                                                                                                                              We further systematically categorized the 92 repurposable drugs
Cardiac glycosides ATPase inhibitor                     .  The predicted list of drugs con-                                             into 70 categories based on their domains of application and molecu-
tains three cardiac glycosides ATPase inhibitors: Digoxin, Digitoxi-                                                                    lar mechanism. According to this, we identified and highlighted sev-
genin, and Ouabain. These drugs have been reported to be effective                                                                      eral  drugs  that  target  host  proteins  that  the  virus  needs  to  enter
against different viruses such as herpes simplex, influenza, chikun-                                                                    and subsequently hijack human cells. One such example is Capto-
gunya, coronavirus, and respiratory syncytial virus [47].                                                                               pril, which directly inhibits the production of Angiotensin-Converting
                                                                                                                                        Enzyme-2 (ACE-2), in turn already known to be a crucial host factor
Mg132,resveratrolandcaptopril                   .  MG132,aproteasomalinhibitor,is                                                       for SARS-CoV-2. Further, we identified Primaquine, as an antimalaria
astronginhibitorofSARS-CoVreplicationinearlystage[48].Resvera-                                                                          drug used to prevent the Malaria and also Pneumocystis pneumonia
trolhasalsobeendemonstratedtobeasignificantinhibitorMERS-CoV                                                                            (PCP) relapses, because it interacts with the TIM complex TIMM29
infection [49]. Another drug Captopril is known as Angiotensin II                                                                       and ALG11. Moreover, we have highlighted drugs that act as DNA
receptor blockers (ARB), which directly inhibits the production of                                                                      replication inhibitor (Niclosamide, Anisomycin), glucocorticoid recep-
angiotensin II. In [50], Angiotensin-converting enzyme 2 (ACE2) is                                                                      tor agonists (Medrysone), ATPase inhibitors (Digitoxigenin, Digoxin),
demonstrated as the binding site for SARS-CoV-2. So Angiotensin II                                                                      topoisomerase inhibitors (Camptothecin, Irinotecan), and proteasomal
receptorblockers(ARB)maybegoodcandidatestouseinthetentative                                                                             inhibitors (MG-132). Note that some drugs are known to have rather
treatment for SARS-CoV-2 infections                                                                                                     severe side effects from their original use (Doxorubicin, Vinblastine),

S. Ray et al.
but the disrupting effects of their short-term usage in severe COVID-19                                                                                             [12]Dick  K,  Biggar  KKG,  R.  J.  Comprehensive  prediction  of  the  SARS-CoV-2  vs.
infections may mean sufficient compensation.                                                                                                                                   Human interactome using PIPE4, SPRINT, and PIPE-sites. 2020,http://dx.doi.
       Insummary,wehavecompiledalistofdrugs,whenrepurposed,are                                                                                                                 org/10.5683/SP2/JZ77XA.
of great potential in the fight against the COVID-19 pandemic, where                                                                                                [13]Sadegh S, Matschinske J, Blumenthal DB, Galindez G, Kacprowski T, List M,
                                                                                                                                                                               Nasirigerdeh  R,  Oubounyt  M,  Pichlmair  A,  Rose  TD,  et  al.  Exploring  the
therapy options are still urgently needed. Our list of predicted drugs                                                                                                         SARS-CoV-2 virus-host-drug interactome for drug repurposing. Nature Commun
suggestsbothoptionsthathadbeenidentifiedandthoroughlydiscussed                                                                                                                 2020;11(1):1–9.
before, as well as new opportunities that had not been pointed out                                                                                                  [14]Fiscon G, Conte F, Farina L, Paci P. SAveRUNNER: a network-based algorithm
earlier.Thelatterclassofdrugsmayoffervaluablechancesforpursuing                                                                                                                for  drug  repurposing  and  its  application  to  COVID-19.  PLoS  Comput  Biol
                                                                                                                                                                               2021;17(2):e1008686.
new therapy strategies against COVID-19.                                                                                                                            [15]Gysi DM, Do Valle Í, Zitnik M, Ameli A, Gan X, Varol O, Ghiassian SD, Patten J,
                                                                                                                                                                               Davey  RA,  Loscalzo  J,  et  al.  Network  medicine  framework  for  identifying
Declaration of competing interest                                                                                                                                              drug-repurposing opportunities for COVID-19. Proc Natl Acad Sci 2021;118(19).
                                                                                                                                                                    [16]Ray S, Alberuni S, Maulik U. Computational prediction of HCV-human protein-
       The authors declare that they have no known competing finan-                                                                                                            protein interaction via topological analysis of HCV infected PPI modules. IEEE
                                                                                                                                                                               Transactions on NanoBioscience 2018;17(1):55–61.
cial interests or personal relationships that could have appeared to                                                                                                [17]Ray S, Bandyopadhyay S. A NMF based approach for integrating multiple data
influence the work reported in this paper.                                                                                                                                     sources to predict HIV-1–human PPIs. BMC bioinformatics 2016;17(1):1–13.
                                                                                                                                                                    [18]Grover A, Leskovec J. Node2vec: Scalable feature learning for networks. In:
Availability                                                                                                                                                                   Proceedings of the 22nd ACM SIGKDD international conference on knowledge
                                                                                                                                                                               discovery and data mining. 2016, p. 855–64.
                                                                                                                                                                    [19]Kipf  TN,  Welling  M.  Variational  graph  auto-encoders.  2016,  arXiv  preprint
       All codes and datasets are given in the github link:https://github.                                                                                                     arXiv:1611.07308.
com/sumantaray/Covid19.                                                                                                                                             [20]Salha G, Hennequin R, Remy J-B, Moussallam M, Vazirgiannis M. FastGAE: Fast,
                                                                                                                                                                               scalable and effective graph autoencoders with stochastic subgraph decoding.
                                                                                                                                                                               2020, ArXiv Preprint.
Acknowledgment                                                                                                                                                      [21]Mikolov T, Sutskever I, Chen K, Corrado GS, Dean J. Distributed representations
                                                                                                                                                                               of  words  and  phrases  and  their  compositionality.  In:  Advances  in  neural
       SB acknowledges support from J.C. Bose Fellowship [SB/S1/JCB-                                                                                                           information processing systems. 2013, p. 3111–9.
033/2016 to S.B.] by the DST, Govt. of India; SyMeCProject grant                                                                                                    [22]Law V, Knox C, Djoumbou Y, Jewison T, Guo AC, Liu Y, Maciejewski A, Arndt D,
[BT/Med-II/NIBMG/SyMeC/2014/Vol. II] given to the Indian Statisti-                                                                                                             Wilson M, Neveu V, et al. DrugBank 4.0: shedding new light on drug metabolism.
                                                                                                                                                                               Nucleic Acids Res 2014;42(D1):D1091–7.
calInstitutebytheDepartmentofBiotechnology(DBT),Govt.ofIndia.                                                                                                       [23]Gaulton  A,  Bellis  LJ,  Bento  AP,  Chambers  J,  Davies  M,  Hersey  A,  Light  Y,
Govt. of India; Inspire DST Project. SR acknowledges support from                                                                                                              McGlinchey  S,  Michalovich  D,  Al-Lazikani  B,  et  al.  ChEMBL:  a  large-scale
SERBTAREgrant(FileNoTAR/2021/000072)fromSERB(DST),Govt.                                                                                                                        bioactivity database for drug discovery. Nucleic Acids Res 2012;40(D1):D1100–7.
of India. AM acknowledges support from SERB-MATRICS grant (File                                                                                                     [24]Yang  H,  Qin  C,  Li  YH,  Tao  L,  Zhou  J,  Yu  CY,  Xu  F,  Chen  Z,  Zhu  F,
No. MTR/2020/326) of SERB (DST), Govt. of India; Research project                                                                                                              Chen YZ. Therapeutic target database update 2016: enriched resource for bench
                                                                                                                                                                               to clinical drug target and targeted pathway information. Nucleic Acids Res
grant (0083/RND/ET/KU10 /Jan-2021/1/1) from DST&BT, Govt. of                                                                                                                   2016;44(D1):D1069–74.
West Bengal, India.                                                                                                                                                 [25]Pawson AJ, Sharman JL, Benson HE, Faccenda E, Alexander SP, Buneman OP,
                                                                                                                                                                               Davenport AP, McGrath JC, Peters JA, Southan C, et al. The IUPHAR/BPS guide
Appendix A. Supplementary data                                                                                                                                                 to PHARMACOLOGY: an expert-driven knowledgebase of drug targets and their
                                                                                                                                                                               ligands. Nucleic Acids Res 2014;42(D1):D1098–106.
                                                                                                                                                                    [26]Rual  J-F,  Venkatesan  K,  Hao  T,  Hirozane-Kishikawa  T,  Dricot  A,  Li  N,
       Supplementary material related to this article can be found online                                                                                                      Berriz  GF,  Gibbons  FD,  Dreze  M,  Ayivi-Guedehoussou  N,  et  al.  Towards  a
athttps://doi.org/10.1016/j.artmed.2022.102418.                                                                                                                                proteome-scale map of the human protein–protein interaction network. Nature
                                                                                                                                                                               2005;437(7062):1173–8.
                                                                                                                                                                    [27]Peri  S,  Navarro  JD,  Amanchy  R,  Kristiansen  TZ,  Jonnalagadda  CK,  Suren-
References                                                                                                                                                                     dranath V, Niranjan V, Muthusamy B, Gandhi T, Gronborg M, et al. Development
                                                                                                                                                                               of  human  protein  reference  database  as  an  initial  platform  for  approaching
   [1]Wu F, et al. A new coronavirus associated with human respiratory disease in                                                                                              systems biology in humans. Genome Res 2003;13(10):2363–71.
           China. Nature 2020;579(7798):265–9.                                                                                                                      [28]Subramanian A, Narayan R, Corsello SM, Peck DD, Natoli TE, Lu X, Gould J,
   [2]Forst  CV.  Host–pathogen  systems  biology.  In:  Infectious  disease  informatics.                                                                                     Davis JF, Tubelli AA, Asiedu JK, et al. A next generation connectivity map:
           Springer; 2010, p. 123–47.                                                                                                                                          L1000 platform and the first 1,000,000 profiles. Cell 2017;171(6):1437–52.
   [3]Ray S, Lall S, Bandyopadhyay S. A deep integrated framework for predicting                                                                                    [29]Yu H, Tardivo L, Tam S, Weiner E, Gebreab F, Fan C, Svrzikapa N, Hirozane-
           SARS-CoV2–human protein-protein interaction. IEEE Transactions on Emerging                                                                                          Kishikawa T, Rietman E, Yang X, et al. Next-generation sequencing to generate
           Topics in Computational Intelligence 2022.                                                                                                                          interactome datasets. Nature Methods 2011;8(6):478.
   [4]Kaufmann SH, Dorhoi A, Hotchkiss RS, Bartenschlager R. Host-directed therapies                                                                                [30]Beckett SJ. Improved community detection in weighted bipartite networks. R
           for bacterial and viral infections. Nat Rev Drug Discov 2018;17(1):35.                                                                                              Soc Open Sci 2016;3(1):140536.
   [5]de Chassey B, Meyniel-Schicklin L, Aublin-Gex A, André P, Lotteau V. New hori-                                                                                [31]Wishart DS, Knox C, Guo AC, Shrivastava S, Hassanali M, Stothard P, Chang Z,
           zons for antiviral drug discovery from virus–host protein interaction networks.                                                                                     Woolsey J. DrugBank: a comprehensive resource for in silico drug discovery and
           Curr Opin Virol 2012;2(5):606–13.                                                                                                                                   exploration. Nucleic Acids Res 2006;34(suppl_1):D668–72.
   [6]Doolittle JM, Gomez SM. Mapping protein interactions between dengue virus                                                                                     [32]Nepusz  T,  Yu  H,  Paccanaro  A.  Detecting  overlapping  protein  complexes  in
           and its human and insect hosts. PLoS Negl Trop Dis 2011;5(2).                                                                                                       protein-protein interaction networks. Nature Methods 2012;9(5):471.
   [7]Bandyopadhyay  S,  Ray  S,  Mukhopadhyay  A,  Maulik  U.  A  review  of  in  sil-                                                                             [33]Salehi  B,  Venditti  A,  Sharifi-Rad  M,  Kregiel  D,  Sharifi-Rad  J,  Durazzo  A,
           ico  approaches  for  analysis  and  prediction  of  HIV-1-human  protein–protein                                                                                   Lucarini M, Santini A, Souto EB, Novellino E, et al. The therapeutic potential of
           interactions. Brief Bioinform 2015;16(5):830–51.                                                                                                                    apigenin. Int J Mol Sci 2019;20(6):1305.
   [8]Cao H, Zhang Y, Zhao J, Zhu L, Wang Y, Li J, Feng Y-M, Zhang N. Prediction of                                                                                 [34]Zheng B-J, Chan K-W, Lin Y-P, Zhao G-Y, Chan C, Zhang H-J, Chen H-L, Wong SS,
           the Ebola virus infection related human genes using protein-protein interaction                                                                                     Lau SK, Woo PC, et al. Delayed antiviral plus immunomodulator treatment still
           network. Combin Chem High Throughput Screen 2017;20(7):638–46.                                                                                                      reduces mortality in mice infected by high inoculum of influenza A/H5N1 virus.
   [9]Zhou  Y,  Hou  Y,  Shen  J,  Huang  Y,  Martin  W,  Cheng  F.  Network-based                                                                                             Proc Natl Acad Sci 2008;105(23):8091–6.
           drug  repurposing  for  novel  coronavirus  2019-nCoV/SARS-CoV-2.  Cell  Discov                                                                          [35]Horwitz  SB,  Chang  C-K,  Grollman  AP.  Antiviral  action  of  camptothecin.
           2020;6(1):1–18.                                                                                                                                                     Antimicrob Agents Chemother 1972;2(5):395–401.
[10]Li X, Yu J, Zhang Z, Ren J, Peluffo AE, Zhang W, Zhao Y, Yan K, Cohen D,                                                                                        [36]Filion L, Logan D, Gaudreault R, Izaguirre C. Inhibition of HIV-1 replication by
           Wang W. Network bioinformatics analysis provides insight into drug repurposing                                                                                      daunorubicin. Clin Investig Med. Med Clin Exp 1993;16(5):339–47.
           for COVID-2019. 2020, Preprints.                                                                                                                         [37]Kaptein SJ, De Burghgraeve T, Froeyen M, Pastorino B, Alen MM, Mondotte JA,
[11]Gordon DE, Jang GM, Bouhaddou M, Xu J, Obernier K, White KM, O’Meara MJ,                                                                                                   Herdewijn P, Jacobs M, De Lamballerie X, Schols D, et al. A derivate of the
           Rezelj VV, Guo JZ, Swaney DL, et al. A SARS-CoV-2 protein interaction map                                                                                           antibiotic doxorubicin is a selective inhibitor of dengue and yellow fever virus
           reveals targets for drug repurposing. Nature 2020;1–13.                                                                                                             replication in vitro. Antimicrob Agents Chemother 2010;54(12):5269–80.

S. Ray et al.
[38]Huang Q, Hou J, Yang P, Yan J, Yu X, Zhuo Y, He S, Xu F. Antiviral activity of                                                                                         [45]Liu  J,  Cao  R,  Xu  M,  Wang  X,  Zhang  H,  Hu  H,  Li  Y,  Hu  Z,  Zhong  W,
            mitoxantrone dihydrochloride against human herpes simplex virus mediated by                                                                                                Wang M. Hydroxychloroquine, a less toxic derivative of chloroquine, is effective
            suppression of the viral immediate early genes. BMC Microbiol 2019;19(1):274.                                                                                              in inhibiting SARS-CoV-2 infection in vitro. Cell Discov 2020;6(1):1–4.
[39]Archin NM, Kirchherr JL, Sung JA, Clutton G, Sholtis K, Xu Y, Allard B, Stuelke E,                                                                                     [46]Vöhringer  H-F,  Arastéh  K.  Pharmacokinetic  optimisation  in  the  treatment  of
            Kashuba AD, Kuruc JD, et al. Interval dosing with the HDAC inhibitor vorinostat                                                                                            Pneumocystis carinii pneumonia. Clin Pharmacokinet 1993;24(5):388–412.
            effectively reverses HIV latency. J Clin Investig 2017;127(8):3126–35.                                                                                         [47]Amarelle  L,  Lecuona  E.  The  antiviral  effects  of  na,  K-ATPase  inhibition:  A
[40]Ju H-Q, Xiang Y-F, Xin B-J, Pei Y, Lu J-X, Wang Q-L, Xia M, Qian C-W, Ren Z,                                                                                                       minireview. Int J Mol Sci 2018;19(8):2154.
            Wang S-Y, et al. Synthesis and in vitro anti-HSV-1 activity of a novel Hsp90                                                                                   [48]Schneider M, Ackermann K, Stuart M, Wex C, Protzer U, Schätzl HM, Gilch S.
            inhibitor BJ-B11. Bioorg Med Chem Lett 2011;21(6):1675–7.                                                                                                                  Severe acute respiratory syndrome coronavirus replication is severely impaired
[41]Shim HY, Quan X, Yi Y-S, Jung G. Heat shock protein 90 facilitates formation                                                                                                       by  MG132  due  to  proteasome-independent  inhibition  of  M-calpain.  J  Virol
            of the HBV capsid via interacting with the HBV core protein dimers. Virology                                                                                               2012;86(18):10112–22.
            2011;410(1):161–9.                                                                                                                                             [49]Lin S-C, Ho C-T, Chuo W-H, Li S, Wang TT, Lin C-C. Effective inhibition of
[42]DeDiego ML, Nieto-Torres JL, Jiménez-Guardeño JM, Regla-Nava JA, Alvarez E,                                                                                                        MERS-CoV infection by resveratrol. BMC Infect Dis 2017;17(1):144.
            Oliveros JC, Zhao J, Fett C, Perlman S, Enjuanes L. Severe acute respiratory syn-                                                                              [50]Shang J, Ye G, Shi K, Wan Y, Luo C, Aihara H, Geng Q, Auerbach A, Li F.
            drome coronavirus envelope protein regulates cell stress response and apoptosis.                                                                                           Structural basis of receptor recognition by SARS-CoV-2. Nature 2020;1–4.
            PLoS Pathogens 2011;7(10).
[43]Wang Y, Jin F, Wang R, Li F, Wu Y, Kitazato K, Wang Y. HSP90: a promising
            broad-spectrum antiviral drug target. Arch Virol 2017;162(11):3269–82.
[44]Sultan I, Howard S, Tbakhi A. Drug repositioning suggests a role for the heat
            shock protein 90 inhibitor geldanamycin in treating COVID-19 infection. 2020,
            ArXiv.

