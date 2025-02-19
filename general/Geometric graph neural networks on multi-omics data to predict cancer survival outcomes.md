             HHS Public Access
             Author manuscript
             Comput Biol Med. Author manuscript; available in PMC 2023 November 11.
Published in final edited form as:
 Comput Biol Med. 2023 September ; 163: 107117. doi:10.1016/j.compbiomed.2023.107117.
Geometric graph neural networks on multi-omics data to predict
cancer survival outcomes
*Corresponding author at: Department of Applied Mathematics & Statistics, Stony Brook University, NY, USA.
allen.tannenbaum@stonybrook.edu (A. Tannenbaum).
Declaration of competing interest
Larry Norton discloses the following relationships and financial interests:
     •        Agenus Inc.
              Provision of Services
     •        American Society of Clinical Oncology (ASCO)
              Provision of Services (uncompensated)
     •        Breast Cancer Research Foundation
              Provision of Services (uncompensated)
     •        Celgene
              Provision of Services
     •        Codagenix, Inc
              Ownership / Equity Interests; Provision of Services
     •        Cold Spring Harbor Laboratory
              Provision of Services
     •        Cure Breast Cancer Foundation
              Intellectual Property Rights; Provision of Services (uncompensated)
     •        Immix Biopharma, Inc.
              Ownership / Equity Interests; Provision of Services
     •        Martell Diagnostic Laboratories, Inc.
              Ownership / Equity Interests
     •        NewStem Ltd.
              Provision of Services (uncompensated)
     •        QLS Advisors, LLC
              Provision of Services
     •        Samus Therapeutics LLC
              Fiduciary Role/Position; Ownership / Equity Interests; Provision of Services (uncompensated)
     •        Springer Nature Limited
              Provision of Services (uncompensated)
     •        Translational Breast Cancer Research Consortium
              Provision of Services (uncompensated)
     •        U.S. Department of Justice
              Provision of Services (uncompensated)

Zhu et al.                                                                                                                                                                                                    Page 2
   Jiening Zhua       , Jung Hun Ohb         , Anish K. Simhalb        , Rena Elkinb      , Larry Nortonc, Joseph O.
   Deasyb     , Allen Tannenbauma,d,*
   aDepartment of Applied Mathematics & Statistics, Stony Brook University, NY, USA
   bDepartment of Medical Physics, Memorial Sloan Kettering Cancer Center, NY, USA
   cDepartment of Medicine, Memorial Sloan Kettering Cancer Center, NY, USA
   dDepartment of Computer Science, Stony Brook University, NY, USA
   Abstract
         The advance of sequencing technologies has enabled a thorough molecular characterization of the
         genome in human cancers. To improve patient prognosis predictions and subsequent treatment
         strategies, it is imperative to develop advanced computational methods to analyze large-scale,
         high-dimensional genomic data. However, traditional machine learning methods face a challenge
         in handling the high-dimensional, low-sample size problem that is shown in most genomic
         data sets. To address this, our group has developed geometric network analysis techniques on
         multi-omics data in connection with prior biological knowledge derived from protein-protein
         interactions (PPIs) or pathways. Geometric features obtained from the genomic network, such
         as Ollivier-Ricci curvature and the invariant measure of the associated Markov chain, have been
         shown to be predictive of survival outcomes in various cancers. In this study, we propose a
         novel supervised deep learning method called geometric graph neural network (GGNN) that
         incorporates such geometric features into deep learning for enhanced predictive power and
         interpretability. More specifically, we utilize a state-of-the-art graph neural network with sparse
         connections between the hidden layers based on known biology of the PPI network and pathway
         information. Geometric features along with multi-omics data are then incorporated into the
         corresponding layers. The proposed approach utilizes a local-global principle in such a manner
         that highly predictive features are selected at the front layers and fed directly to the last layer
         for multivariable Cox proportional-hazards regression modeling. The method was applied to multi-
         omics data from the CoMMpass study of multiple myeloma and ten major cancers in The Cancer
         Genome Atlas (TCGA). In most experiments, our method showed superior predictive performance
         compared to other alternative methods.
   Keywords
         Geometric graph neural network; Ollivier-Ricci curvature; Survival prediction; Optimal transport;
         Multi-omics; TCGA
   1.   Introduction
                      The rapid advance of sequencing technologies has provided a comprehensive molecular
                      portrait of human cancers. In particular, large-scale cancer projects, such as The Cancer
                      Genome Atlas (TCGA), have enabled better understanding of cancer biology by producing
                      multi-omics data, including RNA-Seq gene expression profiles, copy number alteration
                      (CNA), and DNA methylation, that portray the human genome at different levels [1].
                      Each single-omic data can provide a complementary perspective of the complex biological
                       Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                    Page 3
                      system, facilitating new biological insights and the identification of integrative biomarkers.
                      A number of machine learning methods have been developed to analyze the large-scale
                      multi-omics data. However, the high-dimension, low-sample size (HDLSS) nature of the
                      genomic data poses challenges for typical machine learning algorithms. Therefore, there
                      is a strong need to develop advanced computational methods to not only improve patient
                      prognosis predictions but also to better understand the driving factors for treatment.
                      Geometric techniques, applied to multi-omics data in connection with a protein-protein
                      interaction (PPI) network, are well suited for HDLSS data analysis and have demonstrated
                      their potential to inform on the molecular basis of cancer. Mathematically defined quantities
                      that describe the network geometry provide a different perspective to understand biological
                      mechanisms. The non-parametric geometric analyses, such as Ollivier-Ricci curvature and
                      the invariant measure associated with the underlying Markov chain, developed by our group
                      have been proven to be successful for predicting survival outcomes and understanding the
                      underlying biological mechanisms in multiple cancers. Weistuch et al. explored invariant
                      measures on RNA-Seq data of triple-negative breast cancer, which identified known driving
                      interactions including TP53/SRC and BRCA1/PTPN11 [2]. Elkin et al. employed the total
                      curvature to investigate survival of patients with high-grade serous carcinoma of the ovary
                      treated with immune checkpoint inhibitors, which outperformed other commonly used
                      metrics like tumor mutational burden, fraction of genome altered, and large-scale state
                      transitions [3]. Simhal et al. used Ollivier-Ricci curvature as a profile for patient clustering,
                      which identified poor prognosis subtypes in multiple myeloma that are associated with
                      specific genomic features including CCND1 and MAF/MAFB translocations [4]. Zhu et
                      al. extended Ollivier-Ricci curvature for integrated multi-omics data analysis employing a
                      multi-layer vector-valued graph [5].
                      Recently, deep learning has demonstrated a powerful capability in a wide range of fields
                      including computer vision, bioinformatics, nature language processing, and recommendation
                      system [6–    10]. In bioinformatics, many deep learning-based methods have been developed
                      for cancer survival analysis. For example, Oh et al. proposed an interpretable convolutional
                      neural network called PathCNN for survival prediction and pathway analysis, using a newly
                      defined pathway image [7]. Simon et al. proposed a regularized Cox proportional-hazards
                      model with L1 and L2 elastic net penalties (Cox-EN) [11]. Yousefi et al. proposed a
                      Bayesian optimized deep survival modeling method called SurvivalNet that enables the
                      interpretation of deep survival models [12]. Lee et al. introduced a deep learning-based
                      autoencoder to integrate multi-omics data for survival analysis in lung cancer (Cox-AE)
                      [13]. Li et al. developed a similar autoencoder-based multi-omics integration technique
                      that concatenates the hidden features learned from multi-omics data and demonstrated its
                      improved overall survival prediction in breast cancer [14]. Chai et al. proposed a denoising
                      autoencoder algorithm called DCAP and its extension DCAP-XGB in connection with
                      XGboost to obtain robust representative features from multi-omics data [15].
                      A graph neural network (GNN) is a class of artificial neural networks for processing data
                      represented as graphs [16], rapidly extending its application to various graph problems
                      in biochemistry, cyber security, and social networks [17–                20]. Generalized from the grid-
                      structured convolution, there are two main streams of GNN: spectral and non-spectral.
                        Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                    Page 4
                       Spectral approaches define the convolution operation in the spectral/Fourier domain via the
                       graph Laplacian [21–        23], whereas non-spectral approaches define convolutions directly on
                       the graph, operating on groups of spatially close neighbors [24–                   26].
                       In this work, we propose a novel survival deep learning model called geometric graph
                       neural network (GGNN), a variant of GNN combining the aforementioned geometric
                       methods, which benefits from the flexibility provided by deep learning techniques while
                       still preserving much of the interpretability of the geometric analysis. Sparse connections
                       between the hidden layers were inspired by the known biology of the PPI network from the
                       Human Protein Reference Database (HPRD) [27] and pathway information from the Kyoto
                       Encyclopedia of Genes and Genomes (KEGG) database [28], supplemented with geometric
                       features that are fed into the corresponding layers of the network. This is likely to mitigate
                       overfitting and enhance interpretability of the model. The survival prediction is based on
                       a local–global principle, where highly predictive features are selected at the front layers
                       of the network and fed directly to the last layer to produce a multivariable Cox regression
                       model. The predictive power of the proposed method was compared with several alternative
                       methods in multiple cancers, using multi-omics data from the TCGA database as well as the
                       CoMMpass study of multiple myeloma.
   2.   Background
   2.1.   Graph neural network
                       Message passing is a non-spectral based approach of GNNs where a feature of a given node
                       is updated according to feature aggregation from the node’s neighbors. For each node                            u, the
                                                                                    u
                       message passing of its corresponding feature ℎk at layer k                 to layer k+1 can be written as
                       follows:
                                                                   ℎ k +1u    =ϕℎ ku,φ ∥ v ∈  N u ℎ kv,                                      (1)
                       where ϕ     and φ   are differentiable functions, Nu is the set of neighbors of node u, and
                       ∥ denotes the vector concatenation. φ            can be sum, mean or a sampling function to
                       accommodate the arbitrary graph structure where the number of neighbors of a node is
                       not determined. In this manner, message passing serves as a generalization of grid-based
                       convolution.
                       For a task where the graph topology is fixed, the message passing function can be defined
                       individually for each node:
                                                                       ℎ k +1u    =ϕ u∥ v ∈  N‾ u ℎ kv,                                      (2)
                       where N‾u is the extended neighbor set of u             which includes u itself.
                       More generally, the features on each layer may represent different components of a graph.
                       For example, features on nodes, edges or any subset of nodes can be considered as follows:
                                                                      ℎ k +1w    =ϕ w∥ v ∈  S w ℎ kv,                                        (3)
                         Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                    Page 5
                        where w     represents any component of the graph, Sw                 is a set determined by the inclusion
                        relation on the graph whose elements, v              , represent the graph components of the previous
                        layer.
    2.2.   Invariant measure
                        A Markov chain models a stochastic process where the probability of a given event depends
                        only on the state of the previous event [29]. The probability of each node interacting with its
                        neighbors is predefined by transition probabilities. In the gene network setting [30,31], we
                        set the probability of moving from a node i to another node j to be:
                                                                                   gj        i,
                                                                    μ ij  =     ∑    kigk  j                                                     (4)
                                                                               0           otherwise,
                        where gk>0 is the weight of node k               (Fig. 1).
                        Given an initial state π0, the state πt+1 at the next time step is obtained from the state πt at
                        the current time step via a transition matrix p, as follows:
                                                                              π t+1 =π tp .                                                      (5)
                        After an infinite number of time steps, the initial distribution will converge to an invariant
                        (stationary) distribution π        such that
                                                                                 π  =πp  .                                                       (6)
                        The invariant (stationary) distribution has a closed form solution:
                                                                          π i =  1              gk,
                                                                                Z gi  ∑                                                          (7)
                                                                                      k ∈  N  i
                        where Z     is the normalization factor to ensure π             is a probability distribution.
                        This Markov process on the gene network, which takes RNA-Seq gene expression,
                        CNA or methylation (as well potentially other data types) as nodal weights, mimics the
                        interactions among genes and the invariant distribution gives a distribution that represents
                        the information each gene has which includes not only its own value, but the interactions
                        with its neighbors.
    2.3.   Ollivier-Ricci curvature
                        Curvature is a basic geometric concept which describes the “shape” of a Riemannian
                        manifold [32,33]. The Ollivier-Ricci curvature is a natural analogue of Ricci curvature on
                        Riemannian manifolds to discrete a graph G  =                  V ,E     based on optimal transport [34]. For
                        two nodes i,j∈V         , the Ollivier-Ricci curvature κ          i,j   is defined as:
                         Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                    Page 6
                                                                       κ  i,j   = 1−     W     1μ i,μ j,                                          (8)
                                                                                           d  i,j
                        where W     1  ⋅, ⋅   is the Wasserstein distance on the graph, μi and μj are defined according to
                        Eq. (4), and d     i,j   is the w-hop graph distance between the nodes i and j, which is a standard
                        graph distance we previously used in [3]. On a discrete graph, the W1 Wasserstein distance
                        may be defined as follows:
                                                           W     1μ i,μ j=      min        ∑    d  p,q   π  p,q   ,                               (9)
                                                                            π  ∈  Π  μ i,μ jp,q
                        where Π      μi,μj  is the set of matrices        π  ∈ ℝ      +n ×  n ∣π 1                 , 1    is an n -dimensional
                                                                                             n =μ i,πT 1   n =μ j       n
                        vector of all ones, n=         V  , and d   p,q   is the same graph distance as that in (8). The objective
                        function describes the total cost of moving the source distribution μi to the target distribution
                        μj.
    3.   Methods
                        We constructed a novel deep learning architecture to build a predictive model of survival
                        outcomes in cancer employing the concept of biological network geometry based on
                        known biological information derived from the PPI network and pathways. Features which
                        represent the different level of biological or geometric information were used for the Cox
                        proportional-hazards model.
    3.1.   Main components
                              •        Dual-track update
                                       The first two feed-forward steps use the dual-track update, where the update
                                       functions are composed of two parts: geometry and machine learning. The
                                       update function may be written as follows:
                                                          ℎ k +1w    =α wGϕ wG∥ v ∈  S w ℎ v+α wM ϕ wM∥ v ∈  S w ℎ kv,                           (10)
                                       where ϕwG     is a nonlinear function of the given graph geometric property and ϕwM                        is
                                       a nonlinear function with the same input as the geometric part, but with learnable
                                       parameters. αwG      and αwM   are mixed parameters, which are also learnable. In this
                                       way, we combine the information from both geometry and machine learning.
                              •        Local-global feature extraction
                                       Many biomarkers and their interactions or relevant pathways are responsible for
                                       prognosis in cancer patients. To improve predictive power, we use geometric
                                       features as well as multi-omics data for model building, taking into account their
                                       non-linear interactions that are naturally identified in a training process of deep
                                       learning.
                         Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                    Page 7
   3.2.   Architecture
                       3.2.1.   Single-omic sub-network
                             1.       Gene Layer
                                      The gene layer is the input layer for original gene expression data                      X   with g
                                      genes of n     samples, i.e., X  =      x 1,…,   x n∈ ℝ   g.
                             2.       1-neighbor Layer
                                      The 1-neighbor layer is an input layer for invariant measure values for each gene
                                      and aggregates neighboring gene expression information from neighbors of each
                                      gene.
                                      The sparse connection between the gene and 1-neighbor layers is implemented
                                      via a binary adjacency matrix. From the PPI network, the binary adjacency
                                      matrix     A  1 ∈  B g ×  g is provided, where an element aij is one if genes i and j are
                                      connected; otherwise, it is zero.
                             3.       Edge Layer
                                      The edge layer is an input layer for pre-computed Ollivier-Ricci curvature values
                                      for each edge and aggregates information of neighbors of the two ends of each
                                      edge. The way we aggregate information is inspired by the definition of Ollivier-
                                      Ricci curvature, which considers the distributions around the two ends of an edge
                                      (Fig. 1).
                                      The sparse connection between the 1-neighbor and edge layers is given by a
                                      binary matrix (A  2 ∈  B g ×  e,e: number of edges), where an element aij is one if
                                      gene i is adjacent to either of the two ends of edge j.
                             4.       Pathway Layer
                                      The pathway layer has 186 nodes. Each node of the pathway layer represents
                                      a specific biological pathway. Pathway databases (e.g., KEGG) provide a set of
                                      genes that are involved in a specific pathway.
                                      The sparse connection between the edge and pathway layers is given by a binary
                                      matrix (A  3 ∈  B e ×  p,p: number of pathways), where an element aij is one if edge i
                                      belongs to pathway j.
                             5.       Feature Layer
                                      The feature layer consists of 50 nodes that represent features from all other
                                      layers mentioned above. To be specific, the feature layer combines the top 10
                                      features from each front layer (gene, 1-neighbor, edge, and pathway layers).
                                      There is a fully connected layer with 10 nodes between the pathway and feature
                                      layers, which represents the interaction of pathways. The 10 nodes are combined
                                      into the feature layer.
                             6.       Cox Layer
                         Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                    Page 8
                                      The Cox layer is the output layer which has only one node. The node produces a
                                      risk score, i.e., Prognostic Index (PI) for each sample. There is a fully connected
                                      layer with 10 nodes between the feature and Cox layers.
                       The number of nodes in the gene, 1-neighbor, and edge layers may differ, depending on
                       the number of available genes in each dataset. The training epoch was set to 1500, and the
                       batch size was set to the number of training samples. Adam was used as an optimizer with a
                       learning rate of 5e-5. Fig. 2 illustrates the deep learning architecture for single-omic data.
                       3.2.2.   Multi-omics data integration network—  In this work, we analyzed multi-
                       omics data including RNA-Seq gene expression, CNA, and methylation data represented
                       in a gene level. To integrate these multi-omics data in the proposed deep learning network,
                       we first trained the sub-network for each omic data and concatenated the three feature
                       layers in the sub-network of each omic type. The concatenated 150 features (50 features for
                       each omic type) are connected to a fully connected layer with 10 nodes that, in turn, are
                       connected to the Cox node to predict a risk score for each sample (Fig. 3).
   3.3.   Objective function and evaluation
                       The training of the survival network model was based on the negative log partial likelihood
                       loss in the Cox proportional-hazards model [35] with L2-regularization of the parameters:
                                  ℓ   Θ   = −     1         ℱ   Θ ,ℎ i  −log    ∑            exp   ℱ   Θ ,ℎ j    +λ     ∥ Θ  ∥ 2  ,       (11)
                                                 nE ∑i ∈  E                      j ∣T j >T i
                       where i∈E       indicates the occurrence of the event for patient i and nE              is the total number
                       of events in the training set. The neural network is represented by a function ℱ                      Θ,ℎi
                       parameterized by Θ        and ℎi is the input from patient i.Ti and Tj are the survival time for
                       patient i and patient j, respectively.
                       The Harrell’s concordance index (C-index) [              36] quantifies how well the predicted risk
                       scores for any pair align with the actual outcomes along with survival time. The C-index
                       ranges from 0 to 1 where a higher C-index value indicates a better concordance/prediction
                       and C-index = 0.5 means a completely random prediction. During the training process,
                       the C-index was used to identify the top-ranked features at each layer using the values of
                       ∣C  −  index   −0.5 ∣  . For the final evaluation, the setaside test data were fed into the trained
                       network to predict risk scores and compute the C-index.
                       Details of the model implementation and training can be found at Github repository (https://
                       github.com/MSK-MOI/GGNN).
   4.   Experiments
   4.1.   Datasets
                       We tested the proposed method on multi-omics data for 10 major cancers (breast, colon,
                       head & neck, kidney, low-grade glioma, liver, lung, pancreas, sarcoma, and skin cancers)
                       from the TCGA and the CoMMpass study of multiple myeloma. TCGA multi-omics data
                        Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                    Page 9
                      including RNA-Seq gene expression, CNA, and methylation data were downloaded from the
                      cBioPortal database [37,38]. For the CoMMpass study, RNA-Seq gene expression and CNA
                      data were downloaded at www.research.mmrf.org.
                      The input genomic data were structured as a graph, utilizing known biological information:
                      PPI network and pathways. The PPI network was obtained from the HPRD, which is
                      a manually curated database for reliable PPIs evidenced from published literature [27].
                      Key pathway information was obtained from the KEGG database, which is a widely used
                      bioinformatics resource that provides information about biological pathways, networks, and
                      functional annotations of genes and proteins [28]. For each cancer and each omic type, the
                      input gene list is the intersection of the gene list provided in data and the gene lists from
                      HPRD and KEGG. (See Table 1 for the number of samples and input genes for each cancer
                      and each omic type.).
                      For computational convenience in deep learning modeling, invariant measures and Ollivier-
                      Ricci curvature values were pre-computed. Because the Markov chain process requires all
                      the nodal values in the network to be positive, CNA values were exponentiated to ensure
                      their positivity. Because methylation (ranges from 0 to 1) is known to repress transcription,
                      1- (methylation values) were used for the invariant measure and Ollivier-Ricci curvature
                      calculation. In contrast, as input to the gene layer, original CNA and methylation values and
                      Z-score transformed values for RNA-Seq gene expression were used.
   4.2.   Benchmark tests
                      The prediction performance of the proposed method (GGNN) was compared with four other
                      methods: Cox-EN [11], Cox-AE [13], DCAP [15], and DCAP-XGB [15]. We used default
                      parameters for these methods. For each cancer type, we randomly split the data into 60% for
                      training, 20% for validation, and 20% for test. A C-index was computed for the validation
                      and test sets, separately. We repeated the random splitting process 30 times and computed an
                      average C-index for each experiment.
   4.3.   Results
                      As shown in Table 2, GGNN achieved similar C-index values on validation and test sets
                      with the tight range of 95% confidence intervals in all 11 cancers, suggesting that the models
                      were not overfitted and are stable.
                      We compared the prediction performance of C-index obtained by our method (GGNN) with
                      other alternative methods. As shown in Table 3, GGNN achieved the highest C-index values
                      for all cancers, except for colon cancer.
                      To investigate the effect of the geometry and machine learning components individually,
                      we conducted the GGNN modeling using one of the two components to update the
                      first two layers of GGNN, fixing other components. In all cancers except for low-grade
                      glioma and sarcoma, the model combining geometry and machine learning components
                      improved predictive power (Table 4). We noticed that neither of the two components alone
                      outperformed the other.
                        Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 10
                      To investigate the contribution of the sub-network of each omic type in modeling, we
                      compared the predictive performance of individual omic types with the combined network
                      (Table 5). We noticed that except for colon cancer and sarcoma, the combined network
                      resulted in better predictive performance compared to single-omic data, meaning that the
                      integration of multi-omics data enhances predictive power. In all cancers, RNA-Seq data
                      showed better predictive performance than CNA and methylation.
   4.4.   Case study of multiple myeloma
                      We conducted Kaplan–Meier survival analysis on the CoMMpass dataset of multiple
                      myeloma as a case study, using predicted risk scores. As shown in Fig. 4, the predicted
                      high and low-risk groups dichotomized by the median risk score were significantly separated
                      with log-rank p     -values < 10−3     in all training, validation, and test sets.
                      As we mentioned in the Methods section, our model utilized a local-global feature extraction
                      strategy. This enabled the identification of important risk factors that can be further
                      investigated in biological analysis. Table 6 shows the top-ranked features that could be
                      interpreted as potential risk factors associated with multiple myeloma prognosis.
                      We noticed that some of the genes in Table 6 have been found to be relevant to multiple
                      myeloma in previous studies. Liang et al. reported that overexpression of the cell cycle
                      regulatory kinase Wee1 is associated with poor survival in multiple myeloma and inhibition
                      of Wee1 is being explored as a potential anti-cancer therapeutic target [39]. Quan et al.
                      reported that MCM expression is associated with prognosis in multiple myeloma [40]. The
                      interaction DUSP-MAPK3 appears to be an important pathway involved in the pathogenesis
                      of multiple myeloma, and targeting this pathway may have the therapeutic potential in the
                      disease [41]. We also extracted the top ten genes from alternative methods in multiple
                      myeloma and found that some genes are common with those obtained via GGNN (See
                      Appendix).
   5.   Discussion
                      In this paper, we proposed a novel deep learning model called GGNN that enables the
                      integration of multi-omics data as well as geometric information derived from the known
                      biology of a given PPI network. Overall, the performance of GGNN in predicting survival
                      outcomes in 10 major cancers was better than four other alternative methods: Cox-EN
                      [11], Cox-AE [13], DCAP [15], and DCAP-XGB [15]. Our experimental results indicated
                      that the integration of multi-omics data improves predictive power and adding geometric
                      information into the model further increases prediction performance. Except for colon
                      cancer and sarcoma, the model applied to multi-omics data resulted in better predictive
                      performance compared to single-omic data. In analysis of single-omic data, RNA-Seq gene
                      expression data showed better predictive power than either CNA or methylation data.
                      The proposed architecture is extensible. For example, it is possible to incorporate other omic
                      types such as proteomic and microRNA data with the similar sub-network structure. Further,
                      we plan to add medical imaging and radiomic data into our model. This will help correlate
                        Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 11
                       multi-omics data with imaging data, enhancing biological interpretation of the imaging data,
                       and further improving predictive power.
                       Unlike other deep learning methods, GGNN can identify key biomarkers during the
                       modeling process, increasing the interpretability of biological findings. Moreover, use of
                       only important biomarkers in modeling can reduce the size of layers, which is likely to
                       mitigate overfitting.
                       In a downstream biological analysis of multiple myeloma, we identified several key
                       biomarkers likely associated with survival outcomes. We plan to further investigate these
                       biomarkers to elucidate whether they could be utilized as therapeutic targets in multiple
                       myeloma.
   6.   Conclusion
                       In this work, we proposed a novel deep learning framework that incorporates geometric
                       information into a deep learning-based model for prognosis prediction in cancer using
                       multi-omics data. Geometric information such as invariant measure and Ollivier-Ricci
                       curvature has been found to be useful in our previous studies for the prediction modeling
                       and identification of key biomarkers in multiple cancers with enhanced interpretability. We
                       demonstrated that the integration of geometric information and multi-omics data in a deep
                       learning model significantly improves predictive power.
                       Our method improves interpretability of the model by using known biological information
                       such as PPI and pathways as well as the local–global feature extraction strategy. The
                       analysis of multiple myeloma data showed that the proposed method has the potential to
                       identify risk factors associated with survival outcomes.
   Acknowledgments
                       This study was in part supported by AFOSR, USA grants (FA9550-20-1-0029 and FA9550-23-1-0096), NIH,
                       USA grant (R01-AG048769), Cure Alzheimer’s Fund, USA grant, Breast Cancer Research Foundation, USA grant
                       (BCRF-17-193), Army Research Office, USA grant (W911NF2210292), and MSK Cancer Center, USA Support
                       grant (P30 CA008748).
   Appendix.: Significant genes identified by alternative methods in multiple
   myeloma
                                                                            Table A.1
                       The top ten genes identified by alternative methods in multiple myeloma based on the
                       significant difference in gene expression between the high and low-risk groups. Genes that
                       are in common with those in GGNN were marked as bold.
                              Cox-EN      Cox-AE       DCAP         DCAP-XGB
                        1     PER1        TK1          RBM8A        TPM3
                        2     SMURF1      TTK          TPM3         RBM8A
                        3     BET1        PKMYT1       BIRC5        WEE1
                        Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 12
                               Cox-EN       Cox-AE       DCAP          DCAP-XGB
                         4     PPM1A        TUBA1B       TK1           TK1
                         5     RPE          BIRC5        PKMYT1        MCM4
                         6     SSH2         EXO1         WEE1          BIRC5
                         7     SIAH1        STMN1        CCNA2         PKMYT1
                         8     NLGN3        RBM8A        SNRPE         RAD21
                         9     SNRPA1       BUB1B        UBE2C         CCNA2
                         10    PHKG1        MCM2         ELK4          EXO1
                       Table A.1 shows the top ten genes from alternative methods in multiple myeloma, which
                       were identified based on the significant differences in gene expression between the high and
                       low-risk groups, using the t-test. We noticed that there are some genes in common with
                       those identified in GGNN, suggesting that those genes are likely associated with prognosis
                       in multiple myeloma.
   References
                       [1]. The Cancer Genome Atlas Research Network, Comprehensive genomic characterization defines
                              human glioblastoma genes and core pathways, Nature 455 (7216) (2008) 1061–1068 [PubMed:
                              18772890]
                       [2]. Weistuch Corey, Zhu Jiening, Deasy Joseph O., Tannenbaum Allen R., The maximum entropy
                              principle for compositional data, BMC Bioinformatics 23 (1) (2022) 449. [PubMed: 36309638]
                       [3]. Elkin Rena, Oh Jung Hun, Liu Ying L. Selenica Pier, Weigelt Britta, Reis-Filho Jorge S., Zamarin
                              Dmitriy, Deasy Joseph O., Norton Larry, Levine Arnold J., Tannenbaum Allen R., Geometric
                              network analysis provides prognostic information in patients with high grade serous carcinoma of
                              the ovary treated with immune checkpoint inhibitors, Npj Gen. Med 6 (1) (2021) 99.
                       [4]. Simhal Anish K, Maclachlan Kylee H, Elkin Rena, Zhu Jiening, Usmani Saad, Keats Jonathan J,
                              Norton Larry, Deasy Joseph O., Jung Hun Oh, Tannenbaum Allen, Geometric Network Analysis
                              Defines Poor-Prognosis Subtypes in Multiple Myeloma, Blood 140 (Supplement 1) (2022) 9991–
                              9992.
                       [5]. Zhu Jiening, Tran Anh Phong, Deasy Joseph O., Tannenbaum Allen, Multi-omic integrated
                              curvature study on pan-cancer genomic data, Cold Spring Harbor Laboratory, 2022, BioRxiv.
                       [6]. He Kaiming, Zhang Xiangyu, Ren Shaoqing, Sun Jian, Deep residual learning for image
                              recognition, in: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR,
                              2016, pp. 770–778.
                       [7]. Jung Hun Oh Wookjin Choi, Ko Euiseong, Kang Mingon, Tannenbaum Allen, Deasy Joseph
                              O, PathCNN: Interpretable convolutional neural networks for survival prediction and pathway
                              analysis applied to glioblastoma, Bioinformatics 37 (Supplement_1) (2021) i443–i450. [PubMed:
                              34252964]
                       [8]. Li Yu, Huang Chao, Ding Lizhong, Li Zhongxiao, Pan Yijie, Gao Xin, Deep learning in
                              bioinformatics: Introduction, application, and perspective in the big data era, Methods 166 (2019)
                              4–21. [PubMed: 31022451]
                       [9]. Vaswani Ashish, Shazeer Noam, Parmar Niki, Uszkoreit Jakob, Jones Llion, Gomez Aidan N.,
                              Kaiser Lukasz, Polosukhin Illia, Attention is all you need, 2017, CoRR abs/1706.03762
                       [10]. Cheng Heng-Tze, Koc Levent, Harmsen Jeremiah, Shaked Tal, Chandra Tushar, Aradhye Hrishi,
                              Anderson Glen, Corrado Greg, Chai Wei, Ispir Mustafa, Anil Rohan, Haque Zakaria, Hong
                              Lichan, Jain Vihan, Liu Xiaobing, Shah Hemal, Wide & deep learning for recommender systems,
                              2016, ArXiv:1606.07792.
                       [11]. Simon Noah, Friedman Jerome H., Hastie Trevor, Tibshirani Rob, Regularization paths for Cox’s
                              proportional hazards model via coordinate descent, J. Stat. Softw 39 (5) (2011) 1–13.
                         Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 13
                        [12]. Yousefi Safoora, Amrollahi Fatemeh, Amgad Mohamed, Dong Chengliang, Lewis Joshua E.,
                               Song Congzheng, Gutman David A., Halani Sameer H., Velazquez Vega Jose Enrique, Brat
                               Daniel J., Cooper Lee A.D., Predicting clinical outcomes from large scale cancer genomic
                               profiles with deep survival models, Sci. Rep 7 (1) (2017) 11707. [PubMed: 28916782]
                        [13]. Lee Tzong-Yi, Huang Kai-Yao, Chuang Cheng-Hsiang, Lee Cheng-Yang, Chang Tzu-Hao,
                               Incorporating deep learning and multi-omics autoencoding for analysis of lung adenocarcinoma
                               prognostication, Comput. Biol. Chem 87 (2020) 107277. [PubMed: 32512487]
                        [14]. Tong Li, Mitchel Jonathan, Chatlin Kevin, Wang May D., Deep learning based feature-level
                               integration of multi-omics data for breast cancer patients survival analysis, BMC Med. Inform.
                               Decis. Mak 20 (1) (2020) 225. [PubMed: 32933515]
                        [15]. Chai Hua, Zhou Xiang, Zhang Zhongyue, Rao Jiahua, Zhao Huiying, Yang Yuedong, Integrating
                               multi-omics data through deep learning for accurate cancer prognosis prediction, Comput. Biol.
                               Med 134 (2021) 104481. [PubMed: 33989895]
                        [16]. Scarselli Franco, Gori Marco, Chung Tsoi Ah, Hagenbuchner Markus, Monfardini Gabriele,
                               The graph neural network model, IEEE Trans. Neural Netw 20 (1) (2009) 61–80. [PubMed:
                               19068426]
                        [17]. Jumper John, Evans Richard, Pritzel Alexander, Green Tim, Figurnov Michael, Ronneberger
                               Olaf, Tunyasuvunakool Kathryn, Bates Russ, Žídek Augustin, Potapenko Anna, Bridgland
                               Alex, Meyer Clemens, Kohl Simon A.A., Ballard Andrew J., Cowie Andrew, Romera-Paredes
                               Bernardino, Nikolov Stanislav, Jain Rishub, Adler Jonas, Back Trevor, Petersen Stig, Reiman
                               David, Clancy Ellen, Zielinski Michal, Steinegger Martin, Pacholska Michalina, Berghammer
                               Tamas, Bodenstein Sebastian, Silver David, Vinyals Oriol, Senior Andrew W., Kavukcuoglu
                               Koray, Kohli Pushmeet, Hassabis Demis, Highly accurate protein structure prediction with
                               AlphaFold, Nature 596 (7873) (2021) 583–589. [PubMed: 34265844]
                        [18]. Fout Alex, Byrd Jonathon, Shariat Basir, Ben-Hur Asa, Protein interface prediction using
                               graph convolutional networks, in: Advances in Neural Information Processing Systems, Vol.
                               30, NeurIPS, 2017.
                        [19]. Wang Su, Wang Zhiliang, Zhou Tao, Sun Hongbin, Yin Xia, Han Dongqi, Zhang Han, Shi
                               Xingang, Yang Jiahai, THREATRACE: Detecting and tracing host-based threats in node level
                               through provenance graph learning, IEEE Trans. Inf. Forensics Secur 17 (2022) 3972–3987.
                        [20]. Fan Wenqi, Ma Yao, Li Qing, He Yuan, Zhao Eric, Tang Jiliang, Yin Dawei, Graph neural
                               networks for social recommendation, WWW ‘19, 2019, pp. 417–426.
                        [21]. Bruna Joan, Zaremba Wojciech, Szlam Arthur, Lecun Yann, Spectral networks and locally
                               connected networks on graphs, in: International Conference on Learning Representations, ICLR,
                               2014.
                        [22]. Defferrard Michaël, Bresson Xavier, Vandergheynst Pierre, Convolutional neural networks on
                               graphs with fast localized spectral filtering, in: Advances in Neural Information Processing
                               Systems, NeurIPS, 2016, pp. 3844–3852
                        [23]. Kipf Thomas N., Welling Max, Semi-supervised classification with graph convolutional
                               networks, in: International Conference on Learning Representations, ICLR, 2016.
                        [24]. Monti Federico, Boscaini Davide, Masci Jonathan, Rodola Emanuele, Svoboda Jan, Bronstein
                               Michael M., Geometric deep learning on graphs and manifolds using mixture model CNNs, in:
                               2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2017, pp. 5425–
                               5434.
                        [25]. Duvenaud David K, Maclaurin Dougal, Iparraguirre Jorge, Bombarell Rafael, Hirzel Timothy,
                               Aspuru-Guzik Alan, Adams Ryan P, Convolutional networks on graphs for learning molecular
                               fingerprints, in: Advances in Neural Information Processing Systems, Vol. 28, NeurIPS, 2015.
                        [26]. Gilmer Justin, Schoenholz Samuel S., Riley Patrick F., Vinyals Oriol, Dahl George E., Neural
                               message passing for quantum chemistry, in: International Conference on Machine Learning,
                               ICML, 2017, pp. 1263–1272.
                        [27]. Goel Renu, Harsha HC, Pandey Akhilesh, Keshava Prasad TS, Human protein reference database
                               and human proteinpedia as resources for phosphoproteome analysis, Mol. BioSyst 8 (2012) 453–
                               463. [PubMed: 22159132]
                          Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 14
                        [28]. Kanehisa Minoru, Sato Yoko, Kawashima Masayuki, Furumichi Miho, Tanabe Mao, KEGG as
                               a reference resource for gene and protein annotation, Nucleic Acids Res. 44 (D1) (2015) D457–
                               D462. [PubMed: 26476454]
                        [29]. Jones Peter W., Smith Peter, Stochastic Processes: An Introduction, third ed., Chapman & Hall/
                               CRC, 2020.
                        [30]. Chen Yongxin, Dela Cruz Filemon, Sandhu Romeil, Kung Andrew L., Mundi Prabhjot, Deasy
                               Joseph O., Tannenbaum Allen, Pediatric sarcoma data forms a unique cluster measured via the
                               earth mover’s distance, Sci. Rep 7 (1) (2017) 7035. [PubMed: 28765612]
                        [31]. Zhu Jiening, Jung Hun Oh, Deasy Joseph O., Tannenbaum Allen R., vWCluster: Vector-valued
                               optimal transport for network based clustering using multi-omics data in breast cancer, PLOS
                               ONE 17 (3) (2022) 1–15.
                        [32]. do Carmo Manfredo P. , Differential Geometry of Curves and Surfaces, Prentice Hall, 1976, pp.
                               I–VIII, 1–503.
                        [33]. Jost Jürgen, Riemannian Geometry and Geometric Analysis, Vol. 42005, Springer, 2008.
                        [34]. Ollivier Yann, Ricci curvature of Markov chains on metric spaces, J. Funct. Anal 256 (3) (2009)
                               810–864.
                        [35]. Cox DR, Regression models and life-tables, J. R. Stat. Soc. Ser. B Stat. Methodol 34 (2) 187–
                               202.
                        [36]. Uno Hajime, Cai Tianxi, Pencina Michael J., D’Agostino Ralph B., Wei LJ, On the C-statistics
                               for evaluating overall adequacy of risk prediction procedures with censored survival data, Stat.
                               Med 30 (10) (2011) 1105–1117. [PubMed: 21484848]
                        [37]. Cerami Ethan, Gao Jianjiong, Dogrusoz Ugur, Gross Benjamin E., Sumer Selcuk Onur, Aksoy
                               Bülent Arman, Jacobsen Anders, Byrne Caitlin J., Heuer Michael L., Larsson Erik, Antipin
                               Yevgeniy, Reva Boris, Goldberg Arthur P., Sander Chris, Schultz Nikolaus, The cBio cancer
                               genomics portal: An open platform for exploring multidimensional cancer genomics data, Cancer
                               Discov. 2 (5) (2012) 401–404. [PubMed: 22588877]
                        [38]. Gao Jianjiong, Aksoy Bülent Arman, Dogrusoz Ugur, Dresdner Gideon, Gross Benjamin, Sumer
                               S Onur, Sun Yichao, Jacobsen Anders Sinha Rileen, Larsson Erik Cerami Ethan, Sander Chris,
                               Schultz Nikolaus, Integrative analysis of complex cancer genomics and clinical profiles using the
                               cBioPortal, Sci. Signal 6 (269) (2013) pl1. [PubMed: 23550210]
                        [39]. Liang Long, He Yanjuan, Wang Haiqin, Zhou Hui, Xiao Ling, Ye Mao, Kuang Yijin, Luo
                               Saiqun, Zuo Yuna, Feng Peifu, Yang Chaoying, Cao Wenjie, Liu Taohua, Roy Mridul, Xiao
                               Xiaojuan, Liu Jing, The Wee1 kinase inhibitor MK1775 suppresses cell growth, attenuates
                               stemness and synergises with bortezomib in multiple myeloma, Br. J. Haematol 191 (1) (2020)
                               62–76. [PubMed: 32314355]
                        [40]. Quan Liang, Qian Tingting, Cui Longzhen, Liu Yan, Fu Lin, Si Chaozeng, Prognostic role of
                               minichromosome maintenance family in multiple myeloma, Cancer Gene Therapy 27 (10–11)
                               (2020) 819–829. [PubMed: 31959909]
                        [41]. Bermudez O, Pagès G, Gimond C, The dual-specificity MAP kinase phosphatases: Critical roles
                               in development and cancer, Am. J. Physiol. Cell Physiol 299 (2) (2010) C189–C202. [PubMed:
                               20463170]
                          Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 15
                                Fig. 1.
                                Computational range for Ollivier-Ricci curvature for an edge                                                     i,j   .
                                  Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 16
                                 Fig. 2.
                                 A deep learning architecture with single-omic data.
                                   Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 17
                                 Fig. 3.
                                 A deep learning architecture with multi-omic data.
                                   Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 18
                            Fig. 4.
                            Kaplan Meier survival analysis (all log-rank p                            -values < 10−3         ) between high and low-risk
                            groups split by the median predicted risk score for training, validation, and test sets in the
                            CoMMpass multiple myeloma data.
                              Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 19
                                       Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 20
                                                                                   Table 2
The C-index values by our proposed method (GGNN) on the validation and test sets in 11 cancers. CI:
confidence interval.
  Cancer       Validation       Test       Test 95% CI
  brca         0.683            0.661      (0.634, 0.689)
  coad         0.608            0.593      (0.559, 0.627)
  hnsc         0.618            0.615      (0.587, 0.643)
  kirc         0.727            0.690      (0.667, 0.712)
  lgg          0.828            0.817      (0.801, 0.832)
  lihc         0.697            0.672      (0.656, 0.687)
  luad         0.674            0.638      (0.620, 0.656)
  paad         0.662            0.625      (0.605, 0.645)
  sarc         0.714            0.661      (0.641, 0.681)
  skcm         0.650            0.604      (0.591, 0.617)
  mm           0.667            0.641      (0.629, 0.654)
                               Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 21
                                                                                  Table 3
Comparison of prediction performance with C-index values.
            Cox-EN        Cox-AE        DCAP         DCAP-XGB           GGNN
 brca       0.533         0.599         0.607        0.596              0.661
 coad       0.541         0.515         0.536        0.610              0.592
 hnsc       0.545         0.568         0.585        0.590              0.615
 kirc       0.631         0.606         0.655        0.679              0.690
 lgg        0.763         0.693         0.777        0.797              0.817
 lihc       0.578         0.589         0.641        0.626              0.672
 luad       0.548         0.562         0.601        0.632              0.638
 paad       0.538         0.506         0.582        0.622              0.625
 sarc       0.602         0.625         0.648        0.648              0.662
 skcm       0.561         0.563         0.560        0.601              0.604
 mm         0.566         0.555         0.576        0.602              0.641
                              Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 22
                                                                                  Table 4
The contribution of geometry track and machine learning track in GGNN modeling in terms of C-index.
            GGNN-ml          GGNN-geometry             GGNN
 brca       0.577            0.593                     0.661
 coad       0.588            0.512                     0.592
 hnsc       0.602            0.587                     0.615
 kirc       0.657            0.664                     0.690
 lgg        0.826            0.809                     0.817
 lihc       0.650            0.623                     0.672
 luad       0.587            0.629                     0.638
 paad       0.590            0.592                     0.625
 sarc       0.649            0.622                     0.662
 skcm       0.607            0.622                     0.606
 mm         0.623            0.631                     0.641
                              Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 23
                                                                                    Table 5
The C-index values for each omic type.
            RNA-net         CNA-net         Methyl-net        Combined-net
  brca      0.654           0.509           0.603             0.661
  coad      0.596           0.566           0.537             0.592
  hnsc      0.595           0.570           0.593             0.615
  kirc      0.687           0.559           0.665             0.690
  lgg       0.808           0.749           0.807             0.817
  lihc      0.660           0.557           0.634             0.672
  luad      0.635           0.536           0.623             0.638
  paad      0.617           0.569           0.596             0.625
  sarc      0.656           0.585           0.587             0.662
  skcm      0.625           0.514           0.615             0.604
  mm        0.640           0.559           NA                0.641
                               Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

Zhu et al.                                                                                                                                                                                                  Page 24
                                       Comput Biol Med. Author manuscript; available in PMC 2023 November 11.

