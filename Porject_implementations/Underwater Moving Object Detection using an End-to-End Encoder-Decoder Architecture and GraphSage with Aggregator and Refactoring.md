   UnderwaterMovingObjectDetectionusinganEnd-to-EndEncoder-Decoder
               ArchitectureandGraphSagewithAggregatorandRefactoring
                                Meghna Kapoor                                              Suvam Patra
                 Indian Institute of Technology Jammu                         Manipal Institute of Technology
                             Jammu and Kashmir                                         Manipal, Karnatka
                           meghna@iitjammu.ac.in                                 suvampatra8802@gmail.com
                       Badri Narayan Subudhi                                             Vinit Jakhetiya
             Indian Institute of Technology Jammu                        Indian Institute of Technology Jammu
                         Jammu and Kashmir                                           Jammu and Kashmir
                  subudhi.badri@iitjammu.ac.in                              vinit.jakhetiya@iitjammu.ac.in
                                                            Ankur Bansal
                                           Indian Institute of Technology Jammu
                                                       Jammu and Kashmir
                                                ankur.bansal@iitjammu.ac.in
                          Abstract
   Underwater environments are greatly affected by sev-                 1.Introduction
eral factors, including low visibility, high turbidity, back-
scattering, dynamic background, etc., and hence pose chal-                  Detection of moving objects in a video scene is one of
lengesinobjectdetection. Severalalgorithmsconsidercon-                  the most fundamental problems in computer vision.   Al-
volutionalneuralnetworkstoextractdeepfeaturesandthen                    thoughseveralsurveillance-basedtechniquesaredeveloped
object detection using the same. However, the dependency                foroutdoorscenesandveryfewtechnologiesaredeveloped
on the kernel’s size and the network’s depth results in fad-            for underwater applications till the early twenty-first cen-
ing relationships of latent space features and also are un-             tury.  Most of the underwater object detection techniques
able to characterize the spatial-contextual bonding of the              are employed for tracking marine life for estimating the
pixels.  Hence, they are unable to procure satisfactory re-             spread of diseases [15] among the marine animals, cracks
sults in complex underwater scenarios. To re-establish this             in oil and gas pipelines [12], drowning detection [14], etc.
relationship, we propose a unique architecture for under-               Theseapplicationsmakeitinterestingformanyunderwater
water object detection where U-Net architecture is consid-              surveillance tasks too. Further, state-of-the-art moving ob-
eredwiththeResNet-50backbone. Further,thelatentspace                    ject detection algorithms focus on detecting the shape and
features from the encoder are fed to the decoder through a              structure of the object.
GraphSage model. GraphSage-based model is explored to                       Themovingobjectdetectiontaskismorecomplexinthe
reweightthenoderelationshipinnon-euclideanspaceusing                    underwaterscenarioascomparedtoconventionalabovewa-
different aggregator functions and hence characterize the               ter due to the intrinsic properties of water.  There are two
spatio-contextual bonding among the pixels.  Further, we                mainfactorsthataffectunderwaterimagesgreatly. Thefor-
explored the dependency on different aggregator functions:              mer includes when the light coming from the objects in the
mean,  max,  and LSTM, to evaluate the model’s perfor-                  scene is absorbed and scattered due to the suspended parti-
mance. We evaluated the proposed model on two underwa-                  cle present in the water, which produces a haze in the un-
ter benchmark databases:  F4Knowledge and underwater                    derwater scene. The latter is due to the salinity of the water
change detection. The performance of the proposed model                 where the optical light gets attenuated due to the difference
is evaluated against eleven state-of-the-art techniques in              in viscosity of the water, which creates the color cast prob-
terms of both visual and quantitative evaluation measures.              leminthescene. Further,thepoorvisibilityanddecoloriza-

tioninunderwaterconditionsposechallengesfortraditional                             space variables to graph space using GrapSage is pro-
computer vision techniques to accurately analyze underwa-                          posed. Here, each element of latent space is projected
ter images and videos.                                                             as anode of an unorderedand unstructured graph, and
   Although many of the conventional object detection al-                          training is done to learn the edge relationships.
gorithms are used in underwater surveillance;  very few                         • Further, we used different aggregator functions like
works are reported which are specifically designed to de-                          LSTM, mean,  and max to refactor the relationship
tectunderwatermovingobjectsagainsttheunderwaterchal-                               among the neighboring nodes of latent variables.
lenges including haze,  color cast,  poor visibility,  decol-
orization, etc.   In the state-of-the-art (SOTA) techniques                     The organization of this paper is as follows.  Section 2
deep convolutional neural networks (CNN) [1,3,16] are                        depicts the discussions on state-of-the-art techniques.  The
used to extract the deep features from the underwater im-                    proposed work with the motivation of the same is provided
age sequences and draw a projection map from RGB color                       in Section 3.  Section 4 describes the experimental results
image to a binary classification of the images as object and                 and analysis of the proposed work.  The conclusions and
background.  The projections from the encoder to the de-                     future works are provided in Section 5.
coderarenon-invertibleduetopoolinglayers. Theassump-
tion of symmetricity impediments the extraction of spatial-                  2.State-of-the-artTechniques
contextual information among pixels. This motivates to re-                      The main idea of moving object detection is to classify
factor the latent space variables to define the relationship                 each pixel of an underwater video frame as foreground or
among the nodes that are necessary to preserve the infor-                    backgroundhenceperceivingtheshapeandstructureofthe
mation and minute details of the object.   In SOTA tech-                     object. Based on the study of SOTA techniques we devise
niques, graph convolution network (GCN) [13] is found to                     underwater object detection techniques into the following
be effective in exploring the convolution network in non-                    sub-categories.
euclidean space.  The GCN assumes the neighborhood in
non-euclidean space and integrates the information using a                   2.1.Statisticalmethods
mean aggregator [2,30]. We further, broaden the perspec-                        Statistical methods were used to statistically model the
tive using GraphSage, i.e., the Graph sampling and aggre-                    pixel information and further estimate the parameters with
gator.  The learning is based on a function from the local                   the relative changes in subsequent frames to detect the ob-
neighborhood, and information among nodes is shared us-                      ject’s movement.   The process of finding the changes in
ing different aggregator functions.                                          pixel intensity from two consecutive image frames of a
   In this article, we propose a simple yet efficient end-to-                video helps in detecting the foreground from the back-
end hybrid deep learning architecture that uses both deep                    ground. Routetal.[25]proposedamethodforlocalchange
learning and graph theory for underwater object detection.                   detection to detect underwater moving objects. In the said
Intheproposedtechnique,weadheredtotheuseofaU-Net                             work, the authors used a difference of 5 frames to detect
architecturewhichiscomposedofanencoderandadecoder                            the local changes. Vasamsetti et al. [28] proposed a multi-
part. The U-Net architecture is designed with a ResNet-50                    frame triplet pattern (MFTP) model to detect underwater
backbone. Further, the encoder part is connected to the de-                  moving objects. However, the said method failed in the dy-
coder part through a GraphSage technique.  In traditional                    namic background condition.  Javed et al. [10] proposed a
CNNs, the dependency on the kernel’s size and the net-                       robust principal component analysis-based model for mov-
work’s depth results in fading relationships of latent space                 ing object detection.   The authors decomposed the input
features. Hence, they are unable to procure satisfactory re-                 data matrix into a low-rank matrix representing the back-
sults in complex underwater scenarios.  Hence in the pro-                    groundimageandasparsecomponentidentifyingthemov-
posed scheme, we explored the utilization of refactoring                     ing objects.   Rout et al. [26] proposed a spatio-temporal
of latent space vectors using GraphSage network. Further,                    Gaussian-integratedWronskianmodeltodetectmovingob-
weexploreddifferentaggregatorfunctionsinGraphSageto                          jects from a given video scene. The said method considers
check the refactorization of latent space features.                          the background modeling by exploiting the spatial depen-
   The main contributions of this article are listed below:                  dencyamongthepixelsinWronskianframeworkandmulti-
   • We explored the hypothesis that projections by con-                     temporal background in the temporal direction with a mix-
     volutional neural networks lose information in latent                   ture of Gaussian probability density functions.  However,
     spaceandutilizerefactoringoflatentspacevectorsus-                       consideringtheunderwaterchallengesthefocushasshifted
     ing a novel refactoring algorithm i.e.  GraphSage for                   towarddeeplearning-basedmethods. Thedeepfeaturesare
     moving object detection.                                                extractedandgiventothedecodertore-projecttheinforma-
                                                                             tion to image space passing through a non-linear activation
   • A novel projection method of high-dimensional latent                    map to infer the moving object from the frame.

2.2.Deeplearningbasedmethods                                                tecture is the encoder part, and the right side of the archi-
   In  SOTA,  the  Encoder-decoder-based  deep  learning-                   tecture is the decoder part.  As discussed in the previous
based methods are popularly used for moving object de-                      section, several algorithms are reported in the state-of-the-
tection.  These methods extract the deep features from un-                  art techniques for underwater moving object detection.  It
derwater video scenes using deep architectures like CNN,                    may be noted that state-of-the-art techniques use CNN ar-
transformers, etc in the encoder part of the network.  The                  chitecture to extract the deep features from underwater im-
extractedfeaturescontaintheobjectinformationandarere-                       ages.  The convolutional layers project the data from the
tained during the training of the end-to-end model. Chen et                 image domain to a higher dimensional latent space. CNNs
al.[3]proposedamodelusinganovelattentionmodelcom-                           arenotfullyconnectednetworks,andthenodeconnections
prising long short-term memory. The said method is tested                   depend on the spatial neighborhoods.  The non-euclidean
on CDnet and may fail to incorporate underwater dynam-                      space doesn’t preserve the spatial information, which leads
ics. Lin et al. [18] proposed a mask RCNN-based method                      to ill-formulated connections in higher dimensional space.
to detect objects in the underwater environment. However,                   Aswegodeeper,thespacebecomesnon-euclidean,andthe
the said method doesn’t preserve the minute details of the                  information in non-euclidean latent space is loosely con-
moving object.  Further, Li et al. [17] proposed a method                   nected in terms of spatial relationship.
for underwater marine life detection using Faster R-CNN.                        Fig:2column (b-c) depicts an example of two standard
Recently,Bajpaietal.[1]proposedaUNet-basedmodelfor                          SOTA techniques used for underwater moving object de-
underwatermovingobjectdetectionusingtheResNetback-                          tection:  ML-BGS [31] and SubSENSEBGS [27].   It can
bone. The proposed methods fail to retain spatial informa-                  be seen clearly that, both models fail to detect the objects
tion.  Hence, a re-weighting module is expected to restore                  in case of complex backgrounds.  The structural informa-
the connections in latent space.  Fan et al. [4] proposed a                 tion of the object is lost.  In a higher dimensional space,
methodformulti-scalecontextualfeaturesusingaugmenta-                        the spatial relationships are not maintained as the projec-
tion of the receptive field. The proposed model has a com-                  tionwithaconvolutionalneuralnetworktransformstheeu-
posite connection backbone to deal with the distortion in                   clideanspaceintoanon-euclideanspace. Hence,thelossof
texture and blurring due to the scattering effect.                          minute details is observed.
                                                                                To maintain the relationship among the nodes,  a re-
2.3.Graphbasedmethods                                                       factoring module is required.  In the proposed scheme, we
   Recently,graphconvolutionalneuralnetworks(GCNNs)                         have used a combination of deep CNN and GraphSage al-
are found to be effective in various computer vision tasks                  gorithms.  The deep CNN network extracts the spatial in-
suchasimageclassificationandsemanticsegmentation. Xu                        formation and projects the extracted features in higher di-
et al. [30] proposed a method based on graph learning to                    mensional space.  A projection from latent space to graph
extract relevant contextual information from sparse graph                   space is made using GraphSage.  GraphSage is used as a
structures. To increase spatial awareness, learnable spatial                reweighting module to re-establish the connection between
Gaussian kernels performed the graph inference on graphs.                   nodes or feature vector elements. A deep decoder projects
Chen et al. [2] proposed a combination of semantic seg-                     the information from feature space to image space to de-
mentationnetworksforfeatureextractiononlabelsandim-                         tect the moving objects in the scene with spatio-contextual
ages, and the inferred features were used to initialize the                 neighborhood information.
adjacency matrix of the graphs.   GCNNs [29] are a nat-                         We are aware that the underwater complexities are enor-
ural choice for analyzing irregularly structured input data                 mous,  which include poor illumination,  underwater dy-
represented in non-euclidean space. Giraldo et al. [6] pro-                 namic  environments,  objects  with  different  shapes  and
posed a graph CNN-based model for moving object detec-                      sizes, and cluttered background. GraphSage [7] is a graph
tionincomplexenvironmentsfromunseenvideos. Thesaid                          CNN method that can handle irregular and unstructured
method uses mask R-CNN, motion, texture, and color fea-                     data by updating the node features in a graph and can bet-
turestoinitializethegraph. Oneofthemajordisadvantages                       ter deal the underwater uncertainties. Hence, it is expected
of the said model is its dependency on handcrafted feature                  that the GraphSage method will be better suited for object
selection.  Moreover, the existing state-of-the-art methods                 detection in underwater conditions. Further, to describe the
are computationally intensive.                                              latent space variables we have considered various aggrega-
                                                                            torsforreweighing. Weexploreddifferentaggregatorfunc-
3.ProposedMethod                                                            tions: mean, max, and LSTM over node relations.
   We propose an encoder-decoder architecture for under-                    3.1.EncoderforFeatureExtraction
water moving object detection as shown in Fig:1.   We                           IntheproposedschemeweadheredtotheuseofaU-Net
use a U-net architecture where the left part of the archi-                  architecture with a ResNet-50 encoder for the feature ex-

                                   Figure 1. Proposed model for moving object detection using GraphSage
traction. In the ResNet-50 network, CNNs at different lev-                   3.2.GraphSage
els are used to extract the features from images and project                     The existing state-of-the-art methods focuses on using
them in higher dimensional space. The deep CNN network                       graph learning on features extracted by assigning each el-
extracts the spatial relationship of pixels assuming the in-                 ement as a node of the graph making the archaic methods
formation to be in euclidean space.   Though convolution                     computationally intensive. However, sampling of graph ac-
projectstheinformationfromlow-dimensionalimagespace                          cording to the neighbourhood and then aggregating to de-
to high-dimensional latent space but unable to preserve the                  visearelationshipisnotexplored. Inourproposedmethod,
spatio-contextual entity of the image in higher dimensional                  the learning strategy of GraphSage is adapted to re-factor
space. Hence,afeaturepoolingmoduleorfeaturereweight-                         the latent feature vector. The latent vector from the feature
ing module is required to re-establish the latent space con-                 spaceisfedtoinitializethegraph. Everyelementofthefea-
nections.  Although several algorithms use feature pooling                   ture vector is considered a node of the graph. Graph archi-
modules, but are failing to preserve the spatial entity and                  tecture tries to find relationships among them. Images have
hencetheerrorinobjectdetectionresults. Herewepropose                         a spatial relationship that can be modeled better in CNN.
the use of GraphSage for the same.                                           Usinggraphsonimagesleadstohighcomputationtimeand
                                                                             space. Hence deploying graphs on high dimensional space
                                                                             torefactortherelationshipratherthanonfullimageisabet-
                                                                             ter way to get the best of both worlds.  GraphSage is used
                                                                             for classification in literature.  Liu et al. [19] proposed a
                                                                             GraphSage model for forecasting traffic speed. Graphsage
                                                                             isinitializedwiththehistoricaltrafficspeedsandgeometri-
                                                                             calinformation. Loetal.[20]proposedaGraphSage-based
                                                                             method for intrusion detection. The graph is initialized and
          Figure 2. Underwater moving object detection                       trained for edge classification.  To the best of our knowl-
                                                                             edge, no work has been done on node refactorization using
                                                                             GraphSage. Tothebestofourknowledge,noworkhasbeen
                                                                             done on node refactorization using GraphSage.
                                                                                 A graphG  can be defined as an unordered set of tuples
                                                                             defined over vertices (V ) and edges (E ). The nodes or ver-
                                                                             tices are connected with links or edges.  In the proposed
                                                                             scheme, GraphSage (Graph sample and aggregate) is used
                                                                             forlargegraphsforinductivereasoning. Thebasicarchitec-
                                                                             ture of the GraphSage algorithm is given in Fig:3. Further,
                                                                             different stages of the GraphSage are given as follows.
                                                                             3.2.1   Sampling
                  Figure 3. GraphSage Algorithm                              TheneighborhoodN   isdefinedasthedirecthopsconnected
                                                                             by a pathway as shown in Fig:3(a). The neighborhood is

defined as a fixed-size subset from the sample set using a                      3.3.Decoder
uniform draw. The neighbors are updated in each iteration.
Working with neighbors helps in reducing the computation                            The re-weighted features are mapped using an inverse-
timeandsize. Theinformationfromtheneighborsisaggre-                             mappingfunctionbythedecoder. Inordertopreservemost
gated and given to the node of the next stage.                                  information, an identical mapping is obtained using U-Net
                                                                                architecture.  There are skip connections between the en-
3.2.2   Aggregatorfunctions                                                     coder and decoder to preserve the information. The model
                                                                                is initialized using ImageNet data, and later, the weights
Aggregator  functions af   define  the  relationship  among                     are updated using the F4Knowledge dataset using transfer
nodes.  The information among the neighboring nodes are                         learning.
shared and updated according to the aggregator function as                          The algorithmic enumeration of the proposed scheme is
shown in Fig:3(b).  In GraphSage, the neighbors (       j) in                   provided in Algorithm1
latent space layerl represented byh l− 1j     have no order, and
theaggregatorfunctionrepresentstheparticularnodewhile
being trainable.
                                                                                  Algorithm 1: Proposed Algorithm for Object De-
                h lN  i ←   af{h l− 1j     ∀j ∈N   (i)}.             (1)          tection
                                                                                    Input: RGB video frame
   In the proposed model, three different aggregator func-                          Output: A binary segmented frame
tions are used to re-weight the latent space relationship.                       1 for k = 1 to number of epochsdo
Mean aggregator:  The mean of neighborhood nodes is                              2       capture frame f
taken into account to evaluate the information at the cur-                       3      bi ←   f
rent node.                                                                       4       fori = 1 to 3do
Maxaggregator: Themaxorthepoolingfunctionoperates                                5           ci ←   conv  2d(bi,kernel   )
by doing element-wise max across the neighboring nodes.                          6           bi ←   pool  (ci)
LSTMaggregator: Compared to the above two functions,                             7      m  i ←   bi
the third is the most complex function and is inherently not                     8      x i ←   flatten   (m  i)
symmetric.  Random permutation among the neighbors is                            9      x i ←   sigmoid    (x i)
applied.                                                                        10       Graph Initialization; A graphG , Latent space
                                                                                          vectorx i,i∈V   , layersL , neighbourhood
3.2.3   RefactoringusingGraphSage                                                         functionN  : i→   2i, weight matricesW   l
                                                                                          ∀l =1  ···L
Every element of the feature space is considered as a node.                     11      h i ←   x i   ∀i∈V
The information from neighboring nodesN   is defined over                       12       forl = 1 to Ldo
the information from previous nodes. An aggregator func-                        13            fori∈V   do
tion from the set{mean, max, LSTM} is applied over the                          14                h lN  i ←
obtainedinformationfromneighborsandisdenotedash lN  .                                              aggregator       function      l{h l− 1j     ∀j ∈
The current information and information from the neigh-                                            N  (i)} h li ←
borhood are concatenated.  The obtained vector is passed                                           σ (W   l.concatenate      (h l− 1i   ,h lN  (i)))
through a non-linear activation (sigmoid in our case). The
updated representation of nodei in layerl is given as:                          15      yi ←   h li   ∀i∈V
                                                                                16      ki ←   conv  2d(yi)
                                                                                17       for i = 1 to 3do
                    h li =  fupdate   (h lN  i,h l− 1j   ).          (2)        18           u i ←   upsample      (ki)+  bi
                                                                                19           di ←   conv  2d(u i)
Here, fupdate     can be simply an aggregator operator or any
complex function.  We have used the update function as a                        20      di ←   sigmoid    (di)P  N
concatenate operator. The algorithmic representation of the                     21     L  =  −   1N     k =0  tk∗log (ˆt)+(1  − ti)∗log (1 −  ˆti)
proposedGraphSageschemeisprovidedinAlgorithm1. A                                22       compute gradient
graphG  is initialized using a latent vector and iterated over                  23       update weights and bias
the k-hop neighborhood. At every iteration, the aggregated
informationamongthenodesisupdatedtolearnthespatial-
contextual information among non-euclidean space.

                                            Figure 4. Qualitative measure of F4Knowledge dataset on proposed model
  Table 1. Quantitative analysis of proposed method with different aggregator functions on different challenges of F4Knowledge dataset
                      Challenge         Aggregator Function            Accuracy            Recall    Precision    F measure
                                                                                     Training    Testing
                      ComplexBkg                Mean                     99.60       98.59      98.99          99.58            99.28
                                                            Max                      99.58       58.77      99.68          59.00            74.13
                                                           LSTM                    99.48       99.61      99.03          97.07            98.04
                      Crowded                       Mean                     98.83       97.41      99.37          98.02            98.69
                                                            Max                      98.88       96.78      99.36          97.39            98.37
                                                           LSTM                    98.46       96.96      99.56          97.38            98.46
                      DynamicBkg                Mean                     98.76       97.27      99.27          99.99            99.63
                                                            Max                      98.77       97.08      97.08        100.00            98.52
                                                           LSTM                    98.75       96.90      97.14          99.75            98.43
                      Hybrid                          Mean                     99.14       97.84      98.06          99.78            98.91
                                                            Max                      99.13       97.94      98.10          99.84            98.96
                                                           LSTM                    99.14       98.66      98.92          99.73            99.32
                      Standard                        Mean                     98.74       98.92      99.37          99.54            99.46
                                                            Max                      98.73       98.92      99.37          99.54            99.46
                                                           LSTM                    98.72       98.29      98.89          99.38            99.13
                      Aggregate                                                    98.98       95.33      98.81          96.40            97.25
Table 2. Quantitative analysis in terms of F-measure with six SOTA techniques. The red color indicates the best, and blue indicates the
second best
         Challenge          Texture-BGS[9]   MLBGS[31]   MultiCueBGS[21]   SubSENSEBGS[27]   SILTP[8]   MFI[28]   PM(mean)   PM(max)   PM(LSTM)
   complexbackground           0.69               0.58                 0.48                     0.21                0.73         0.83    99.28  74.13   98.04
          crowded                   0.54               0.74                 0.68                     0.67                0.67         0.69    98.69  98.37   98.46
   dynamicbackground           0.43               0.32                 0.33                     0.81                0.32         0.64    99.63  98.52   98.43
  camouflageforeground          0.42               0.66                 0.77                     0.42                0.66         0.72    99.46  99.46   99.13
           hybrid                    0.49               0.46                 0.72                     0.42                0.69          0.8      98.91  98.96   99.32
Table 3. Quantitative analysis in terms of F-measure with five SOTA architectures. The red color indicates the best, and the blue indicates
the second best.
                              Challenge         GSMM[24]    AGMM[32]    ABMM[11]    ADE[33]    GWFT[22]    PM(mean)
                              fishswarm              0.57               0.30               0.06             0.59       0.85       0.99
                             marinesnow            0.84               0.82               0.65             0.82       0.91       0.99
                         smallaquaculture         0.77               0.74               0.43             0.88       0.93       0.99
                                caustics                0.55               0.74               0.67       0.75       0.67       0.99
                              twofishes              0.79               0.79               0.76             0.71       0.82       0.95

4.ExperimentalResultsandAnalysis                                             obtainedbytheproposedschemeasshowninFig:4column
   The  proposed  technique  is  executed  on  an  NVIDIA                    (i) are able to detect the object correctly.
A100  80  GB  GPU  with  128  GB  RAM.  It  is  imple-                       Table 4.   Quantitative measure on underwater change detection
mented by python programming with the PyTorch frame-                         dataset
work on the Linux operating system.  We have evaluated
theperformanceoftheproposedschemeontwobenchmark                                 Challenges     Accuracy    Precision    Recall    F-measure
databases: F4KnowledgeandUnderwaterchangedetection.                              Caustics         99.61         99.95       99.65       99.80
Intheproposedscheme,abatchsizeof2 isconsidereddur-                                Marine          98.78         99.80       98.97       99.39
ing training. In GraphSage, a hop of two neighbors is con-                      Fishswarm       98.18         99.60       98.51       99.05
                                                                                 TwoFish         98.60         99.38       99.22       99.30
sidered.  We used Adam optimizer with a learning rate of                       AquaCulture      90.97         93.48       97.36       95.38
e− 3 toconvergeourmodel. TheU-Netarchitectureusesbi-
narycrossentropyasalossfunctiontocomputethegradient
and update the hyperparameters. The model’s performance                      4.3.QuantitativeAnalysisResults
istestedusingdifferentaggregatefunctionsusingvisualand                           In this article, the evaluation metrics considered to eval-
quantitative evaluation measures.  The performance of the                    uate the quantitative performance of the proposed moving
proposed scheme is corroborated by comparing its results                     object detection model are accuracy, precision, recall, and
withthoseoftheelevenstate-of-the-art(SOTA)techniques:                        f-measure. Accuracyistheratioofacorrectlylabeledpixel
Texture-BGS [9], MLBGS [31], MLCB [21], Subsense-                            as foreground among all the pixels. Precision is the ratio of
BGS[27],SILTP[8],MFI[28],GSMM[24],AGMM[32],                                  pixelscorrectlylabeledasaforegroundtothedetectedtotal
ABMM [11], ADE [33], GWFT [22].                                              foreground pixels.  The recall is the ratio of pixels labeled
4.1.DescriptiononDatabases                                                   astheforeground tothosethatbelong to theforeground. F-
                                                                             measure is the harmonic mean of precision and recall.  As
   We  have  evaluated  the  performance  of  the  proposed                  the number of background and foreground pixels is not the
schemeontwobenchmarkdatabases: F4Knowledge[5]and                             same, the f-measure is the most reliable metric.
Underwater change detection [23].  The Fish4Knowledge
dataset has video sequences captured from 10 cameras.                                          F 1 =              TPTP   +    1                 (3)
We considered five challenges from the Fish4Knowledge                                                          2 (FP   +  FN   ).
dataset:  complex background, crowded scenes, dynamic                            The results obtained by the proposed scheme on the
background, camouflaged foreground, and hybrid scenes.                       F4Knowledge dataset, using the different aggregator func-
Thenumberofsamplesvariesbetweendifferentchallenges.                          tions are provided in Table:1in terms of accuracy, recall,
The second dataset considered in our experiment is under-                    precision, and F-measure quantitative evaluation measures.
water change detection.  The said dataset has five videos                    In this table, we have provided the considered evaluation
withdifferentchallenges: caustics,marine,fishswarm,two                       measuresobtainedbytheproposedschemeonfivedifferent
fish, and aquaculture. Fish4Knowledge has less correlation                   challenges of the Fish4Knowledge dataset. The results are
among the frames corresponding to ground truth, while a                      found to be very effective and produce a higher accuracy
high correlation among the frames can be observed in the                     withaverygoodprecisionrecordonallthechallengeswith
underwater change detection dataset.                                         different aggregator functions like mean, max, and LSTM.
4.2.VisualAnalysisofResults                                                  We also observed that for the “complex backgrounds” se-
                                                                             quence,themaxoperatorwasfoundtobeprovidingalesser
   The visual analysis of the proposed architecture for un-                  accuracy.
derwatermovingobjectdetectioniscarriedoutondifferent                             Further, the proposed model is compared with those of
challenging sequences of the F4Knowledge database and                        the different state-of-the-art techniques along with consid-
underwaterchangedetection. A visual illustration, of the re-                 ered three aggregator functions:  mean, max, and LSTM.
sults on F4Knowledge are shown in Fig:4columns (a) and                       The proposed model is compared with the six state-of-the-
(b) represent the original and ground-truth images of se-                    art techniques:  Texture-BGS [9],  MLBGS [31],  MLCB
quences.   Fig:4columns (c) to (h) represent the results                     [21], SubsenseBGS [27], SILTP [8], MFI [28] techniques
obtained on the considered sequence of the F4Knowldge                        in terms of F-measure and are shown in Table:2. It can be
database  using  Texture-BGS  [9],  MLBGS  [31],  MLCB                       clearly observed that the proposed model provides the best
[21], SubsenseBGS [27], SILTP [8], MFI [28] techniques.                      results compared to all SOTA techniques. Hence, it corrob-
It may be observed that most of SOTA methods failed to                       orates our hypothesis. It is also observed that the mean ag-
provide the complete object region.  Even many instances                     gregator surpasses the F-measure as compared to max and
the moving object region are missed. However, the results                    LSTM aggregators and other considered SOTA techniques.

   We  also  verified  the  effectiveness  of  the  proposed                   Table 5.  Ablation study of different aggregator functions on the
schemeontheunderwaterchangedetectiondatasetwiththe                             F4Knowledge dataset. Red indicates best, and blue indicates sec-
meanaggregator. Table:3hasquantitativeresultscompared                          ond best.
to five SOTA methods:  Gaussian switch mixture model                                  Function    Accuracy    Precision    Recall    F-measure
(GSMM) [24], adaptive Gaussian mixture model (AGMM)                                     Mean 98.51 99.3891.1598.66
[32], adaptive background mixture model (ABMM) [11],                                    Max         94.46         99.0198.7298.66
adaptive density estimation (ADE) [33], Gaussian switch                                LSTM98.50 99.1993.8998.68
with flux tensor (GWFT) [22] in terms of F-measure. Our
proposed model was found to be performing best as com-                         5.ConclusionsandFutureWorks
pared to all the SOTA architectures. Table:4contains the
quantitative results for the proposed model using the mean                         In this article, we propose a novel hybrid deep learn-
aggregator function in terms of accuracy, precision, and re-                   ing and GraphSage architecture for underwater object de-
call on five challenges of underwater change detection.                        tection.   The proposed model consists of an end-to-end
                                                                               encoder-decoder-basedU-NetarchitecturewiththeResNet-
                                                                               50 backbone.  To reduce the effects of misclassification in
                                                                               object detection, a novel GraphSage-based model is sand-
                                                                               wiched between the encoder and decoder of the U-Net ar-
                                                                               chitecture. Threeaggregatorfunctions,namely,mean,max,
                                                                               and LSTM, are verified to retain the missing information.
                                                                               Theproposedschemeistestedontwobenchmarkunderwa-
                                                                               ter databases: F4Knowledge and underwater change detec-
                                                                               tion.  The effectiveness of the proposed scheme is verified
                                                                               with eleven state-of-the-art techniques. It is verified that in
                                                                               non-euclidean space, only convolution operation is insuffi-
     (a) Aggregated quantitative measure of F4Knowledge dataset                cient to retain the information. Refactoring the relationship
     on proposed model                                                         among nodes is necessary.  Further, mean based aggrega-
                                                                               tor is found to be providing the best results. In the future,
                                                                               we would like to improve the performance of the proposed
                                                                               schemeusingthefirstgenericobjectneuralnetworktracker
                                                                               for its possible real-time implementation.
                                                                               References
                                                                                 [1]Vatsalya Bajpai, Akhilesh Sharma, Badri Narayan Subudhi,
                                                                                     T Veerakumar,  and Vinit Jakhetiya.    Underwater U-Net:
                                                                                     Deep learning with U-Net for visual underwater moving ob-
                                                                                     jectdetection.InProceedingsofOCEANS2021: SanDiego–
            (b) Comparison of different aggregator functions                         Porto, pages 1–4, 2021.2,3
                                                                                 [2]Shengjia Chen, Zhixin Li, and Zhenjun Tang.  Relation R-
Figure 5.  Quantitative measure of F4Knowledge dataset on pro-                       CNN: A graph based relation-aware network for object de-
posed model                                                                          tection.    IEEE Signal Processing Letters,  27:1680–1684,
                                                                                     2020.2,3
                                                                                 [3]Yingying Chen, Jinqiao Wang, Bingke Zhu, Ming Tang, and
                                                                                     Hanqing Lu. Pixel wise deep sequence learning for moving
4.4.Ablationstudy                                                                    objectdetection. IEEETransactionsonCircuitsandSystems
                                                                                     for Video Technology, 29(9):2567–2579, 2019.2,3
   We made an ablation study of the proposed scheme on                           [4]BaojieFan,WeiChen,YangCong,andJiandongTian. Dual
different aggregators using: mean, max, and LSTM meth-                               refinement underwater object detection network.    In An-
ods on the F4Knowledge dataset which are reported in Ta-                             dreaVedaldi,HorstBischof,ThomasBrox,andJan-Michael
                                                                                     Frahm, editors, Computer Vision – ECCV 2020, pages 275–
ble:5. The comparison of different aggregator functions is                           291, Cham, 2020. Springer International Publishing.3
reportedinFig:5. Themeanoperatorhasthehighestaccu-                               [5]Robert B Fisher, Yun-Heh Chen-Burger, Daniela Giordano,
racy, precision, and a comparable F-measure with a differ-                           Lynda Hardman, Fang-Pang Lin, et al.   Fish4Knowledge:
enceof0.02frombest. ThecomputationtimefortheMean                                     collecting and analyzing massive coral reef fish video data,
operator is less than LSTM and hence more preferable.                                volume 104. Springer, 2016.7

 [6]Jhony H Giraldo, Sajid Javed, Naoufel Werghi, and Thierry                       [20]Wai Weng Lo, Siamak Layeghy, Mohanad Sarhan, Marcus
      Bouwmans.GraphCNNformovingobjectdetectionincom-                                     Gallagher, and Marius Portmann.   E-graphsage:  A graph
      plex environments from unseen videos.  In Proceedings of                            neural network based intrusion detection system for iot.  In
      the IEEE/CVF International Conference on Computer Vi-                               NOMS2022-2022IEEE/IFIPNetworkOperationsandMan-
      sion, pages 225–233, 2021.3                                                         agement Symposium, pages 1–9, 2022.4
 [7]Will Hamilton, Zhitao Ying, and Jure Leskovec.  Inductive                       [21]SeungJong Noh and Moongu Jeon.  A new framework for
      representation learning on large graphs. Advances in neural                         background subtraction using multiple cues.  In Asian Con-
      information processing systems, 30, 2017.3                                          ferenceonComputerVision,pages493–506.Springer,2013.
 [8]HongHan,JianfeiZhu,ShengcaiLiao,ZhenLei,andStanZ                                      6,7
      Li. Movingobjectdetectionrevisited: Speedandrobustness.                       [22]Martin Radolko, Fahimeh Farhadifard, and Uwe von Lukas.
      IEEE Transactions on Circuits and Systems for Video Tech-                           Change detection in crowded underwater scenes-via an ex-
      nology, 25(6):910–921, 2014.6,7                                                     tended gaussian switch model combined with a flux tensor
 [9]Marko Heikkila and Matti Pietikainen.    A texture-based                              pre-segmentation. InInternationalConferenceonComputer
      method for modeling the background and detecting moving                             Vision Theory and Applications, volume 5, pages 405–415.
      objects.IEEETransactionsonPatternAnalysisandMachine                                 SCITEPRESS, 2017.6,7,8
      Intelligence, 28(4):657–662, 2006.6,7                                         [23]MartinRadolko,FahimehFarhadifard,andUweFreiherrvon
[10]Sajid  Javed,  Thierry  Bouwmans,  Maryam  Sultana,  and                              Lukas. Datasetonunderwaterchangedetection. InOCEANS
      Soon Ki Jung.  Moving object detection on RGB-D videos                              2016 MTS/IEEE Monterey, pages 1–8. IEEE, 2016.7
      using  graph  regularized  spatiotemporal  RPCA.    In  New                   [24]Martin Radolko and Enrico Gutzeit.   Video segmentation
      Trends  in  Image  Analysis  and  Processing–ICIAP,  pages                          via a gaussian switch background model and higher order
      230–241. Springer, 2017.2                                                           markovrandomfields. InVISAPP(1),pages537–544,2015.
[11]Pakorn KaewTraKulPong and Richard Bowden.    An im-                                   6,7,8
      proved adaptive background mixture model for real-time                        [25]Deepak Kumar Rout, Pranab Gajanan Bhat, T Veerakumar,
      tracking with shadow detection.   Video-based surveillance                          Badri Narayan Subudhi, and Santanu Chaudhury.  A novel
      systems: Computer vision and distributed processing, pages                          five-frame difference scheme for local change detection in
      135–144, 2002.6,7,8                                                                 underwater video.   In Proceeding of Fourth International
[12]Juhyun  Kim,  Minju  Chae,  Jinju  Han,  Simon  Park,  and                            ConferenceonImageInformationProcessing(ICIIP),pages
      Youngsoo Lee.   The development of leak detection model                             1–6. IEEE, 2017.2
      in subsea gas pipeline using machine learning.  Journal of                    [26]Deepak Kumar Rout, Badri Narayan Subudhi, T. Veeraku-
      Natural Gas Science and Engineering, 94:104134, 2021.1                              mar, and Santanu Chaudhury.   Spatio-contextual Gaussian
[13]Thomas N Kipf and Max Welling.  Semi-supervised classi-                               mixture  model  for  local  change  detection  in  underwater
      fication with graph convolutional networks.  arXiv preprint                         video. ExpertSystemswithApplications,97:117–136,2018.
      arXiv:1609.02907, 2016.2                                                            2
[14]Fei  Lei,  Hengyu  Zhu,  Feifei  Tang,  and  Xinyuan  Wang.                     [27]Pierre-Luc St-Charles, Guillaume-Alexandre Bilodeau, and
      Drowning behavior detection in swimming pool based on                               RobertBergevin. Flexiblebackgroundsubtractionwithself-
      deep learning.  Signal, Image and Video Processing, pages                           balanced local sensitivity. In Proceedings of the IEEE Con-
      1–8, 2022.1                                                                         ference on Computer Vision and Pattern Recognition Work-
[15]Daoliang Li and Ling Du.  Recent advances of deep learn-                              shops, pages 408–413, 2014.3,6,7
      ingalgorithmsforaquaculturalmachinevisionsystemswith                          [28]Srikanth  Vasamsetti,  Supriya  Setia,  Neerja  Mittal,  Har-
      emphasisonfish. ArtificialIntelligenceReview,pages1–40,                             ish K Sardana, and Geetanjali Babbar. Automatic underwa-
      2022.1                                                                              ter moving object detection using multi-feature integration
[16]Guohao  Li,  Matthias  Muller,  Ali  Thabet,  and  Bernard                            framework in complex backgrounds.  IET Computer Vision,
      Ghanem.  Deepgcns: Can GCNs go as deep as CNNs?   In                                12(6):770–778, 2018.2,6,7
      Proceedings of the IEEE/CVF international conference on                       [29]Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long,
      computer vision, pages 9267–9276, 2019.2                                            Chengqi Zhang, and S Yu Philip.  A comprehensive survey
[17]XiuLi,MinShang,HongweiQin,andLianshengChen. Fast                                      on graph neural networks. IEEE transactions on neural net-
      accuratefishdetectionandrecognitionofunderwaterimages                               works and learning systems, 32(1):4–24, 2020.3
      with Fast R-CNN.   In OCEANS 2015 Marine Technology                           [30]Hang Xu, Chenhan Jiang, Xiaodan Liang, and Zhenguo Li.
      Society/IEEE Washington, pages 1–5.3                                                Spatial-aware graph relation network for large-scale object
[18]Wei-Hong Lin, Jia-Xing Zhong, Shan Liu, Thomas Li, and                                detection.    In Proceedings of the IEEE/CVF Conference
      Ge Li. Roimix: Proposal-fusion among multiple images for                            on Computer Vision and Pattern Recognition, pages 9298–
      underwater object detection. In IEEE International Confer-                          9307, 2019.2,3
      ence on Acoustics, Speech and Signal Processing (ICASSP),                     [31]Jian Yao and Jean-Marc Odobez.   Multi-layer background
      pages 2588–2592, 2020.3                                                             subtraction based on color and texture.  In 2007 IEEE con-
[19]Jielun Liu, Ghim Ping Ong, and Xiqun Chen.  Graphsage-                                ference on computer vision and pattern recognition, pages
      based traffic speed forecasting for segment network with                            1–8. IEEE, 2007.3,6,7
      sparse data.  IEEE Transactions on Intelligent Transporta-                    [32]Zoran Zivkovic. Improved adaptive gaussian mixture model
      tion Systems, 23(3):1755–1766, 2022.4                                               for background subtraction.  In Proceedings of the 17th In-

      ternational Conference on Pattern Recognition, 2004. ICPR
      2004., volume 2, pages 28–31. IEEE, 2004.6,7,8
[33]Zoran  Zivkovic  and  Ferdinand  Van  Der  Heijden.    Effi-
      cient adaptive density estimation per image pixel for the
      task of background subtraction.  Pattern recognition letters,
      27(7):773–780, 2006.6,7,8

