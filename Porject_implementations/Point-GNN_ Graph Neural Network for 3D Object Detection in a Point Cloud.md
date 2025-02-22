   Point-GNN:GraphNeuralNetworkfor3DObjectDetectioninaPointCloud
                                        Weijing Shi and Ragunathan (Raj) Rajkumar
                                                    Carnegie Mellon University
                                                         Pittsburgh, PA 15213
                                                  {weijings,  rajkumar}@cmu.edu
                           Abstract
   In this paper, we propose a graph neural network to
detect objects from a LiDAR point cloud.   Towards this
end, we encode the point cloud efﬁciently in a ﬁxed ra-
dius near-neighbors graph. We design a graph neural net-
work,namedPoint-GNN,topredictthecategoryandshape                           Figure1.Threepointcloudrepresentationsandtheircommonpro-
of the object that each vertex in the graph belongs to.  In                cessing methods.
Point-GNN, we propose an auto-registration mechanism to
reduce translation variance, and also design a box merg-                   iteratively to create a point set representation.   The re-
ing and scoring operation to combine detections from mul-                  peated grouping and sampling on a large point cloud can
tiple vertices accurately.   Our experiments on the KITTI                  becomputationallycostly. Recent3Ddetectionapproaches
benchmark show the proposed approach achieves leading                      [10][21][16] often take a hybrid approach to use a grid and
accuracy using the point cloud alone and can even sur-                     asetrepresentationindifferentstages. Althoughtheyshow
pass fusion-based algorithms. Our results demonstrate the                  some promising results, such hybrid strategies may suffer
potential of using the graph neural network as a new ap-                   the shortcomings of both representations.
proach for 3D object detection.  The code is available at                     In this work, we propose to use a graph as a compact
https://github.com/WeijingShi/Point-GNN.                                   representation of a point cloud and design a graph neural
                                                                           network called Point-GNN to detect objects.  We encode
1.Introduction                                                             thepointcloudnativelyinagraphbyusingthepointsasthe
                                                                           graph vertices.  The edges of the graph connect neighbor-
   Understandingthe3Denvironmentisvitalinroboticper-                       hood points that lie within a ﬁxed radius, which allows fea-
ception. Apointcloudthatcomposesasetofpointsinspace                        ture information to ﬂow between neighbors. Such a graph
isawidely-usedformatfor3DsensorssuchasLiDAR.De-                            representation adapts to the structure of a point cloud di-
tecting objects accurately from a point cloud is crucial in                rectly without the need to make it regular. A graph neural
applications such as autonomous driving.                                   network reuses the graph edges in every layer, and avoids
   Convolutional neural networks that detect objects from                  grouping and sampling the points repeatedly.
images rely on the convolution operation.  While the con-                     Studies [15] [9] [2] [17] have looked into using graph
volution operation is efﬁcient, it requires a regular grid as              neural network for the classiﬁcation and the semantic seg-
input. Unlikeanimage,apointcloudistypicallysparseand                       mentation of a point cloud.   However, little research has
notspacedevenlyonaregulargrid. Placingapointcloudon                        looked into using a graph neural network for the 3D object
a regular grid generates an uneven number of points in the                 detection in a point cloud. Our work demonstrates the fea-
gridcells. Applyingthesameconvolutionoperationonsuch                       sibilityofusingaGNNforhighlyaccurateobjectdetection
a grid leads to potential information loss in the crowded                  in a point cloud.
cells or wasted computation in the empty cells.                               Our proposed graph neural network Point-GNN takes
   Recent breakthroughs in using neural networks [3] [22]                  the point graph as its input.   It outputs the category and
allow an unordered set of points as input.   Studies take                  bounding boxes of the objects to which each vertex be-
advantage of this type of neural network to extract point                  longs. Point-GNN is a one-stage detection method that de-
cloud features without mapping the point cloud to a grid.                  tects multiple objects in a single shot. To reduce the trans-
However, they typically need to sample and group points                    lation variance in a graph neural network, we introduce an
                                                                       1

auto-registration mechanism which allows points to align                     key points.   The features of those subsets are then again
their coordinates based on their features. We further design                 grouped into sets for further feature extraction.  Many 3D
a box merging and scoring operation to combine detection                     object detection approaches take advantage of such neural
results from multiple vertices accurately.                                   networks to process a point cloud without mapping it to a
   We evaluate the proposed method on the KITTI bench-                       grid.  However, the sampling and grouping of points on a
mark.  On the KITTI benchmark, Point-GNN achieves the                        largescaleleadtoadditionalcomputationalcosts. Mostob-
state-of-the-art accuracy using the point cloud alone and                    ject detection studies only use the neural network on sets
even surpasses sensor fusion approaches.  Our Point-GNN                      as a part of the pipeline.  [13] generates object proposals
shows the potential of a new type 3D object detection ap-                    fromcameraimagesanduses[14]toseparatepointsthatbe-
proach using graph neural network, and it can serve as a                     longtoanobjectfromthebackgroundandpredictabound-
strong baseline for the future research. We conduct an ex-                   ing box. [16] uses [14] as a backbone network to generate
tensive ablation study on the effectiveness of the compo-                    bounding box proposals directly from a point cloud. Then,
nents in Point-GNN.                                                          it uses a second-stage point network to reﬁne the bound-
   In summery, the contributions of this paper are:                          ing boxes.  Hybrid approaches such as [23] [19] [10] [21]
  •We propose a new object detection approach using                          use [3] to extract features from local point sets and place
     graph neural network on the point cloud.                                the features on a regular grid for the convolutional opera-
                                                                             tion. Althoughtheyreducethelocalirregularityofthepoint
  •WedesignPoint-GNN,agraphneuralnetworkwithan                               cloudtosomedegree,theystillsufferthemismatchbetween
     auto-registration mechanism that detects multiple ob-                   a regular grid and the overall point cloud structure.
     jects in a single shot.                                                 Point cloud in graphs. Research on graph neural network
                                                                             [18]seekstogeneralizetheconvolutionalneuralnetworkto
  •We achieve state-of-the-art 3D object detection accu-                     agraphrepresentation. AGNNiterativelyupdatesitsvertex
     racy in the KITTI benchmark and analyze the effec-                      features by aggregating features along the edges. Although
     tiveness of each component in depth.                                    theaggregationschemesometimesissimilartothatindeep
                                                                             learning on sets, a GNN allows more complex features to
2.RelatedWork                                                                be determined along the edges.  It typically does not need
   Prior work in this context can be grouped into three cat-                 to sample and group vertices repeatedly.  In the computer
egories, as shown in Figure1.                                                vision domain, a few approaches represent the point cloud
Point cloud in grids. Many recent studies convert a point                    as a graph.   [15] uses a recurrent GNN for the semantic
cloud to a regular grid to utilize convolutional neural net-                 segmentationonRGBDdata. [9]partitionsapointcloudto
works. [20] projects a point cloud to a 2D Bird’s Eye View                   simplegeometricalshapesandlinkthemintoagraphforse-
(BEV) image and uses a 2D CNN for object detection. [4]                      mantic segmentation. [2] [17] look into classifying a point
projects a point cloud to both a BEV image and a Front                       cloud using a GNN. So far, few investigations have looked
View(FV)imagebeforeapplyinga2DCNNonboth. Such                                into designing a graph neural network for object detection,
projection induces a quantization error due to the limited                   where an explicit prediction of the object shape is required.
image resolution.  Some approaches keep a point cloud in                        Our work differs from previous work by designing a
3Dcoordinates. [23]representspointsin3Dvoxelsandap-                          GNN for object detection.   Instead of converting a point
plies3Dconvolutionforobjectdetection. Whentheresolu-                         cloud to a regular gird, such as an image or a voxel, we
tion of the voxels grows, the computation cost of 3D CNN                     use a graph representation to preserve the irregularity of a
grows cubically, but many voxels are empty due to point                      point cloud.  Unlike the techniques that sample and group
sparsity. Optimizations such as the sparse convolution [19]                  thepointsintosetsrepeatedly,weconstructthegraphonce.
reduce the computation cost. Converting a point cloud to a                   TheproposedPoint-GNNthenextractsfeaturesofthepoint
2D/3Dgridsuffersfromthemismatchbetweentheirregular                           cloud by iteratively updating vertex features on the same
distribution of points and the regular structure of the grids.               graph.  Our work is a single-stage detection method with-
Pointcloudinsets. Deep learning techniques on sets such                      out the need to develop a second-stage reﬁnement neural
as PointNet [3] and DeepSet[22] show neural networks can                     networks like those in [4][16][21][11][13].
extract features from an unordered set of points directly. In                3. Point-GNN for 3D Object Detection in a
suchamethod,eachpointisprocessedbyamulti-layerper-                                PointCloud
ceptron (MLP) to obtain a point feature vector. Those fea-
turesareaggregatedbyanaverageormaxpoolingfunction                               Inthissection,wedescribetheproposedapproachtode-
toformaglobalfeaturevectorofthewholeset. [14]further                         tect 3D objects from a point cloud. As shown in Figure2,
proposes the hierarchical aggregation of point features, and                 the overall architecture of our method contains three com-
generates local subsets of points by sampling around some                    ponents: (a) graph construction, (b) a GNN ofT iterations,

Figure 2. The architecture of the proposed approach. It has three main components: (a) graph construction from a point cloud, (b) a graph
neural network for object detection, and (c) bounding box merging and scoring.
and (c) bounding box merging and scoring.                                    Topreservetheinformationwithintheoriginalpointcloud,
3.1.GraphConstruction                                                        we encode the dense point cloud in the initial state valuesi
                                                                             of the vertex.  More speciﬁcally, we search the raw points
   Formally, we deﬁne a point cloud ofN  points as a set                     withinar0 radiusofeachvertexandusetheneuralnetwork
P ={p1,...,pN}, wherepi= (xi,si) is a point with both                        on sets to extract their features. We follow  [10]  [23] and
3D coordinatesxi∈R 3  and the state valuesi∈Rk ak-                           embed the lidar reﬂection intensity and the relative coordi-
length vector that represents the point property.  The state                 nates using anMLP  and then aggregate them by theMax
valuesican be the reﬂected laser intensity or the features                   function.  We use the resulting features as the initial state
which encode the surrounding objects. Given a point cloud                    value of the vertex.  After the graph construction, we pro-
P, weconstruct agraphG = (P,E) byusingP asthe ver-                           cess the graph with a GNN, as shown in Figure2b.
tices and connecting a point to its neighbors within a ﬁxed                  3.2.GraphNeuralNetworkwithAuto-Registration
radiusr, i.e.
                                                                                A typical graph neural network reﬁnes the vertex fea-
               E ={(pi,pj)|∥xi−xj∥2<r}              (1)                      tures  by  aggregating  features  along  the  edges.    In  the
                                                                             (t+1) thiteration,itupdateseachvertexfeatureintheform:
   Theconstructionofsuchagraphisthewell-knownﬁxed
radiusnear-neighborssearchproblem. Byusingacelllistto                                      vt+1i    = gt(ρ({etij|(i,j)∈E}),vti)
ﬁnd point pairs that are within a given cut-off distance, we                                 etij= ft(vti,vtj)                                           (2)
canefﬁcientlysolvetheproblemwitharuntimecomplexity
ofO(cN) wherecis the max number of neighbors within                          whereetandvtare the edge and vertex features from the
the radius [1].                                                              tthiteration. Afunctionft(.) computestheedgefeaturebe-
   In practice, a point cloud commonly comprises tens of                     tween two vertices.ρ(.) is a set function which aggregates
thousands of points.   Constructing a graph with all the                     theedgefeaturesforeachvertex.gt(.) takestheaggregated
points as vertices imposes a substantial computational bur-                  edge features to update the vertex features. The graph neu-
den. Therefore,weuseavoxeldownsampledpointcloud ˆP                           ral network then outputs the vertex features or repeats the
for the graph construction. It must be noted that the voxels                 process in the next iteration.
hereareonlyusedtoreducethedensityofapointcloudand                               Inthecaseofobjectdetection,wedesigntheGNNtore-
they are not used as the representation of the point cloud.                  ﬁne a vertex’s state to include information about the object
Westilluseagraphtopresentthedownsampledpointcloud.                           where the vertex belongs.  Towards this goal, we re-write

Equation (2) to reﬁne a vertex’s state using its neighbors’                 If a vertex is outside any bounding boxes, we assign the
states:                                                                     background class to it.  We use the average cross-entropy
    st+1                                                                    loss as the classiﬁcation loss.
      i    = gt(ρ({ft(xj−xi,stj)|(i,j)∈E}),sti)      (3)                                                   N∑   M∑
Note that we use the relative coordinates of the neighbors                                  lcls=−1                 yicjlog(picj)                 (6)
as input toft(.) for the edge feature extraction.  The rel-                                            N  i=1  j=1
ative coordinates induce translation invariance against the
global shift of the point cloud.  However, it is still sensi-               wherepicandyicare the predicted probability and the one-
tive to translation within the neighborhood area.  When a                   hot class label for thei-th vertex respectively.
small translation is added to a vertex, the local structure of                 For the object bounding box,  we predict it in the 7
its neighbors remains similar.  But the relative coordinates                degree-of-freedom formatb =  (x,y,z,l,h,w,θ), where
of the neighbors are all changed, which increases the input                 (x,y,z) represent the center position of the bounding box,
variance toft(.).  To reduce the translation variance, we                   (l,h,w) represent the box length, height and width respec-
propose aligning neighbors’ coordinates by their structural                 tively,andθistheyawangle. Weencodetheboundingbox
features instead of the center vertex coordinates.  Because                 with the vertex coordinates(xv,yv,zv) as follows:
the center vertex already contains some structural features
fromthepreviousiteration,wecanuseittopredictanalign-                             δx= x−xvlm ,δy= y−yvhm ,δz= z−zvwm
ment offset, and propose anauto-registrationmechanism:
         ∆xit= ht(sti)                                                           δl= log(llm ),δh= log(hhm ),δw = log(wwm )                   (7)
          st+1i    = gt(ρ({f(xj−xi+∆ xit,stj)},sti)        (4)                   δθ= θ−θ0
∆xtiis the coordination offset for the vertices to register                               θm
their coordinates.ht(.) calculates the offset using the cen-                wherelm,hm,wm,θ0,θm are constant scale factors.
tervertexstatevaluefromthepreviousiteration. Bysetting                         The localization branch predicts the encoded bounding
ht(.) tooutputzero,theGNNcandisabletheoffsetifneces-                        boxδb= (δx,δy,δz,δl,δh,δw,δθ) for each class. If a ver-
sary. In that case, the GNN returns to Equation (3). We an-                 texiswithinaboundingbox,wecomputetheHuberloss[7]
alyze the effectiveness of this auto-registration mechanism                 between the ground truth and our prediction. If a vertex is
in Section4.                                                                outside any bounding boxes or it belongs to a class that we
   As shown in Figure2b, we model     ft(.),gt(.) andht(.)                  do not need to localize, we set its localization loss as zero.
using multi-layer perceptrons (MLP ) and add a residual                     We then average the localization loss of all the vertices:
connection ingt(.).   We chooseρ(.)  to beMax  for its                                     N∑
robustness[3]. A single iteration in the proposed graph net-                  lloc=   1        1 (vi∈binterest) ∑         lhuber(δ−δgt)   (8)
work is then given by:                                                                 N  i=1
                                                                                                                   δ∈δbi
       ∆xit= MLPth(sti)                                                        Topreventover-ﬁtting,weaddL1 regularizationtoeach
         etij= MLPtf([xj−xi+∆ xit,stj])                           (5)       MLP. The total loss is then:
       st+1i    = MLPtg(Max({eij|(i,j)∈E}))+ sti                                             ltotal= αlcls+ βlloc+ γlreg                 (9)
where[,]represents the concatenation operation.
   Every iterationtuses a different set ofMLPt, which                       whereα,βandγareconstantweightstobalanceeachloss.
is not shared among iterations.  AfterT iterations of the                   3.4.BoxMergingandScoring
graph neural network, we use the vertex state value to pre-
dict both the category and the bounding box of the object                      As multiple vertices can be on the same object, the neu-
where the vertex belongs. A classiﬁcation branchMLPcls                      ralnetworkcanoutputmultipleboundingboxesofthesame
computes a multi-class probability.  Finally, a localization                object. It is necessary to merge these bounding boxes into
branchMLPloccomputes a bounding box for each class.                         oneandalsoassignaconﬁdencescore. Non-maximumsup-
3.3.Loss                                                                    pression(NMS)hasbeenwidelyusedforthispurpose. The
                                                                            common practice is to select the box with the highest clas-
   For the object category, the classiﬁcation branch com-                   siﬁcation score and suppress the other overlapping boxes.
putes a multi-class probability distribution{pc1,...,pcM}                   However, the classiﬁcation score does not always reﬂect
foreachvertex.M  isthetotalnumberofobjectclasses,in-                        the localization quality.  Notably, a partially occluded ob-
cludingtheBackground class. Ifavertexiswithinabound-                        ject can have a strong clue indicating the type of the object
ingboxofanobject,weassigntheobjectclasstothevertex.                         butlacksenoughshapeinformation. ThestandardNMScan

 Algorithm1:NMS with Box Merging and Scoring                              only use the point cloud in our approach. Since the dataset
   Input:B={b1,...,bn},D={d1,...,dn},Th                                   only annotates objects that are visible within the image, we
   Bis the set of detected bounding boxes.                                process the point cloud only within the ﬁeld of view of the
   Dis the corresponding detection scores.                                image.  The KITTI benchmark evaluates the average pre-
   This an overlapping threshold value.                                   cision (AP) of three types of objects: Car, Pedestrian and
   Greencolor marks the main modiﬁcations.                                Cyclist. Duetothescaledifference,wefollowthecommon
 1M←{},Z←{}                                                               practice [10][23][19][21] and train one network for the Car
 2 whileB̸= emptydo                                                       and another network for the Pedestrian and Cyclist.   For
 3   i←argmaxD                                                            training, we remove samples that do not contain objects of
 4  L←{}                                                                  interest.
 5       forbjinBdo                                                       4.2.ImplementationDetails
 6             ifiou(bi,bj)>Ththen
 7     L←L∪bj                                                                We use three iterations (T = 3  ) in our graph neural net-
 8     B←B−bj,D←D−dj                                                      work.  During training, we limit the maximum number of
 9             end                                                        input edges per vertex to 256. During inference, we use all
10       end                                                              the input edges. All GNN layers perform auto-registration
11   m←median(L)                                                          using a two-layerMLPh of units (64,3) . TheMLPclsis
12   o←occlusion(m)                                                       ofsize(64,#( classes)). Foreachclass,MLPlocisofsize
13   z←(o+1)  ∑          bk∈L IoU(m,bk)dk                                 (64,64,7) .
14  M←M∪m,Z←Z∪z                                                           Car: Weset(lm,hm,wm) tothemediansizeofCarbound-
15 end                                                                    ing boxes (3.88m,1.5m,1.63m). We treat a side-view car
16 returnM,Z                                                              withθ∈[−π/4,π/4] andafront-viewcarθ∈[π/4,3π/4]
                                                                          as two different classes.   Therefore, we setθ0   =  0    and
                                                                          θ0  = π/2  respectively.  The scaleθm  is set asπ/2.  To-
pick an inaccurate bounding box base on the classiﬁcation                 gether with the Background class and DoNotCare class, 4
score alone.                                                              classes are predicted. We construct the graph withr= 4 m
   Toimprovethelocalizationaccuracy,weproposetocal-                       andr0  = 1 m.  We set  ˆP  as a downsampled point cloud
culate the merged box by considering the entire overlapped                by  a  voxel  size  of  0.8  meters  in  training  and  0.4  me-
box cluster. More speciﬁcally, we consider the median po-                 ters in inference. MLPf andMLPg, are both of sizes
sition and size of the overlapped bounding boxes. We also                 (300 ,300)   .  For the initial vertex state, we use anMLP
compute the conﬁdence score as the sum of the classiﬁ-                    of(32,64,128 ,300)    for embedding raw points and another
cation scores weighted by the Intersection-of-Union (IoU)                 MLP   of (300 ,300)    after theMax  aggregation.  We set
factor and an occlusion factor. The occlusion factor repre-               Th= 0 .01  in NMS.
sents the occupied volume ratio. Given a boxbi, letli,wi,                 PedestrianandCyclist. Again,weset(lm,hm,wm) tothe
hibe its length, width and height, and letvli,vwi,vhi be the              median bounding box size. We set(0.88m,1.77m,0.65m)
unit vectors that indicate their directions respectively. xj              for Pedestrian and(1.76m,1.75m,0.6m) for Cyclist. Sim-
are the coordinates of pointpj.  The occlusion factoroiis                 ilar to what we did with the Car class,  we treat front-
then:                                                                     view and side-view objects as two different classes.  To-
                     ∏                                                    getherwiththeBackground classandtheDoNotCareclass,
 oi=      1                   maxpj∈bi(vTxj)−minpj∈bi(vTxj)  (10)         6classesarepredicted. Webuildthegraphusingr= 1 .6m,
       liwihiv∈{vli,vwi,vhi }                                             and downsample the point cloud by a voxel size of 0.4 me-
                                                                          ters in training and 0.2 meters in inference. MLPf and
   We modify standard NMS as shown in Algorithm1.  It                     MLPg are both of sizes (256 ,256)   .  For the vertex state
returnsthemergedboundingboxesM andtheirconﬁdence                          initialization,  we setr0   =  0 .4m.   We use a anMLP
scoreZ. We will study its effectiveness in Section4.                      of (32,64,128 ,256 ,512)    for embedding and anMLP   of
                                                                          (256 ,256)    to process the aggregated feature. We setTh =
4.Experiments                                                             0.2 in NMS.
4.1.Dataset                                                                  WetraintheproposedGNNend-to-endwithabatchsize
                                                                          of 4. The loss weights areα = 0 .1,β = 10   ,γ = 5 e−7.
   WeevaluateourdesignusingthewidelyusedKITTIob-                          We use stochastic gradient descent (SGD) with a stair-case
ject detection benchmark [6].  The KITTI dataset contains                 learning-rate decay. For Car, we use an initial learning rate
7481trainingsamplesand7518testingsamples. Eachsam-                        of 0.125    and a decay rate of 0.1  every 400 K  steps.  We
pleprovidesboththepointcloudandthecameraimage. We                         trained the network for 1400 K  steps.  For Pedestrian and

                                    Table 1. The Average Precision (AP) comparison of 3D object detection on the KITTItest dataset.
                        Table 2. The Average Precision (AP) comparison of Bird’s Eye View (BEV) object detection on the KITTItest dataset.
             Cyclist, we use an initial learning rate of 0.32  and a decay                      els: Easy, Moderate, and Hard. Our approach achieves the
             rate of 0.25   every 400 K  steps.  We trained it for 1000 K                       leading results on the Car detection of Easy and Moderate
             steps.                                                                             level and also the Cyclist detection of Moderate and Hard
             4.3.DataAugmentation                                                               level.  Remarkably, on the Easy level BEV Car detection,
                                                                                                we surpass the previous state-of-the-art approach by 3.45.
                 Topreventoverﬁtting,weperformdataaugmentationon                                Also, we outperform fusion-based algorithms in all cate-
             thetrainingdata. Unlikemanyapproaches[19][10][16][21]                              gories except for Pedestrian detection. In Figure3, we pro-
             thatusesophisticatedtechniquestocreatenewgroundtruth                               vide qualitative detection results on all categories. The re-
             boxes,wechooseasimpleschemeofglobalrotation,global                                 sults on both the camera image and the point cloud can be
             ﬂipping, box translation and vertex jitter.   During train-                        visualized.  It must be noted that our approach uses only
             ing, we randomly rotate the point cloud by yaw ∆θ∼                                 the point cloud data.  The camera images are purely used
             N(0,π/8)  and then ﬂip thex-axis by a probability of 0.5.                          for visual inspection since the test dataset does not provide
             After that, each box and points within 110%      size of the                       groundtruthlabels. AsshowninFigure3,ourapproachstill
             box randomly shift by (∆ x∼N(0,3),∆y = 0 ,∆z∼                                      detectsPedestrianreasonablywelldespitenotachievingthe
             N(0,3)) .  We use a 10%     larger box to select the points to                     topscore. OnelikelyreasonwhyPedestriandetectionisnot
             preventcuttingtheobject. Duringthetranslation,wecheck                              asgoodasthatforCarandCyclististhattheverticesarenot
             and avoid collisions among boxes, or between background                            dense enough to achieve more accurate bounding boxes.
             points and boxes. During graph construction, we use a ran-                         4.4.AblationStudy
             dom voxel downsample to induce vertex jitter.
                                                                                                   For the ablation study, we follow the standard practice
             4.3.1   Results                                                                    [10][21][5] and split the training samples into a training
                                                                                                splitof3712samplesandavalidationsplitof3769samples.
             We have submitted our results to the KITTI 3D object de-                           We use the training split to train the network and evaluate
             tection benchmark and the Bird’s Eye View (BEV) object                             its accuracy on the validation split.  We follow the same
             detection benchmark. In Table1and Table2, we compare                               protocol and assess the accuracy by AP. Unless explicitly
             our results with the existing literature.  The KITTI dataset                       modiﬁed for a controlled experiment, the network conﬁgu-
             evaluatestheAveragePrecision(AP)onthreedifﬁcultylev-                               ration and training parameters are the same as those in the
                                                                   Car                           Pedestrian                          Cyclist
         Method                    Modality                        Car                           Pedestrian                          Cyclist
         Method                    Modality           Easy    Moderate    Hard          Easy    Moderate    Hard          Easy    Moderate    Hard
UberATG-ContFuse[12]           LiDAR+Image           82.54       66.22       64.04Easy    Moderate    HardEasy    Moderate    HardN/A         N/A         N/AEasy    Moderate    HardN/A         N/A         N/A
UberATG-ContFuse[12]           LiDAR+Image           88.81       85.83       77.33      N/A         N/A         N/A       N/A         N/A         N/A
     AVOD-FPN[8]               LiDAR+Image           81.94       71.88       66.38     50.80       42.81       40.88      64.00       52.18       46.61
     AVOD-FPN[8]               LiDAR+Image           88.53       83.79        77.9     58.75       51.05       47.54      68.06       57.48       50.77
     F-PointNet[13]            LiDAR+Image           81.20       70.39       62.19     51.21       44.89       40.23      71.96       56.77       50.39
     F-PointNet[13]            LiDAR+Image           88.70      84.00      75.33       58.09       50.22       47.20      75.38       61.96       54.68
  UberATG-MMF[11]              LiDAR+Image           86.81       76.75       68.41      N/A         N/A         N/A       N/A         N/A         N/A
  UberATG-MMF[11]              LiDAR+Image           89.49       87.47       79.10      N/A         N/A         N/A       N/A         N/A         N/A
      VoxelNet[23]                  LiDAR            81.97       65.46       62.85     57.86      53.42      48.87        67.17       47.65       45.11
      VoxelNet[23]                  LiDAR            89.60       84.81       78.57     65.95      61.05      56.98        74.41       52.18       50.49
     SECOND[19]                     LiDAR            83.13       73.66       66.20     51.07       42.56       37.29      70.51       53.85       53.85
     SECOND[19]                     LiDAR            88.07       79.37       77.95     55.10       46.27       44.76      73.67       56.04       48.78
    PointPillars[10]                LiDAR            79.05       74.99       68.30     52.08       43.53       41.49      75.78       59.07       52.92
    PointPillars[10]                LiDAR            88.35       86.10       79.83     58.66       50.23       47.19      79.14       62.25       56.00
    PointRCNN[16]                   LiDAR            85.94       75.76       68.32     49.43       41.78       38.63      73.93       59.60       53.59
        STD[21]                     LiDAR            89.66       87.76      86.89      60.99       51.39       45.89      81.04       65.32       57.85
        STD[21]                     LiDAR            86.61       77.63      76.06      53.08       44.24       41.97      78.89       62.53       55.77
    OurPoint-GNN                    LiDAR            93.11      89.17        83.9      55.36       47.07       44.61      81.17      67.28      59.67
    OurPoint-GNN                    LiDAR            88.33      79.47       72.29      51.92       43.77       40.14      78.60      63.48      57.08

              Figure 3. Qualitative results on the KITTI test dataset using Point-GNN. We show the predicted 3D bounding box of Cars (green), Pedes-
              trians (red) and Cyclists (blue) on both the image and the point cloud. Best viewed in color.
                                                                                               individual accuracy in every category. Similarly, when not
                                                                                               using auto-registration, box merging and box scoring (Row
                                                                                               5) also achieve higher accuracy than standard NMS (Row
                                                                                               1).  These results demonstrate the effectiveness of the box
                                                                                               scoring and merging.
                   Table 3. Ablation study on theval. split of KITTI data.                     Auto-registration mechanism. Table3also shows the ac-
                                                                                               curacyimprovementfromtheauto-registrationmechanism.
              previous section. We focus on the detection of Car because                       As shown in Row 2, by using auto-registration alone, we
              of its dominant presence in the dataset.                                         also surpass the baseline without auto-registration (Row 1)
              Boxmergingandscoring. In Table3, we compare the ob-                              on every category of 3D detection and the moderate and
              ject detection accuracy with and without box merging and                         hard categories of BEV detection. The performance on the
              scoring. For the test without box merging, we modify line                        easy category of BEV detection decreases slightly but re-
              11 in Algorithm1.  Instead of taking the       medianbound-                      mains close.  Combining the auto-registration mechanism
              ingbox,wedirectlytaketheboundingboxwiththehighest                                with box merging and scoring (Row 6), we achieve higher
              classiﬁcation score as in standard NMS. For the test with-                       accuracy than using the auto-registration alone (Row 2).
              out box scoring, we modify lines 12 and 13 in Algorithm1                         However, the combination of all three modules (Row 6)
              to set the highest classiﬁcation score as the box score. For                     does not outperform box merging and score (Row 5).  We
              the test without box merging or scoring, we modify lines                         hypothesize that the regularization likely needs to be tuned
              11, 12, and 13, which essentially leads to standard NMS.                         after adding the auto-registration branch.
              Row2ofTable3showsabaselineimplementationthatuses                                     We further investigate the auto-registration mechanism
              standard NMS with the auto-registration mechanism.  As                           by visualizing the offset∆xin Equation4. We extract               ∆x
              shown in Row 3 and Row 4 of Table3, both box merging                             from different GNN iterations and add them to the vertex
              and box scoring outperform the baseline. When combined,                          position. Figure4shows the vertices that output detection
              as shown in Row 6 of the table, they further outperform the                      results and their positions with added offsets.  We observe
    Box     Box   Auto       BEVAP(Car)               3DAP(Car)
   Merge   Score   Reg.  Easy   Moderate   Hard  Easy   Moderate   Hard
1    -        -       - 89.11     87.14     86.1885.46     76.80     74.89
2    -        -    ✓    89.03     87.43     86.3985.58     76.98     75.69
3    ✓         -    ✓   89.33     87.83     86.6386.59     77.49     76.35
4    -    ✓   ✓         89.60     88.02     86.9787.40     77.90     76.75
5    ✓    ✓         -   90.03     88.27     87.1288.16     78.40     77.49
6    ✓    ✓   ✓         89.82     88.31     87.1687.89     78.34     77.38

                                                                                            Table 5. Average precision on downsampled KITTIval. split.
             Figure 4. An example from the val. split showing the vertex loca-              a breakdown of the current inference time helps with fu-
             tions with added offsets. The blue dot indicates the original posi-            ture optimization. Our implementation is written in Python
             tion of the vertices. The orange, purple, and red dots indicate the            and uses Tensorﬂow for GPU computation.  We measure
             original position with added offsets from the ﬁrst, the second, and            the inference time on a desktop with Xeon E5-1630 CPU
             the third graph neural network iterations. Best viewed in color.               and GTX 1070 GPU. The average processing time for one
                                                                                            sampleinthevalidationsplitis643ms. Readingthedataset
                                                                                            andrunningthecalibrationtakes11.0%time(70ms). Creat-
                                                                                            ingthegraphrepresentationconsumes18.9%time(121ms).
                                                                                            The inference of the GNN takes 56.4% time (363ms). Box
                                                                                            merging and scoring take 13.1% time (84ms).
                                                                                            Robustness on LiDAR sparsity.  The KITTI dataset col-
             Table 4. Average precision on the KITTI val. split using different             lects point cloud data using a 64-scanning-line LiDAR.
             number of Point-GNN iterations.                                                Such a high-density LiDAR usually leads to a high cost.
                                                                                            Therefore,itisofinteresttoinvestigatetheobjectdetection
             that the vertex positions with added offsets move towards                      performanceinalessdensepointcloud. TomimicaLiDAR
             the center of vehicles.  We see such behaviors regardless                      systemwithfewerscanninglines,wedownsamplethescan-
             of the original vertex position.  In other words, when the                     ning lines in the KITTI validation dataset. Because KITTI
             GNN gets deeper, the relative coordinates of the neighbor                      givesthepointcloudwithoutthescanninglineinformation,
             vertices depend less on the center vertex position but more                    weusek-meanstoclustertheelevationanglesofpointsinto
             on the property of the point cloud.  The offset ∆xcancels                      64 clusters, where each cluster represents a LiDAR scan-
             thetranslationofthecentervertex,andthusreducesthesen-                          ning line. We then downsample the point cloud to 32, 16, 8
             sitivity to the vertex translation.  These qualitative results                 scanning lines by skipping scanning lines in between. Our
             demonstrate that Equation4helps to reduce the translation                      test results on the downsampled KITTI validation split are
             variance of vertex positions                                                   shown in Table5. The accuracy for the moderate and hard
             Point-GNN iterations.  Our Point-GNN reﬁne the vertex                          levels drops fast with downsampled data, while the detec-
             states iteratively.  In Table4, we study the impact of the                     tion for the easy level data maintains a reasonable accuracy
             number of iterations on the detection accuracy.  We train                      untilitisdownsampledto8scanninglines. Thisisbecause
             Point-GNNs withT = 1  ,T = 2  , and compare them with                          that the easy level objects are mostly close to the LiDAR,
             T = 3  , which is the conﬁguration in Section4.3.1. Addi-                      and thus have a dense point cloud even if the number of
             tionally, we train a detector using the initial vertex state di-               scanning lines drops.
             rectly without any Point-GNN iteration. As shown in Table
             4,theinitialvertexstatealoneachievesthelowestaccuracy                          5.Conclusion
             since it only has a small receptive ﬁeld around the vertex.
             Without Point-GNN iterations, the local information can-                           Wehavepresentedagraphneuralnetwork,namedPoint-
             not ﬂow along the graph edges, and therefore its receptive                     GNN, to detect 3D objects from a graph representation
             ﬁeld cannot expand.  Even with a single Point-GNN itera-                       of the point cloud.   By using a graph representation, we
             tionT = 1  , the accuracy improves signiﬁcantly.T = 2                          encode the point cloud compactly without mapping to a
             has higher accuracy thanT = 3  , which is likely due to the                    grid or sampling and grouping repeatedly. Our Point-GNN
             training difﬁculty when the neural network goes deeper.                        achievestheleadingaccuracyinboththe3DandBird’sEye
             Running-time analysis.  The speed of the detection algo-                       View object detection of the KITTI benchmark. Our exper-
             rithm is important for real-time applications such as au-                      iments show the proposed auto-registration mechanism re-
             tonomousdriving. However,multiplefactorsaffecttherun-                          duces transition variance, and the box merging and scoring
             ning time, including algorithm architecture, code optimiza-                    operationimprovesthedetectionaccuracy. Inthefuture,we
             tion and hardware resource.  Furthermore, optimizing the                       plantooptimizetheinferencespeedandalsofusetheinputs
             implementation is not the focus of this work.   However,                       from other sensors.
NumberofNumberof   BEVAP(Car)BEVAP(Car)          3DAP(Car)3DAP(Car)
scanninglineiterationsEasy    Moderate    HardEasy    Moderate    HardEasy    Moderate    HardEasy    Moderate    Hard
  T=064      87.24      77.39      75.8489.82      88.31      87.1673.90      64.42      59.9187.89      78.34      77.38
  T=132      89.83      87.67      86.3089.62      79.84      78.7788.00      77.89      76.1485.31      69.02      67.68
  T=216      90.00      88.37      87.2286.56      61.69      60.5788.34      78.51      77.6766.67      50.23      48.29
  T=38       89.82      88.31      87.1649.72      34.05      32.8887.89      78.34      77.3826.88      21.00      19.53

References                                                                         [15]X. Qi, R. Liao, J. Jia, S. Fidler, and R. Urtasun.  3d graph
 [1]Jon L. Bentley, Donald F. Stanat, and E.Hollins Williams.                             neural networks for rgbd semantic segmentation.   In 2017
      The complexity of ﬁnding ﬁxed-radius near neighbors.  In-                           IEEEInternationalConferenceonComputerVision(ICCV),
      formation Processing Letters, 6(6):209 – 212, 1977.3                                pages 5209–5218, Oct 2017.1,2
 [2]YinBi,AaronChadha,AlhabibAbbas,EirinaBourtsoulatze,                            [16]Shaoshuai Shi, Xiaogang Wang, and Hongsheng Li. Pointr-
      and Yiannis Andreopoulos.  Graph-based object classiﬁca-                            cnn: 3d object proposal generation and detection from point
      tion for neuromorphic vision sensing.  In The IEEE Inter-                           cloud. InTheIEEEConferenceonComputerVisionandPat-
      national Conference on Computer Vision (ICCV), October                              tern Recognition (CVPR), June 2019.1,2,6
      2019.1,2                                                                     [17]Yue  Wang,  Yongbin  Sun,  Ziwei  Liu,  Sanjay  E.  Sarma,
 [3]R.Q.Charles,H.Su,M.Kaichun,andL.J.Guibas.Pointnet:                                    Michael M. Bronstein, and Justin M. Solomon.   Dynamic
      Deeplearningonpointsetsfor3dclassiﬁcationandsegmen-                                 graph cnn for learning on point clouds.  ACM Transactions
      tation.  In 2017 IEEE Conference on Computer Vision and                             on Graphics (TOG), 2019.1,2
      Pattern Recognition (CVPR), pages 77–85, July 2017.1,2,                      [18]Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long,
      4                                                                                   Chengqi Zhang, and Philip S. Yu.  A Comprehensive Sur-
                                                                                          vey  on  Graph  Neural  Networks.     arXiv  e-prints,  page
 [4]X. Chen, H. Ma, J. Wan, B. Li, and T. Xia.  Multi-view 3d                             arXiv:1901.00596, Jan 2019.2
      object detection network for autonomous driving.  In 2017                    [19]Yan Yan, Yuxing Mao, and Bo Li.  Second:  Sparsely em-
      IEEE Conference on Computer Vision and Pattern Recogni-                             beddedconvolutionaldetection. Sensors,18(10),2018.2,5,
      tion (CVPR), pages 6526–6534, July 2017.2                                           6
 [5]Yilun Chen, Shu Liu, Xiaoyong Shen, and Jiaya Jia.  Fast                       [20]B.Yang,W.Luo,andR.Urtasun. Pixor: Real-time3dobject
      point r-cnn. In The IEEE International Conference on Com-                           detection from point clouds. In2018 IEEE/CVF Conference
      puter Vision (ICCV), October 2019.6                                                 on Computer Vision and Pattern Recognition, pages 7652–
 [6]Andreas Geiger, Philip Lenz, and Raquel Urtasun.  Are we                              7660, June 2018.2
      ready for autonomous driving?  the kitti vision benchmark                    [21]ZetongYang,YananSun,ShuLiu,XiaoyongShen,andJiaya
      suite. InConferenceonComputerVisionandPatternRecog-                                 Jia. Std: Sparse-to-dense 3d object detector for point cloud.
      nition (CVPR), 2012.5                                                               In The IEEE International Conference on Computer Vision
 [7]Peter J. Huber.  Robust estimation of a location parameter.                           (ICCV), October 2019.1,2,5,6
      Ann. Math. Statist., 35(1):73–101, 03 1964.4                                 [22]Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barn-
 [8]J.Ku,M.Moziﬁan,J.Lee,A.Harakeh,andS.L.Waslander.                                      abas  Poczos,  Ruslan  R  Salakhutdinov,  and  Alexander  J
      Joint 3d proposal generation and object detection from view                         Smola.  Deep sets.  In I. Guyon, U. V. Luxburg, S. Bengio,
      aggregation. In2018IEEE/RSJInternationalConferenceon                                H.Wallach,R.Fergus,S.Vishwanathan,andR.Garnett,edi-
      IntelligentRobotsandSystems(IROS),pages1–8,Oct2018.                                 tors,AdvancesinNeuralInformationProcessingSystems30,
      6                                                                                   pages 3391–3401. Curran Associates, Inc., 2017.1,2
 [9]Loic Landrieu and Martin Simonovsky.   Large-scale point                       [23]Y. Zhou and O. Tuzel.  Voxelnet:  End-to-end learning for
      cloudsemanticsegmentationwithsuperpointgraphs. InThe                                point cloud based 3d object detection.  In 2018 IEEE/CVF
      IEEE Conference on Computer Vision and Pattern Recogni-                             Conference on Computer Vision and Pattern Recognition,
      tion (CVPR), June 2018.1,2                                                          pages 4490–4499, June 2018.2,3,5,6
[10]Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou,
      Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders
      for object detection from point clouds. In The IEEE Confer-
      ence on Computer Vision and Pattern Recognition (CVPR),
      June 2019.1,2,3,5,6
[11]MingLiang,BinYang,YunChen,RuiHu,andRaquelUrta-
      sun.  Multi-task multi-sensor fusion for 3d object detection.
      In The IEEE Conference on Computer Vision and Pattern
      Recognition (CVPR), June 2019.2,6
[12]MingLiang,BinYang,ShenlongWang,andRaquelUrtasun.
      Deepcontinuousfusionformulti-sensor3dobjectdetection.
      In The European Conference on Computer Vision (ECCV),
      September 2018.6
[13]C. R. Qi, W. Liu, C. Wu, H. Su, and L. J. Guibas.  Frus-
      tum pointnets for 3d object detection from rgb-d data.   In
      2018 IEEE/CVF Conference on Computer Vision and Pat-
      tern Recognition, pages 918–927, June 2018.2,6
[14]Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. Point-
      net++: Deep hierarchical feature learning on point sets in a
      metric space. arXiv preprint arXiv:1706.02413, 2017.2

