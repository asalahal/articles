     Graph Networks as Learnable Physics Engines for Inference and Control
    Alvaro Sanchez-Gonzalez1  Nicolas Heess1  Jost Tobias Springenberg1  Josh Merel1  Martin Riedmiller1
                                                   Raia Hadsell1  Peter Battaglia1
                            Abstract
     Understanding  and  interacting  with  everyday
     physical scenes requires rich knowledge about
     thestructureoftheworld,representedeitherim-
     plicitly in a value or policy function, or explic-
     itly in a transition model.  Here we introduce a
     new class of learnable models—based ongraph
     networks—which implement an inductive bias
     forobject- andrelation-centric representationsof
     complex, dynamical systems. Our results show
     that as a forward model, our approach supports
     accuratepredictions from realandsimulated data,
     and surprisingly strong and efﬁcient generaliza-
     tion, acrosseight distinctphysical systems which
     we varied parametrically and structurally.  We
     alsofoundthatourinferencemodelcanperform
     system identiﬁcation. Our models are also differ-
     entiable,andsupportonlineplanningviagradient-
     based trajectory optimization, as well as ofﬂine
     policy optimization. Our framework offers new
     opportunities for harnessing and exploiting rich                     Figure1.(Top)Ourexperimentalphysicalsystems. (Bottom)Sam-
     knowledge about theworld, and takes akey step                        plesofparametrizedversionsofthesesystems(seevideos:link).
     toward building machines with more human-like
     representations of the world.
                                                                          of objects2 and their relations, applying the same object-
                                                                          wisecomputationstoallobjects,andthesamerelation-wise
1. Introduction                                                           computations to all interactions. This allows for combina-
                                                                          torialgeneralizationtoscenariosneverbeforeexperienced,
Manydomains,suchasmathematics,language,andphysical                        whose underlying components and compositional rules are
systems, are combinatorially complex.  The possibilities                  well-understood. For example, particle-based physics en-
scale rapidlywith the numberof elements. Forexample,a                     gines make the assumption that bodies follow the same dy-
multi-linkchain canassumeshapesthat areexponentialin                      namics, and interact with each other following similar rules,
the number of angles each link can take, and a box full of                e.g., via forces, which is how they can simulate limitless
bouncing balls yields trajectories which are exponential in               scenarios given different initial conditions.
the number of bounces that occur. How can an intelligent                  Here we introduce a new approach for learning and con-
agent understand and control such complex systems?                        trolling complex systems, by implementing a structural in-
Apowerfulapproachistorepresentthesesystemsinterms                         ductive bias for object- and relation-centric representations.
   1DeepMind,London,UnitedKingdom. Correspondenceto: Al-                  Ourapproach uses“graph networks”(GNs),a classof neu-
varoSanchez-Gonzalez<alvarosg@google.com>,PeterBattaglia                  ral networks that can learn functions on graphs (Scarselli
<peterbattaglia@google.com>.                                              et al.,2009b;Li et al.,2015;Battaglia et al.,2016;Gilmer
                                                                          etal.,2017). In aphysicalsystem,the GNletsusrepresent
Proceedings of the 35   th  International Conference on Machine
Learning, Stockholm, Sweden, PMLR 80, 2018. Copyright 2018                    2“Object”herereferstoentitiesgenerally,ratherthanphysical
by the author(s).                                                         objects exclusively.

                               Graph Networks as Learnable Physics Engines for Inference and Control
the bodies (objects) with the graph’s nodes and the joints                  ters et al.,2017;Ehrhardt et al.,2017;Amos et al.,2018).
(relations)withitsedges. Duringlearning,knowledgeabout                      Our inferencemodel sharessimilar aimsas approaches for
body dynamics is encoded in the GN’s node update func-                      learning system identiﬁcation explicitly (Yu et al.,2017;
tion, interaction dynamics are encoded in the edge update                   Pengetal.,2017),learningpoliciesthatarerobusttohidden
function, and global system properties are encoded in the                   property variations (Rajeswaran et al.,2016),and learning
globalupdatefunction. Learnedknowledgeissharedacross                        exploration strategies in uncertain settings (Schmidhuber,
theelementsofthesystem,whichsupportsgeneralization                          1991;Sun et al.,2011;Houthooft et al.,2016).  We use
to new systems composed of the same types of body and                       our learned models for model-based planning in a similar
joint building blocks.                                                      spirit to classic approaches which use pre-deﬁned models
Across seven complex, simulated physicalsystems, and one                    (Li&Todorov,2004;Tassaetal.,2008;2014),andourwork
real robotic system (see Figure1), our experimental results                 also relates tolearning-based approaches for model-based
show that our GN-basedforwardmodels support accurate                        control (Atkeson & Santamaria,1997;Deisenroth & Ras-
and generalizable predictions, inference models3 support                    mussen,2011;Levine & Abbeel,2014). We also explore
systemidentiﬁcationinwhichhiddenpropertiesareabduced                        jointlylearningamodelandpolicy(Heessetal.,2015;Gu
fromobservations,andcontrolalgorithmsyieldcompetitive                       etal.,2016;Nagabandietal.,2017). Notablerecent,concur-
performance against strong baselines.   This work repre-                    rentwork(Wangetal.,2018)usedaGNNtoapproximatea
sentstheﬁrstgeneral-purpose,learnablephysicsenginethat                      policy, which complements our use of a related architecture
can handle complex, 3D physical systems. Unlike classic                     to approximate forward and inference models.
physics engines, our model has no speciﬁc a priori knowl-
edge ofphysical laws, butinstead leverages itsobject- and                   3. Model
relation-centric inductive bias to learn toapproximate them                 Graph representation of a physical system.    Our ap-
via supervised training on current-state/next-state pairs.                  proach  is  founded  on  the  idea  of  representing  phys-
Our work makes three technical contributions: GN-based                      ical  systems  as  graphs:   the  bodies  and  joints  corre-
forward models, inference models, and control algorithms.                   spond  to  the  nodes  and  edges,   respectively,   as  de-
The forward and inference models are based on treating                      picted  in  Figure2a.    Here  a  (directed)              graph  is  de-
physical systems as graphs and learning about them using                    ﬁnedasG = (  g,{ni}i=1···Nn,{ej,sj,rj}j=1···Ne), where
GNs. Our controlalgorithmuses ourforwardandinference                        g is a vector of global features,{ni}i=1···Nn is a set of
models for planning and policy learning.                                    nodes where each niis a vector of node features, and
(For full algorithm, implementation, and methodological                     {ej,sj,rj}j=1···Neisasetofdirectededgeswhereejisa
details,aswellasvideosfromallofourexperiments,please                        vector ofedge features, andsjandrjare theindices of the
see the Supplementary Material.)                                            sender and receiver nodes, respectively.
                                                                            We distinguish between static and dynamic properties in a
2. Related Work                                                             physical scene, which we represent in separate graphs. A
                                                                            static graphGscontains static information about the param-
Our work draws on several lines of previous research. Cog-                  etersofthesystem,includingglobalparameters(suchasthe
nitive scientists have longpointed to richgenerativemodels                  timestep,viscosity,gravity,etc.),perbody/nodeparameters
as central to perception, reasoning, and decision-making                    (such as mass, inertia tensor, etc.), and per joint/edge pa-
(Craik,1967;Johnson-Laird,1980;Miall&Wolpert,1996;                          rameters(suchasjointtypeandproperties, motortypeand
Spelke & Kinzler,2007;Battaglia et al.,2013). Our core                      properties,etc.). AdynamicgraphGdcontainsinformation
modelimplementationisbasedonthebroaderclassofgraph                          about the instantaneous state of the system. This includes
neural networks (GNNs) (Scarselli et al.,2005;2009a;b;                      eachbody/node’s3DCartesianposition,4Dquaternionori-
Bruna et al.,2013;Li et al.,2015;Henaff et al.,2015;Du-                     entation, 3D linear velocity, and 3D angular velocity.4 Ad-
venaud etal.,2015;Dai et al.,2016;Defferrard et al.,2016;                   ditionally, it contains the magnitude of the actions applied
Niepertetal.,2016;Kipf&Welling,2016;Battagliaetal.,                         to the different joints in the corresponding edges.
2016;Watters et al.,2017;Raposo et al.,2017;Santoro                             4Somephysicsengines,suchasMujoco(Todorovetal.,2012),
etal.,2017;Bronsteinetal.,2017;Gilmeretal.,2017). One                       represent systems using “generalized coordinates”, which sparsely
of our key contributions is an approach for learning physi-                 encode degrees of freedom rather than full body states.  Gen-
cal dynamics models (Grzeszczuk et al.,1998;Fragkiadaki                     eralized coordinates offer advantages such as preventing bodies
etal.,2015;Battagliaetal.,2016;Chang etal.,2016;Wat-                        connectedbyjointsfromdislocating(becausethereisnodegreeof
                                                                            freedomforsuchdisplacement). Inourapproach,however,such
   3We use the term “inference” in the sense of “abductive                  representations do not admit sharing asnaturally because there are
inference”—roughly,constructingexplanationsfor(possiblypar-                 different input and output representations for a body depending on
tial) observations—and not probabilistic inference, per se.                 the system’s constraints.

                              Graph Networks as Learnable Physics Engines for Inference and Control
Figure2.Graph representations and GN-based models. (a) A physical system’s bodies and joints can be represented by a graph’s nodes
and edges, respectively. (b) A GN block takes a graph as input and returns a graph with the same structure but different edge, node,
and global features as output (see Algorithm1). (c) A feed-forward GN-based forward model for learning one-step predictions. (d) A
recurrent GN-based forward model. (e) A recurrent GN-based inference model for system identiﬁcation.
Algorithm 1Graph network, GN                                              concatenated5 withG (e.g., a graph skip connection), and
   Input: Graph,G = (  g,{ni},{ej,sj,rj})                                 providedasinputtothesecondGN,whichreturnsanoutput
   for each edge{ej,sj,rj}do                                              graph,G∗. OurforwardmodeltrainingoptimizestheGNso
      Gather sender and receiver nodesnsj,nrj                             thatG∗’s{ni}featuresreﬂectpredictionsaboutthestatesof
      Compute output edges,e∗j= fe(g,nsj,nrj,ej)                          eachbodyacrossatime-step. ThereasonweusedtwoGNs
   end for                                                                wastoallowallnodes andedgesto communicate witheach
   for each node{ni}do                                                    otherthroughtheg′outputfromtheﬁrstGN.Preliminary
      Aggregatee∗jper receiver,ˆei=∑          j/rj=ie∗j                   tests suggested this provided large performance advantages
      Compute node-wise features,n∗i= fn(g,ni,ˆei)                        over a single IN/GN (see ablation study in SM FigureH.2).
   end for                                                                We also introduce a second, recurrent GN-based forward
   Aggregate all edges and nodesˆe =∑           je∗j, ˆn=∑    in∗i        model, which contains three RNN sub-modules (GRUs,
   Compute global features,g∗= fg(g,ˆn,ˆe)                                (Cho et al.,2014)) applied across all edges, nodes, and
   Output: Graph,G∗= (  g∗,{n∗i},{e∗j,sj,rj})                             globalfeatures,respectively,beforebeingcomposedwitha
                                                                          GN block (see Figure2d).
                                                                          Our forward models were all trained to predict state dif-
Graph networks.    TheGNarchitectures introducedhere                      ferences, so to compute absolute state predictions we up-
generalizeinteractionnetworks(IN)(Battagliaetal.,2016)                    datedtheinputstatewiththepredictedstatedifference. To
in several ways. They include global representations and                  generatealong-rangerollout trajectory,werepeatedlyfed
outputsforthestateofasystem,aswellasper-edgeoutputs.                      absolute state predictions and externally speciﬁed control
They are deﬁned as “graph2graph” modules (i.e., they map                  inputsbackintothemodelasinput,iteratively. Asdatapre-
inputgraphstooutputgraphswithdifferentedge,node,and                       and post-processing steps, we normalized the inputs and
global features),which can becomposed indeep and recur-                   outputs to the GN model.
rentneuralnetwork(RNN)conﬁgurations. AcoreGNblock
(Figure2b) contains three sub-functions—edge-wise,     fe,                Inference models.   System identiﬁcation refers to infer-
node-wise,fn, and global,fg—which can be implemented                      ences about unobserved properties of a dynamic system
using standard neural networks. Here we use multi-layer                   based on its observed behavior.  It is important for con-
perceptrons (MLP). A single feedforward GN pass can be                    trollingsystemswhoseunobservedpropertiesinﬂuencethe
viewed as one step of message-passing on a graph (Gilmer                  controldynamics. Herewe consider“implicit”systemiden-
etal.,2017),where     feisﬁrstappliedtoupdatealledges,fn                  tiﬁcation,inwhichinferencesaboutunobservedproperties
isthenappliedtoupdateallnodes,andfgisﬁnallyapplied                        are not estimated explicitly,but are expressed inlatent rep-
to update the global feature. See Algorithm1for details.                  resentationswhich aremadeavailable toothermechanisms.
                                                                          We introduce a recurrent GN-based inference model, which
Forward models.    For prediction, we introduce a GN-                     observes only the dynamic states of a trajectory and con-
based forward model for learning to predict future states                     5We deﬁne theterm “graph-concatenation”as combining two
fromcurrentones. Itoperatesononetime-step,andcontains                     graphs by concatenating their respective edge, node, and global
two GNs composed sequentially in a “deep” arrangement                     features. We deﬁne “graph-splitting” as splitting the edge, node,
(unshared parameters; see Figure2c). The ﬁrst GN takes an                 and global features of one graph to form two new graphs with the
inputgraph,G,and producesalatent graph,G′. ThisG′is                       same structure.

                              Graph Networks as Learnable Physics Engines for Inference and Control
Figure3.Evaluation rollout in a Swimmer6. Trajectory videos are here:link-P.F.S6. (a) Frames of ground truth and predicted states
over a 100 step trajectory.  (b-e) State sequence predictions for link #3 of the Swimmer.  The subplots are (b)x,y,z-position, (c)
q0,q1,q2,q3 -quaternion orientation, (d)x,y,z-linear velocity, and (e)x,y,z-angular velocity. [au] indicates arbitrary units.
structsalatentrepresentationoftheunobserved,staticprop-                    tems, and recording the state transitions. We also trained
erties(i.e., performs implicitsystemidentiﬁcation). It takes               models from recorded trajectories of a real JACO robotic
as input a sequence of dynamic state graphs,Gd, under                      under human control during a stacking task.
some control inputs, and returns an output,G∗(T), afterT                   In experiments that examined generalization and system
timesteps. ThisG∗(T) isthenpassedtoaone-stepforward                        identiﬁcation,wecreatedadatasetofversionsofseveralof
model by graph-concatenating it with an input dynamic                      our systems—Pendulum, Cartpole, Swimmer, Cheetah and
graph,Gd. The recurrent core takes as input,Gd, and hid-                   JACO— with procedurally varied parameters and structure.
den graph,Gh, which are graph-concatenated5 and passed                     Wevariedcontinuouspropertiessuchaslinklengths,body
to a GN block (see Figure2e). The graph returned by the                    masses, and motor gears. In addition, we also varied the
GN block is graph-split5 to form an output,G∗, and up-                     numberoflinksintheSwimmer’sstructure,from3-15(we
datedhiddengraph,G∗h. Thefullarchitecturecanbetrained                      refer to a swimmer withN  links as SwimmerN ).
jointly,and learns to infer unobserved properties of the sys-
tem fromhow thesystem’s observed features behave, and                      MPCplanning.    WeusedourGN-basedforwardmodelto
use them to make more accurate predictions.                                implement MPC planning by maximizing a dynamic-state-
                                                                           dependent reward along a trajectory from a given initial
Control algorithms.    For control, we exploit the fact that               state. We used our GN forward model to predict theN -step
the GN is differentiable to use our learned forward and                    trajectories(N istheplanninghorizon)inducedbyproposed
inference models for model-based planning within a clas-                   action sequences, as well as the total reward associated
sic, gradient-based trajectory optimization regime, also                   withthetrajectory. Weoptimizedtheseactionsequencesby
knownasmodel-predictivecontrol(MPC).Wealsodevelop                          backpropagatinggradientsofthetotalrewardwithrespectto
an agent which simultaneously learns a GN-based model                      theactions,andminimizingthenegativerewardbygradient
and policy function via Stochastic Value Gradients (SVG)                   descent, iteratively.
(Heess et al.,2015).                      6
                                                                           Model-based  reinforcement  learning.   To  investigate
4. Methods                                                                 whether our GN-based model can beneﬁt reinforcement
                                                                           learning (RL) algorithms, we used our model within an
Environments.   Our  experiments  involved  seven  actu-                   SVG regime (Heess et al.,2015). The GN forward model
ated Mujoco simulation environments (Figure1).   Six                       wasusedasadifferentiableenvironmentsimulatortoobtain
were from the “DeepMind Control Suite” (Tassa et al.,                      a gradient of the expected return (predicted based on the
2018)—Pendulum,Cartpole,Acrobot,Swimmer,Cheetah,                           next state generated by a GN) with respect to a parame-
Walker2d—and one was a model of a JACO commercial                          terized, stochastic policy, which was trained jointly with
robotic arm.  We generated training data for our forward                   the GN. For our experiments we used a single step predic-
models by applying simulated random controls to the sys-                   tion (SVG(1)) andcompared tosample-efﬁcient model-free
   6MPC and SVG are deeply connected: in MPC the control                   RL baselines using either stochastic policies (SVG(0)) or
inputsareoptimizedgiventheinitialconditionsinasingleepisode,               deterministic policies via the Deep Deterministic Policy
while in SVG a policy function that maps states to controls is             Gradients (DDPG) algorithm (Lillicrapet al.,2016) (which
optimized over states experienced during training.                         is also used as a baseline in the MPC experiments).

                              Graph Networks as Learnable Physics Engines for Inference and Control
                                                                          Figure5.Prediction errors, on (a) one-step and (b) 20-step evalua-
                                                                           tions,betweenthebestMLPbaselineandthebestGNmodelafter
Figure4.(a)One-step and(b)100-step rollouterrorsfor different              72hoursoftraining. Swimmer6predictionerrors,on(c)one-step
models and training (different bars) on different test data (x-axis        and (d) 20-step evaluations, between the best MLP baseline and
labels),relativetotheconstantpredictionbaseline(blackdashed                the best GN model for data in the training set (dark), data in the
line). Blue bars are GN models trained on single systems. Red              validation set (medium), and data from DDPG agent trajectories
and yellow bars are GN modelstrained on multiple systems, with            (light). Thenumbersabovethebarsindicatetheratiobetweenthe
(yellow) and without (red) parametric variation. Note that includ-         corresponding generalization test error (medium or light) and the
ing Cheetah in multiple system training caused performance to              training error (dark).
diminish(lightredvsdarkredbars),whichsuggestssharingmight
not always be beneﬁcial.
                                                                           baseline. After normalization, the errors wereaveraged to-
Baseline comparisons.    As a simple baseline, we com-                     gether. Allerrorsreportedarecalculatedfor1000100-step
pared our forward models’ predictions to a constant pre-                   sequences from the test set.
dictionbaseline, which copiedthe input state as the output
state. WealsocomparedourGN-basedforwardmodelwith                           5. Results: Prediction
a learned, MLP baseline, which we trained to make for-                     Learning a forward model for a single system.   Our re-
wardpredictionsusingthesamedataastheGNmodel. We                            sultsshow thatthe GN-basedmodelcan betrained tomake
replaced thecore GN withan MLP,and ﬂattened andcon-                       veryaccurateforwardpredictionsunderrandomcontrol. For
catenatedthegraph-structuredGNinputandtargetdatainto                       example, theground truthand model-predictedtrajectories
a vector suitable for input to the MLP. We swept over 20                   forSwimmer6werebothvisuallyandquantitativelyindistin-
unique hyperparameter combinations for the MLP architec-                   guishable(seeFigure3). Figure4’sblackbarsshowthatthe
ture, with up to 9 hidden layers and 512 hidden nodes per                  predictionsacrossmostothersystemswerefarbetterthan
layer.                                                                     theconstantpredictionbaseline. Asastrongerbaselinecom-
As an MPC baseline, with a pre-speciﬁed physical model,                    parison, Figures5a-b show that our GN model had lower
we used a Differential Dynamic Programming algorithm                       error than the MLP-based model in 6 of the 7 simulated
(Tassa et al.,2008;2014) that had access to the ground-                    controlsystemswetested. Thiswasespeciallypronounced
truthMujocomodel. Wealsousedthetwomodel-freeRL                             forsystemswithmuchrepeatedstructure,suchastheSwim-
agents mentioned above, SVG(0) and DDPG, as baselines                      mer, while for systems with little repeated structure, such
in sometests. Some ofthe trajectoriesfrom aDDPG agent                      as Pendulum, there was negligible difference between the
inSwimmer6werealsousedtoevaluategeneralizationof                           GN and MLP baseline.  These results suggest that a GN-
the forward models.                                                        basedforwardmodelisveryeffectiveatlearningpredictive
                                                                           dynamics in a diverse range of complex physical systems.
Prediction performance evaluation.    Unless otherwise                    We also found that the GN generalized better than the MLP
speciﬁed, we evaluated our models on squared one-step                      baselinefromtrainingtotestdata,aswellasacrossdifferent
dynamicstatedifferences(one-steperror)andsquaredtra-                       action distributions. Figures5c-d show that for Swimmer6,
jectory differences (rollout error) between the prediction                 therelativeincreaseinerrorfrom trainingtotestdata,and
and the ground truth.  We calculated independent errors                    to data recordedfrom a learned DDPGagent, was smaller
for position, orientation, linear velocity angular velocity,               for theGN modelthan forthe MLP baseline. We speculate
andnormalizedthemindividuallytotheconstantprediction                       thattheGN’ssuperiorgeneralization isaresultofimplicit

                              Graph Networks as Learnable Physics Engines for Inference and Control
Figure6.Zero-shotdynamicsprediction. Thebarsshowthe100-
step rollout error of a model trained on a mixture of 3-6 and 8-9          Figure7.Real and predicted test trajectoriesof aJACO robotarm.
linkSwimmers,andtestedonSwimmerswith3-15links. Thedark                     The recurrent model tracks the ground truth (a) orientations and
barsindicatetestSwimmerswhosenumberoflinksthemodelwas                      (b) angular velocities closely. (c) The total 100-step rollout error
trainedon(video:link-P.F.SN),thelightbarsindicate Swimmers                 was muchbetter forthe recurrent model, though the feed-forward
it was not trained on (video:link-P.F.SN(Z)).                              model was still well below the constant prediction baseline.  A
                                                                           videoofaMujocorenderingofthetrueandpredictedtrajectories:
regularizationduetoitsinductivebiasforsharingparame-                       link-P.F.JR.
tersacrossallbodiesandjoints;theMLP,inprinciple,could
devotedisjointsubsetsofitscomputationstoeachbodyand                        jectoriesarevisuallyveryclosetothegroundtruth(video:
joint, which might impair generalization.                                  link-P.F.SN(Z)).
Learning a forward model for multiple systems.   An-                       Realrobotdata.    Toevaluateourapproach’sapplicability
other important feature of our GN model is that it is very                 tothe realworld,we trainedGN-basedforwardmodels on
ﬂexible, able to handle wide variation across a system’s                   real JACO proprioceptive data; under manual control by
properties,and acrosssystemswith differentstructure. We                    a human performing a stacking task.  We found the feed-
tested how it learned forward dynamics of systems with                     forward GNperformance was not asaccurate asthe recur-
continuously varying static parameters, using a new dataset                rent GN forward model7: Figure7shows a representative
where the underlying systems’ bodies and joints had differ-                predicted trajectoryfrom the test set,as well as overallper-
ent masses, body lengths, joint angles, etc.  These static                 formance. These results suggest that our GN-based forward
state features were provided to the model via the input                    model is a promising approach for learning models in real
graphs’ node and edge attributes. Figure4shows that the                    systems.
GNmodel’sforwardpredictionswereagainaccurate,which
suggests it can learn well even when the underlying system                 6. Results: Inference
properties vary.
We next explored the GN’s inductive bias for body- and                     In many real-world settings the system’s state is partially
joint-centric learning by testing whether a single model                   observable. Robot arms often use joint angle and velocity
can make predictions across multiple systems that vary in                  sensors,butotherpropertiessuchasmass,jointstiffness,etc.
their number of bodies and the joint structure.  Figure6                   are often not directly measurable. We applied our system
shows that when trained on a mixed dataset of Swimmers                     identiﬁcation inference model (see Model Section3) to a
with 3-6, 8-9 links, the GN model again learned to make                    settingwhereonlythedynamicstatevariables(i.e.,position,
accurateforwardpredictions. Wepushedthisevenfurther                        orientation,andlinearandangularvelocities)wereobserved,
by training a single GN model on multiple systems, with                    and found it could support accurate forward predictions
completelydifferentstructures,andfoundsimilarlypositive                    (during its “prediction phase”) after observing randomly
results(see Figure4, redand yellow bars). This highlights                  controlled system dynamics during an initial 20-step “ID
akeydifference,intermsofgeneralapplicability,between                       phase” (see Figure8).
GN and MLP models:  the GN can naturally operate on                        TofurtherexploretheroleofourGN-basedsystemidenti-
variably structured inputs, while the MLP requires ﬁxed-                   ﬁcation, wecontrasted themodel’s predictionsafter an ID
size inputs.                                                               phase, which containeduseful controlinputs, against anID
The GN model can even generalize, zero-shot, to systems                    phasethatdidnotapplycontrolinputs,acrossthreediffer-
whosestructurewasheldoutduringtraining,aslongasthey                        ent Pendulum systems with variable, unobserved lengths.
are composed ofbodies and jointssimilar to those seendur-                  Figure9shows that the GN forward model with an identiﬁ-
ingtraining. FortheGNmodeltrainedonSwimmerswith                            ableIDphasemakesveryaccuratepredictions,butwithan
3-6, 8-9 links, wetested on held-out Swimmerswith 7 and                    unidentiﬁable ID phase its predictions are very poor.
10-15 links. Figure6shows that zero-shot generalization                        7This mightresult from lagor hysteresis whichinduces long-
performance isveryaccurate for 7and 10 linkSwimmers,                       rangetemporaldependenciesthatthefeed-forwardmodelcannot
and degrades gradually from 11-15 links.  Still, their tra-                capture.

                               Graph Networks as Learnable Physics Engines for Inference and Control
Figure8.System identiﬁcation performance.  The y-axis repre-
sents100-steprollout error,relativetothetrivial constantpredic-
tionbaseline(blackdashedline). ThebaselineGN-basedmodel
(black bars) with no system identiﬁcationmodule performs worst.
A model which was always provided the true static parameters
(mediumblue bars)andthusdid notrequiresystem identiﬁcation                 Figure10.Framesfroma40-stepGN-basedMPCtrajectoryofthe
performed best. A model without explicit access to the true static         simulatedJACOarm. (a)Imitationoftheposeofeachindividual
parameters, but with a system identiﬁcation module (light blue             bodyofthearm(13variablesx9bodies). (b)Imitationofonlythe
bars),performedgenerallywell,sometimesveryclosetothemodel                  palm’spose(13variables). Thefullvideosarehere:link-C.F.JA(o)
which observed the true parameters. But when that same model               andlink-C.F.JA(a).
was presented with an ID phase whose hidden parameters were
different(butfromthesamedistribution)fromitspredictionphase                proachesforexploitingourGNmodelincontinuouscontrol.
(red bars), its performance was similar or worse than the model
(black) with no ID information available.  (The N/A column is
because our Swimmer experiments always varied the number of                Model-predictive   control   for   single   systems.    We
linksaswellasparameters,whichmeanttheinferredstaticgraph                   trained a GN forward model and used it for MPC by opti-
could not be concatenated with the initial dynamic graph.)                 mizingthecontrolinputsviagradientdescenttomaximize
                                                                           predicted reward under a known reward function. We found
                                                                           our GN-based MPC could support planning in all of our
                                                                           control systems, across a range of reward functions.  For
                                                                           example, Figure10shows frames of simulated JACO tra-
                                                                           jectories matching a target pose and target palm location,
                                                                           respectively, under MPC with a 20-step planning horizon.
                                                                           IntheSwimmer6systemwitharewardfunctionthatmax-
Figure9.SystemidentiﬁcationanalysisinPendulum. (a)Control                  imized the head’s movement toward a randomly chosen
inputs areapplied to threePendulums with different, unobservable           target, GN-based MPC with a 100-step planning horizon se-
lengths during the 20-step ID phase, which makes the system                lectedcontrolinputsthatresultedincoordinated,swimming-
identiﬁable. (b) The model’s predicted trajectories (dashed curves)        like movements. Despite the fact that the Swimmer6 GN
track the ground truth (solid curves) closely in the subsequent            modelused forMPCwastrainedto makeone-steppredic-
80-step prediction phase.  (c) No control inputs are applied to            tions under random actions, its swimming performance was
the same systems during the ID phase, which makes the system               close to both that of a more sophisticated planning algo-
identiﬁable. (d)Themodel’spredictedtrajectoriesacrosssystems               rithmwhichusedthetrueMujocophysicsasitsmodel,as
are very different from the ground truth.                                  well as that of a learned DDPG agent trained on the sys-
AkeyadvantageofoursystemID approachisthatoncethe                           tem(see Figure11a). And whenwetrained theGNmodel
IDphasehasbeenperformedforsomesystem,theinferred                           using a mixture of both random actions and DDPG agent
representation can be storedand reused to make trajectory                  trajectories, there was effectively no difference in perfor-
                                                                           mance between our approach, versus the Mujoco planner
predictionsfromdifferentinitialstatesofthesystem. This                     andlearned DDPGagentbaselines (seevideo:link-C.F.S6).
contrasts with an approach that would use an RNN to both                   For Cheetah with reward functions for maximizing forward
infer the system properties and use them throughout the                    movement,maximizingheight,maximizingsquaredverti-
trajectory, which thus would require identifying the same                  cal speed, and maximizing squared angular speed of the
system from data each time a new trajectory needs to be                    torso, MPC with a 20-step horizon using a GN model re-
predicted given different initial conditions.                              sultedinrunning,jumping,andotherreasonablepatternsof
7. Results: Control                                                        movements (see video:link-C.F.Ch(k)).
Differentiablemodelscanbevaluableformodel-basedse-                         Model-predictivecontrolformultiplesystems.   Similar
quential decision-making, and here we explored two ap-                     to how our forward models learned accurate predictions

                              Graph Networks as Learnable Physics Engines for Inference and Control
                                                                          Figure12.Learning curves for Swimmer6 SVG agents. The GN-
                                                                          basedagent(blue)asymptotesearlier,andatahigherperformance,
                                                                          thanthemodel-freeagent(red). Thelinesrepresentmedianperfor-
Figure11.GN-basedMPCperformance(%distancetotargetafter                    mancefor6randomseeds,with25and75%conﬁdenceintervals.
700 steps) for (a) model trained on Swimmer6 and (b) model
trained on Swimmers with 3-15 links (see Figure6). In (a), GN-
based MPC (blue point) is almost as good as the Mujoco-based
planner(blackline)andtrainedDDPG(greyline)baselines. When                 the model and policy were trained simultaneously.8 Com-
theGN-basedMPC’smodelistrainedonamixtureofrandomand                       pared to a model-free agent (SVG(0)), our GN-based SVG
DDPGagentSwimmer6trajectories(redpoint),itsperformanceis                  agent (SVG(1)) achieved a higher level performance af-
as good as the strong baselines. In (b) the GN-based MPC (blue            terfewerepisodes(Figure12). ForGN-basedagentswith
point) (video:link-C.F.SN) is competitive with a Mujoco-based             more than one forward step (SVG(2-4)), however, the per-
planner baseline (black) (video:link-C.F.SN(b)) for 6-10 links,           formance was not signiﬁcantly better, and in some cases
but is worse for 3-5 and 11-15 links.  Note, the model was not
trained on the open points, 7 and 10-15 links, which correspond           was worse (SVG(5+)).
tozero-shotmodelgeneralizationforcontrol. Errorbarsindicate
mean and standard deviation across 5 experimental runs.                   8. Discussion
                                                                          This workintroduced anew classof learnableforward and
across multiple systems, we also found they could support                 inferencemodels,basedon“graphnetworks”(GN),which
MPC across multiple systems (inthis video, a single model                 implement an object- and relation-centric inductive bias.
is used for MPC in Pendulum, Cartpole, Acrobot, Swim-                     Acrossarangeofexperimentswefoundthatthesemodels
mer6 and Cheetah:link-C.F.MS). We also found GN-based                     are surprisingly accurate, robust, and generalizable when
MPCcouldsupportzero-shotgeneralizationinthecontrol                        usedfor prediction,system identiﬁcation,andplanning in
setting, for a single GN model trained on Swimmers with                   challenging, physical systems.
3-6, 8-9 links, and tested on MPC on Swimmers with 7,                     WhileourGN-basedmodelsweremosteffectiveinsystems
10-15links. Figure11b showsthatit performedalmostas                       withcommonstructureamongbodiesandjoints(e.g.,Swim-
well as the Mujoco baseline for many of the Swimmers.                     mers), they were less successful when there was not much
                                                                          opportunityforsharing(e.g.,Cheetah). Ourapproachalso
Model-predictive  control  with  partial  observations.                   does not address a commonproblem for model-based plan-
Because real-world control settings are often partially ob-               ners that errors compound over long trajectory predictions.
servable,weusedthesystemidentiﬁcationGNmodel(see                          Some key future directions include using our approach for
Sections3and5) for MPC under partial observations in                      controlinreal-worldsettings,supportingsimulation-to-real
Pendulum,Cartpole,SwimmerN,Cheetah,andJACO.The                            transferviapre-trainingmodelsinsimulation,extendingour
modelwastrained asintheforwardprediction experiments,                     models to handle stochastic environments, and performing
withan IDphasethat applied20random controlinputs to                       system identiﬁcation over the structure of the system as
implicitly infer the hidden properties. Our results show that             well as the parameters. Our approach may also be useful
ourGN-basedforwardmodelwithasystemidentiﬁcation                           within imagination-based planning frameworks (Hamrick
module is able to control these systems (Cheetah video:                   et al.,2017;Pascanu et al.,2017), as well as integrated
link-C.I.Ch. All videos are in SM TableA.2).                              architectures with GN-like policies (Wang et al.,2018).
                                                                          This work takes a key step towards realizing the promise of
Model-based reinforcement learning.    In our second ap-                  model-basedmethodsbyexploitingcompositionalrepresen-
proach to model-based control, we jointly trained a GN                    tationswithinapowerfulstatisticallearningframework,and
modelandapolicyfunctionusingSVG(Heessetal.,2015),                         opens new paths for robust, efﬁcient, and general-purpose
where the model was used to backpropagate error gradients                 patterns of reasoning and decision-making.
tothepolicyinordertooptimizeitsparameters. Crucially,                         8In preliminary experiments, we found little beneﬁt of pre-
ourSVGagentdoesnotuseapre-trainedmodel,butrather                          training the model, though further exploration is warranted.

                             Graph Networks as Learnable Physics Engines for Inference and Control
References                                                               Ehrhardt, S., Monszpart, A., Mitra, N. J., and Vedaldi, A.
Amos, B., Dinh, L., Cabi, S., Rothrl, T., Muldal, A., Erez,                 Learningaphysicallong-termpredictor. arXivpreprint
  T., Tassa, Y., de Freitas, N., and Denil, M.  Learning                    arXiv:1703.00247, 2017.
  awareness models. ICLR, 2018.                                          Fragkiadaki, K., Agrawal, P., Levine, S., and Malik, J.
Atkeson, C. G. and Santamaria, J. C. A comparison of di-                    Learning visual predictive models of physics for play-
  rectandmodel-basedreinforcementlearning. InRobotics                       ing billiards. CoRR, abs/1511.07404, 2015.
  andAutomation,1997.Proceedings.,1997IEEEInterna-                       Gilmer,J., Schoenholz,S. S.,Riley, P. F., Vinyals,O., and
  tional Conference on, volume 4, pp. 3557–3564. IEEE,                      Dahl,G.E. Neuralmessagepassingforquantumchem-
  1997.                                                                     istry. InICML, pp. 1263–1272, 2017.
Battaglia, P., Pascanu, R., Lai, M., Rezende, D. J., et al.              Grzeszczuk, R., Terzopoulos, D., and Hinton, G. Neuroan-
  Interactionnetworksforlearningaboutobjects,relations                      imator:  Fast neural network emulation and control of
  and physics. InNIPS, pp. 4502–4510, 2016.                                 physics-based models.  In Proceedings of the 25th an-
Battaglia,P.W.,Hamrick,J.B.,andTenenbaum,J.B. Sim-                          nual conference on Computer graphics and interactive
  ulation as an engine of physical scene understanding.                     techniques, pp. 9–20. ACM, 1998.
  Proceedings of the National Academy of Sciences, 110                   Gu, S., Lillicrap, T. P., Sutskever, I., and Levine, S. Con-
  (45):18327–18332, 2013.                                                   tinuous deep q-learning with model-based acceleration.
Bronstein,M.M.,Bruna,J.,LeCun,Y.,Szlam,A.,andVan-                           CoRR, abs/1603.00748, 2016.
  dergheynst, P. Geometric deep learning: going beyond                   Hamrick, J. B., Ballard, A. J., Pascanu, R., Vinyals, O.,
  euclideandata. IEEESignalProcessingMagazine,34(4):                        Heess, N., and Battaglia, P. W.  Metacontrol for adap-
  18–42, 2017.                                                              tive imagination-based optimization.    arXiv preprint
Bruna, J., Zaremba, W., Szlam, A., and LeCun, Y.  Spec-                     arXiv:1705.02670, 2017.
  tral networks and locally connected networks on graphs.                Heess, N., Wayne, G., Silver, D., Lillicrap, T., Erez, T.,
  arXivpreprintarXiv:1312.6203, 2013.                                       and Tassa, Y.  Learning continuous control policies by
Chang, M. B., Ullman, T., Torralba, A., and Tenenbaum,                      stochasticvaluegradients. InNIPS,pp.2944–2952,2015.
  J.B. Acompositionalobject-basedapproachtolearning                      Henaff,  M.,  Bruna,  J.,  and LeCun,  Y.    Deep convolu-
  physical dynamics.  arXiv preprint arXiv:1612.00341,                      tionalnetworkson graph-structureddata. arXivpreprint
  2016.                                                                     arXiv:1506.05163, 2015.
Cho,K.,VanMerri¨enboer,B.,Bahdanau,D.,andBengio,Y.                       Houthooft, R., Chen, X., Duan, Y., Schulman, J., Turck,
  Onthepropertiesofneuralmachinetranslation: Encoder-                       F. D., and Abbeel, P.  Curiosity-driven exploration in
  decoder approaches.  arXiv preprint arXiv:1409.1259,                      deepreinforcementlearningviabayesianneuralnetworks.
  2014.                                                                     CoRR, abs/1605.09674, 2016.
Craik,K.J.W. Thenatureofexplanation,volume445. CUP                       Johnson-Laird, P. N. Mental models in cognitive science.
  Archive, 1967.                                                            Cognitivescience, 4(1):71–115, 1980.
Dai,H.,Dai,B.,andSong,L. Discriminativeembeddings                        Kingma, D.P. andWelling, M. Auto-encodingvariational
  of latent variable models for structured data. In ICML,                   bayes. InICLR, 2014.
  pp. 2702–2711, 2016.                                                   Kipf, T. N. and Welling, M.  Semi-supervised classiﬁca-
Defferrard, M., Bresson, X., and Vandergheynst, P.  Con-                    tion with graph convolutional networks. arXiv preprint
  volutional neural networks on graphs with fast localized                  arXiv:1609.02907, 2016.
  spectral ﬁltering. InNIPS, pp. 3844–3852, 2016.                        Levine, S. and Abbeel, P.  Learning neural network poli-
Deisenroth, M. and Rasmussen, C. Pilco: A model-based                       cieswithguidedpolicysearchunderunknowndynamics.
  and data-efﬁcient approach to policysearch. InICML28,                     In Ghahramani, Z., Welling, M., Cortes, C., Lawrence,
  pp. 465–472. Omnipress, 2011.                                             N.D.,andWeinberger,K.Q.(eds.),NIPS27,pp.1071–
Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bom-                     1079. Curran Associates, Inc., 2014.
  barell, R., Hirzel,T., Aspuru-Guzik, A.,and Adams, R. P.               Li, W. andTodorov,E. Iterative linearquadratic regulator
  Convolutional networks on graphs for learning molecular                   design for nonlinear biological movement systems.  In
  ﬁngerprints. InNIPS, pp. 2224–2232, 2015.                                 ICINCO(1), pp. 222–229, 2004.

                             Graph Networks as Learnable Physics Engines for Inference and Control
Li,  Y.,  Tarlow,  D.,  Brockschmidt,  M.,  and  Zemel,  R.             Scarselli,F.,Gori,M.,Tsoi,A.C.,Hagenbuchner,M.,and
  Gated graph sequence neural networks. arXiv preprint                     Monfardini,G. Thegraphneuralnetworkmodel. IEEE
  arXiv:1511.05493, 2015.                                                  TransactionsonNeuralNetworks, 20(1):61–80, 2009b.
Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T.,        Schmidhuber, J. Curious model-building control systems.
  Tassa,Y.,Silver,D.,andWierstra,D. Continuouscontrol                      In Proc. Int. J. Conf. Neural Networks, pp. 1458–1463.
  with deep reinforcement learning. InICLR, 2016.                          IEEE Press, 1991.
Miall,R.C.andWolpert,D.M. Forwardmodelsforphysio-                       Spelke, E.S. and Kinzler, K. D. Core knowledge. Develop-
  logicalmotorcontrol. Neuralnetworks,9(8):1265–1279,                      mentalscience, 10(1):89–96, 2007.
  1996.                                                                 Sun, Y., Gomez, F. J., and Schmidhuber, J.  Planning to
Nagabandi, A., Kahn, G., Fearing, R. S., and Levine, S.                    be surprised: Optimal bayesian exploration in dynamic
  Neural network dynamics for model-based deep rein-                       environments. InAGI, volume6830 ofLectureNotesin
  forcement learning with model-free ﬁne-tuning.  arXiv                    ComputerScience, pp. 41–51. Springer, 2011.
  preprintarXiv:1708.02596, 2017.                                       Tassa, Y., Erez, T., and Smart, W. D.  Receding horizon
Niepert,M.,Ahmed,M.,andKutzkov,K. Learningconvo-                           differential dynamic programming. In NIPS, pp. 1465–
  lutionalneuralnetworksforgraphs. InICML,pp.2014–                         1472, 2008.
  2023, 2016.                                                           Tassa, Y., Mansard, N., and Todorov, E.  Control-limited
Pascanu, R., Li, Y., Vinyals, O., Heess, N., Buesing, L.,                  differentialdynamic programming. InRoboticsandAu-
  Racani`ere,S.,Reichert,D.,Weber,T.,Wierstra,D.,and                       tomation (ICRA), 2014 IEEE International Conference
  Battaglia,P. Learningmodel-basedplanningfromscratch.                     on, pp. 1168–1175. IEEE, 2014.
  arXivpreprintarXiv:1707.06170, 2017.                                  Tassa,Y.,Doron,Y.,Muldal,A.,Erez,T.,Li,Y.,Casas, D.
Peng, X. B., Andrychowicz, M., Zaremba, W., andAbbeel,                     d. L., Budden, D., Abdolmaleki, A., Merel, J., Lefrancq,
  P. Sim-to-real transfer of robotic control with dynamics                 A.,  et al.    Deepmind control suite.    arXiv preprint
  randomization. CoRR, abs/1710.06537, 2017.                               arXiv:1801.00690, 2018.
Rajeswaran, A., Ghotra, S., Levine, S., and Ravindran, B.               Todorov, E., Erez, T., and Tassa, Y.  Mujoco: A physics
  Epopt:  Learning robust neural network policies using                    engine for model-based control.  In Intelligent Robots
  model ensembles. CoRR, abs/1610.01283, 2016.                             andSystems(IROS),2012IEEE/RSJInternationalCon-
                                                                           ferenceon, pp. 5026–5033. IEEE, 2012.
Raposo, D., Santoro, A., Barrett, D., Pascanu, R., Lilli-               Wang, T., Liao, R., Ba, J., and Fidler, S. Nervenet: Learn-
  crap, T., and Battaglia, P. Discovering objects and their                ingstructuredpolicywithgraphneuralnetworks. ICLR,
  relations from entangled scene representations.  arXiv                   2018.
  preprintarXiv:1702.05068, 2017.
Rezende, D. J., Mohamed, S., and Wierstra, D.  Stochas-                 Watters, N., Zoran, D., Weber, T., Battaglia, P., Pascanu, R.,
  tic backpropagation and approximate inference in deep                    andTacchetti,A. Visualinteractionnetworks: Learning
  generative models. InICML31, 2014.                                       a physics simulator from video. InNIPS, pp. 4542–4550,
                                                                           2017.
Santoro, A., Raposo, D., Barrett, D. G., Malinowski, M.,                Yu, W., Liu, C. K., and Turk, G.  Preparing for the un-
  Pascanu, R., Battaglia, P., and Lillicrap, T.  A simple                  known: Learning a universal policy with online system
  neural network module for relational reasoning. InNIPS,                  identiﬁcation. CoRR, abs/1702.02453, 2017.
  pp. 4974–4983, 2017.
Scarselli,F.,Yong,S.L.,Gori,M.,Hagenbuchner,M.,Tsoi,
  A.C., andMaggini,M. Graph neuralnetworksfor rank-
  ingwebpages. InWebIntelligence,2005.Proceedings.
  The2005IEEE/WIC/ACMInternationalConferenceon,
  pp. 666–672. IEEE, 2005.
Scarselli,F.,Gori,M.,Tsoi,A.C.,Hagenbuchner,M.,and
  Monfardini,G. Computationalcapabilitiesofgraphneu-
  ralnetworks. IEEETransactionsonNeuralNetworks,20
  (1):81–102, 2009a.

