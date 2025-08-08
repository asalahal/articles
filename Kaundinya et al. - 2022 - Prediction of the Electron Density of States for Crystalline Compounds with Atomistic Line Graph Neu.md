JOM, Vol. 74, No. 4, 2022
https://doi.org/10.1007/s11837-022-05199-y
� 2022 The Minerals, Metals & Materials Society


COMPUTATIONAL DESIGN OF ALLOYS FOR ENERGY TECHNOLOGIES

# Prediction of the Electron Density of States for Crystalline Compounds with Atomistic Line Graph Neural Networks (ALIGNN)


PRATHIK R. KAUNDINYA, [1] KAMAL CHOUDHARY, [2,3]

and SURYA R. KALIDINDI 1,4,5


1.—School of Computational Science and Engineering, Georgia Institute of
Technology, Atlanta, GA 30332, USA. 2.—Materials Science and Engineering Division, National
Institute of Standards and Technology, Gaithersburg, MD 20899, USA. 3.—Theiss
Research, La Jolla, CA 92037, USA. 4.—G.W. Woodruff School of Mechanical
Engineering, Georgia Institute of Technology, Atlanta, GA 30332, USA. 5.—email: surya.kalidindi@me.gatech.edu


Machine learning (ML)-based models have greatly enhanced the traditional
materials discovery and design pipeline. Specifically, in recent years, surrogate ML models for material property prediction have demonstrated success in
predicting discrete scalar-valued target properties to within reasonable
accuracy of their DFT-computed values. However, accurate prediction of
spectral targets, such as the electron density of states (DOS), poses a much
more challenging problem due to the complexity of the target, and the limited
amount of available training data. In this study, we present an extension of
the recently developed atomistic line graph neural network to accurately
predict DOS of a large set of material unit cell structures, trained to the
publicly available JARVIS-DFT dataset. Furthermore, we evaluate two
methods of representation of the target quantity: a direct discretized spectrum, and a compressed low-dimensional representation obtained using an
autoencoder. Through this work, we demonstrate the utility of graph-based
featurization and modeling methods in the prediction of complex targets that
depend on both chemistry and directional characteristics of material structures.



screening of the extremely large materials design

INTRODUCTION

space. One of the primary potential applications of

Although physics-based modeling approaches high-throughput screening with ML-based surrosuch as density functional theory (DFT) have been gate models is in the identification of materials that
used extensively in the development of material are well suited for use in energy storage applicastructure–property relationships, they generally tions. Materials selected for such applications genincur a high computational cost. As such, they are erally have a desired stability (i.e., target formation
not well suited to materials discovery efforts that energy range), exhibit a particular conductive/insudemand high-throughput screening to identify novel lating capability (i.e., target electronic bandgap
chemistries and material structures possessing cer- range), and have a particular specific heat capacity
tain desired combinations of physical and chemical (i.e., phononic and electronic heat capacity ranges).
properties. In this context, machine learning (ML)- Thus, a suitable workflow for identifying such
based surrogate models trained on available DFT materials is to shortlist a set of candidate materials
datasets offer an attractive avenue for the rapid (derived from the extensive materials search space)

and perform DFT calculations for even more accurate property prediction on this shortlisted set of
(Received December 11, 2021; accepted January 27, 2022; candidate materials.
published online March 2, 2022)



INTRODUCTION



Although physics-based modeling approaches
such as density functional theory (DFT) have been
used extensively in the development of material
structure–property relationships, they generally
incur a high computational cost. As such, they are
not well suited to materials discovery efforts that
demand high-throughput screening to identify novel
chemistries and material structures possessing certain desired combinations of physical and chemical
properties. In this context, machine learning (ML)based surrogate models trained on available DFT
datasets offer an attractive avenue for the rapid



1395


1396 Kaundinya, Choudhary, and Kalidindi



ML-based surrogate models have been successfully employed in predicting scalar bulk material
properties, such as optical and electronic bandgaps, [1][,][2] formation energy, [3][–][9] Debye temperature, [10]

atomization energies, [11][–][14] and polarizability of crystalline compounds. [15] ML models have also been
developed to predict electronic properties, such as
the position of Wannier centers, [16] electron charge
density, [17][–][21] and electron density of states. [22][–][24]

These electronic properties can then be used to
predict other physical properties of the material
(e.g., the estimation of exchange–correlation
energy [25] from the electron density). Of particular
interest in this study is the electron density of states
(DOS), which reflects the number of states that may
be occupied by the electrons in the material at a
spectrum of energy levels. The DOS can be used to
compute directly many other useful physical properties of the material, such as the effective mass of
electrons in charge carriers, [26][,][27] the electronic contribution to heat capacity in metals, [28][,][29] and the
electronic bandgap of crystal structures. The salient
features of the DOS spectrum include the positions
and magnitudes of prominent peaks, and the value
of the DOS at the Fermi energy (i.e., at the origin of
the plot).
Much of the prior work in ML-based predictions of
the DOS spectra of materials has been limited to
predicting certain features of it, learning patterns in
the DOS across similar compounds, or restricting
analysis to datasets with limited chemical diversity.
This is largely due to the complexity arising from
the target being a spectrum (as opposed to a scalarvalued quantity) and the availability of limited
training data. For example, Schutt et al. compared
two different information encoding schemes [the
Coulomb matrix and partial radial distribution
function (PRDF)] and proposed a model using
kernel ridge regression to predict the value of the
DOS at the Fermi energy (E f ), which is closely
related to the Seebeck coefficient and the critical
temperature of superconductors. [30] As another
example, Broderick and Rajan extracted the main
features of the DOS spectra of single-element
crystal systems with principal component analysis
(PCA) and identified 16 spectral patterns that could
be weighted differently to yield close approximations to the DOS spectra of transition metals. [31]

Another effort demonstrated a method for ‘‘pattern
learning’’ the DOS spectra of multi-component alloy
systems [23] and nanoparticles. [24] In these studies,
PCA was used to generate weighted components of
several basis spectra, which were then used to
adequately represent the DOS spectra of the different material systems. In a more sophisticated
approach, Chandrasekaran et al. [19] developed a
rotationally invariant grid-based fingerprinting
scheme using Gaussian functions of varying widths
at each energy level. This representation was used
to model the local density of states (LDOS) at each
energy level within polyethylene and aluminum



crystal systems, which were then summed to produce the overall DOS spectra. Mahmoud et al. [22]

compared different approaches for representing the
DOS (with principal components, a pointwise discretization or cumulative distribution function), and
developed predictors for the LDOS of several configurations of silicon.
The prior studies discussed above are not readily
extendable to ML-based predictions of DOS for large
collections of diverse crystalline materials exhibiting different symmetries and chemical elements.
This difficulty stems from the need to develop an
efficient featurization scheme for the diverse material structures (i.e., inputs) present in such large
collections. It is important to note that electronic
properties such as the DOS are sensitive to spatial
and directional characteristics of the material structure (such as interatomic distances, bond angles,
and localized distortions). As such, the first step in a
broadly applicable ML-based framework to predict
the DOS involves developing a compact (but comprehensive) representation that accounts for the
diverse structural and chemical information needed
to represent complex material structures. Typically,
such representations should seek to quantify the
salient details of the local atomic neighborhoods as
the main features (i.e., regressors) controlling the
complex interatomic interactions underlying the
computations of the DOS. Some of these representations are kernel-based, accounting for individual
interactions between pairs of atoms or a collection of
atoms belonging to selected species (e.g., smooth
overlap of atomic positions power spectrum, [32]

PRDF, [30] and grid-based descriptors proposed by
Chandrasekaran et al. [19] ). In an alternate approach,
the molecular structure is comprehensively featureengineered using the framework of n-point spatial
correlations, [33][–][35] which utilize voxelized representations and benefit from the computational efficiency of the fast Fourier transform v algorithm. [7]

This framework aims to capture comprehensively
the salient features in the molecular structure,
including the directional information (e.g., bond
angles) that is lost in the pairwise descriptors
mentioned earlier.
Recently, a new approach using graph embeddings has been proposed to capture the short-range
and long-range atomic interactions in molecular
structures. Specifically, the use of crystal graphs
(CG) as inputs to a Graph Convolutional Neural
Network (GCNN) has been termed as CGCNN
(Crystal Graph Convolutional Neural Network). [36]

Multiple variants of CGCNN have already shown
promise for materials discovery problems [e.g.,
iCGCNN (improved CGCNN), [37] OGCNN (Orbital
GCNN) [38] and GATGNN (Graph Attention GNN) [39] ].
In these approaches, the structure is typically
represented as a graph, with the nodes capturing
information about the individual constituent atoms
and the edges encoding interatomic information
(e.g., interatomic distances). The node


Prediction of the Electron Density of States for Crystalline Compounds with Atomistic Line 1397
Graph Neural Networks (ALIGNN)



representations are transformed over several layers
(referred to as graph convolution layers) utilizing
information from its close neighbors in the graph.
Finally, the transformed node representations are
aggregated and further transformed in a feedforward neural network comprising of fully-connected (FC) layers, with the target property of
interest being the output of these layers. These
models have been successfully applied to predict the
scalar properties of molecular structures with good
accuracy. However, these early efforts did not
capture directional information in crystal systems,
and instead utilized limited rotationally invariant
features such as interatomic distances, bond
valences, and atomic radii. [36][–][38] The Atomistic Line
Graph Neural Network (ALIGNN) [40] is a recently
developed extension of GCNN approach that
addresses the shortcomings described above by
capturing interatomic distances and bond angles
in two separate graphs, known as the crystal graph
and line graph, respectively. As a result of being
able to capture the directional aspect of the local
environment, ALIGNN models perform significantly
better than CGCNN in the prediction of properties
mentioned above (e.g., 50% improvement in the
prediction of the formation energy of crystals, 30%
improvement in the prediction of the electronic
bandgap). [40]

In this study, we utilize the open-source implementation of the ALIGNN framework to effectively
capture unit cell structural information of a broad
range of crystalline materials (comprising different
structural symmetries and chemical elements) and
predict their corresponding DOS spectra. Since the
regression target is a spectral quantity (i.e., DOS),
we also evaluate two methods of representation of
the output: (1) a primitive discretization of the DFTcomputed spectrum with a vector of 300 evenly

�
spaced points in the energy range of ( 5 to 10 eV),
and (2) an autoencoder network to ‘‘learn’’ a concise
low-dimensional representation of the high-dimensional target. In the second representation, the lowdimensional output of ALIGNN is passed through
the decoder segment of the autoencoder network to
recover the desired target (i.e., DOS). High-fidelity
data for � 56 k crystalline materials was obtained
from the publicly available Joint Automated Repository for Various Integrated Simulations–Density
Functional Theory (JARVIS-DFT) dataset [41][–][45] and
used to train the models described in this work.


ALIGNN MODELING FRAMEWORK


As mentioned earlier, ALIGNN encodes crystal
structure information in the form of a crystal graph
and a separate line graph. The crystal graph is
constructed in the same manner as the CGCNN, [36]

and is comprised of two parts: (1) a set of I nodes
representing the individual atoms present in the
crystal unit cell, and (2) edges that quantify the
connectivity between the nodes using suitably



in which K 1 and K 2 are trainable weight matrices, f
indicates a nonlinear activation (such as the ReLU
or SiLU), and � indicates the elementwise multiplication operation. Note that the same weight
matrix K 2 is shared between all neighbors in the
second term of Eq. 2. u [0] ij [l] [represents an intermediate]
update of the node vectors of the line graph using



defined measures. Each node is associated with a
vector embedding v i (i 2 I), which is formed by
concatenating the following five attributes of the
specific atom: electronegativity, electron affinity,
number of valence electrons, first ionization energy,
and atomic radius. Each edge, between a pair of
atoms indexed by i and j, is associated with another
vector embedding u ij . In this work, the edges embed
a simple scalar Euclidean distance measure
between the atomic centers. Only the first 16
nearest neighbors of each atom included in the
crystal graph are considered in this work.
The line graph L is constructed from the crystal
graph G described above. Each node in the line
graph corresponds to an edge in the crystal graph.
In other words, the node embeddings in the line
graph and the edge embeddings in the crystal graph
share the same latent representation. Edges are
then drawn between these nodes when a common
atom is shared (e.g., between u ij and u jk Þ, with the
edge embedding denoted by t ijk and reflecting the
bond angle cosine for the interatomic angle formed
between the ordered triplet of atoms indexed by
i; j; and k. In other words, for each triplet of nodes
(represented by the node indices v i, v j and v k, and
edge vectors u ij and u jk in the crystal graph), the
corresponding edge vector in the line graph, t ijk,
captures an angular measure of the bond angle
involved.
As mentioned earlier, the node and edge representations are updated sequentially over several
graph convolution layers. Specifically, the update
consists of iterative modifications of the line graph
and crystal graph, in order. Each update is known
as an edge-gated graph convolution, and operates on
a node and its local environment. The update
process is described next for the line graph, with a
similar procedure applied to update the crystal
graph. Let l index the updates of the line graph,
operating on nodes u ij and their edges t ijk . We start
by computing normalized edge contributions as:



^t [l] ijk [¼]



r t � [l] ijk � ð1Þ
~~P~~ m2K [r][ t] ~~�~~ [l] ijm ~~�~~ þe



where r �ð Þ indicates the sigmoid function, and e
denotes a small constant added for numerical
stability (taken as 1e � 6). The node vector of the
line graph is then updated as:



^

u [0] ij [l] [¼][ K] 1 [l] [u] [l] ij [þ][ f] P t [l] ijk [�] [K] 2 [l] [u] [l] ij

j2J



!



ð2Þ


1398 Kaundinya, Choudhary, and Kalidindi



only the information from its neighbors in it. The
update from u [0] ij [l] [to][ u] [l] ij [þ][1] occurs with graph convolution
over the crystal graph, as will be described later.
Each edge in the line graph t ijk is updated as:

t ijk [l][þ][1] [¼][ t] [l] ijk [þ][ f][ X] � 1 [l] [u] [l] ij [0] [þ][ X] 2 [l] [u] [l] jk [0] [þ][ X] t [l] [t] [l] ijk � ð3Þ


where X 1, X 2, and X t represent trainable weight
matrices. As mentioned earlier, the update steps for
the crystal graph are similar to those used for the
line graph. For the crystal graph, the normalized
edge contribution as applied in Eq. 1 yields the
normalized intermediate edge vector ^u [0] ij [l] [. The nor-]
malized intermediate edge vector is then used to
update the node vector of the crystal graph as
v [l] i [!][ v] [l] i [þ][1], in a manner similar to Eqs. 2 and 3. The
overall ALIGNN training schedule consists of L
graph convolution updates that are performed
sequentially on the line graph and crystal graph,
respectively. The final output of the crystal graph is
obtained by performing an average pooling operation on all node vectors in it. The averaged node
vector is then used as an input to a series of FC
layers to produce a prediction of the target. The
architecture of the ALIGNN model used in this
work is similar to the original model proposed by
Choudhary and DeCost, [40] extended to support
predictions for a vectorial target (the DOS spectrum). Specifically, the implementation in this work
entails two main extensions: (1) an increase in the
number of neurons in the output of the FC layers
equal to the number of bins chosen for the vector
target, and (2) the consideration of a larger number
of neighbors for each atom included in the crystal
graph (16 in this study vs. 12 in the earlier study).


APPLICATION


As mentioned earlier, a dataset comprising � 56 k
k crystal structures and their corresponding DOS
spectra was obtained from the publicly available
JARVIS-DFT repository and used to train an
ALIGNN model. This dataset included crystal
structural information (atomic centers, species,
lattice constant, lattice type, etc.) and several
DFT-computed chemical properties. Specifically,
DOS spectra were generated for all crystal structures with the OptB88vdW functional, [43] with an
automatic convergence for k-points and a Gaussian
smearing of 0:01 eV. This dataset included crystalline compounds covering 89 species with varied
chemistries. Figure 1 summarizes the frequency of
occurrence of the different species in the compounds
included in the dataset. It can be seen that there is a
broad distribution of chemical species in the compounds included, with most species occurring in �
3000 compounds. Oxygen was the most frequently
observed species (14; 970 compounds) due to it being
a part of several compound classes such as oxides,
sulfates, carbonates, and nitrates.



The dataset was randomly partitioned into a 70–
10–20% split for use as train, validation (during
training), and (fully blind) test sets, respectively.
The primary purpose of the validation set was to
implement an automated early stopping criterion
during the model training phase. The input to the
ALIGNN model comprised the chemical structure
features (species and coordinates) and the target
was the DFT-computed DOS. Figure 2 depicts the
pipeline implemented to train the ALIGNN model.
As shown in Fig. 2a, a nearest-neighbor search was
performed for each atom to build the initial crystal
and line graphs (similar to the original CGCNN). [36]

Each atom was connected to its 16 nearest neighbors, and the crystal and line graphs were initialized using the chemical and structural features
described earlier in ‘‘ALIGNN Modeling Framework’’ section (Fig. 2b). Following this, four
ALIGNN edge-gated graph convolution updates
were performed on both graphs, using the methodology detailed in ‘‘ALIGNN Modeling Framework’’
section (Fig. 2c). Average pooling was performed on
all nodes to yield a globally averaged node vector
(Fig. 2d), which was further connected to four FC
layers (Fig. 2e). Finally, the DOS spectrum was
predicted as the output of the FC layers (Fig. 2f)
using the two different representations described in
the next section.


REPRESENTATION OF THE DOS


As mentioned earlier, since DOS is inherently a
high-dimensional entity, it has presented significant challenges to prior machine learning efforts in
literature. In this work, we explored two different
representations of the DOS spectrum: (1) a primitive discretization of the DOS using 300 uniform
bins on the energy values, and (2) a low-dimensional


Fig. 1. The frequency of occurrence of the different chemical species
in the dataset used in this study.


Prediction of the Electron Density of States for Crystalline Compounds with Atomistic Line 1399
Graph Neural Networks (ALIGNN)


Fig. 2. A flow diagram depicting the ALIGNN pipeline built in this study to predict the DOS spectrum of the compound. (a) The atoms indicated by
the pink marker represent the local neighborhood (i.e., connected atoms in the graph) corresponding to the reference atom. (b) The embedding
procedure creates the node and edge vectors of the crystal graph and line graph. (c) The ALIGNN updates described in ‘‘ALIGNN Modeling
Framework’’ section are used to update the graphs. (d) An average pooling operation is performed on the set of all node vectors present in the
crystal graph I to produce an averaged feature set. (e) A series of fully connected neural networks are used to map the features to the target DOS
spectrum. (f) The predicted DOS spectrum is the output of the model (Color figure online).



representation of the DOS learned using an autoencoder network. Further details of these two
approaches are presented next.


i. Discretized representation of the DOS Since
the DFT-computed DOS spectra were obtained
with varying energygrids for each material,an
interpolation scheme was necessary to standardize the representations for training and
testing the ALIGNN models developed in this
work. The discretized representation of the
DOS was established by interpolating the
DFT-computed DOS values between �5 and
10 eV on a uniform grid with a spacing of
0:05 eV. This discretization interval was selected through multiple trials with the goal of
achieving adequate resolution to unambiguously capture the salient features of the DOS
spectra (e.g., major peak locations and their
intensities) with the minimum number of bins.
Specifically, each trial consisted of implementing the interpolation scheme with a different
energy spacing on a set of randomly chosen
candidate materials, and evaluating the reconstruction performance.
ii. Further, the DOS intensities were normalized by the total number of valence electrons
in the crystal, thereby bounding the target
DOS values to lie in the range ½0; 1�. The
ALIGNN model was trained on the normalized DOS spectra.
iii. Low-dimensional representation of the DOS
As an alternate to the primitive discretized
representation described above, we have also
explored the utility of an autoencoder network
for establishing high-value low-dimensional



latent representations of the normalized DOS.
This is because the 300 discretized values of
the DOS described above are expected to
exhibit some degree of correlation (i.e., dependency) among themselves. This observation is
further supported by recent studies that have
identified several patterns in the DOS spectra
of different alloy compositions. [23][,][24] Autoencoder networks have been shown to be ideal for
addressing this task in other similar problems. [46] These networks typically consist of
two connected components: (1) an encoder to
map the high-dimensional input to a lowdimensional latent embedding, and (2) followed by a decoder that reconstructs the highdimensional input data from the learned lowdimensional latent embedding. Figure 3
shows the architecture of the autoencoder
network trained in this study. The encoder
comprises of several fully connected layers,
with a decreasing number of output neurons
with each layer, as shown. Each layer of the
encoder uses a ReLU activation function. The
architecture of the decoder is designed to
reverse the mapping of the encoder, with all
layers comprising ReLU activation except for
the last layer, which employs a sigmoid activation function to ensure that the reconstructed DOS spectrum only has values in
the range of ½0; 1�. The autoencoder was
trained on the same training data used to
train ALIGNN, with the training objective set
to minimizing the MSE (mean squared error)
between the decoder-reconstructed and the
actual (input) DOS spectra.


1400 Kaundinya, Choudhary, and Kalidindi


Fig. 3. The architecture of the autoencoder used in this study. The encoder was designed to produce a low-dimensional representation of the
300-dimensional DOS vector, while the decoder was designed to fully reconstruct the 300-dimensional DOS vector from the low-dimensional
representation.



RESULTS AND DISCUSSION


Separate ALIGNN models [referred to as Discretized ALIGNN (D-ALIGNN)) and autoencoder
ALIGNN (AE-ALIGNN)] were trained for both
representations of DOS presented in the previous
section. The performance of the models on each test
sample was quantified using two metrics, the mean
absolute error (MAE) and the relative error (RE).
These error metrics were computed using the
predicted values of the DOS discretized over N ¼
300 points denoted as p � fp 1 ; p 1 ; . . . p N g and the
corresponding DFT-computed values denoted as
a � fa 1 ; a 1 ; . . . a N g:


N
MAE ¼ N [1] Pjp i � a i j ð4Þ

i¼1


RE ¼ j ~~[p]~~ [�] ~~a~~ ~~[a]~~ ~~j~~ ð5Þ


where ~~p~~ and ~~a~~ denote the mean DOS of the
predicted and DFT-computed spectra. Based on
the above metrics, a baseline prediction was made
to compare with the performance of the trained
models. The baseline prediction is defined as the
mean DOS spectrum generated from all the materials present in the training set. In other words, the
baseline prediction corresponds to the prediction
made by a model without any parameters or learning capability, simply as the mean of the training
data. The metrics described above were also computed for the baseline prediction.
Figure 4 summarizes the predictive accuracy of
the D-ALIGNN model for the test set. Figure 4a
presents a histogram of the MAE values for the test
samples, with the shaded regions indicating different quartiles in the data. It can be observed that the
DOS spectra for 92% of the test samples was
predicted to within an MAE of 0:02 states/eV/electron, indicating the high accuracy of D-ALIGNN
model. Most interestingly, the first two quartiles in



Fig. 4a (comprising of 2782 samples) exhibit predictive errors below 0:008 states/eV/electron and display reasonable agreement with the DFT-computed
values. In addition, the average MAE over the
entire test set was found to be 0:009 states/eV/electron, with only 6% of the materials having a
predictive error greater than 0:02 states/eV/electron. The average MAE corresponded to a � 3:5
times improvement over the baseline model (which
had an MAE of 0:031 states/eV/electron). Additionally, we observed a strong correlation of the higher
prediction errors with the lack of adequate number
of training points involving certain elements.
Specifically, it was observed that the highest prediction errors occurred in compounds containing one
of the following five elements: Cs, La, Ar, Ce, and W.
These specific elements were in less than 1:5% of the
compounds included in the training set. However,
the predictions are expected to improve as more
data are added to the training set. It was also noted
that the average MAE values for the different
classes of crystal structures (i.e., orthorhombic,
cubic, hexagonal, monoclinic, triclinic, trigonal,
and tetragonal) were in a close range of 0:007–
0:009 states/eV/electron, indicating that the model
produced in this work exhibits good predictions
across all the crystal classes considered. This observation confirms that the implicit feature engineering in the D-ALIGNN model is capable of
identifying the salient features across a diverse set
of crystal structures. The histogram of the relative
errors shown in Fig. 4b reaffirms the excellent
predictive capability of the model, with 85% of the
predictions exhibiting an RE under 0:2. In order to
better visualize the predictions of the model, Fig. 4c
depicts the DFT-computed and model-predicted
DOS spectra for four random samples, one from
each of the quartiles in Fig. 4a. As seen from these
comparisons, the general characteristics in the DOS
spectra (such as peak locations and trends in the
curve) are well captured by the model predictions.


Prediction of the Electron Density of States for Crystalline Compounds with Atomistic Line 1401
Graph Neural Networks (ALIGNN)


Fig. 4. (a) The MAE values for the D-ALIGNN predictions for the test set. The shaded colors represent four quartiles of the prediction error. (b)
The relative errors in the test set. (c) Four comparisons of the predicted DOS for randomly selected test points from each quartile in (a). In these
comparisons, the actual DOS is shown as a black curve, and the predicted DOS is depicted by the colored curve matching the quartile color in (a)
(Color figure online).



However, the model appears to sometimes predict
non-existent peaks (as in InH 2 ) or understate the
magnitudes of existing peaks (as in LaCo 2 Ge 2 ).
Additionally, the DOS at the Fermi energy (i.e., the
y-intercept of the plot) appears to be computed
accurately, with 90% of the materials having an
absolute prediction error under 0:02 states/eV/atom.
This allows for the accurate characterization of
material’s conductive nature (i.e., as insulators,
semiconductors, or conductors).
Table I shows the variation in the average MAE
for the AE-ALIGNN model with the size of the
encoded representation. It is important to note that
the model complexity (i.e., number of parameters) is
directly related to the size of the low-dimensional
representation used as a target for the model (recall



that the output is an FC layer with the same size as
the target vector). As such, it is desirable to choose a
latent dimension that is as small as possible to avoid
a potential model overfit, but large enough to
capture the complexity of the spectral output. As
seen from Table I, a latent vector of length 12 offers
an optimal choice for adequately representing the
DOS spectra in the dataset, since further improvement in the accuracy of the DOS prediction beyond
this number of dimensions is fairly limited. Additionally, the average MAE of the AE-ALIGNN
models appears to be higher than the D-ALIGNN
model. This can be attributed to the loss in information through the compression achieved by the
autoencoder. On average, the reconstruction MAE
of the autoencoder compression for the case with a


1402 Kaundinya, Choudhary, and Kalidindi



latent dimension of 12 was found to be 0:003 states/
eV/electron, which contributed to the higher error in
the AE-ALIGNN models than the D-ALIGNN
model.


Table I. The variation of the average MAE for the
test predictions from the AE-ALIGNN model as a
function of number of autoencoder features used
for the representation of the DOS


Number of dimensions Average MAE


20 0.0112

16 0.0130

12 0.0134

8 0.019



Figure 5 depicts the results of the predictions of
the AE-ALIGNN model with a target dimension of
12 on the materials in the test set. As seen from
Fig. 5a, the first quartile spans a larger range of
MAE values in comparison with the D-ALIGNN
model, indicating that the D-ALIGNN is more
accurate for a larger number of samples. This
reaffirms the hypothesis that the reconstruction
error leads to loss of information and consequently a
higher average MAE in prediction. Additionally, the
relative error histogram depicted in Fig. 5b is
similar to the trend observed for the D-ALIGNN
model, with 82% of the predictions of the DOS
spectra exhibiting an RE under 0:2. The plots of
DFT-computed and predicted DOS spectra of four
randomly chosen samples, one from each of the
quartiles in Fig. 5a, indicates that the AE-ALIGNN



Fig. 5. (a) The MAE values for the AE-ALIGNN predictions (with 12 latent dimensions) for the test set. (b) The relative errors in the test
predictions. (c) Four comparisons of the AE-ALIGNN predicted and DFT-computed DOS spectra from each quartile shown in (a).


Prediction of the Electron Density of States for Crystalline Compounds with Atomistic Line 1403
Graph Neural Networks (ALIGNN)


Fig. 6. A heatmap of the periodic table, with elements shaded by the average bandgap of binary compounds containing them as computed from
the DOS spectra predicted by D-ALIGNN.



model does indeed provide good predictions. However, it is also clear that there is a slight loss in
predictive accuracy of the AE-ALIGNN model compared to the D-ALIGNN model.
The models trained in this work can also provide
useful insights into the electronic properties (that
are derived from the DOS) of crystal structures and
their relationships to the constituent species. For
example, one of the primary criteria in selecting
materials for photovoltaic applications is a desirable
band gap range (usually between 1:4 and 1:6 eV).
The band gap for a non-conducting material may be
easily computed by calculating the energy difference
between the first non-zero DOS values on either
side of the y-axis in the DOS spectrum. In order to
demonstrate the utility of the developed ALIGNN
model to guide the search for materials with a
required bandgap, we consider all binary compounds occurring in the test set and perform
computations of the estimated bandgap using the
predicted DOS spectra of the materials. Figure 6
depicts a heatmap of the periodic table, with each
element displaying the average (non-zero) bandgap
of all binary compounds containing the selected
element. Note that elements not occurring in any
binary compound in our dataset are shaded gray. In
general, two major trends can be observed in the
Table. First, the average bandgap of binary



compounds having one element from groups (i.e.,
columns) 1–2 and 16–18 appears to be relatively
higher than those containing only transition metals
(i.e., d-block elements). This reaffirms the characteristic that binary crystals with a larger difference
in electronegativity between the two species typically have a larger bandgap. Interestingly, 8 of the
top 15 binary compounds with the largest bandgaps
were found to contain fluorine (the most electronegative element) as one of the elements. Second, the
average bandgap generally reduces with the
increased atomic number down a group, with materials consisting of at least one element belonging to
periods 1–3 demonstrating a higher bandgap. The
higher bandgap may be attributed to the smaller
atomic radius of these elements that contributes to
stronger bonds in the binary compound. Consequently, a larger amount of energy is required to
move electrons to the conduction band, thus increasing the bandgap in these materials. It is remarkable
that the D-ALIGNN model developed in this work
learned these insights implicitly in automated
protocols.


CONCLUSION


In this study, we have proposed and evaluated the
utility of ALIGNN models in predicting the DOS
spectra of crystal structures. Specifically, the


1404 Kaundinya, Choudhary, and Kalidindi



benefit of utilizing a graph-based featurization
scheme that captures directional information is
shown by demonstrating that the model can accurately predict the salient features of the DOS, which
is a physical property that is inherently dependent
on this information. We have also evaluated two
different representational approaches for the DOS,
a primitive discretization, and a compressed representation that is generated by using a separately
trained autoencoder. Although the D-ALIGNN
model performed better, both models exhibited
sufficient accuracy to be used for high-throughput
DOS spectrum predictions for new crystals. Most
importantly, both modeling frameworks are scalable
to include complex crystal systems that have varied
structural and chemical diversity.


ACKNOWLEDGEMENTS


P.R.K. and S.R.K. gratefully acknowledge support
from ONR N00014-18-1-2879. The Hive cluster at
Georgia Institute of Technology (supported by NSF
1828187) was used for this work. We would like to
thank Brian DeCost for the helpful discussion.


FUNDING


The authors declare that no known competing financial interests have influenced the work reported
in this paper.


CONFLICT OF INTEREST


The authors declare that there is no conflict of
interest.


REFERENCES


1. J. Lee, A. Seko, K. Shitara, K. Nakayama, and I. Tanaka,
[Phys. Rev. B. https://doi.org/10.1103/PhysRevB.93.115104](https://doi.org/10.1103/PhysRevB.93.115104)
(2016).
2. G. Pilania, A. Mannodi-Kanakkithodi, B.P. Uberuaga, R.
Ramprasad, J.E. Gubernatis, and T. Lookman, Sci. Rep. 6,
[19375. https://doi.org/10.1038/srep19375 (2016).](https://doi.org/10.1038/srep19375)
3. S. Kirklin, J.E. Saal, B. Meredig, A. Thompson, J.W. Doak,
[M. Aykol, S. Ru¨hl, and C. Wolverton, npj Comput. Mater. h](https://doi.org/10.1038/npjcompumats.2015.10)
[ttps://doi.org/10.1038/npjcompumats.2015.10 (2015).](https://doi.org/10.1038/npjcompumats.2015.10)
4. K. Choudhary, B. DeCost, and F. Tavazza, Phys. Rev. Mater.
[https://doi.org/10.1103/physrevmaterials.2.083801 (2018).](https://doi.org/10.1103/physrevmaterials.2.083801)
5. A.M. Deml, R. O’Hayre, C. Wolverton, and V. Stevanovic´,
[Phys. Rev. B. https://doi.org/10.1103/PhysRevB.93.085142](https://doi.org/10.1103/PhysRevB.93.085142)
(2016).
6. F. Faber, A. Lindmaa, O.A. von Lilienfeld, and R. Armiento,
[Int. J. Quantum Chem. 115, 1094. https://doi.org/10.1002/](https://doi.org/10.1002/qua.24917)
[qua.24917 (2015).](https://doi.org/10.1002/qua.24917)
7. P.R. Kaundinya, K. Choudhary, and S.R. Kalidindi, Phys.
[Rev. Mater. 5, 063802. https://doi.org/10.1103/PhysRevMate](https://doi.org/10.1103/PhysRevMaterials.5.063802)
[rials.5.063802 (2021).](https://doi.org/10.1103/PhysRevMaterials.5.063802)
8. W. Ye, C. Chen, Z. Wang, I.H. Chu, and S.P. Ong, Nat
[Commun 9, 3800. https://doi.org/10.1038/s41467-018-06322-](https://doi.org/10.1038/s41467-018-06322-x)
[x (2018).](https://doi.org/10.1038/s41467-018-06322-x)
9. L. Ward, R. Liu, A. Krishna, V.I. Hegde, A. Agrawal, A.
[Choudhary, and C. Wolverton, Phys. Rev. B. https://doi.org/](https://doi.org/10.1103/PhysRevB.96.024104)
[10.1103/PhysRevB.96.024104 (2017).](https://doi.org/10.1103/PhysRevB.96.024104)
10. A.N. Filanovich and A.A. Povzner, in 2020 Ural Symposium
on Biomedical Engineering, Radioelectronics and Information Technology (USBEREIT), (2020), pp. 0414–0416.



11. M. Rupp, A. Tkatchenko, K.R. Muller, and O.A. von
[Lilienfeld, Phys. Rev. Lett. 108, 058301. https://doi.org/10.](https://doi.org/10.1103/PhysRevLett.108.058301)
[1103/PhysRevLett.108.058301 (2012).](https://doi.org/10.1103/PhysRevLett.108.058301)
12. H. Huo and M. Rupp (2017).
13. L. Ward, B. Blaiszik, I. Foster, R.S. Assary, B. Narayanan,
[and L. Curtiss, MRS Commun. 9, 891. https://doi.org/10.15](https://doi.org/10.1557/mrc.2019.107)
[57/mrc.2019.107 (2019).](https://doi.org/10.1557/mrc.2019.107)
14. G. Montavon, K. Hansen, S. Fazli, M. Rupp, F. Biegler, A.
Ziehe, A. Tkatchenko, A. von Lilienfeld and K.-R. Mu¨ller
(2012), pp. 449–457.
15. D.M. Wilkins, A. Grisafi, Y. Yang, K.U. Lao, R.A. DiStasio,
[and M. Ceriotti, Proc. Natl. Acad. Sci. 116, 3401. https://doi.](https://doi.org/10.1073/pnas.1816132116)
[org/10.1073/pnas.1816132116 (2019).](https://doi.org/10.1073/pnas.1816132116)
16. L. Zhang, M. Chen, X. Wu, H. Wang, and R. Car, Phys. Rev.
[B 102, 041121. https://doi.org/10.1103/PhysRevB.102.04112](https://doi.org/10.1103/PhysRevB.102.041121)
[1 (2020).](https://doi.org/10.1103/PhysRevB.102.041121)
17. A. Grisafi, A. Fabrizio, B. Meyer, D.M. Wilkins, C.
[Corminboeuf, and M. Ceriotti, ACS Cent. Sci. 5, 57–64. h](https://doi.org/10.1021/acscentsci.8b00551)
[ttps://doi.org/10.1021/acscentsci.8b00551 (2019).](https://doi.org/10.1021/acscentsci.8b00551)
18. J.M. Alred, K.V. Bets, Y. Xie, and B.I. Yakobson, Compos.
[Sci. Technol. 166, 3–9. https://doi.org/10.1016/j.compscitech.](https://doi.org/10.1016/j.compscitech.2018.03.035)
[2018.03.035 (2018).](https://doi.org/10.1016/j.compscitech.2018.03.035)
19. A. Chandrasekaran, D. Kamal, R. Batra, C. Kim, L. Chen,
[and R. Ramprasad, npj Comput. Mater. 5, 22. https://doi.org/](https://doi.org/10.1038/s41524-019-0162-7)
[10.1038/s41524-019-0162-7 (2019).](https://doi.org/10.1038/s41524-019-0162-7)
20. F. Brockherde, L. Vogt, L. Li, M.E. Tuckerman, K. Burke,
[and K.-R. Mu¨ller, Nat. Commun. 8, 872. https://doi.org/10.](https://doi.org/10.1038/s41467-017-00839-3)
[1038/s41467-017-00839-3 (2017).](https://doi.org/10.1038/s41467-017-00839-3)
21. S. Gong, T. Xie, T. Zhu, S. Wang, E.R. Fadel, Y. Li, and J.C.
[Grossman, Phys. Rev. B 100, 184103. https://doi.org/10.110](https://doi.org/10.1103/PhysRevB.100.184103)
[3/PhysRevB.100.184103 (2019).](https://doi.org/10.1103/PhysRevB.100.184103)
22. C. Ben Mahmoud, A. Anelli, G. Csa´nyi, and M. Ceriotti,
[Phys. Rev. B 102, 235130. https://doi.org/10.1103/PhysRevB.](https://doi.org/10.1103/PhysRevB.102.235130)
[102.235130 (2020).](https://doi.org/10.1103/PhysRevB.102.235130)
23. [B.C. Yeo, D. Kim, C. Kim, and S.S. Han, Sci. Rep. 9, 5879. h](https://doi.org/10.1038/s41598-019-42277-9)
[ttps://doi.org/10.1038/s41598-019-42277-9 (2019).](https://doi.org/10.1038/s41598-019-42277-9)
24. K. Bang, B.C. Yeo, D. Kim, S.S. Han, and H.M. Lee, Sci.
[Rep. 11, 11604. https://doi.org/10.1038/s41598-021-91068-8](https://doi.org/10.1038/s41598-021-91068-8)
(2021).
25. P. Borlido, J. Schmidt, A.W. Huran, F. Tran, M.A.L. Mar[ques, and S. Botti, npj Comput. Mater. 6, 96. https://doi.org/](https://doi.org/10.1038/s41524-020-00360-0)
[10.1038/s41524-020-00360-0 (2020).](https://doi.org/10.1038/s41524-020-00360-0)
26. [J. Singh, J. Non-Cryst. Solids 299–302, 444. https://doi.org/](https://doi.org/10.1016/S0022-3093(01)00957-7)
[10.1016/S0022-3093(01)00957-7 (2002).](https://doi.org/10.1016/S0022-3093(01)00957-7)
27. [K.F. Garrity, Phys. Rev. B 94, 045122. https://doi.org/10.11](https://doi.org/10.1103/PhysRevB.94.045122)
[03/PhysRevB.94.045122 (2016).](https://doi.org/10.1103/PhysRevB.94.045122)
28. G. Hamaoui, N. Horny, Z. Hua, T. Zhu, J.-F. Robillard, A.
[Fleming, H. Ban, and M. Chirtoc, Sci. Rep. 8, 11352. http](https://doi.org/10.1038/s41598-018-29505-4)
[s://doi.org/10.1038/s41598-018-29505-4 (2018).](https://doi.org/10.1038/s41598-018-29505-4)
29. [Z. Lin, L.V. Zhigilei, and V. Celli, Phys. Rev. B 77, 075133. h](https://doi.org/10.1103/PhysRevB.77.075133)
[ttps://doi.org/10.1103/PhysRevB.77.075133 (2008).](https://doi.org/10.1103/PhysRevB.77.075133)
30. K.T. Schu¨tt, H. Glawe, F. Brockherde, A. Sanna, K.R.
[Mu¨ller, and E.K.U. Gross, Phys. Rev. B 89, 205118. https://d](https://doi.org/10.1103/PhysRevB.89.205118)
[oi.org/10.1103/PhysRevB.89.205118 (2014).](https://doi.org/10.1103/PhysRevB.89.205118)
31. S.R. Broderick, and K. Rajan, EPL (Europhysics Letters) 95,
[57005. https://doi.org/10.1209/0295-5075/95/57005 (2011).](https://doi.org/10.1209/0295-5075/95/57005)
32. A.P. Barto´k, R. Kondor, and G. Csa´nyi, Phys. Rev. B 87,
[184115. https://doi.org/10.1103/PhysRevB.87.184115 (2013).](https://doi.org/10.1103/PhysRevB.87.184115)
33. A. Cecen, T. Fast, and S.R. Kalidindi, Integr. Mater. Manuf.
Innov. 5, 1. [https://doi.org/10.1186/s40192-015-0044-x](https://doi.org/10.1186/s40192-015-0044-x)
(2016).
34. S. Kalidindi, Hierarchical Materials Informatics: Novel
Analytics for Materials Data, (2015).
35. [S.R. Kalidindi, ISRN Mater. Sci. 2012, 305692. https://doi.](https://doi.org/10.5402/2012/305692)
[org/10.5402/2012/305692 (2012).](https://doi.org/10.5402/2012/305692)
36. T. Xie, and J.C. Grossman, Phys. Rev. Lett. 120, 145301.
(2018).
37. [C.W. Park, and C. Wolverton, Phys. Rev. Mat. 4, 063801. h](https://doi.org/10.1103/PhysRevMaterials.4.063801)
[ttps://doi.org/10.1103/PhysRevMaterials.4.063801 (2020).](https://doi.org/10.1103/PhysRevMaterials.4.063801)
38. M. Karamad, R. Magar, Y. Shi, S. Siahrostami, I.D. Gates,
[and A. Barati Farimani, Phys. Rev. Mater. 4, 093801. http](https://doi.org/10.1103/PhysRevMaterials.4.093801)
[s://doi.org/10.1103/PhysRevMaterials.4.093801 (2020).](https://doi.org/10.1103/PhysRevMaterials.4.093801)


Prediction of the Electron Density of States for Crystalline Compounds with Atomistic Line 1405
Graph Neural Networks (ALIGNN)



39. S.-Y. Louis, Y. Zhao, A. Nasiri, X. Wang, Y. Song, F. Liu, and
[J. Hu, Phys. Chem. Chem. Phys. 22, 18141. https://doi.org/](https://doi.org/10.1039/D0CP01474E)
[10.1039/D0CP01474E (2020).](https://doi.org/10.1039/D0CP01474E)
40. [K. Choudhary, and B. DeCost, npj Comput. Mater. 7, 185. h](https://doi.org/10.1038/s41524-021-00650-1)
[ttps://doi.org/10.1038/s41524-021-00650-1 (2021).](https://doi.org/10.1038/s41524-021-00650-1)
41. K. Choudhary, I. Kalish, R. Beams, and F. Tavazza, Sci.
Rep. 7, 1. (2017).
42. K. Choudhary, and F. Tavazza, Comput. Mater. Sci. 161,
300. (2019).
43. K. Choudhary, Q. Zhang, A.C. Reid, S. Chowdhury, N. Van
Nguyen, Z. Trautt, M.W. Newrock, F.Y. Congo, and F.
Tavazza, Sci. Data 5, 180082. (2018).
44. K. Choudhary, K.F. Garrity, and F. Tavazza, J. Phys. Con[dens. Matter 32, 475501. https://doi.org/10.1088/1361-648x/](https://doi.org/10.1088/1361-648x/aba06b)
[aba06b (2020).](https://doi.org/10.1088/1361-648x/aba06b)



45. K. Choudhary, K.F. Garrity, A.C.E. Reid, B. De Cost, A.J.
Biacchi, A.R. Hight Walker, Z. Trautt, J. Hattrick-Simpers,
A.G. Kusne, A. Centrone, A. Davydov, J. Jiang, R. Pachter,
G. Cheon, E. Reed, A. Agrawal, X. Qian, V. Sharma, H.
Zhuang, S.V. Kalinin, B.G. Sumpter, G. Pilania, P. Acar, S.
Mandal, K. Haule, D. Vanderbilt, K. Rabe and F. Tavazza,
[npj Comput. Mater. 6, 173 (2020). doi:https://doi.org/10.103](https://doi.org/10.1038/s41524-020-00440-1)
[8/s41524-020-00440-1.](https://doi.org/10.1038/s41524-020-00440-1)
46. [Y. Wang, H. Yao, and S. Zhao, Neurocomput 184, 232–242. h](https://doi.org/10.1016/j.neucom.2015.08.104)
[ttps://doi.org/10.1016/j.neucom.2015.08.104 (2016).](https://doi.org/10.1016/j.neucom.2015.08.104)


Publisher’s Note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional
affiliations.


